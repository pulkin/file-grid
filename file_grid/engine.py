import sys
import argparse
import json
import logging
import subprocess
from pathlib import Path
from functools import reduce
from operator import mul
from warnings import warn

from .algorithm import eval_sort, eval_all
from .tools import combinations
from .template import EvalBlock, variable_list_template
from .grid_builtins import builtins
from .files import match_files, match_template_files, write_grid

arg_parser = argparse.ArgumentParser(description="Creates arrays [grids] of similar files and folders")
arg_parser.add_argument("-t", "--static", nargs="+", help="files to be copied", metavar="FILE", default=tuple())
arg_parser.add_argument("-r", "--recursive", help="visit sub-folders when matching file names", action="store_true")
arg_parser.add_argument("-p", "--pattern", help="naming pattern", metavar="PATTERN", default="grid{id}/{name}")
arg_parser.add_argument("-m", "--max", help="maximum allowed grid size", metavar="N", default=10_000)
arg_parser.add_argument("-f", "--force", help="force overwrite", action="store_true")
arg_parser.add_argument("-d", "--dry", help="dry run", action="store_true")
arg_parser.add_argument("-e", "--exec", nargs="+", help="execute per grid")
arg_parser.add_argument("--list", help="save list of created files and folders", metavar="FILE", default=".grid")
arg_parser.add_argument("--json", help="save json file with information", metavar="FILE", default=".grid.json")
arg_parser.add_argument("--log", help="save log file", metavar="FILE", default=".grid.log")
arg_parser.add_argument("--root", help="root folder for scanning/placing grid files", default=".")
arg_parser.add_argument("action", help="action to perform", choices=["new", "run", "cleanup"])
arg_parser.add_argument("extra", nargs="*", help="extra action arguments for 'new' and 'run'")


class Engine:
    def __init__(self, action, extra, static_files, root, recursive, naming_pattern, max_size, list_fn, state_fn,
                 log_fn, force_overwrite, dry_run, do_exec):
        self.action = action
        self.extra = extra
        self.static_files = static_files
        self.root = root
        self.recursive = recursive
        self.naming_pattern = naming_pattern
        self.max_size = max_size
        self.list_fn = list_fn
        self.state_fn = state_fn
        self.log_fn = log_fn
        self.force_overwrite = force_overwrite
        self.dry_run = dry_run
        self.do_exec = do_exec

    @classmethod
    def from_argparse(cls, options):
        return cls(
            action=options.action,
            extra=options.extra,
            static_files=options.static,
            root=options.root,
            recursive=options.recursive,
            naming_pattern=options.pattern,
            max_size=options.max,
            list_fn=options.list,
            state_fn=options.json,
            log_fn=options.log,
            force_overwrite=options.force,
            dry_run=options.dry,
            do_exec=options.exec,
        )

    def setup_logging(self):
        logging.basicConfig(filename=self.log_fn, filemode="w", level=logging.INFO)

    def load_state(self):
        """Reads the grid state"""
        logging.info(f"Loading grid state from '{self.state_fn}'")
        try:
            with open(self.state_fn, "r") as f:
                return json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Grid file does not exit: {repr(e.filename)}") from e

    def save_state(self, state):
        """Saves the grid state"""
        with open(self.state_fn, "w") as f:
            json.dump(state, f, indent=4)

    def save_paths(self, paths):
        """Saves path list"""
        if len(paths) > 0:
            with open(self.list_fn, 'a') as f:
                for i in paths:
                    f.write(str(i.absolute()) + "\n")

    def load_paths(self):
        """Loads path list"""
        logging.info(f"Loading grid state from '{self.state_fn}'")
        try:
            with open(self.list_fn, 'r') as f:
                return list(i[:-1] for i in f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Grid file does not exit: {repr(e.filename)}") from e

    def match_static(self):
        logging.info("Matching static files")
        result = match_files(self.static_files, allow_empty=True, recursive=self.recursive)
        for i in result:
            logging.info(f"  {str(i)}")
        logging.info(f"Total: {len(result)} files")
        return result

    def match_templates(self, exclude):
        logging.info("Matching template files")
        if len(self.extra) == 0:
            request = "*",
        else:
            request = self.extra
        result = match_template_files(request, recursive=self.recursive, exclude=exclude)
        for i in result:
            logging.info(f"  {str(i)}")
        logging.info(f"Total: {len(result)} files")
        return result

    @staticmethod
    def collect_statements(files_grid):
        statements = {}
        for grid_file in files_grid:
            for chunk in grid_file.chunks:
                if isinstance(chunk, EvalBlock):
                    if chunk.name in statements:
                        raise ValueError(f"duplicate statement {chunk} (also {statements[chunk.name]}")
                    else:
                        statements[chunk.name] = chunk
        return statements

    def group_statements(self, statements):
        statements_core = {}
        statements_dependent = {}

        for name, statement in statements.items():
            logging.info(repr(statement))
            if len(statement.names_missing(builtins)) == 0:
                logging.info("  core, evaluating ...")
                result = statement.eval(builtins)
                if "__len__" not in dir(result):
                    result = [result]
                logging.info(f"  result: {result} (len={len(result)})")
                statements_core[name] = result

            else:
                logging.info(f"  depends on: {', '.join(map(repr, statement.required))}")
                statements_dependent[name] = statement
        total = reduce(mul, map(len, statements_core.values())) if len(statements_core) else 1
        logging.info(f"Total: {len(statements_core)} core statement(s) ({total} combination(s)), "
                     f"{len(statements_dependent)} dependent statement(s)")
        if total > self.max_size:
            raise RuntimeError(f"the total grid size {total} is above threshold {self.max_size}")
        return statements_core, statements_dependent, total

    def run_new(self, builtins=builtins):
        """
        Performs the new action.

        Creates an array of grid files.
        """
        logging.info("Creating a new grid")

        files_static = self.match_static()
        files_grid = self.match_templates(files_static)
        statements = self.collect_statements(files_grid)

        reserved_names = set(builtins) | {"__grid_id__"}
        overlap = set(statements).intersection(reserved_names)
        if len(overlap) > 0:
            raise ValueError(f"the following names used in the grid are reserved: {', '.join(overlap)}")

        statements_core, statements_dependent, total = self.group_statements(statements)

        # Read previous run
        grid_state = {"grid": [], "names": list(statements)}

        if len(statements_core) == 0:
            warn(f"No fixed groups found")

        # Figure out order
        ordered_statements = eval_sort(statements_dependent, reserved_names | set(statements_core))
        # Add variables template file
        files_grid.append(variable_list_template(sorted(statements.keys())))
        # Iterate over possible combinations and write a grid
        files_created = []
        exceptions = []
        for index, stack in enumerate(combinations(statements_core)):
            scratch = str(Path(self.naming_pattern.format(id=index, name="")))
            stack["__grid_id__"] = index

            values = eval_all(ordered_statements, {**stack, **builtins})
            stack.update({statement.name: v for statement, v in zip(ordered_statements, values)})
            grid_state["grid"].append({"stack": stack, "location": scratch})
            logging.info(f"  composing {scratch}")
            files_created.extend(write_grid(self.naming_pattern.format(id=index, name="{name}"), stack, files_static,
                                            files_grid, self.root, self.force_overwrite, self.dry_run))
            if self.do_exec is not None:
                commands = tuple(i.format(id=index) for i in self.do_exec)
                commands_joined = ' '.join(commands)
                logging.info(f"  running {commands_joined}")
                try:
                    print(f"{index} > {commands_joined}")
                    subprocess.check_call(commands_joined, cwd=self.root, stdout=sys.stdout, stderr=sys.stderr, shell=True)

                except subprocess.CalledProcessError as e:
                    logging.exception(f"{commands_joined}: process error")
                    exceptions.append(e)

        if not self.dry_run:
            # Save state
            self.save_state(grid_state)
            # Save files created
            self.save_paths(files_created)

        if len(exceptions) > 0:
            raise exceptions[-1]

    def run_exec(self):
        """
        Performs the run action.

        Executes a command in all grid folders.
        """
        logging.info(f"Executing {' '.join(self.extra)}")
        current_state = self.load_state()
        logging.info(f"   grid folders: {len(current_state['grid'])}")
        exceptions = []
        for grid_info in current_state["grid"]:
            cwd = grid_info["location"]
            try:
                print(f'{cwd}: {" ".join(self.extra)}')
                print(subprocess.check_output(self.extra, cwd=cwd, stderr=subprocess.PIPE, text=True))

            except FileNotFoundError as e:
                print(f"{' '.join(self.extra)}: file not found (working directory {repr(cwd)})")
                logging.exception(f"{' '.join(self.extra)}: file not found (working directory {repr(cwd)})")
                exceptions.append(e)

            except subprocess.CalledProcessError as e:
                print(f"{' '.join(self.extra)}: process error (working directory {repr(cwd)})")
                print(e.stdout, end="")
                print(e.stderr, end="")
                logging.exception(f"{' '.join(self.extra)}: process error (working directory {repr(cwd)})")
                exceptions.append(e)
        if len(exceptions) > 0:
            raise exceptions[-1]

    def run_cleanup(self):
        """
        Performs the cleanup action.

        Removes all grid folders and grid state file.
        """
        logging.info("Cleaning up")
        exceptions = []
        processed = set()  # there may be duplicates
        for line in self.load_paths()[::-1]:
            if line not in processed:
                processed.add(line)
                path = Path(line)
                logging.info(f"  {str(path)}")
                if not self.dry_run:
                    try:
                        if path.is_dir():
                            path.rmdir()
                        else:
                            path.unlink()
                    except Exception as e:
                        exceptions.append(e)
                        logging.exception(f"Error while removing {str(path)}")
        if len(exceptions):
            logging.error(f"{len(exceptions)} errors occurred while removing grid files")
        logging.info("Removing the data file")
        if not self.dry_run:
            Path(self.state_fn).unlink()
            Path(self.list_fn).unlink()
        if len(exceptions):
            raise exceptions[-1]

    def run(self):
        self.setup_logging()
        if self.action == "new":
            self.run_new()
        elif self.action == "run":
            self.run_exec()
        elif self.action == "cleanup":
            self.run_cleanup()
        else:
            raise NotImplementedError(f"action '{self.action}' not implemented")


def grid_run(options=None):
    """Parses command line arguments and runs the desired grid action"""
    if options is None:
        options = arg_parser.parse_args()

    if options.action in ("new", "run"):
        if len(options.extra) == 0:
            arg_parser.error(f"usage: grid {options.action} COMMAND or FILE(s)")
    elif options.action == "cleanup":
        if len(options.extra) > 0:
            arg_parser.error("usage: grid cleanup (no extra arguments)")

    return Engine.from_argparse(options).run()
