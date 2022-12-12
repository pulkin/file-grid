import argparse
import json
import logging
import shutil
import subprocess
import sys
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
arg_parser.add_argument("-f", "--files", nargs="+", help="files to be processed", metavar="FILE", default=tuple())
arg_parser.add_argument("-t", "--static", nargs="+", help="files to be copied", metavar="FILE", default=tuple())
arg_parser.add_argument("-r", "--recursive", help="visit sub-folders when matching file names", action="store_true")
arg_parser.add_argument("-n", "--name", help="grid folder naming pattern", metavar="PATTERN", default="grid%d")
arg_parser.add_argument("-m", "--max", help="maximum allowed grid size", metavar="N", default=10_000)
arg_parser.add_argument("-s", "--state", help="state file name", metavar="FILE", default=".grid")
arg_parser.add_argument("-l", "--log", help="log file name", metavar="FILE", default=".grid.log")
arg_parser.add_argument("--root", help="root folder for scanning/placing grid files", default=".")
arg_parser.add_argument("action", help="action to perform", choices=["new", "run", "cleanup", "distribute"])
arg_parser.add_argument("extra", nargs="*", help="extra action arguments for 'run' and 'distribute'")


class Engine:
    def __init__(self, action, extra, template_files, static_files, root, recursive, name, max_size, state_fn, log_fn):
        self.action = action
        self.extra = extra
        self.template_files = template_files
        self.static_files = static_files
        self.root = root
        self.recursive = recursive
        self.name = name
        self.max_size = max_size
        self.state_fn = state_fn
        self.log_fn = log_fn

    @classmethod
    def from_argparse(cls, options):
        return cls(
            action=options.action,
            extra=options.extra,
            template_files=options.files,
            static_files=options.static,
            root=options.root,
            recursive=options.recursive,
            name=options.name,
            max_size=options.max,
            state_fn=options.state,
            log_fn=options.log,
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

    def folder_name(self, index):
        """Folder name convention"""
        return self.name % index

    def match_static(self):
        logging.info("Matching static files")
        result = match_files(self.static_files, allow_empty=True, recursive=self.recursive)
        for i in result:
            logging.info(f"  {str(i)}")
        logging.info(f"Total: {len(result)} files")
        return result

    def match_templates(self, exclude):
        logging.info("Matching template files")
        request = []
        if self.action == "distribute":
            request = (*self.template_files, *self.extra)
        elif self.action == "new":
            request = self.template_files
            if len(request) == 0:
                request = "*",
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
            if len(statement.names_missing(builtins)) == 0 and self.action != "distribute":
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

    def check_folder_conflicts(self, *args):
        for i in range(*args):
            p = Path(self.name % i)
            if p.exists():
                raise RuntimeError(f"file or folder '{str(p)}' already exists")

    def run_new(self, builtins=builtins):
        """
        Performs the new action.

        Creates an array of grid folders.
        """
        logging.info("Creating a new grid")

        files_static = self.match_static()
        files_grid = self.match_templates(files_static)
        statements = self.collect_statements(files_grid)

        reserved_names = set(builtins) | {"__grid_folder_name__", "__grid_id__"}
        overlap = set(statements).intersection(reserved_names)
        if len(overlap) > 0:
            raise ValueError(f"the following names used in the grid are reserved: {', '.join(overlap)}")

        statements_core, statements_dependent, total = self.group_statements(statements)

        # Read previous run
        self.check_folder_conflicts(total)
        grid_state = {"grid": {}, "names": list(statements)}
        index = 0

        if len(statements_core) == 0:
            warn(f"No fixed groups found")

        # Figure out order
        ordered_statements = eval_sort(statements_dependent, reserved_names | set(statements_core))
        # Add variables template file
        files_grid.append(variable_list_template(sorted(statements.keys())))
        # Iterate over possible combinations and write a grid
        for stack in combinations(statements_core):
            scratch = self.folder_name(index)
            stack["__grid_folder_name__"] = scratch
            stack["__grid_id__"] = index

            values = eval_all(ordered_statements, {**stack, **builtins})
            stack.update({statement.name: v for statement, v in zip(ordered_statements, values)})
            grid_state["grid"][scratch] = {"stack": stack}
            logging.info(f"  composing {scratch}")
            write_grid(scratch, stack, files_static, files_grid, self.root)
            index += 1

        # Save state
        self.save_state(grid_state)

    def run_distribute(self):
        """
        Runs grid distribute.

        Distributes one or many files over the existing grid.
        """
        logging.info("Distributing over the existing grid")
        current_state = self.load_state()
        logging.info(f"   grid folders: {len(current_state['grid'])}")

        files_static = self.match_static()
        files_grid = self.match_templates(files_static)
        statements = self.collect_statements(files_grid)

        reserved_names = set(builtins) | {"__grid_folder_name__", "__grid_id__"}
        overlap = set(statements).intersection(reserved_names)
        if len(overlap) > 0:
            raise ValueError(f"the following names used in the grid are reserved: {', '.join(overlap)}")

        statements_core, statements_dependent, total = self.group_statements(statements)
        if len(statements_core) > 0:
            raise ValueError(f"(new) core statements are not allowed when distributing: {', '.join(statements_core)}")
        overlap = set(current_state["names"]).intersection(set(statements))
        if len(overlap) > 0:
            raise ValueError(f"new names overlap with the ones previously defined: {', '.join(overlap)}")
        if len(statements_dependent) == 0:
            warn("No dependent statements found. File(s) will be distributed as-is.")

        logging.info(f"Distributing files into {len(current_state['grid'])} folders")
        exceptions = []

        # Figure out order
        ordered_statements = eval_sort(statements_dependent, set(reserved_names | set(current_state["names"])))

        for k, v in current_state["grid"].items():
            if not Path(k).is_dir():
                logging.exception(f"Grid folder {k} does not exist")
                exceptions.append(FileNotFoundError(f"No such file or directory: {repr(k)}"))
                continue
            try:
                stack = v["stack"]
                values = eval_all(ordered_statements, v["stack"])
                stack.update({statement.name: v for statement, v in zip(ordered_statements, values)})
                write_grid(k, stack, files_static, files_grid, self.root)
            except Exception as e:
                exceptions.append(e)
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
        for cwd in current_state["grid"]:
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
        current_state = self.load_state()
        logging.info("Removing grid folders")
        exceptions = []
        for f in current_state["grid"]:
            try:
                shutil.rmtree(f)
                logging.info(f"  {f}")
            except Exception as e:
                exceptions.append(e)
                logging.exception(f"Error while removing {f}")
        if len(exceptions):
            logging.error(f"{len(exceptions)} errors occurred while removing grid folders")
        logging.info("Removing the data file")
        Path(self.state_fn).unlink()
        if len(exceptions):
            raise exceptions[-1]

    def run(self):
        self.setup_logging()
        if self.action == "new":
            self.run_new()
        elif self.action == "distribute":
            self.run_distribute()
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

    if options.action == "new":
        if len(options.extra) > 0:
            arg_parser.error("usage: grid new (with no extra arguments)")
    elif options.action == "run":
        if len(options.extra) == 0:
            arg_parser.error("usage: grid run COMMAND")
    elif options.action == "cleanup":
        if len(options.extra) > 0:
            arg_parser.error("usage: grid cleanup (no extra arguments)")
    elif options.action == "distribute":
        if len(options.extra) == 0:
            arg_parser.error("usage: grid distribute FILE [FILE ...]")

    return Engine.from_argparse(options).run()
