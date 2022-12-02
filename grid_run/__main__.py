#!/usr/bin/env python3
import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from warnings import warn

from .algorithm import eval_sort, eval_all
from .tools import combinations
from .template import EvalBlock
from .grid_builtins import builtins
from .files import match_files, match_template_files, write_grid

filename_data = ".grid"
filename_log = ".grid.log"
root = "."

logging.basicConfig(filename=filename_log, filemode="w", level=logging.INFO)

parser = argparse.ArgumentParser(description="Creates an array [grid] of similar jobs and executes [submits] them")
parser.add_argument("-f", "--files", nargs="+", help="files to be processed", metavar="FILENAME")
parser.add_argument("-t", "--static", nargs="+", help="files to be copied", metavar="FILENAME")
parser.add_argument("-n", "--name", help="grid folder naming pattern", metavar="STRING")
parser.add_argument("-g", "--target", help="target tolerance for optimization", metavar="FLOAT", type=float)
parser.add_argument("action", help="action to perform", choices=["new", "run", "cleanup", "distribute"])
parser.add_argument("command", nargs="*", help="command to execute for 'run' action")

options = parser.parse_args()
logging.info(' '.join(sys.argv))


def get_grid_state():
    """Reads the grid state"""
    try:
        with open(filename_data, "r") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Grid file does not exit: {repr(e.filename)}") from e


def save_grid_state(state):
    """Saves the grid state"""
    with open(filename_data, "w") as f:
        json.dump(state, f, indent=4)


def folder_name(index):
    """Folder name convention"""
    return options.name % index


# ----------------------------------------------------------------------
#   New grid, distribute
# ----------------------------------------------------------------------

if options.action in ("new", "distribute"):

    # Errors
    if len(options.command) == 0 and options.action == "distribute":
        parser.error("nothing to distribute")

    # Defaults

    if not options.static:
        options.static = []

    if not options.name:
        options.name = 'grid%d'

    if options.action == "distribute":
        if not os.path.exists(filename_data):
            print("No existing grid found.")
            logging.error("Grid does not exist")
            sys.exit(1)

    else:
        if os.path.exists(filename_data):
            print(
                "The grid is already present in this folder. Use 'grid cleanup' to cleanup previous run or -c to continue previous run.")
            logging.error("Previous run found, exiting")
            sys.exit(1)

    # ------------------------------------------------------------------
    #   Common part
    # ------------------------------------------------------------------

    # Match static items
    logging.info("Matching static part")
    files_static = match_files(options.static, allow_empty=True)
    for i in files_static:
        logging.info(f"  {str(i)}")
    logging.info("Total: {n} items".format(n=len(files_static)))

    if options.files or options.action == "distribute":
        # Match files
        logging.info("Files provided explicitly")
        files_grid = match_template_files(options.command if options.action == "distribute" else options.files,
                                          exclude=files_static)

    else:
        # Find files, default behavior
        logging.info("Searching for grid files in current folder")
        files_grid = match_template_files(["*"], exclude=files_static)

    for i in files_grid:
        logging.info(f"  {i}")
    logging.info(f"Total: {len(files_grid)} files")

    # Collect all statements into dict
    logging.info("Processing grid-formatted files")

    statements = {}
    for grid_file in files_grid:
        for chunk in grid_file.chunks:
            if isinstance(chunk, EvalBlock):
                if chunk.name in statements:
                    raise ValueError(f"duplicate statement {chunk} (also {statements[chunk.name]}")
                else:
                    statements[chunk.name] = chunk

    reserved_names = set(builtins) | {"__grid_folder_name__", "__grid_id__"}
    overlap = set(statements).intersection(reserved_names)
    if len(overlap) > 0:
        raise ValueError(f"the following names used in the grid are reserved: {', '.join(overlap)}")

    # Split statements by type
    total = 1
    statements_core = {}
    statements_dependent = {}

    for name, statement in statements.items():
        logging.info(repr(statement))
        if len(statement.names_missing(builtins)) == 0 and options.action != "distribute":
            logging.info("  core, evaluating ...")
            result = statement.eval(builtins)
            if "__len__" not in dir(result):
                result = [result]
            logging.info(f"  result: {result} (len={len(result)})")
            total = total * len(result)
            if total > 1e6:
                raise RuntimeError(f"grid size is too large: {total}")
            statements_core[name] = result

        else:
            logging.info(f"  depends on: {', '.join(map(repr, statement.required))}")
            statements_dependent[name] = statement
    logging.info(f"Total: {len(statements_core)} core statement(s) ({total} combination(s)), "
                 f"{len(statements_dependent)} dependent statement(s)")

    # Read previous run
    if options.action == "distribute":
        grid_state = get_grid_state()
        logging.info("Continue with previous {n} instances".format(n=len(grid_state["grid"])))

        overlap = set(grid_state["names"]).intersection(set(statements))
        if len(overlap) > 0:
            raise ValueError(f"new statement names overlap with previously defined ones: {', '.join(overlap)}")
    else:
        grid_state = {"grid": {}, "names": list(statements)}

    index = len(grid_state["grid"])

    # Check if folders already exist
    # TODO: this checks if there are any folders starting with [former] prefix
    # for x in glob.glob(options.prefix + "*"):
    #     if not x in grid_state["grid"]:
    #         print(
    #             "File or folder {name} may conflict with the grid. Either remove it or use a different prefix through '--prefix' option.".format(
    #                 name=x))
    #         logging.error("{name} already exists".format(name=x))
    #         sys.exit(1)

    # ------------------------------------------------------------------
    #   New
    # ------------------------------------------------------------------

    if options.action == "new":
        if len(statements_core) == 0:
            warn(f"No fixed groups found")

        # Figure out order
        ordered_statements = eval_sort(statements_dependent, reserved_names | set(statements_core))
        # Iterate over possible combinations and write a grid
        for stack in combinations(statements_core):
            scratch = folder_name(index)
            stack["__grid_folder_name__"] = scratch
            stack["__grid_id__"] = index

            values = eval_all(ordered_statements, {**stack, **builtins})
            stack.update({statement.name: v for statement, v in zip(ordered_statements, values)})
            grid_state["grid"][scratch] = {"stack": stack}
            logging.info(f"  composing {scratch}")
            write_grid(scratch, stack, files_static, files_grid, root)
            index += 1

        # Save state
        save_grid_state(grid_state)

    # ------------------------------------------------------------------
    #   Distribute
    # ------------------------------------------------------------------

    elif options.action == "distribute":
        assert len(statements_core) == 0
        if len(statements_dependent) == 0:
            warn("No dependent statements found. File(s) will be distributed as-is.")

        logging.info(f"Distributing files into {len(grid_state)} folders")
        exceptions = []

        # Figure out order
        ordered_statements = eval_sort(statements_dependent, set(grid_state["names"]))

        for k, v in grid_state["grid"].items():
            if not Path(k).is_dir():
                logging.exception(f"Grid folder {k} does not exist")
                exceptions.append(FileNotFoundError(f"No such file or directory: {repr(k)}"))
            else:
                stack = v["stack"]
                values = eval_all(ordered_statements, v["stack"])
                stack.update({statement.name: v for statement, v in zip(ordered_statements, values)})
                write_grid(k, stack, files_static, files_grid, root)
        if len(exceptions) > 0:
            raise exceptions[-1]

# ----------------------------------------------------------------------
#   Execute in context of grid
# ----------------------------------------------------------------------

elif options.action == "run":
    if options.files or options.static or options.name or options.target:
        parser.error(f"-f, --files, -t, --static-files, -n, --name, -g, --target options "
                     f"are irrelevant to {repr(options.action)}")
    if len(options.command) == 0:
        parser.error("missing command to run")

    current_state = get_grid_state()
    logging.info(f"Executing {' '.join(options.command)} in {len(current_state['grid'])} grid folders")
    exceptions = []
    for cwd in current_state["grid"]:

        try:
            print(cwd)
            print(subprocess.check_output(options.command, cwd=cwd, stderr=subprocess.PIPE, text=True))
        except (FileNotFoundError, subprocess.SubprocessError) as e:
            print(f"Failed to execute {' '.join(options.command)} (working directory {repr(cwd)})")
            logging.exception(f"Failed to execute {' '.join(options.command)} (working directory {repr(cwd)})")
            exceptions.append(e)
    if len(exceptions) > 0:
        raise exceptions[-1]

# ----------------------------------------------------------------------
#   Cleanup grid
# ----------------------------------------------------------------------

elif options.action == "cleanup":
    if options.files or options.static or options.name or options.target:
        parser.error(f"-f, --files, -t, --static-files, -n, --name, -g, --target options "
                     f"are irrelevant to {repr(options.action)}")

    current_state = get_grid_state()
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
        logging.error(f"{len(exceptions)} exceptions occurred while removing grid folders")
    logging.info("Removing the data file")
    os.remove(filename_data)
    if len(exceptions):
        raise exceptions[-1]


def dummy():
    pass  # TODO: remove dummy function needed for console_scripts
