#!/usr/bin/env python3
import argparse
import logging
import sys

from .engine import Engine


parser = argparse.ArgumentParser(description="Creates arrays [grids] of similar files and folders")
parser.add_argument("-f", "--files", nargs="+", help="files to be processed", metavar="FILE", default=tuple())
parser.add_argument("-t", "--static", nargs="+", help="files to be copied", metavar="FILE", default=tuple())
parser.add_argument("-r", "--recursive", help="visit sub-folders when matching file names", action="store_true")
parser.add_argument("-n", "--name", help="grid folder naming pattern", metavar="PATTERN", default="grid%d")
parser.add_argument("-m", "--max", help="maximum allowed grid size", metavar="N", default=10_000)
parser.add_argument("-s", "--state", help="state file name", metavar="FILE", default=".grid")
parser.add_argument("-l", "--log", help="log file name", metavar="FILE", default=".grid.log")
parser.add_argument("--root", help="root folder for scanning/placing grid files", default=".")
parser.add_argument("action", help="action to perform", choices=["new", "run", "cleanup", "distribute"])
parser.add_argument("extra", nargs="*", help="extra action arguments for 'run' and 'distribute'")

options = parser.parse_args()

if options.action == "new":
    if len(options.extra) > 0:
        parser.error("usage: grid new (with no extra arguments)")
elif options.action == "run":
    if len(options.extra) == 0:
        parser.error("usage: grid run COMMAND")
elif options.action == "cleanup":
    if len(options.extra) > 0:
        parser.error("usage: grid cleanup (no extra arguments)")
elif options.action == "distribute":
    if len(options.extra) == 0:
        parser.error("usage: grid distribute FILE [FILE ...]")

logging.basicConfig(filename=options.log, filemode="w", level=logging.INFO)
logging.info(' '.join(sys.argv))


grid_engine = Engine.from_argparse(options)

# ----------------------------------------------------------------------
#   New grid, distribute
# ----------------------------------------------------------------------

if options.action == "new":
    grid_engine.run_new()

elif options.action == "distribute":
    grid_engine.run_distribute()

elif options.action == "run":
    grid_engine.run_exec()

elif options.action == "cleanup":
    grid_engine.run_cleanup()


def dummy():
    pass  # TODO: remove dummy function needed for console_scripts
