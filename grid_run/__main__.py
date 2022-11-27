#!/usr/bin/env python3
import argparse
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from operator import attrgetter
from warnings import warn
from itertools import product

from pyparsing import *

filename_data = ".grid"
filename_log = ".grid.log"

logging.basicConfig(filename=filename_log, filemode="w", level=logging.INFO)

parser = argparse.ArgumentParser(description="Creates an array [grid] of similar jobs and executes [submits] them")
parser.add_argument("-f", "--files", nargs="+", help="files to be processed", metavar="FILENAME")
parser.add_argument("-t", "--static-files", nargs="+", help="files to be copied", metavar="FILENAME")
parser.add_argument("-n", "--name", help="grid folder naming pattern", metavar="STRING")
parser.add_argument("-g", "--target", help="target tolerance for optimization", metavar="FLOAT", type=float)
parser.add_argument("-c", action='store_true', help="continue optimization without removing existing grid state")
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

    if not options.static_files:
        options.static_files = []

    if not options.name:
        options.name = 'grid%d'

    if options.action == "distribute":
        if not os.path.exists(filename_data):
            print("No existing grid found.")
            logging.error("Grid does not exist")
            sys.exit(1)

    else:
        if os.path.exists(filename_data) and not options.c:
            print(
                "The grid is already present in this folder. Use 'grid cleanup' to cleanup previous run or -c to continue previous run.")
            logging.error("Previous run found, exiting")
            sys.exit(1)

    # ------------------------------------------------------------------
    #   Language
    # ------------------------------------------------------------------

    # Drivers

    def listRange(*args):
        return list(range(*args))


    def linspace(start, end, steps):
        result = []
        for i in range(steps):
            w = 1.0 * i / (steps - 1)
            result.append(start * (1 - w) + end * w)
        return result

    builtins = {"range": listRange, "linspace": linspace}

    from .language import Variable

    # Language definitions

    def debug(f, name):
        def __(tokens):
            try:
                result = f(tokens)
                return result
            except:
                logging.exception("Exception in unpacking " + name)
                raise

        return __


    Escaped = Suppress("\\") + Literal("{%")("escaped")
    VariableName = Word(alphas + "_" + nums)
    FieldName = Combine(VariableName + ZeroOrMore("." + VariableName))

    # GenericExpression

    GenericExpression = Forward()

    # Primitive types

    PythonQuotedString = (QuotedString('"', escChar="\\") | QuotedString("'", escChar="\\")).setParseAction(
        debug(lambda t: re.sub(r'\\(.)', r'\g<1>', t[0]), "PythonQuotedString"))
    Numeric = Word(nums)
    Integer = Combine(Optional(Literal('+') | Literal('-')) + Numeric).setParseAction(
        debug(lambda t: int(t[0]), "Integer"))
    Float = Combine(Optional(Literal('+') | Literal('-')) + Numeric + Optional("." + Optional(Numeric)) + Optional(
        CaselessLiteral('E') + Integer)).setParseAction(debug(lambda t: float(t[0]), "Float"))
    IorF = Integer ^ Float
    Var = VariableName.copy().setParseAction(debug(lambda t: Variable(t[0]), "Var"))
    Primitive = PythonQuotedString ^ Integer ^ Float ^ Var

    # Args and kwargs

    Arg = GenericExpression + ~FollowedBy("=")
    NamedArg = Group(VariableName("key") + Suppress("=") + GenericExpression("value"))

    Args = Group(Arg + ZeroOrMore(Suppress(",") + Arg))
    KwArgs = Group(NamedArg + ZeroOrMore(Suppress(",") + NamedArg)).setParseAction(
        debug(lambda t: dict((i.key, i.value) for i in t[0]), "KwArgs"))
    MixedArgs = Args("args") + Suppress(",") + KwArgs("kwargs")
    ArbitraryArgs = (MixedArgs | KwArgs("kwargs") | Args("args"))

    # Arrays

    PythonList = (Suppress("[") + Args + Suppress("]")).setParseAction(debug(lambda t: [list(t[0]), ], "PythonList"))


    # Function call
    def translateFunction(f):

        # Numpy functions
        if f.startswith("numpy"):

            # Try to import
            try:
                import numpy
            except ImportError:
                print("Could not import numpy: either install numpy or avoid using '{routine}' in grid files".format(
                    routine=f, ))
                logging.exception("Failed to import numpy")
                sys.exit(1)

            # Pick function
            try:
                return {"numpy": numpy.array, }[f]
            except KeyError:
                print("Could not find numpy function {routine}".format(routine=f, ))
                logging.exception("Failed to execute {routine}".format(routine=f, ))
                sys.exit(1)

        # Simple functions
        else:
            try:
                return {"range": listRange, "linspace": linspace}[f]
            except KeyError:
                print("Could not find function {routine}".format(routine=f, ))
                logging.exception("Failed to execute {routine}".format(routine=f, ))
                sys.exit(1)


    Callable = Group(FieldName("callable") + Suppress("(") + Optional(ArbitraryArgs) + Suppress(")")).setParseAction(
        debug(lambda t: [translateFunction(t[0]["callable"])(*(t[0]["args"] if "args" in t[0] else []),
            **(t[0]["kwargs"] if "kwargs" in t[0] else {}))], "Callable"))

    # Atomic types

    ValueInBrackets = Suppress("(") + GenericExpression + Suppress(")")
    AtomicExpression = Callable | PythonList | Primitive | ValueInBrackets


    # Atomic operations

    def evaluate(t):

        if not isinstance(t, ParseResults):
            return t

        if len(t) % 2 == 0:
            raise Exception("Cannot evaluate {expr}".format(expr=t, ))

        result = evaluate(t[0])

        for i in range(int((len(t) - 1) / 2)):

            operation = t[2 * i + 1]
            arg = t[2 * i + 2]

            if operation == "+":
                result = result + evaluate(arg)
            elif operation == "-":
                result = result - evaluate(arg)
            elif operation == "*":
                result = result * evaluate(arg)
            elif operation == "/":
                result = result / evaluate(arg)
            elif operation == "**":
                result = result ** evaluate(arg)
            else:
                raise ValueError("Unknown operation: " + str(operation))
        return result


    from pyparsing import infix_notation, opAssoc, oneOf, ParseResults

    GenericExpression << infix_notation(AtomicExpression, [("**", 2, opAssoc.RIGHT), (oneOf("* /"), 2, opAssoc.LEFT),
        (oneOf("+ -"), 2, opAssoc.LEFT), ]).setParseAction(debug(lambda x: [evaluate(x)], "GenericExpression"))

    # Grid group

    GroupHeader = Optional(
        Combine(VariableName("id") + ZeroOrMore(Suppress(":") + VariableName)("flags")) + Suppress("="))
    GroupStatement = Escaped("escaped") ^ (Suppress("{%") + GroupHeader + GenericExpression("choices") + Suppress("%}"))


    # ------------------------------------------------------------------
    #   Classes
    # ------------------------------------------------------------------

    from .parsing import split_assignment, iter_template_blocks
    import tempfile
    from types import FunctionType

    class Statement:
        def __init__(self, code, name=None, source="<unknown>"):
            """Represents an expression between {% %}"""
            self.code = code
            if name is None:
                name = str(uuid.uuid4())
            self.name = name
            self.source = source

        @classmethod
        def from_string(cls, text, debug=True, extra_s=""):
            name, text = split_assignment(text)
            if debug:
                f = tempfile.NamedTemporaryFile('w+')
                f.write(text)
                f.seek(0)
                f_name = f.name
            else:
                f_name = "<string>"
            code = compile(text, f_name, "eval")
            return cls(code, name, source=f"{text}{extra_s}")

        @property
        def required(self):
            return set(self.code.co_names)

        def names_missing(self, names):
            return set(self.required) - set(names)

        def eval(self, names):
            delta = self.names_missing(names)
            if len(delta) > 0:
                raise ValueError(f"missing following names: {', '.join(map(repr, delta))}")
            func = FunctionType(self.code, names)
            return func()

        def __str__(self):
            return f"{self.name} = {self.source}"

        def __repr__(self):
            return f"Statement({self.__str__()})"


    class GridFile:

        def __init__(self, name, chunks):
            self.name = name
            self.chunks = chunks

        @classmethod
        def from_text(cls, name, text):
            itr = iter_template_blocks(text)
            chunks = []
            for i in itr:
                chunks.append(i)
                try:
                    chunks.append(Statement.from_string(next(itr), extra_s=f" [defined in {repr(name)}]"))
                except StopIteration:
                    pass
            return cls(name, chunks)

        @classmethod
        def from_file(cls, f):
            return cls.from_text(f.name, f.read())

        def write(self, stack, f):
            for chunk in self.chunks:
                if isinstance(chunk, str):
                    f.write(chunk)
                elif isinstance(chunk, Statement):
                    f.write(str(stack[chunk.name]))  # TODO: proper formatting
                else:
                    raise NotImplementedError(f"unknown {chunk=}")

        def is_trivial(self):
            for chunk in self.chunks:
                if not isinstance(chunk, str):
                    return False
            return True

        def __repr__(self):
            return f"GridFile(name={repr(self.name)}, chunks=[{len(self.chunks)} chunks])"

    def combinations(n):
        keys = n.keys()
        for i in product(*n.values()):
            yield dict(zip(keys, i))


    def copy(s, d, dry=False):

        sh, st = os.path.split(s)

        if len(sh) > 0:
            d = os.path.join(d, sh)

        if not os.path.isdir(d):
            os.makedirs(d)

        if not dry:

            if os.path.isdir(s):
                shutil.copytree(s, os.path.join(d, st))

            elif os.path.isfile(s):
                shutil.copy2(s, d)


    def write_grid(directory_name, stack, files_static, files_grid):

        logging.info("Writing stack {stack} into {file}".format(stack=stack, file=directory_name))

        # Copy static
        logging.info("  Copying static part")

        for s in files_static:
            try:
                copy(s, directory_name)
                logging.info("    copying '{name}'".format(name=s))
            except Exception as e:
                print(e)
                logging.exception("Error while copying {name}".format(name=s))
                sys.exit(1)

        # Copy grid
        logging.info("  Copying grid-formatted")

        for f in files_grid:

            try:
                copy(f.name, directory_name, dry=True)
            except Exception as e:
                print(e)
                logging.exception("Error while creating directory tree for {file}".format(file=f.name))
                sys.exit(1)

            joined = os.path.join(directory_name, f.name)

            try:
                with open(joined, "w") as ff:
                    f.write(stack, ff)
            except Exception as e:
                print(e)
                logging.exception("Error while writing grid-formatted file {file}".format(file=f.name))
                sys.exit(1)

            try:
                shutil.copystat(f.name, joined)
            except Exception as e:
                print(e)
                logging.exception("Error while assigning file attributes")
                sys.exit(1)

            logging.info("    copying '{name}'".format(name=f.name))


    # ------------------------------------------------------------------
    #   Common part
    # ------------------------------------------------------------------

    # Match static items
    logging.info("Matching static part")

    files_static = set()
    for i in options.static_files:

        new = glob.glob(i)
        if len(new) == 0:
            print("File '{pattern}' not found or pattern matched 0 files".format(pattern=i))
            logging.error("File '{pattern}' provided but matched 0 files")
            sys.exit(1)

        for f in new:
            logging.info("  {name}".format(name=f))

        files_static.update(new)

    logging.info("Total: {n} items".format(n=len(files_static)))

    files = set()

    if options.files or options.action == "distribute":

        # Match files
        logging.info("The files are provided explicitly")

        file_list = options.command if options.action == "distribute" else options.files
        for i in file_list:

            new = set(glob.glob(i))
            new.difference_update(files_static)
            if len(new) == 0:
                print("File '{pattern}' not found or pattern matched 0 files"
                      " or all files declared as 'static'".format(pattern=i))
                logging.error("No grid files provided")
                sys.exit(1)

            files.update(new)
        for f in files:
            logging.info("  {name}".format(name=f))

    else:

        # Find files, default behavior
        logging.info("Searching for grid files in current folder")

        for i in os.listdir('.'):
            if os.path.isfile(i) and i not in files_static:
                logging.info(f"  {i}")
                with open(i, 'r') as f:
                    if GridFile.from_file(f).is_trivial():
                        logging.info("    does not contain templates")
                    else:
                        logging.info("    template file")
                        files.add(i)

        if len(files) == 0:
            print("No grid-formatted files found in this folder")
            logging.error("No grid files found in . folder")
            sys.exit(1)

    logging.info("Total: {n} files".format(n=len(files)))

    files_static = sorted(files_static)
    files = sorted(files)

    # Read files

    files_grid = []
    for f in files:
        try:
            with open(f, 'r') as fl:
                files_grid.append(GridFile.from_file(fl))
        except IOError as e:
            print(e)
            logging.exception("Error during reading the file {name}".format(name=f))
            sys.exit(1)
    logging.info("All files have been successfully read")

    # Collect all statements into dict
    logging.info("Processing grid-formatted files")

    statements = {}
    for grid_file in files_grid:
        for chunk in grid_file.chunks:
            if isinstance(chunk, Statement):
                if chunk.name in statements:
                    raise ValueError(f"duplicate statement {chunk} (also {statements[chunk.name]}")
                else:
                    statements[chunk.name] = chunk

    # Split statements by type
    total = 1
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
    if options.c or options.action == "distribute":
        grid_state = get_grid_state()
        logging.info("Continue with previous {n} instances".format(n=len(grid_state["grid"])))
    else:
        grid_state = {"grid": {}}

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

    from .algorithm import eval_all
    if options.action == "new":
        if len(statements_core) == 0:
            warn(f"No fixed groups found")

        # Iterate over possible combinations and write a grid
        for stack in combinations(statements_core):
            scratch = folder_name(index)
            stack["__grid_folder_name__"] = scratch
            stack["__grid_id__"] = index

            stack = eval_all(statements_dependent, stack, builtins)
            grid_state["grid"][scratch] = {"stack": stack}
            write_grid(scratch, stack, files_static, files_grid)
            index += 1

        # Save state

        save_grid_state(grid_state)

    # ------------------------------------------------------------------
    #   Distribute
    # ------------------------------------------------------------------

    elif options.action == "distribute":

        if len(statements_dependent) == 0:
            warn("No dependent statements found. File(s) will be distributed as-is.")

        logging.info(f"Distributing files into {len(grid_state)} folders")
        exceptions = []
        for k, v in grid_state["grid"].items():
            stack = eval_all(statements_dependent, v["stack"])
            if not Path(k).is_dir():
                logging.exception(f"Grid folder {k} does not exist")
                exceptions.append(FileNotFoundError(f"No such file or directory: {repr(k)}"))
            else:
                write_grid(k, stack, files_static, files_grid)
        if len(exceptions) > 0:
            raise exceptions[-1]

# ----------------------------------------------------------------------
#   Execute in context of grid
# ----------------------------------------------------------------------

elif options.action == "run":
    if options.files or options.static_files or options.name or options.target:
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
    if options.files or options.static_files or options.name or options.target:
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
