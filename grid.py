#!/usr/bin/env python3
import logging
from warnings import warn

filename_data = ".grid"
filename_log = ".grid.log"
filename_opt_routine = ".grid.optimize"

logging.basicConfig(filename=filename_log, filemode="w", level=logging.INFO)

import argparse, uuid, logging, json
import re
from pyparsing import *
from operator import attrgetter

import os, glob, shutil, subprocess, sys, math

parser = argparse.ArgumentParser(description="Creates an array [grid] of similar jobs and executes [submits] them.")
parser.add_argument("-f", "--files", nargs="+", help="files to be processed and replicated in each copy of the job",
                    metavar="FILENAME")
parser.add_argument("-t", "--static-files", nargs="+", help="files to be replicated in each copy of the job",
                    metavar="FILENAME")
parser.add_argument("-p", "--prefix", help="prefix of the grid folders", metavar="STRING")
parser.add_argument("-g", "--target", help="target tolerance for optimization", metavar="FLOAT", type=float)
parser.add_argument("-c", action='store_true', help="continue optimization without removing existing grid state")
parser.add_argument("action", help="action to perform",
                    choices=["new", "run", "cleanup", "which", "optimize", "progress", "distribute"])
parser.add_argument("command", nargs="*", help="command to execute or list of folders for 'which' action")

options = parser.parse_args()
logging.info("STARTED LOGGING FOR {action}".format(action=options.action))


def state():
    if not os.path.isfile(filename_data):
        print("Could not find configuration file {name}. Did you run 'grid new'?".format(name=filename_data))
        logging.error("Could not find configuration file {name}".format(name=filename_data))
        sys.exit(1)

    try:
        with open(filename_data, "r") as f:
            return json.load(f)
    except Exception as e:
        print("Failed to read .grid file: wrong format")
        logging.exception("Failed to read grid configuration data")
        sys.exit(1)


def save_state(state):
    try:
        with open(filename_data, "w") as f:
            json.dump(state, f, indent=4)
    except IOError as e:
        print(e)
        logging.exception("Could not save the grid state")
        sys.exit(1)


def folder_name(index):
    return "{prefix}{index}".format(prefix=options.prefix, index=index)


# ----------------------------------------------------------------------
#   New grid, optimize, distribute
# ----------------------------------------------------------------------

if options.action == "new" or options.action == "optimize" or options.action == "distribute":

    # Errors

    if len(options.command) == 0 and options.action == "optimize":
        parser.error("no target executable provided for 'optimize'")

    if options.target and not options.action == "optimize":
        parser.error("-g, --target options are meaningless for {action} action".format(action=options.action))

    if len(options.command) == 0 and options.action == "distribute":
        parser.error("nothing to distribute")

    # Defaults

    if not options.static_files:
        options.static_files = []

    if not options.prefix:
        options.prefix = 'grid'

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


    class OptimizableParameter(object):

        def __init__(self, start, end, tolerance=1e-4):
            if not isinstance(start, (int, float)):
                raise ValueError("The 'start' parameter must be a number")
            if not isinstance(end, (int, float)):
                raise ValueError("The 'end' parameter must be a number")
            if not isinstance(tolerance, (int, float)):
                raise ValueError("The 'tolerance' parameter must be a number")
            self.start = start
            self.end = end
            self.tolerance = tolerance / abs(self.start - self.end)

        def __getitem__(self, i):
            return (1 - i) * self.start + i * self.end

        def __str__(self):
            return "Optimizable range {start}-{end}, finite difference step: {tol:e}".format(
                start=self.start,
                end=self.end,
                tol=self.tolerance,
            )


    class Variable(object):

        def __init__(self, name):
            self.name = name

        def evaluate(self, stack):
            return stack[self.name]

        def ready(self, stack):
            return self.name in stack

        def __add__(self, another):
            return DelayedExpression(lambda x, y: x + y, self, another)

        def __sub__(self, another):
            return DelayedExpression(lambda x, y: x - y, self, another)

        def __mul__(self, another):
            return DelayedExpression(lambda x, y: x * y, self, another)

        def __div__(self, another):
            return DelayedExpression(lambda x, y: x / y, self, another)

        def __truediv__(self, another):
            return DelayedExpression(lambda x, y: x / y, self, another)

        def __floordiv__(self, another):
            return DelayedExpression(lambda x, y: x // y, self, another)

        def __radd__(self, another):
            return DelayedExpression(lambda x, y: x + y, another, self)

        def __rsub__(self, another):
            return DelayedExpression(lambda x, y: x - y, another, self)

        def __rmul__(self, another):
            return DelayedExpression(lambda x, y: x * y, another, self)

        def __rdiv__(self, another):
            return DelayedExpression(lambda x, y: x / y, self, another)

        def __rtruediv__(self, another):
            return DelayedExpression(lambda x, y: x / y, another, self)

        def __rfloordiv__(self, another):
            return DelayedExpression(lambda x, y: x // y, another, self)

        def __pow__(self, another):
            return DelayedExpression(lambda x, y: x ** y, self, another)

        def __repr__(self):
            return self.name

        def __str__(self):
            return "Variable {0}".format(repr(self))


    class DelayedExpression(Variable):

        def __init__(self, function, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.function = function

        def evaluate(self, stack):

            args = []
            for i in self.args:
                if "evaluate" in dir(i):
                    args.append(i.evaluate(stack))
                else:
                    args.append(i)

            kwargs = {}
            for k, v in self.kwargs.items():
                if "evaluate" in dir(v):
                    kwargs[k] = v.evaluate(stack)
                else:
                    kwargs[k] = v

            return self.function(*args, **kwargs)

        def ready(self, stack):
            for i in self.args:
                if "evaluate" in dir(i) and not i.ready(stack):
                    return False
            for k, v in self.kwargs.items():
                if "evaluate" in dir(v) and not v.ready(stack):
                    return False
            return True

        def __repr__(self):
            return "{func}(args: {args}, kwargs: {kwargs})".format(
                func=self.function,
                args=self.args,
                kwargs=self.kwargs,
            )

        def __str__(self):
            return "Expression {0}".format(repr(self))

        @staticmethod
        def evaluateToStack(stack, statements, attr=None, require=False):
            statements = statements.copy()
            done = False

            while not done:

                done = True

                for k, v in statements.items():
                    if not attr is None:
                        v = getattr(v, attr)
                    if not k in stack and v.ready(stack):
                        stack[k] = v.evaluate(stack)
                        done = False

            if require:
                delta = set(statements).difference(set(stack))
                if len(delta) > 0:
                    raise ValueError(f"{len(delta)} expressions cannot be evaluated: {', '.join(sorted(delta))}")


    # Language definitions

    import traceback


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
        debug(lambda t: re.sub(r'\\(.)', r'\g<1>', t[0]), "PythonQuotedString")
    )
    Numeric = Word(nums)
    Integer = Combine(Optional(Literal('+') | Literal('-')) + Numeric).setParseAction(
        debug(lambda t: int(t[0]), "Integer")
    )
    Float = Combine(Optional(Literal('+') | Literal('-')) + Numeric + Optional("." + Optional(Numeric)) + Optional(
        CaselessLiteral('E') + Integer)).setParseAction(
        debug(lambda t: float(t[0]), "Float")
    )
    IorF = Integer ^ Float
    Var = VariableName.copy().setParseAction(
        debug(lambda t: Variable(t[0]), "Var")
    )
    Primitive = PythonQuotedString ^ Integer ^ Float ^ Var

    # Args and kwargs

    Arg = GenericExpression + ~FollowedBy("=")
    NamedArg = Group(VariableName("key") + Suppress("=") + GenericExpression("value"))

    Args = Group(Arg + ZeroOrMore(Suppress(",") + Arg))
    KwArgs = Group(NamedArg + ZeroOrMore(Suppress(",") + NamedArg)).setParseAction(
        debug(lambda t: dict((i.key, i.value) for i in t[0]), "KwArgs")
    )
    MixedArgs = Args("args") + Suppress(",") + KwArgs("kwargs")
    ArbitraryArgs = (MixedArgs | KwArgs("kwargs") | Args("args"))

    # Arrays

    PythonList = (Suppress("[") + Args + Suppress("]")).setParseAction(
        debug(lambda t: [list(t[0]), ], "PythonList")
    )


    # Function call
    def translateFunction(f):

        # Numpy functions
        if f.startswith("numpy"):

            # Try to import
            try:
                import numpy
            except ImportError:
                print("Could not import numpy: either install numpy or avoid using '{routine}' in grid files".format(
                    routine=f,
                ))
                logging.exception("Failed to import numpy")
                sys.exit(1)

            # Pick function
            try:
                return {
                    "numpy": numpy.array,
                }[f]
            except KeyError:
                print("Could not find numpy function {routine}".format(
                    routine=f,
                ))
                logging.exception("Failed to execute {routine}".format(
                    routine=f,
                ))
                sys.exit(1)

        # Simple functions
        else:
            try:
                return {
                    "range": listRange,
                    "linspace": linspace,
                    "optimize": OptimizableParameter,
                }[f]
            except KeyError:
                print("Could not find function {routine}".format(
                    routine=f,
                ))
                logging.exception("Failed to execute {routine}".format(
                    routine=f,
                ))
                sys.exit(1)


    Callable = Group(FieldName("callable") + Suppress("(") + Optional(ArbitraryArgs) + Suppress(")")).setParseAction(
        debug(lambda t: [translateFunction(t[0]["callable"])(
            *(t[0]["args"] if "args" in t[0] else []),
            **(t[0]["kwargs"] if "kwargs" in t[0] else {})
        )], "Callable")
    )

    # Atomic types

    ValueInBrackets = Suppress("(") + GenericExpression + Suppress(")")
    AtomicExpression = Callable | PythonList | Primitive | ValueInBrackets


    # Atomic operations

    def evaluate(t):

        if not isinstance(t, ParseResults):
            return t

        if len(t) % 2 == 0:
            raise Exception("Cannot evaluate {expr}".format(
                expr=t,
            ))

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


    # PowExpression = Forward()
    # PowExpression << ( AtomicExpression + ZeroOrMore( Literal("**") + PowExpression) ).setParseAction(evaluate)
    # MultExpression = Forward()
    # MultExpression << ( PowExpression + ZeroOrMore( (Literal("*") ^ Literal("/")) + MultExpression) ).setParseAction(evaluate)
    # GenericExpression << ( MultExpression + ZeroOrMore( (Literal("+") ^ Literal("-")) + MultExpression)).setParseAction(evaluate)
    from pyparsing import infix_notation, opAssoc, oneOf, ParseResults

    GenericExpression << infix_notation(AtomicExpression, [
        ("**", 2, opAssoc.RIGHT),
        (oneOf("* /"), 2, opAssoc.LEFT),
        (oneOf("+ -"), 2, opAssoc.LEFT),
    ]).setParseAction(debug(lambda x: [evaluate(x)], "GenericExpression"))

    # Grid group

    GroupHeader = Optional(
        Combine(VariableName("id") + ZeroOrMore(Suppress(":") + VariableName)("flags")) + Suppress("="))
    GroupStatement = Escaped("escaped") ^ (Suppress("{%") + GroupHeader + GenericExpression("choices") + Suppress("%}"))


    # Test
    # for i in GroupStatement.searchString("""
    # {% a= optimize(1,2,tolerance=3) %}
    # {% x= a*2**2 %}
    # {% (30*5**2+7*3)*2 %}
    # """):
    # print(i.choices,i.id,i.flags)
    ##print(i[0])
    # exit(0)

    # ------------------------------------------------------------------
    #   Classes
    # ------------------------------------------------------------------

    class InconsistencyException(Exception):
        pass


    class Statement(object):
        """
        Represents an expression between {% %}.
        """

        def __init__(self, owner, start, end, expression, id=None):
            self.file = owner
            self.start = start
            self.end = end
            self.expression = expression

            if id is None or len(id) == 0:
                self.id = str(uuid.uuid4())
            else:
                self.id = id

        def is_regular(self):
            try:
                len(self.expression)
                return True
            except TypeError:
                return False

        def is_optimizable(self):
            return isinstance(self.expression, OptimizableParameter)

        def is_evaluatable(self):
            return "evaluate" in dir(self.expression)

        def line(self):
            return sum(1 for x in self.file.__source__[:self.start] if x == '\n')

        def symbolInLine(self):
            try:
                return next(i for i, x in enumerate(reversed(self.file.__source__[:self.start])) if x == '\n')
            except StopIteration:
                return self.start

        def __str__(self):
            return f"Statement(file={repr(self.file.name)}l{self.line() + 1}:{self.symbolInLine() + 1} expression={repr(self.expression)})"

        __repr__ = __str__

        @staticmethod
        def convert(x):

            if isinstance(x, str):
                return x

            # Convert to float, int if possible
            try:
                if int(x) == x:
                    return int(x)
            except ValueError:
                pass

            try:
                return float(x)
            except ValueError:
                pass

            raise Exception("Internal error: unsupported type {t}".format(
                t=type(x),
            ))

        @staticmethod
        def s2d(s):
            result = {}
            for i in s:
                if i.id in result:
                    raise InconsistencyException("Statements at {g1} and {g2} have the same id '{id}'".format(
                        g1=result[i.id],
                        g2=i,
                        id=i.id,
                    ))
                result[i.id] = i
            return result


    class EscapeStatement(object):

        def __init__(self, owner, start, end, escaped):
            self.file = owner
            self.start = start
            self.end = end
            self.escaped = escaped


    class GridFile(object):

        def __init__(self, s, name=None, floatFormat=".10f"):
            self.__source__ = s
            self.__statements__ = []
            self.__fformatter__ = floatFormat
            self.name = name

            for i, start, end in GroupStatement.scanString(s):

                if "escaped" in dir(i):
                    self.__statements__.append(EscapeStatement(self, start, end, i.escaped))

                elif "choices" in dir(i):
                    self.__statements__.append(Statement(self, start, end, i.choices, id=i.id))

                else:
                    raise Exception("Internal error: unknown parsed element")

            self.__statements__ = sorted(self.__statements__, key=attrgetter('start'))

        def statements(self):
            return list(i for i in self.__statements__ if isinstance(i, Statement))

        def write(self, stack, f):

            start = 0

            for i in self.__statements__:

                f.write(self.__source__[start:i.start])

                if isinstance(i, EscapeStatement):
                    chunk = i.escaped

                elif isinstance(i, Statement):
                    try:
                        chunk = stack[i.id]
                    except KeyError:
                        chunk = self.__source__[i.start:i.end]
                        warn("Expression named '{name}' at {st} could not be evaluated and will be ignored".format(
                            st=i,
                            name=i.id,
                        ))
                        logging.warn(
                            "Expression named '{name}' at {st} could not be evaluated and will be ignored".format(
                                st=i,
                                name=i.id,
                            ))

                if isinstance(chunk, str):
                    f.write(chunk)
                elif isinstance(chunk, float):
                    f.write(("{0:" + self.__fformatter__ + "}").format(chunk))
                elif isinstance(chunk, int):
                    f.write("{0:d}".format(chunk))
                else:
                    raise Exception("Internal error occurred: type of chunk is {type}".format(type=type(chunk)))

                start = i.end

            f.write(self.__source__[start:])

        def is_trivial(self):
            return len(self.statements()) == 0

        def __str__(self):
            return super(GridFile, self).__str__() if self.name is None else self.name


    def combinations(n):

        n_keys = n.keys()
        n_max = tuple(len(n[k].expression) for k in n_keys)
        i = list((0,) * len(n_max))

        while True:
            result = {}
            for k, v in zip(n_keys, i):
                result[k] = n[k].expression[v]
            yield result

            for x, mx in zip(range(len(i)), n_max):
                i[x] += 1
                if i[x] < mx:
                    break
                else:
                    i[x] = 0
                if x == len(i) - 1:
                    return


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

    files = []

    if options.files or options.action == "distribute":

        # Match files
        logging.info("The files are provided explicitly")

        file_list = options.command if options.action == "distribute" else options.files
        for i in file_list:

            new = glob.glob(i)
            if len(new) == 0:
                print("File '{pattern}' not found or pattern matched 0 files".format(pattern=i))
                logging.error("No grid files provided")
                sys.exit(1)
            files = files + new
        for f in files:
            logging.info("  {name}".format(name=f))

    else:

        # Find files, default behavior
        logging.info("Searching for grid files in current folder")

        for i in os.listdir('.'):
            if os.path.isfile(i):
                try:
                    with open(i, 'r') as f:
                        if not GridFile(f.read()).is_trivial():
                            logging.info("  {name}".format(name=i))
                            files.append(i)
                except:
                    logging.exception("Failed to read {name}".format(name=i))

        if len(files) == 0:
            print("No grid-formatted files found in this folder")
            logging.error("No grid files found in . folder")
            sys.exit(1)

    logging.info("Total: {n} files".format(n=len(files)))

    # Match static items
    logging.info("Matching static part")

    files_static = []
    for i in options.static_files:

        new = glob.glob(i)
        if len(new) == 0:
            print("File '{pattern}' not found or pattern matched 0 files".format(pattern=i))
            logging.error("File '{pattern}' provided but matched 0 files")
            sys.exit(1)

        for f in new:
            logging.info("  {name}".format(name=f))

        files_static = files_static + new

    logging.info("Total: {n} items".format(n=len(files_static)))

    # Read files

    files_grid = []
    for f in files:
        try:
            with open(f, 'r') as fl:
                files_grid.append(GridFile(fl.read(), name=f))
        except IOError as e:
            print(e)
            logging.exception("Error during reading the file {name}".format(name=f))
            sys.exit(1)
    logging.info("All files have been successfully read")

    # Collect all statements into dict
    logging.info("Processing grid-formatted files")

    statements = []
    for f in files_grid:
        statements = statements + f.statements()

    try:
        statements = Statement.s2d(statements)
    except InconsistencyException as e:
        print(e)
        logging.exception(str(e))
        sys.exit(1)

    # Split statements by type
    total = 1
    statements_fix = {}
    statements_opt = {}
    statements_dep = {}

    for k, v in statements.items():

        if v.is_optimizable():
            logging.info("  statement {name}: optimizable | {value}".format(name=k, value=str(v)))
            statements_opt[k] = v

        elif v.is_regular():
            logging.info("  statement {name}: {n} | {value}".format(name=k, n=len(v.expression), value=str(v)))
            statements_fix[k] = v
            total = total * len(v.expression)
            if total > 1e6:
                print("Grid size is too large (more than a million)")
                logging.error("Grid size exceeds a million")
                sys.exit(1)

        elif v.is_evaluatable():
            logging.info("  statement {name}: evaluatable | {value}".format(name=k, value=str(v)))
            statements_dep[k] = v

        else:
            raise Exception("Internal error: unknown expression {ex}".format(
                ex=v.expression,
            ))
    logging.info(
        "Total: {fixed} fixed statement(s) ({comb} combination(s)), {opt} optimize statement(s) and {dep} dependent statement(s)".format(
            fixed=len(statements_fix),
            opt=len(statements_opt),
            dep=len(statements_dep),
            comb=total,
        ))

    # Read previous run
    if options.c or options.action == "distribute":
        grid_state = state()
        logging.info("Continue with previous {n} instances".format(n=len(grid_state["grid"])))
    else:
        grid_state = {
            "grid": {}
        }

    index = len(grid_state["grid"])

    # Check if folders already exist
    for x in glob.glob(options.prefix + "*"):
        if not x in grid_state["grid"]:
            print(
                "File or folder {name} may conflict with the grid. Either remove it or use a different prefix through '--prefix' option.".format(
                    name=x))
            logging.error("{name} already exists".format(name=x))
            sys.exit(1)

    # ------------------------------------------------------------------
    #   Optimize
    # ------------------------------------------------------------------

    if options.action == "optimize":

        # Try to import

        try:
            from scipy.optimize import minimize
        except ImportError:
            print("Could not import minimize from scipy.optimize")
            logging.exception("Failed to import minimize from scipy.optimize")
            sys.exit(1)

        # Check if optimizable groups are present

        if len(statements_fix) > 0:
            warn(
                "Found {n} fixed groups which will be ignored during optimize run. Use 'grid new' option if you want to unpack fixed groups.".format(
                    n=len(statements_fix)))
            logging.warn("{n} fixed groups found but action='optimize'".format(n=len(statements_fix)))

        if len(statements_opt) == 0:
            print("No groups to optimize found, exiting.")
            logging.error("'grid optimize' is invoked but no groups to optimize found")
            sys.exit(1)

        # Set optimize groups' names
        optimize_names = statements_opt.keys()

        # Float parser
        cre_float = re.compile(r"([-+]?[0-9]+\.?[0-9]*(?:[eEdD][-+]?[0-9]*)?)|(nan)")


        def function_to_optimize(x, return_folder=False):

            logging.info("Requested vector {vector}".format(vector=x))

            global index

            # Evaluate stack
            stack = {}
            for k, v in zip(optimize_names, x):
                stack[k] = statements_opt[k].expression[v]
            DelayedExpression.evaluateToStack(stack, statements_dep, attr="expression")

            # Attempt to find in cache
            for k, v in grid_state["grid"].items():
                if v["stack"] == stack and "opt_value" in v:
                    result = v["opt_value"]
                    logging.info("Responded cached value {value} from '{folder}'".format(value=result, folder=k))
                    if return_folder:
                        return result, k
                    else:
                        return result

            scratch = folder_name(index)
            grid_state["grid"][scratch] = {"stack": stack}
            write_grid(scratch, stack, files_static, files_grid)
            save_state(grid_state)

            try:
                logging.info("Executing '{ex}'".format(ex=' '.join(options.command)))
                os.environ['CURRENT_GRID_FOLDER'] = scratch
                output = subprocess.check_output(options.command, cwd=scratch, shell=True)
                match = cre_float.finditer(str(output))
                m = None
                for m in match:
                    pass
                if m is None:
                    raise ValueError
                result = float(m.group())
            except ValueError:
                print("Failed to find a float in the output of '{script}' (cwd = {cwd}) script:\n{output}".format(
                    script=" ".join(options.command),
                    output=output,
                    cwd=scratch
                ))
                logging.exception("Failed to match float in the output:\n{output}".format(output=output))
                sys.exit(1)
            except subprocess.CalledProcessError as e:
                print("Process error, see {log} for details".format(log=filename_log))
                logging.exception("Process error: {p}\nOutput: {o}".format(
                    p=" ".join(options.command),
                    o=e.output,
                ))
                sys.exit(1)

            grid_state["grid"][scratch] = {"stack": stack, "opt_value": result}
            save_state(grid_state)

            index += 1
            logging.info("Responded value {value}".format(value=result))
            if return_folder:
                return result, scratch
            else:
                return result


        # Set finite difference epsilon

        eps = list(statements_opt[n].expression.tolerance for n in optimize_names)
        logging.info("Setting finite difference epsilon to {eps}".format(
            eps=eps,
        ))

        if not "optimize" in grid_state:
            grid_state["optimize"] = {}

        # Calculate tolerance if needed
        if not "target_error" in grid_state["optimize"]:

            if options.target:
                grid_state["optimize"]["target_error"] = options.target

            else:
                logging.info("Determining target tolerance")

                vector = [0.5] * len(statements_opt)
                values = []

                v0, f0 = function_to_optimize(vector, return_folder=True)
                values.append(v0)

                for i in range(len(vector)):
                    vector[i] = 0.5 + eps[i]
                    v1, f1 = function_to_optimize(vector, return_folder=True)
                    vector[i] = 0.5
                    values.append(v1)

                    if v0 == v1:
                        print(
                            "Problem with a finite difference gradient: the values from '{g0}' and '{g1}' are equal to {val}".format(
                                g0=f0,
                                g1=f1,
                                val=v0,
                            ))
                        logging.error(
                            "Finite difference gradient vanishes. Value obtained in '{g0}' and '{g1}' is {val}".format(
                                g0=f0,
                                g1=f1,
                                val=v0,
                            ))
                        sys.exit(1)

                grid_state["optimize"]["target_error"] = (max(values) - min(values)) * 10

        # Run minimize
        logging.info("Optimizing to {eps}".format(eps=grid_state["optimize"]["target_error"]))
        result = minimize(function_to_optimize, (0.5,) * len(statements_opt),
                          bounds=((0, 1),) * len(statements_opt),
                          options={
                              "eps": eps,
                          },
                          tol=grid_state["optimize"]["target_error"],
                          method='L-BFGS-B',
                          )

        # Very last call
        logging.info("Performing final computation")
        actual, folder = function_to_optimize(result.x, return_folder=True)

        grid_state["optimize"].update({
            "folder": folder,
            "message": str(result.message),
            "success": bool(result.success),
            "minimized": float(result.fun),
            "minimized_actual": float(actual),
        })
        save_state(grid_state)
        print(actual)

    # ------------------------------------------------------------------
    #   New
    # ------------------------------------------------------------------

    elif options.action == "new":

        # Check if fixed groups are present

        if len(statements_opt) > 0:
            warn("Found {n} optimizable groups which will be ignored during 'new' run.".format(n=len(statements_opt)))
            logging.warn("{n} optimizable groups found but action='new'".format(n=len(statements_opt)))

        if len(statements_fix) == 0:
            print("No fixed groups found, exiting.")
            logging.error("'grid new' is invoked but no fixed groups found")
            sys.exit(1)

        # Iterate over possible combinations and write a grid

        for stack in combinations(statements_fix):
            scratch = folder_name(index)
            stack["__grid_folder_name__"] = scratch
            stack["__grid_id__"] = index

            DelayedExpression.evaluateToStack(stack, statements_dep, attr="expression", require=True)
            grid_state["grid"][scratch] = {"stack": stack}
            write_grid(scratch, stack, files_static, files_grid)
            index += 1

        # Save state

        save_state(grid_state)

    # ------------------------------------------------------------------
    #   Distribute
    # ------------------------------------------------------------------

    elif options.action == "distribute":

        if len(statements_dep) == 0:
            warn("No dependent statements found. The file(s) will be distributed as-is.")

        for k, v in grid_state["grid"].items():
            DelayedExpression.evaluateToStack(v["stack"], statements_dep, attr="expression")
            write_grid(k, v["stack"], files_static, files_grid)

# ----------------------------------------------------------------------
#   Execute in context of grid
# ----------------------------------------------------------------------

elif options.action == "run":

    # Errors

    if options.files or options.static_files or options.prefix or options.target:
        parser.error(
            "-f, --files, -t, --static-files, -p, --prefix, -g, --target options are meaningless for {action} action".format(
                action=options.action))

    if len(options.command) == 0:
        parser.error("no command to run specified")

    current_state = state()

    for f in current_state["grid"]:

        command = " ".join(options.command)
        cwd = os.path.abspath(f)

        try:
            print(cwd)
            p = subprocess.Popen(options.command, cwd=cwd)
            p.wait()
        except:
            print("Failed to execute {command} (working directory {directory})".format(command=command, directory=cwd))
            logging.exception(
                "Failed to execute {command} (working directory {directory})".format(command=command, directory=cwd))
            sys.exit(1)

        logging.info(
            "  executed {command} in {directory} with PID:{pid}".format(command=command, directory=cwd, pid=p.pid))

# ----------------------------------------------------------------------
#   Cleanup grid
# ----------------------------------------------------------------------

elif options.action == "cleanup":

    # Errors

    if options.files or options.static_files or options.prefix or options.target:
        parser.error(
            "-f, --files, -t, --static-files, -p, --prefix, -g, --target options are meaningless for {action} action".format(
                action=options.action))


    def remove(s):

        if os.path.isdir(s):
            shutil.rmtree(s)

        elif os.path.isfile(s):
            os.remove(s)


    current_state = state()

    error = False

    logging.info("Removing folders")

    for f in current_state["grid"]:

        try:
            if os.path.exists(f):
                remove(f)
                logging.info("  {name}".format(name=f))
        except Exception as e:
            print(e)
            logging.exception("Error while removing {name}".format(name=f))
            error = True

    logging.info("Removing data file")

    try:
        remove(filename_data)
    except Exception as e:
        print(e)
        logging.exception("Error while removing {name}".format(name=filename_data))
        sys.exit(1)

    if error:
        sys.exit(1)

# ----------------------------------------------------------------------
#   Which
# ----------------------------------------------------------------------

elif options.action == 'which':

    # Errors

    if options.files or options.static_files or options.prefix or options.target:
        parser.error(
            "-f, --files, -t, --static-files, -p, --prefix, -g, --target options are meaningless for {action} action".format(
                action=options.action))

    grid_state = state()

    if len(options.command) == 0:
        options.command = grid_state["grid"].keys()

    for i in options.command:
        if len(options.command) > 1:
            print(os.path.abspath(i))
        if i in grid_state["grid"]:

            max_key_len = len(max(grid_state["grid"][i].keys(), key=len))
            max_key_len = min(20, max_key_len)

            for k, v in grid_state["grid"][i].items():
                print("{indent}{key}: {value}".format(
                    key=k.rjust(max_key_len) if len(k) <= max_key_len else k[:max_key_len - 3] + '...',
                    value=v,
                    indent='  ' if len(options.command) > 1 else ''
                ))
        else:
            print('  Not found')

# ----------------------------------------------------------------------
#   Progress
# ----------------------------------------------------------------------

elif options.action == "progress":

    # Errors

    if options.files or options.static_files or options.prefix or options.target:
        parser.error(
            "-f, --files, -t, --static-files, -p, --prefix, -g, --target options are meaningless for {action} action".format(
                action=options.action))

    # Try to import

    try:
        from matplotlib import pyplot
        from matplotlib.patches import Patch
        from matplotlib.font_manager import FontManager
        import colorsys
    except ImportError:
        print("Could not import matplotlib")
        logging.exception("Failed to import matplotlib")
        sys.exit(1)

    if len(options.command) > 0:
        filename_data = options.command[0]

    grid_state = state()

    if len(grid_state["grid"]) == 0:
        print("No data available yet")
        logging.error("No data available yet")
        sys.exit(1)

    # Order data

    grid = grid_state["grid"].items()
    grid = [(i[0], i[1]["opt_value"] if "opt_value" in i[1] else None, i[1]["stack"]) for i in grid if "stack" in i[1]]
    grid.sort(key=lambda x: x[0])

    data_x = {}
    for parameter in grid[0][2].keys():
        data_x[parameter] = [g[2][parameter] for g in grid]

    data_x_range = {}
    for k, v in data_x.items():
        data_x_range[k] = (min(v), max(v))

    data_x_step = []
    for p1, p2 in zip(grid[:-1], grid[1:]):
        total = 0
        for parameter in data_x.keys():
            total += (p1[2][parameter] - p2[2][parameter]) ** 2 / (
                        data_x_range[parameter][1] - data_x_range[parameter][0]) ** 2
        data_x_step.append(total ** 0.5)

    data_y = [i[1] for i in grid]
    data_y_numbers = [i for i in data_y if not i is None]
    if len(data_y_numbers) > 0:
        data_y_range = [min(data_y_numbers), max(data_y_numbers)]
    else:
        data_y_range = [0, 1]
    if data_y_range[0] == data_y_range[1]:
        data_y_range[0] -= 0.5
        data_y_range[1] += 0.5
    data_location = [i[0] for i in grid]
    data_colors = [colorsys.hsv_to_rgb(1.0 * i / len(grid), 1, 1) for i in range(len(grid))]

    if len(grid) == 0:
        print("No optimization data found in grid")
        logging.error("No optimization data found in grid")
        sys.exit(1)

    # Ranges of parameters

    data_x_ranges = {}
    for k in grid[0][2].keys():
        data_x_ranges[k] = (
            min([i[2][k] for i in grid]),
            max([i[2][k] for i in grid]),
        )

    range_fraction = 0.02
    ratio = 16. / 9

    # Set plot geometry

    n_keys = len(grid[0][2].keys())
    w = int(math.ceil((n_keys * ratio) ** .5))
    h = int(math.ceil(n_keys / w))
    logging.info("Will arrange {n} plots into {w} by {h} grid".format(
        n=n_keys,
        w=w,
        h=h,
    ))

    # Plot

    pyplot.figure(facecolor='white')

    pyplot.gcf().legend(
        [Patch(alpha=0.7, color=data_colors[i]) for i in range(len(data_location))],
        data_location,
        fancybox=True,
        bbox_to_anchor=(0.9, 0.5),
        loc=10,
    )
    pyplot.subplots_adjust(right=0.8, hspace=0.3)

    for parameter_i, parameter in enumerate(grid[0][2].keys()):

        pyplot.subplot2grid((w, h), (parameter_i % w, parameter_i // w), axisbg=(0.9, 0.9, 0.9))
        ax = pyplot.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.xaxis.grid(color='white', linewidth=2, linestyle='solid')
        ax.yaxis.grid(color='white', linewidth=2, linestyle='solid')
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off",
                       labelleft="on")

        data_x_scatter = []
        data_c_scatter = []

        for x, y, c in zip(data_x[parameter], data_y, data_colors):

            if not y is None:
                data_x_scatter.append(x)
                data_c_scatter.append(c)

            else:

                ax.axvline(x=x, c=c, alpha=0.7)

        ax.scatter(data_x_scatter, data_y_numbers, c=data_c_scatter, s=180, alpha=0.7)

        for x1, x2, y1, y2, step in zip(
                data_x[parameter][:-1],
                data_x[parameter][1:],
                data_y[:-1],
                data_y[1:],
                data_x_step):

            if not y1 is None and not y2 is None and not x1 == x2 and step < range_fraction:
                k = (y1 - y2) / (x1 - x2)
                b = y1 - k * x1
                x_delta = range_fraction * (data_x_range[parameter][1] - data_x_range[parameter][0])
                ax.plot(
                    [0.5 * (x1 + x2 - x_delta), 0.5 * (x1 + x2 + x_delta)],
                    [k * 0.5 * (x1 + x2 - x_delta) + b, k * 0.5 * (x1 + x2 + x_delta) + b],
                    antialiased=True,
                    color='black'
                )

        pyplot.ylim((data_y_range[0] * 1.1 - data_y_range[1] * 0.1, data_y_range[1] * 1.1 - data_y_range[0] * 0.1))
        pyplot.title("Target vs parameter '{p}'".format(p=parameter))

    pyplot.savefig("progress.png")
    pyplot.show()
