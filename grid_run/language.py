
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
        return "{func}(args: {args}, kwargs: {kwargs})".format(func=self.function, args=self.args,
                                                               kwargs=self.kwargs, )

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
