from .parsing import split_assignment
import tempfile
from types import FunctionType
import uuid


class EvalBlock:
    def __init__(self, code, name=None, source="<unknown>"):
        """Represents a statement that can be evaluated"""
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