from types import FunctionType, CodeType
from collections import namedtuple

from .parsing import split_assignment, iter_template_blocks
from .tools import ext_format


class Expression:
    def __init__(self, code: [CodeType, "Expression"]):
        """
        A python expression stored in a code object.

        Parameters
        ----------
        code
            Expression code.
        """
        self.code = code
        if isinstance(code, Expression):
            self.code = code.code

    @classmethod
    def from_string(cls, text: str) -> "Expression":
        """
        Assembles an expression from the provided text.

        Parameters
        ----------
        text
            Text with a valid python expression.

        Returns
        -------
        A compiled expression.
        """
        return cls(compile(text, "<string>", "eval"))

    @property
    def required_names(self) -> set[str]:
        """Names required for this expression"""
        return set(self.code.co_names)

    def get_missing_names(self, names: set[str]) -> set[str]:
        """
        Checks if all required names are covered by
        the names provided.

        Parameters
        ----------
        names
            Names to check against.

        Returns
        -------
        A set of missing names.
        """
        return set(self.required_names) - names

    def eval(self, names: dict[str, object]):
        """
        Computes the value of this expression.

        Parameters
        ----------
        names
            A dictionary with name and argument
            values.

        Returns
        -------
        The computed value.
        """
        delta = self.get_missing_names(set(names))
        if len(delta) > 0:
            raise ValueError(f"missing following names: {', '.join(map(repr, delta))}")
        func = FunctionType(self.code, names)
        return func()


named_expression = namedtuple("named_expression", ("name", "expression", "format"))


def parse_named_expression(text: str, defined_file: str = "unknown", defined_line: str = "u",
                           defined_char: str = "u") -> named_expression:
    """
    Parses text to provide expression, its name and format.

    Parameters
    ----------
    text
        Expression text `[name =] expression [:format]`.
    defined_file
        A file name where the expression was defined.
    defined_line
    defined_char
        Line and char numbers.

    Returns
    -------
    A parsed expression.
    """
    name, fmt, text = split_assignment(text)
    if name is None:
        defined_file = ''.join(i if i.isalnum() else "_" for i in defined_file)
        name = f"anonymous_{defined_file}_l{defined_line}c{defined_char}"
    return named_expression(name, Expression.from_string(text), fmt)


class Template:
    def __init__(self, name, chunks):
        """A file with multiple statements to evaluate"""
        self.name = name
        self.chunks = chunks

    @classmethod
    def from_text(cls, name, text):
        itr = iter_template_blocks(text)
        chunks = []
        for pos, i in itr:
            chunks.append(i)  # regular text block
            try:
                pos, i = next(itr)  # eval block
                chunks.append(parse_named_expression(
                    i,
                    defined_file=name,
                    defined_line=text[:pos].count("\n") + 1,
                    defined_char=pos - text[:pos].rfind("\n"),
                ))
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
            elif isinstance(chunk, named_expression):
                f.write(ext_format(stack[chunk.name], chunk.format))
            else:
                raise NotImplementedError(f"unknown {chunk=}")

    def is_trivial(self):
        for chunk in self.chunks:
            if not isinstance(chunk, str):
                return False
        return True

    def __repr__(self):
        return f"GridFile(name={repr(self.name)}, chunks=[{len(self.chunks)} chunks])"


def variable_list_template(variable_names, name=".variables"):
    """Constructs a template with variable names"""
    # TODO: fix a hack here var = var?
    return Template.from_text(name, '\n'.join(f"{i} = {{% {i} = {i} %}}" for i in variable_names))
