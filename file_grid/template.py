from types import FunctionType

from .parsing import split_assignment, iter_template_blocks


class EvalBlock:
    def __init__(self, code, name, fmt):
        """Represents a statement that can be evaluated"""
        self.code = code
        self.name = name
        self.fmt = fmt

    @classmethod
    def from_string(cls, text, defined_file="unknown", defined_line="u", defined_char="u"):
        name, fmt, text = split_assignment(text)
        if name is None:
            defined_file = ''.join(i if i.isalnum() else "_" for i in defined_file)
            name = f"anonymous_{defined_file}_l{defined_line}c{defined_char}"
        code = compile(text, "<string>", "eval")
        return cls(code, name, fmt)

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

    def format_value(self, val):
        if self.fmt == "suppress":
            return ""
        elif self.fmt is None:
            return str(val)
        else:
            return f"{val:{self.fmt}}"

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"EvalBlock('{self.__str__()}')"


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
                chunks.append(EvalBlock.from_string(
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
            elif isinstance(chunk, EvalBlock):
                f.write(chunk.format_value(stack[chunk.name]))
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
