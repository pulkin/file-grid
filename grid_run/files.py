from .template import EvalBlock
from .parsing import iter_template_blocks


class Template:
    def __init__(self, name, chunks):
        """A file with multiple statements to evaluate"""
        self.name = name
        self.chunks = chunks

    @classmethod
    def from_text(cls, name, text):
        itr = iter_template_blocks(text)
        chunks = []
        for i in itr:
            chunks.append(i)
            try:
                chunks.append(EvalBlock.from_string(next(itr), extra_s=f" [defined in {repr(name)}]"))
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
