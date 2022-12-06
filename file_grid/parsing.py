import re


def repr_pos(text, pos, left=16, width=32, placeholder="..."):
    """Represent position in text"""
    if pos >= len(text):
        raise ValueError(f"invalid {pos=} for text of length {len(text)}")
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos + 1)
    if end == -1:
        end = len(text)
    repr_chunk = text[start:end]
    pos -= start

    if repr_chunk[pos] == "\n":
        two_lines = repr_chunk.split("\n")
        assert len(two_lines) == 2
        two_lines = list(
            "(empty line)" if len(i) == 0 else i if len(i) <= width else f"{i[:width]}{placeholder}"
            for i in two_lines
        )
        return f"Line break between the following lines:\n" + "\n".join(two_lines)

    else:
        if pos > left:
            repr_chunk = f"{placeholder}{repr_chunk[pos - left:]}"
            pos = left + len(placeholder)
            width += len(placeholder)
        if len(repr_chunk) > width:
            repr_chunk = f"{repr_chunk[:width]}{placeholder}"
        return f"{repr_chunk}\n" + ' ' * pos + "^"


def iter_template_blocks(text, left="{%", right="%}", escape="\\"):
    """Iterates over template blocks"""
    len_esc = len(escape)

    def _is_escaped_at(_pos):
        return _pos > len_esc and text[_pos - len_esc:_pos] == escape

    pos = 0

    def _iter_until(_pattern):
        nonlocal pos
        len_pattern = len(_pattern)
        while True:
            pattern_start = text.find(_pattern, pos)
            if pattern_start < 0:
                yield text[pos:]
                pos = len(text)
                yield None  # indicates end of file without finding the pattern
                break
            elif _is_escaped_at(pattern_start):
                yield text[pos:pattern_start - len_esc]
                yield _pattern
                pos = pattern_start + len_pattern
            else:
                yield text[pos:pattern_start]
                pos = pattern_start + len_pattern
                break

    while True:
        # find left
        pieces = list(_iter_until(left))
        if pieces[-1] is None:  # nothing left
            yield ''.join(pieces[:-1])
            break
        else:
            yield ''.join(pieces)
        start = pos

        # find right
        pieces = list(_iter_until(right))
        if pieces[-1] is None:  # did not find end of template
            raise ValueError(f"missing closing bracket for template starting at {start}:\n{repr_pos(text, start)}")
        else:
            yield ''.join(pieces)


def split_assignment(text, pattern=re.compile(r"^\s*(?P<name>[\w_]+)\s*=")):
    match = re.match(pattern, text)
    if match is None:
        return None, text.strip()
    name, text = match.group("name"), text[match.end():]
    if name[0] in "0123456789":
        raise ValueError(f"invalid variable name: {name}")
    return name, text.strip()
