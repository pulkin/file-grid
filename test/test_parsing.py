from file_grid.parsing import iter_template_blocks, repr_pos, split_assignment
import pytest


def test_repr_pos():
    assert repr_pos("cat", 1) == "cat\n ^"
    assert repr_pos("\n\n\ncat", 4) == "cat\n ^"
    assert repr_pos("\n\n\ncat\nsmth", 4) == "cat\n ^"
    s = "cat with a very very very long tail"
    assert repr_pos(s, 1) == f"{s[:32]}...\n ^"
    s = "a little funny cat with a very very very long tail"
    assert repr_pos(s, 16) == f"{s[:32]}...\n" + " " * 16 + "^"
    assert repr_pos(s, 17) == f"...{s[1:33]}...\n" + " " * 19 + "^"
    assert repr_pos(s, 40) == f"...{s[24:]}\n" + " " * 19 + "^"
    assert repr_pos("one\ntwo", 3) == "Line break between the following lines:\none\ntwo"
    assert repr_pos("one\n", 3) == "Line break between the following lines:\none\n(empty line)"
    assert repr_pos("one\n\ntwo", 4) == "Line break between the following lines:\n(empty line)\ntwo"


def test_iter_template_blocks():
    """Tests whether template blocks are properly identified"""
    assert list(iter_template_blocks("casual text")) == [(0, "casual text")]
    assert list(iter_template_blocks("escaped left \\{%")) == [(0, "escaped left {%")]
    assert list(iter_template_blocks("escaped left \\{% ")) == [(0, "escaped left {% ")]
    assert list(iter_template_blocks("prefix{%block%}postfix")) == [(0, "prefix"), (8, "block"), (15, "postfix")]
    assert list(iter_template_blocks("{%block%}")) == [(0, ""), (2, "block"), (9, "")]
    assert list(iter_template_blocks("prefix{%block%}")) == [(0, "prefix"), (8, "block"), (15, "")]
    assert list(iter_template_blocks("{%block%}postfix")) == [(0, ""), (2, "block"), (9, "postfix")]
    assert list(iter_template_blocks("{%%}")) == [(0, ""), (2, ""), (4, "")]
    with pytest.raises(ValueError):
        list(iter_template_blocks("{%}"))
    assert list(iter_template_blocks("{%{%%}")) == [(0, ""), (2, "{%"), (6, "")]
    assert list(iter_template_blocks("{%\\{%%}")) == [(0, ""), (2, "\\{%"), (7, "")]
    assert list(iter_template_blocks("{%\\%}%}")) == [(0, ""), (2, "%}"), (7, "")]
    assert list(iter_template_blocks("{%\\%}\\%}%}")) == [(0, ""), (2, "%}%}"), (10, "")]
    assert list(iter_template_blocks("{%\\{%%}")) == [(0, ""), (2, "\\{%"), (7, "")]


def test_split_assignment():
    assert split_assignment("abc") == (None, None, "abc")
    assert split_assignment("  abc") == (None, None, "abc")
    assert split_assignment("a = b") == ("a", None, "b")
    assert split_assignment("a = b = c") == ("a", None, "b = c")
    with pytest.raises(ValueError):
        split_assignment("2a = b")

    assert split_assignment("abc:format") == (None, "format", "abc")
    assert split_assignment("x = abc : format") == ("x", " format", "abc")
