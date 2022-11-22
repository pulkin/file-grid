import pathlib
from tempfile import mkdtemp
from pathlib import Path
from subprocess import check_output, PIPE, CalledProcessError
import pytest
from pytest_steps import test_steps


def setup_folder(files: dict):
    """Sets up a temporary folder and puts files"""
    root = Path(mkdtemp())
    for path, contents in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        with open(root / path, 'w') as f:
            f.write(contents)
    return root


def run_grid(files, *args, path=Path("grid.py").absolute(), **kwargs):
    """Runs the script"""
    if isinstance(files, (str, pathlib.Path)):
        root = Path(files)
    elif isinstance(files, dict):
        root = setup_folder(files)
    else:
        raise NotImplementedError(f"{files=}")
    try:
        return root, check_output([path, *args], stderr=PIPE, text=True, cwd=root, **kwargs)
    except CalledProcessError as e:
        e.root_folder = root
        raise


def read_folder(root: Path, exclude=(".grid", ".grid.log")):
    """Reads the entire folder"""
    result = {
        str(i.relative_to(root)): open(i, "r").read() if i.is_file() else None
        for i in root.rglob("*")
        if not (i.is_dir() and len(list((root / i).glob("*"))))
    }
    for i in exclude:
        result.pop(i)
    return result


@pytest.mark.parametrize("files", [{}, {"description.txt": "something"}])
def test_empty(files):
    """Dummy setup with a single text file"""
    with pytest.raises(CalledProcessError) as e_info:
        run_grid(files, "new")
    e = e_info.value
    assert e.returncode == 1
    assert e.stderr == ""
    assert e.stdout == "No grid-formatted files found in this folder\n"


@pytest.mark.skip("to be implemented")
def test_const():
    """Constant expressions"""
    base = {"file_with_const": "{% 1 %} {% 'a' %} {% 3 %}"}
    root, output = run_grid(base, "new")
    assert output == ""
    assert read_folder(root) == {**base}  # TODO: update


@test_steps("grid new", "grid run", "grid cleanup")
def test_list():
    """List expressions as well as cleanup"""
    base = {"file_with_list": "{% [1, 2, 'a'] %}", "some_other_file": "abc"}
    root, output = run_grid(base, "new")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_with_list": "1",
        "grid1/file_with_list": "2",
        "grid2/file_with_list": "a",
    }
    yield

    root, output = run_grid(root, "run", "cat", "file_with_list")
    assert output == "\n".join([
        f"{str(root / 'grid0')}",
        "1",
        f"{str(root / 'grid1')}",
        "2",
        f"{str(root / 'grid2')}",
        "a\n",
    ])
    yield

    root, output = run_grid(root, "cleanup")
    assert output == ""
    assert read_folder(root, exclude=(".grid.log",)) == base
    yield


def test_range_1():
    """Range expressions"""
    base = {"file_with_range": "r={% range(5) %}"}
    root, output = run_grid(base, "new")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_with_range": "r=0",
        "grid1/file_with_range": "r=1",
        "grid2/file_with_range": "r=2",
        "grid3/file_with_range": "r=3",
        "grid4/file_with_range": "r=4",
    }


def test_range_2():
    """Range expressions"""
    base = {"file_with_range": "r={% range(5, 6) %}"}
    root, output = run_grid(base, "new")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_with_range": "r=5",
    }


def test_range_3():
    """Range expressions"""
    base = {"file_with_range": "r={% range(5, 10, 3) %}"}
    root, output = run_grid(base, "new")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_with_range": "r=5",
        "grid1/file_with_range": "r=8",
    }


def test_linspace():
    """Linspace expressions"""
    base = {"file_with_linspace": "l={% linspace(5, 10, 3) %}"}
    root, output = run_grid(base, "new")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_with_linspace": "l=5.0000000000",
        "grid1/file_with_linspace": "l=7.5000000000",
        "grid2/file_with_linspace": "l=10.0000000000",
    }


def test_dependency():
    """Dependency expressions"""
    base = {"file_with_dependency": "{% a = [1, 2, 3] %}, {% b = 2 * a %}, {% c = 3 * b %}"}
    root, output = run_grid(base, "new")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_with_dependency": "1, 2, 6",
        "grid1/file_with_dependency": "2, 4, 12",
        "grid2/file_with_dependency": "3, 6, 18",
    }


def test_loop_dependency():
    """Loop dependency expressions"""
    base = {"file_with_dependency": "{% a = b %}, {% b = 2 * a %} {% x = [1, 2] %}"}
    with pytest.raises(CalledProcessError) as e_info:
        run_grid(base, "new")
    e = e_info.value
    assert e.returncode == 1
    assert e.stderr.endswith("ValueError: 2 expressions cannot be evaluated: a, b\n")
    assert e.stdout == ""


def test_explicit_files():
    """Explicit files spec"""
    base = {
        "file_include": "{% [1, 2] %}",
        "file_include_static": "abc",
        "file_exclude": "{% [3, 4] %}",
        "file_exclude_static": "def",
    }
    root, output = run_grid(base, "new", "--files", "file_include", "file_include_static")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_include": "1",
        "grid0/file_include_static": "abc",
        "grid1/file_include": "2",
        "grid1/file_include_static": "abc",
    }


def test_static_files():
    """Explicit files spec"""
    base = {
        "file_with_list": "{% [1, 2] %}",
        "file_include_static": "abc {% [3, 4] %}",
    }
    root, output = run_grid(base, "new", "--static-files", "file_include_static")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_with_list": "1",
        "grid0/file_include_static": "abc {% [3, 4] %}",
        "grid1/file_with_list": "2",
        "grid1/file_include_static": "abc {% [3, 4] %}",
    }


def test_prefix():
    """Prefix option"""
    base = {"file_with_list": "{% [1, 2, 'a'] %}"}
    root, output = run_grid(base, "new", "-p", "custom")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "custom0/file_with_list": "1",
        "custom1/file_with_list": "2",
        "custom2/file_with_list": "a",
    }
    root, output = run_grid(root, "cleanup")
    assert output == ""
    assert read_folder(root, exclude=(".grid.log",)) == base
