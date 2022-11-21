from tempfile import mkdtemp
from pathlib import Path
from subprocess import check_output, PIPE, CalledProcessError
import pytest


def setup_folder(files: dict):
    """Sets up a temporary folder and puts files"""
    root = Path(mkdtemp())
    for path, contents in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        with open(root / path, 'w') as f:
            f.write(contents)
    return root


def run_grid(files: dict, *args, path=Path("grid.py").absolute(), **kwargs):
    """Runs the script"""
    root = setup_folder(files)
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


def test_list():
    """List expressions"""
    base = {"file_with_list": "{% [1, 2, 'a'] %}"}
    root, output = run_grid(base, "new")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_with_list": "1",
        "grid1/file_with_list": "2",
        "grid2/file_with_list": "a",
    }


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
