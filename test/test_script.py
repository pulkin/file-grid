import pathlib
import subprocess
from tempfile import mkdtemp
from pathlib import Path
from subprocess import check_output, PIPE, CalledProcessError
import shutil
import pytest
from pytest_steps import test_steps


@pytest.fixture(scope="session")
def grid_script(pytestconfig):
    return pytestconfig.getoption("grid_script")


def setup_folder(files: dict, root=None):
    """Sets up a temporary folder and puts files"""
    if root is None:
        root = Path(mkdtemp())
    for path, contents in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        with open(root / path, 'w') as f:
            f.write(contents)
    return root


def run_grid(files, grid_script, *args, **kwargs):
    """Runs the script"""
    if isinstance(files, (str, pathlib.Path)):
        root = Path(files)
    elif isinstance(files, dict):
        root = setup_folder(files)
    else:
        raise NotImplementedError(f"{files=}")
    try:
        return root, check_output([*grid_script.split(), *args], stderr=PIPE, text=True, cwd=root, **kwargs)
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
def test_raise_empty(grid_script, files):
    """Dummy setup with a single text file"""
    with pytest.raises(CalledProcessError) as e_info:
        run_grid(files, grid_script, "new")
    e = e_info.value
    assert e.returncode == 1
    assert e.stderr == ""
    assert e.stdout == "No grid-formatted files found in this folder\n"


def test_raise_non_existent(grid_script):
    """Dummy setup with a single text file"""
    with pytest.raises(CalledProcessError) as e_info:
        run_grid({}, grid_script, "run", "something")
    e = e_info.value
    assert e.returncode == 1
    assert e.stderr.endswith("Grid file does not exit: '.grid'\n")
    assert e.stdout == ""


@pytest.mark.skip("to be implemented")
def test_const(grid_script):
    """Constant expressions"""
    base = {"file_with_const": "{% 1 %} {% 'a' %} {% 3 %}"}
    root, output = run_grid(base, grid_script, "new")
    assert output == ""
    assert read_folder(root) == {**base}  # TODO: update


@test_steps("grid new", "grid run", "grid distribute", "grid cleanup")
def test_list(grid_script):
    """List expressions as well as cleanup"""
    base = {"file_with_list": "{% a = [1, 2, 'a'] %}", "some_other_file": "abc"}
    root, output = run_grid(base, grid_script, "new")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_with_list": "1",
        "grid1/file_with_list": "2",
        "grid2/file_with_list": "a",
    }
    yield

    root, output = run_grid(root, grid_script, "run", "cat", "file_with_list")
    assert output == "\n".join([
        "grid0",
        "1",
        "grid1",
        "2",
        "grid2",
        "a\n",
    ])
    yield

    payload = {"additional_file": "{% a * 2 %}"}
    setup_folder(payload, root)
    base = {**base, **payload}
    root, output = run_grid(root, grid_script, "distribute", "additional_file")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_with_list": "1",
        "grid0/additional_file": "2",
        "grid1/file_with_list": "2",
        "grid1/additional_file": "4",
        "grid2/file_with_list": "a",
        "grid2/additional_file": "aa",
    }
    yield

    root, output = run_grid(root, grid_script, "cleanup")
    assert output == ""
    assert read_folder(root, exclude=(".grid.log",)) == base
    yield


@test_steps("grid new", "grid run", "grid distribute", "grid cleanup")
def test_list_missing(grid_script):
    """List expressions as well as cleanup"""
    base = {"file_with_list": "{% a = [1, 2, 'a'] %}", "some_other_file": "abc"}
    root, output = run_grid(base, grid_script, "new")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_with_list": "1",
        "grid1/file_with_list": "2",
        "grid2/file_with_list": "a",
    }
    yield

    # remove one of grid folders
    shutil.rmtree(root / "grid1")

    with pytest.raises(subprocess.SubprocessError) as e_info:
        run_grid(root, grid_script, "run", "cat", "file_with_list")
    e = e_info.value
    assert e.returncode == 1
    assert e.stderr.endswith("No such file or directory: \'grid1\'\n")
    assert e.stdout == "\n".join([
        "grid0",
        "1",
        "grid1",
        "Failed to execute cat file_with_list (working directory 'grid1')",
        "grid2",
        "a\n",
    ])
    yield

    payload = {"additional_file": "{% a * 2 %}"}
    setup_folder(payload, root)
    base = {**base, **payload}
    with pytest.raises(subprocess.SubprocessError) as e_info:
        run_grid(root, grid_script, "distribute", "additional_file")
    e = e_info.value
    assert e.returncode == 1
    assert e.stderr.endswith("No such file or directory: \'grid1\'\n")
    assert e.stdout == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_with_list": "1",
        "grid0/additional_file": "2",
        "grid2/file_with_list": "a",
        "grid2/additional_file": "aa",
    }
    yield

    with pytest.raises(subprocess.SubprocessError) as e_info:
        run_grid(root, grid_script, "cleanup")
    e = e_info.value
    assert e.returncode == 1
    assert e.stderr.endswith("No such file or directory: \'grid1\'\n")
    assert e.stdout == ""
    assert read_folder(root, exclude=(".grid.log",)) == base
    yield


def test_range_1(grid_script):
    """Range expressions"""
    base = {"file_with_range": "r={% range(5) %}"}
    root, output = run_grid(base, grid_script, "new")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_with_range": "r=0",
        "grid1/file_with_range": "r=1",
        "grid2/file_with_range": "r=2",
        "grid3/file_with_range": "r=3",
        "grid4/file_with_range": "r=4",
    }


def test_range_2(grid_script):
    """Range expressions"""
    base = {"file_with_range": "r={% range(5, 6) %}"}
    root, output = run_grid(base, grid_script, "new")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_with_range": "r=5",
    }


def test_range_3(grid_script):
    """Range expressions"""
    base = {"file_with_range": "r={% range(5, 10, 3) %}"}
    root, output = run_grid(base, grid_script, "new")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_with_range": "r=5",
        "grid1/file_with_range": "r=8",
    }


def test_linspace(grid_script):
    """Linspace expressions"""
    base = {"file_with_linspace": "l={% linspace(5, 10, 3) %}"}
    root, output = run_grid(base, grid_script, "new")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_with_linspace": "l=5.0000000000",
        "grid1/file_with_linspace": "l=7.5000000000",
        "grid2/file_with_linspace": "l=10.0000000000",
    }


def test_dependency(grid_script):
    """Dependency expressions"""
    base = {"file_with_dependency": "{% a = [1, 2, 3] %}, {% b = 2 * a %}, {% c = 3 * b %}"}
    root, output = run_grid(base, grid_script, "new")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_with_dependency": "1, 2, 6",
        "grid1/file_with_dependency": "2, 4, 12",
        "grid2/file_with_dependency": "3, 6, 18",
    }


def test_loop_dependency(grid_script):
    """Loop dependency expressions"""
    base = {"file_with_dependency": "{% a = b %}, {% b = 2 * a %} {% x = [1, 2] %}"}
    with pytest.raises(CalledProcessError) as e_info:
        run_grid(base, grid_script, "new")
    e = e_info.value
    assert e.returncode == 1
    assert e.stderr.endswith("ValueError: 2 expressions cannot be evaluated: a, b\n")
    assert e.stdout == ""


def test_explicit_files(grid_script):
    """Explicit files spec"""
    base = {
        "file_include": "{% [1, 2] %}",
        "file_include_static": "abc",
        "file_exclude": "{% [3, 4] %}",
        "file_exclude_static": "def",
    }
    root, output = run_grid(base, grid_script, "new", "--files", "file_include", "file_include_static")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_include": "1",
        "grid0/file_include_static": "abc",
        "grid1/file_include": "2",
        "grid1/file_include_static": "abc",
    }


def test_static_files(grid_script):
    """Explicit files spec"""
    base = {
        "file_with_list": "{% [1, 2] %}",
        "file_include_static": "abc {% [3, 4] %}",
    }
    root, output = run_grid(base, grid_script, "new", "--static-files", "file_include_static")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_with_list": "1",
        "grid0/file_include_static": "abc {% [3, 4] %}",
        "grid1/file_with_list": "2",
        "grid1/file_include_static": "abc {% [3, 4] %}",
    }


@test_steps("grid new", "grid cleanup")
def test_name(grid_script):
    """Prefix option"""
    base = {"file_with_list": "{% [1, 2, 'a'] %}"}
    root, output = run_grid(base, grid_script, "new", "-n", "custom%d")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "custom0/file_with_list": "1",
        "custom1/file_with_list": "2",
        "custom2/file_with_list": "a",
    }
    yield

    root, output = run_grid(root, grid_script, "cleanup")
    assert output == ""
    assert read_folder(root, exclude=(".grid.log",)) == base
    yield
