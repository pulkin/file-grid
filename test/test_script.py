import pathlib
import subprocess
from tempfile import mkdtemp
from pathlib import Path
from subprocess import check_output, PIPE, CalledProcessError
import shutil
import sys
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
        if contents is None:
            (root / path).mkdir(parents=True, exist_ok=True)
        else:
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
        sys.stdout.write(e.stdout)
        sys.stderr.write(e.stderr)
        e.root_folder = root
        raise


def read_folder(root: Path, exclude=(".grid", ".grid.json", ".grid.log")):
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
        run_grid(files, grid_script, "new", "*")
    e = e_info.value
    assert e.returncode == 1
    assert e.stderr.endswith(f"pattern '*' in '.' matched 0 files (matched total: {len(files) + 1}, "
                             f"files: {len(files)})\n")
    assert e.stdout == ""


def test_raise_run_no_grid(grid_script):
    """Empty setup"""
    with pytest.raises(CalledProcessError) as e_info:
        run_grid({}, grid_script, "run", "something")
    e = e_info.value
    assert e.returncode == 1
    assert e.stderr.endswith("Grid file does not exit: '.grid.json'\n")
    assert e.stdout == ""


def test_raise_run_no_executable(grid_script):
    """Errors with grid run"""
    base = {"file_with_list": "{% a = [1, 2, 'a'] %}"}
    root, output = run_grid(base, grid_script, "new", "*")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_with_list": "1",
        "grid0/.variables": "a = 1",
        "grid1/file_with_list": "2",
        "grid1/.variables": "a = 2",
        "grid2/file_with_list": "a",
        "grid2/.variables": "a = a",
    }

    with pytest.raises(CalledProcessError) as e_info:
        run_grid(root, grid_script, "run", "something-non-existent")
    e = e_info.value
    assert e.returncode == 1
    assert e.stderr.endswith("No such file or directory: 'something-non-existent'\n")
    assert e.stdout == "\n".join([
        "grid0: something-non-existent",
        "something-non-existent: file not found (working directory 'grid0')",
        "grid1: something-non-existent",
        "something-non-existent: file not found (working directory 'grid1')",
        "grid2: something-non-existent",
        "something-non-existent: file not found (working directory 'grid2')\n",
    ])


def test_raise_run_subprocess_error(grid_script):
    """Errors with grid run"""
    base = {"file_with_list": "{% a = [1, 2, 'a'] %}"}
    root, output = run_grid(base, grid_script, "new", "*")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_with_list": "1",
        "grid0/.variables": "a = 1",
        "grid1/file_with_list": "2",
        "grid1/.variables": "a = 2",
        "grid2/file_with_list": "a",
        "grid2/.variables": "a = a",
    }

    with pytest.raises(CalledProcessError) as e_info:
        run_grid(root, grid_script, "run", "cat", "non-existent")
    e = e_info.value
    assert e.returncode == 1
    assert e.stderr.endswith("Command \'[\'cat\', \'non-existent\']\' returned non-zero exit status 1.\n")
    assert e.stdout == "\n".join([
        "grid0: cat non-existent",
        "cat non-existent: process error (working directory 'grid0')",
        "cat: non-existent: No such file or directory",
        "grid1: cat non-existent",
        "cat non-existent: process error (working directory 'grid1')",
        "cat: non-existent: No such file or directory",
        "grid2: cat non-existent",
        "cat non-existent: process error (working directory 'grid2')",
        "cat: non-existent: No such file or directory\n",
    ])


def test_raise_grid_folder_exists(grid_script):
    """Conflicting folder setup"""
    with pytest.raises(CalledProcessError) as e_info:
        run_grid({"some_list": "{% [1, 2, 3] %}", "grid1/some_list": ""}, grid_script, "new", "*")
    e = e_info.value
    assert e.returncode == 1
    assert e.stderr.endswith("file or folder 'grid1/some_list' already exists\n")
    assert e.stdout == ""


def test_const(grid_script):
    """Constant expressions"""
    base = {"file_with_const": "{% 1 %} {% 'a' %} {% 3 %}"}
    root, output = run_grid(base, grid_script, "new", "*")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_with_const": "1 a 3",
        "grid0/.variables": "anonymous_file_with_const_l1c11 = a\n"
                            "anonymous_file_with_const_l1c21 = 3\n"
                            "anonymous_file_with_const_l1c3 = 1",
    }


@test_steps("new", "run", "update", "cleanup")
def test_list(grid_script):
    """List expressions as well as cleanup"""
    base = {"file_with_list": "{% a = [1, 2, 'a'] %}", "some_other_file": "abc"}
    root, output = run_grid(base, grid_script, "new", "*")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_with_list": "1",
        "grid0/.variables": "a = 1",
        "grid1/file_with_list": "2",
        "grid1/.variables": "a = 2",
        "grid2/file_with_list": "a",
        "grid2/.variables": "a = a",
    }
    yield

    root, output = run_grid(root, grid_script, "run", "cat", "file_with_list")
    assert output == "\n".join([
        "grid0: cat file_with_list",
        "1",
        "grid1: cat file_with_list",
        "2",
        "grid2: cat file_with_list",
        "a\n",
    ])
    yield

    payload = {"additional_file": "{% a * 2 %}"}
    setup_folder(payload, root)
    base = {**base, **payload}
    root, output = run_grid(root, grid_script, "new", "*", "-f")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_with_list": "1",
        "grid0/additional_file": "2",
        "grid0/.variables": "a = 1\n"
                            "anonymous_additional_file_l1c3 = 2",
        "grid1/file_with_list": "2",
        "grid1/additional_file": "4",
        "grid1/.variables": "a = 2\n"
                            "anonymous_additional_file_l1c3 = 4",
        "grid2/file_with_list": "a",
        "grid2/additional_file": "aa",
        "grid2/.variables": "a = a\n"
                            "anonymous_additional_file_l1c3 = aa",
    }
    yield

    root, output = run_grid(root, grid_script, "cleanup")
    assert output == ""
    assert read_folder(root, exclude=(".grid.log",)) == base
    yield


@test_steps("new", "run", "update", "cleanup")
def test_list_missing(grid_script):
    """List expressions as well as cleanup"""
    base = {"file_with_list": "{% a = [1, 2, 'a'] %}", "some_other_file": "abc"}
    root, output = run_grid(base, grid_script, "new", "*")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_with_list": "1",
        "grid0/.variables": "a = 1",
        "grid1/file_with_list": "2",
        "grid1/.variables": "a = 2",
        "grid2/file_with_list": "a",
        "grid2/.variables": "a = a",
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
        "grid0: cat file_with_list",
        "1",
        "grid1: cat file_with_list",
        "cat file_with_list: file not found (working directory 'grid1')",
        "grid2: cat file_with_list",
        "a\n",
    ])
    yield

    payload = {"additional_file": "{% a * 2 %}"}
    setup_folder(payload, root)
    base = {**base, **payload}
    root, output = run_grid(root, grid_script, "new", "*", "-f")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_with_list": "1",
        "grid0/additional_file": "2",
        "grid0/.variables": "a = 1\n"
                            "anonymous_additional_file_l1c3 = 2",
        "grid1/file_with_list": "2",
        "grid1/additional_file": "4",
        "grid1/.variables": "a = 2\n"
                            "anonymous_additional_file_l1c3 = 4",
        "grid2/file_with_list": "a",
        "grid2/additional_file": "aa",
        "grid2/.variables": "a = a\n"
                            "anonymous_additional_file_l1c3 = aa",
    }
    yield

    root, output = run_grid(root, grid_script, "cleanup")
    assert output == ""
    assert read_folder(root, exclude=(".grid.log",)) == base
    yield


def test_range_1(grid_script):
    """Range expressions"""
    base = {"file_with_range": "r={% range(5) %}"}
    root, output = run_grid(base, grid_script, "new", "*")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_with_range": "r=0",
        "grid0/.variables": "anonymous_file_with_range_l1c5 = 0",
        "grid1/file_with_range": "r=1",
        "grid1/.variables": "anonymous_file_with_range_l1c5 = 1",
        "grid2/file_with_range": "r=2",
        "grid2/.variables": "anonymous_file_with_range_l1c5 = 2",
        "grid3/file_with_range": "r=3",
        "grid3/.variables": "anonymous_file_with_range_l1c5 = 3",
        "grid4/file_with_range": "r=4",
        "grid4/.variables": "anonymous_file_with_range_l1c5 = 4",
    }


def test_range_2(grid_script):
    """Range expressions"""
    base = {"file_with_range": "r={% range(5, 6) %}"}
    root, output = run_grid(base, grid_script, "new", "*")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_with_range": "r=5",
        "grid0/.variables": "anonymous_file_with_range_l1c5 = 5",
    }


def test_range_3(grid_script):
    """Range expressions"""
    base = {"file_with_range": "r={% range(5, 10, 3) %}"}
    root, output = run_grid(base, grid_script, "new", "*")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_with_range": "r=5",
        "grid0/.variables": "anonymous_file_with_range_l1c5 = 5",
        "grid1/file_with_range": "r=8",
        "grid1/.variables": "anonymous_file_with_range_l1c5 = 8",
    }


def test_linspace(grid_script):
    """Linspace expressions"""
    base = {"file_with_linspace": "l={% linspace(5, 10, 3) %}"}
    root, output = run_grid(base, grid_script, "new", "*")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_with_linspace": "l=5.0",
        "grid0/.variables": "anonymous_file_with_linspace_l1c5 = 5.0",
        "grid1/file_with_linspace": "l=7.5",
        "grid1/.variables": "anonymous_file_with_linspace_l1c5 = 7.5",
        "grid2/file_with_linspace": "l=10.0",
        "grid2/.variables": "anonymous_file_with_linspace_l1c5 = 10.0",
    }


def test_dependency(grid_script):
    """Dependency expressions"""
    base = {"file_with_dependency": "{% a = [1, 2, 3] %}, {% b = 2 * a %}, {% c = 3 * b %}"}
    root, output = run_grid(base, grid_script, "new", "*")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_with_dependency": "1, 2, 6",
        "grid0/.variables": "a = 1\nb = 2\nc = 6",
        "grid1/file_with_dependency": "2, 4, 12",
        "grid1/.variables": "a = 2\nb = 4\nc = 12",
        "grid2/file_with_dependency": "3, 6, 18",
        "grid2/.variables": "a = 3\nb = 6\nc = 18",
    }


def test_loop_dependency(grid_script):
    """Loop dependency expressions"""
    base = {"file_with_dependency": "{% a = b %}, {% b = 2 * a %} {% x = [1, 2] %}"}
    with pytest.raises(CalledProcessError) as e_info:
        run_grid(base, grid_script, "new", "*")
    e = e_info.value
    assert e.returncode == 1
    assert e.stderr.endswith("2 expressions cannot be evaluated:\n"
                             "a: missing \'b\'\n"
                             "b: missing \'a\'\n")
    assert e.stdout == ""


def test_explicit_files(grid_script):
    """Explicit files spec"""
    base = {
        "file_include": "{% [1, 2] %}",
        "file_include_static": "abc",
        "file_exclude": "{% [3, 4] %}",
        "file_exclude_static": "def",
    }
    root, output = run_grid(base, grid_script, "new", "file_include", "--static", "file_include_static")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_include": "1",
        "grid0/.variables": "anonymous_file_include_l1c3 = 1",
        "grid0/file_include_static": "abc",
        "grid1/file_include": "2",
        "grid1/.variables": "anonymous_file_include_l1c3 = 2",
        "grid1/file_include_static": "abc",
    }


def test_static_files(grid_script):
    """Explicit files spec"""
    base = {
        "file_with_list": "{% [1, 2] %}",
        "file_include_static": "abc {% [3, 4] %}",
    }
    root, output = run_grid(base, grid_script, "new", "*", "--static", "file_include_static")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_with_list": "1",
        "grid0/.variables": "anonymous_file_with_list_l1c3 = 1",
        "grid0/file_include_static": "abc {% [3, 4] %}",
        "grid1/file_with_list": "2",
        "grid1/.variables": "anonymous_file_with_list_l1c3 = 2",
        "grid1/file_include_static": "abc {% [3, 4] %}",
    }


@test_steps("new", "cleanup")
def test_pattern(grid_script):
    """Prefix option"""
    base = {"file_with_list": "{% [1, 2, 'a'] %}"}
    root, output = run_grid(base, grid_script, "new", "*", "-p", "custom{id}/{name}")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "custom0/file_with_list": "1",
        "custom0/.variables": "anonymous_file_with_list_l1c3 = 1",
        "custom1/file_with_list": "2",
        "custom1/.variables": "anonymous_file_with_list_l1c3 = 2",
        "custom2/file_with_list": "a",
        "custom2/.variables": "anonymous_file_with_list_l1c3 = a",
    }
    yield

    root, output = run_grid(root, grid_script, "cleanup")
    assert output == ""
    assert read_folder(root, exclude=(".grid.log",)) == base
    yield


@test_steps("new", "cleanup")
def test_pattern_no_folders(grid_script):
    """Prefix option"""
    base = {"file_with_list": "{% [1, 2, 'a'] %}"}
    root, output = run_grid(base, grid_script, "new", "*", "-p", "{name}.{id}")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "file_with_list.0": "1",
        ".variables.0": "anonymous_file_with_list_l1c3 = 1",
        "file_with_list.1": "2",
        ".variables.1": "anonymous_file_with_list_l1c3 = 2",
        "file_with_list.2": "a",
        ".variables.2": "anonymous_file_with_list_l1c3 = a",
    }
    yield

    root, output = run_grid(root, grid_script, "cleanup")
    assert output == ""
    assert read_folder(root, exclude=(".grid.log",)) == base
    yield


def test_recursive(grid_script):
    """Prefix option"""
    base = {"sub/file_with_list": "{% [1, 2, 'a'] %}"}
    root, output = run_grid(base, grid_script, "new", "*", "-r")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/sub/file_with_list": "1",
        "grid0/.variables": "anonymous_sub_file_with_list_l1c3 = 1",
        "grid1/sub/file_with_list": "2",
        "grid1/.variables": "anonymous_sub_file_with_list_l1c3 = 2",
        "grid2/sub/file_with_list": "a",
        "grid2/.variables": "anonymous_sub_file_with_list_l1c3 = a",
    }


def test_non_recursive(grid_script):
    """Prefix option"""
    base = {"sub/file_with_list": "{% [1, 2, 'a'] %}"}
    with pytest.raises(CalledProcessError) as e_info:
        run_grid(base, grid_script, "new", "*")
    e = e_info.value
    assert e.returncode == 1
    assert e.stderr.endswith("pattern '*' in '.' matched 0 files (matched total: 2, files: 0)\n")
    assert e.stdout == ""


def test_formatting(grid_script):
    """Formatting"""
    base = {"file_with_linspace": "{% linspace(0, 1, 4):.2f %}"}
    root, output = run_grid(base, grid_script, "new", "*")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_with_linspace": "0.00",
        "grid0/.variables": "anonymous_file_with_linspace_l1c3 = 0.0",
        "grid1/file_with_linspace": "0.33",
        "grid1/.variables": "anonymous_file_with_linspace_l1c3 = 0.3333333333333333",
        "grid2/file_with_linspace": "0.67",
        "grid2/.variables": "anonymous_file_with_linspace_l1c3 = 0.6666666666666666",
        "grid3/file_with_linspace": "1.00",
        "grid3/.variables": "anonymous_file_with_linspace_l1c3 = 1.0",
    }


def test_suppress(grid_script):
    """Formatting"""
    base = {"file_with_suppressed": "nothing {% a = [1, 2, 3]:suppress %} here"}
    root, output = run_grid(base, grid_script, "new", "*")
    assert output == ""
    assert read_folder(root) == {
        **base,
        "grid0/file_with_suppressed": "nothing  here",
        "grid0/.variables": "a = 1",
        "grid1/file_with_suppressed": "nothing  here",
        "grid1/.variables": "a = 2",
        "grid2/file_with_suppressed": "nothing  here",
        "grid2/.variables": "a = 3",
    }


def test_dry_new(grid_script):
    """Test dry run"""
    base = {"file_with_list": "{% a = [1, 2, 'a'] %}"}
    root, output = run_grid(base, grid_script, "new", "*", "--dry")
    assert output == ""
    assert read_folder(root, exclude=(".grid.log",)) == base


def test_dry_cleanup(grid_script):
    """Test dry run"""
    base = {"file_with_list": "{% a = [1, 2, 'a'] %}"}
    root, _ = run_grid(base, grid_script, "new", "*")
    ref_list = read_folder(root)
    root, output = run_grid(root, grid_script, "cleanup", "--dry")
    assert output == ""
    assert read_folder(root) == ref_list
