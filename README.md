[![build](https://github.com/pulkin/file-grid/actions/workflows/test.yml/badge.svg)](https://github.com/pulkin/file-grid/actions)
[![pypi](https://img.shields.io/pypi/v/file-grid)](https://pypi.org/project/file-grid/)

# ![icon](resources/icon-full.svg)

Clone and multiply your files seamlessly using a simple template language.

What it is
----------

`grid` is a command-line tool to create *many* copies of *similar* files
using a **template** language.

`grid` scans current folder for template files with evaluate blocks and
expands them.
`grid` creates copies of template files and substitutes template blocks
(eval blocks) with their corresponding computed values.
As a result, text files with eval blocks such as
`something = {% range(3, 8) %}` replicate into 5 separate files containing
`something = 3`, `something = 4`, etc.

Install
-------

Install from pypi

```bash
pip install file-grid
```

Install from git using pip

```bash
pip install git+https://github.com/pulkin/file-grid.git#egg=file_grid
```

Build and install from source manually

```bash
git clone https://github.com/pulkin/file-grid.git
pip install build
python -m build
pip install dist/*.tar.gz
```

Run
---

```bash
python -m file_grid --help
```

or simply

```bash
grid --help
```

if your `PATH` includes python `bin` folder.

Example
-------

Suppose you have a single file `run.sh` in the folder `root`:

```
root
|- run.sh
```

The contents of `run.sh` includes the following script

```
run-some-program --some-arg=arg-value
```

You would like to make copies of this file where `--some-arg`
takes values from 0 to 9.
You turn `run.sh` into a template where `arg-value` is replaced with
eval block `{% range(10) %}` like this:

```
run-some-program --some-arg={% range(10) %}
```

Afterwards you invoke `grid new *` which takes care of interpreting
your template and creating copies of the file `run.sh` in 10 separate
folders named `grid0`, `grid1`, etc.

```
grid new
```

```
root
|- run.sh
|- grid0
|  |- run.sh
|
|- grid1
|  |- run.sh
|
|- grid2
|  |- run.sh
...
```

While the file `run.sh` in the root folder remains untouched, each copy
of the root folder `grid0` to `grid9` contains the file `run.sh` where
the `{% ... %}` eval block is substituted with one of its values `range(10)`:
i.e. `0`, `1`, `2`, etc.
For example, the contents of `root/grid4/run.sh` is

```
run-some-program --some-arg=4
```

To execute each copy of `run.sh` simply add `--exec` argument as
`grid new * --exec grid{id}/run.sh`
which runs 10 copies of the file one after another.

Template language
-----------------

By default, `grid` scans for all files and attempts to locate brackets
`{% ... %}`.
The expression inside has to be a valid python `compile(..., 'eval')`
statement.

### Grid: multiple brackets

Consider the following file.

```
run-some-program --some-arg={% range(10) %} --another-arg={% range(3) %}
```

It will be expanded into 30 copies with all possible combinations of the
two arguments / eval block values.

### Dependent statements

It is possible to re-use computed eval blocks as a part of an expression
in another eval block.
For this, named blocks are available as `{% name = expression %}`.
For example,

```
run-some-program --some-arg={% a = range(10) %} --another-arg={% a + 3 %}
```

In this case, 10 copies are created where the value of the second block
(`--another-arg`) is always the value substituted in the first block plus 3.

### Multiple files

Multiple files are treated as if it is a single file (i.e. all dependent
blocks belong to the same scope and all named blocks are shared).

### Formatting

Supported through the usual `{% [1, 2, 3]:.3f %}` postfix notation.
The supress `supress` postfix will format into empty string:
`{% block = [1, 2, 3]:supress %}`.

### Useful implementation details

- All python types are supported: integers, floats, strings, objects, etc.
  For example, this is a valid eval block: `{% ['a', 2, 3.] %}`.
- Anonymous eval blocks such as the above are assigned an
  `anonymous_{file}_l{line}c{char_in_line}` name.
- Currently, only `range` and `linspace` are available as builtins.
  TBD: will be fixed.
- To see variable values after the grid was created simply look into
  the corresponding `.variables` file.
- A two-phase scheme is used when evaluating blocks.
  At the first stage, blocks without dependencies are identified and
  computed.
  At the second stage, all dependent templates are computed.
- Under the hood, blocks are compiled into python code objects in `eval`
  mode and name dependencies are determined via `code.co_names`.
- `__grid_id__` with grid sequence id is injected into eval scope
  at the second stage.
- The grid size (shape) is defined by the (cartesian) product of all
  values of independent eval blocks.
  If the computed template value results in an object with `__len__`
  attribute it will be treated as-is.
  Otherwise, the object (for example, integer or float) will be replaced
  with a single-element list. I.e. the effect of `{% a = 1 %}` and
  `{% a = [1] %}` is the same.
  Instead, `{% a = 'abc' %}` will iterate over individual characters 'a',
  'b', 'c' while `{% a = ['abc'] %}` will produce a single value 'abc'.
- For the sake of simplicity, the closing bracket `%}` has the highest
  priority when parsing.
  In the following template `{% "%}" %}` the eval block part is `{% "%}`.
  To make a valid expression, escaping is necessary `{% "\%}" %}` resulting
  in `"%}"` as its computed value.
  Both `"{%"` inside the template block or `"%}"` outside of it are treated
  as-is without the need of escaping.
