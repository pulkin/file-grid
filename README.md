[![build](https://github.com/pulkin/file-grid/actions/workflows/test.yml/badge.svg)](https://github.com/pulkin/file-grid/actions)
[![pypi](https://img.shields.io/pypi/v/file-grid)](https://pypi.org/project/file-grid/)

# ![icon](resources/icon-full.svg)

Clone and multiply your files seamlessly using a simple template language.

What it is
----------

`grid` is a command-line tool to create *many* copies of *similar* files
using a **template** language.

`grid` searches for text files with specially-formatted expressions
and explodes them.
`grid` creates copies of the files where expressions are replaced
with their corresponding evaluations.
For example, a text file with the following expression
`Hi my name is {% ['Alice', 'Bob', 'Carol' %}`
will explode into 3 separate files `Hi my name is Alice`,
`Hi my name is Bob`, `Hi my name is Carol`.

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

assuming your `PATH` includes python `bin` folder.

Example
-------

Suppose you have a single text file `run.sh` in the `root` folder:

```
root
|- run.sh
```

`run.sh` includes the following text

```
run-some-program --some-arg=arg-value
```

You would like to make copies of `run.sh` where `--some-arg`
takes integer values from 0 to 9.
You introduce a python expression within brackets `{%` and `%}`
to define a "grid" as follows

```
run-some-program --some-arg={% range(10) %}
```

Afterwards you invoke `grid new *` in the folder `root` which
takes care of interpreting the template and creating
10 different files `run.sh` in 10 different folders named
`grid0`, `grid1`, etc.

```
grid new *
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

For example, the file `root/grid4/run.sh` includes the following
expanded text

```
run-some-program --some-arg=4
```

To execute each copy of `run.sh` simply add `--exec` argument as
below

`grid new * --exec grid{id}/run.sh`

It will run each copy of the file one after another.

Template language
-----------------

`grid` attempts to locate bracketed expressions `{% ... %}`.
Each expression has to be a valid python `compile(..., 'eval')`
statement.

### Multiple expressions

Consider the following file

```
run-some-program --some-arg={% range(10) %} --another-arg={% range(3) %}
```

It will be expanded into 30 copies (`10x3` cartesian product or a `10x3`
grid) with all possible combinations of the two arguments `range(10)`
and `range(3)`.

### Dependent statements

It is possible to re-use computed values in *dependent* expressions.
Example:

```
run-some-program --some-arg={% a = range(10) %} --another-arg={% a + 3 %}
```

In this case 10 copies are created:

```
run-some-program --some-arg=0 --another-arg=3
run-some-program --some-arg=1 --another-arg=4
run-some-program --some-arg=2 --another-arg=5
...
run-some-program --some-arg=9 --another-arg=12
```

### Multiple files

All expressions share the same scope across all files.

### Formatting

Standard python formatting is supported through the
`{% [1, 2, 3]:.3f %}` postfix notation.
An additional `suppress` postfix will always format
into empty string, example:
`{% block = [1, 2, 3]:suppress %}`.

### Implementation details

- All python types are supported: integers, floats, strings, objects, etc.
  For example, this is a valid eval block: `{% ['a', 2, 3.] %}`.
- Anonymous eval blocks such as the above are assigned an
  `anonymous_{file}_l{line}c{char_in_line}` name.
- Currently, only `range` and `linspace` are available as builtins.
- An implicit `.variables` file includes all expressions computed.
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
  In the following template `{% "%}" %}` the expression part is `{% "%}`.
  To make a valid expression, escaping is necessary `{% "\%}" %}` resulting
  in `"%}"` as its computed value.
  Both `"{%"` inside the template block or `"%}"` outside of it are treated
  as-is without the need of escaping.
