[![build](https://github.com/pulkin/grid-run/actions/workflows/test.yml/badge.svg)](https://github.com/pulkin/grid-run/actions)

grid-run
========

Clone and multiply your HPC jobs seamlessly using a simple template language.

What it is
----------

`grid-run` is a program to create copies of files based on templates.
It scans the given folder or file for template pieces and expands all templates by creating copies of
the folder or file where templates are substituted with their corresponding computed values.

Install
-------

Install from git

```bash
pip install git+https://github.com/pulkin/grid-run.git#egg=grid_run
```

Build and install from sources

```bash
pip install build
python -m build
pip install dist/*.tar.gz
```

Run
---

```bash
python -m grid_run
```

or simply

```bash
grid
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
the following template:

```
run-some-program --some-arg={% range(10) %}
```

Afterwards you invoke `grid new` which takes care of interpreting
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
the `{% ... %}` block is substituted with one of the values of `range(10)`,
i.e. `0`, `1`, `2`, etc.
For example, the contents of `root/grid4/run.sh` is

```
run-some-program --some-arg=4
```

Now, executing each copy of `run.sh` is as simple as `grid run ./run.sh`
which runs 10 copies one after another within their corresponding grid
folders.

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
two arguments / template values.

### Dependent statements

It is possible to re-use computed template values as a part of expression
in another template.
For this, named expressions are available as `{% name = expression %}`.
For example,

```
run-some-program --some-arg={% a = range(10) %} --another-arg={% a + 3 %}
```

In this case, 10 copies are created where the value of the second template
(`--another-arg`) is always the value substituted in the first template plus 3.

### Multiple files

Multiple files are treated as if it is a single file (i.e. all dependent
statements belong to the same scope and all named expressions are shared).

### Formatting

TBD: not implemented yet.
Currently, `str(...)` is used when writing expanded expressions.

### Other details

- All python types are supported: integers, floats, strings, objects, etc.
  For example, this is a valid template: `{% ['a', 2, 3.] %}`.
- Anonymous expressions such as the above are assigned a random UUID.
- Currently, only `range` and `linspace` are available as builtins.
  TBD: will be fixed.
- A two-phase scheme is used when evaluating templates.
  At the first stage, templates without dependencies are identified and
  computed.
  At the second stage, all dependent templates are computed.
- `__grid_folder_name__` with the grid folder name and
  `__grid_id__` with grid sequence id are injected into template scope
  at the second stage.
- The grid size (shape) is defined by the (cartesian) product of all
  values of independent templates.
  If the computed template value results in an object with `__len__`
  attribute it will be iterated over.
  Otherwise, the object (for example, integer or float) will be replaced
  with a list. I.e. the effect of `{% a = 1 %}` and `{% a = [1] %}` is the
  same.
  Instead, `{% a = 'abc' %}` will iterate over individual characters 'a',
  'b', 'c' while `{% a = ['abc'] %}` will produce a single value 'abc'.
- For the sake of simplicity, the closing bracket `%}` has the highest
  priority when parsing.
  In the following `{% "%}" %}` the template part is `{% "%}` only
  while the rest is just a text.
  To make a valid expression, escaping is necessary `{% "\%}" %}` resulting
  in `"%}"` as its computed value.
  Both `"{%"` inside the template block or `"%}"` outside it are treated
  as-is without the need of escaping.
