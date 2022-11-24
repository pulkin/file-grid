[![build](https://github.com/pulkin/grid-run/actions/workflows/test.yml/badge.svg)](https://github.com/pulkin/grid-run/actions)

grid-run
========

Distribute your HPC jobs seamlessly using a simple template language.

Build
-----

```commanline
pip install build
python -m build
```

Install
-------

```commandline
pip install dist/*.tar.gz
```

Run
---

```commandline
python -m grid_run
```

or simply

```commandline
grid
```

if your `PATH` includes python `bin` folder.
