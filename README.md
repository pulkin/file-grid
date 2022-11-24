[![build](https://github.com/pulkin/grid-run/actions/workflows/test.yml/badge.svg)](https://github.com/pulkin/grid-run/actions)

grid-run
========

Distribute your HPC jobs seamlessly using a simple template language.

One-liner install
-----------------

```bash
pip install git+https://github.com/pulkin/grid-run.git#egg=grid_run
```

Build
-----

```bash
pip install build
python -m build
```

Install
-------

```bash
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
