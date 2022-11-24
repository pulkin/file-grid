def pytest_addoption(parser):
    parser.addoption("--grid-script", action="store", default="python -m grid_run")
