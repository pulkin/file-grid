def pytest_addoption(parser):
    parser.addoption("--grid-script", action="store", default="python -m file_grid")
