import pytest


# for integration tests that run slow
def pytest_addoption(parser):
    parser.addoption('--runslow', action='store_true', default=False, help='run slow tests')


def pytest_collection_modifyitems(config, items):
    # runslow passed, do now skip slow tests
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="slow tests, need --runslow cli option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
