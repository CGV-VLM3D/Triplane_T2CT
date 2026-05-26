def pytest_configure(config):
    config.addinivalue_line(
        "markers", "tier0: tier-0 single-batch sanity overfit (fast)"
    )
