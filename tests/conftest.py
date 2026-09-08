def pytest_configure(config):
  config.addinivalue_line(
    "markers", "slow: builds a simulation environment (needs a GPU; ~30s each)"
  )
