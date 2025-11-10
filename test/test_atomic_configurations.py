from srhf.atomic_configurations import configurations

def test_all_configs_length_four():
    """Ensure every atomic configuration has exactly four entries."""
    for idx, conf in enumerate(configurations):
        assert isinstance(conf, list), f"Configuration {idx} is not a list"
        assert len(conf) == 4, f"Configuration {idx} length != 4 (got {len(conf)})"

