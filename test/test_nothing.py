"""
just wanted to see if I could run GHA successfully with dependencies that are only available on GH
ignore this robust test
"""
import pytest
import molsym

def test_it():
    sky = "blue"
    assert sky == "blue"

