import toml

from pyloras import __version__


def test_version():
    setup = toml.load('./pyproject.toml')
    assert __version__ == setup['tool']['poetry']['version']
