"""Top-level package for award_pynder."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("award-pynder")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Eva Maxfield Brown"
__email__ = "evamxb@uw.edu"
