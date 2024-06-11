try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("lightquakevisualizer")
except PackageNotFoundError:
    # If the package is not installed, don't add __version__
    pass
