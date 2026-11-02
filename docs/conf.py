"""Sphinx configuration for the trickle-down documentation."""

# For a full list of options see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from importlib.metadata import version as _get_version

# The repository root, so autodoc and the doctests can import both
# trickle_down and the paper/ reconstruction.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# -- Project information -----------------------------------------------------

project = "trickle-down"
copyright = "2024-2026, Bob Wilson"
author = "Bob Wilson"

# The full version, including alpha/beta/rc tags
release = _get_version("trickle-down")


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    # `make doctest` executes every `>>>` example in the guides, so an
    # exact result quoted from the paper cannot drift from the library.
    "sphinx.ext.doctest",
    # The usage guide is trickle_down/USAGE.md, included rather than copied.
    "myst_parser",
]

myst_enable_extensions = ["dollarmath"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "user_guides/_generated"]

autodoc_member_order = "bysource"

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = f"trickle-down {release}"
html_theme_options = {
    "github_url": "https://github.com/rwilson4/trickle-down",
    "show_prev_next": False,
}
