# Configuration file for the Sphinx documentation builder.
# See https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from pysad.utils import Data
from pysad.version import __version__

# The docs write the number of datasets as |num_datasets|, so that it follows Data.datasets().
rst_epilog = f"\n.. |num_datasets| replace:: {len(Data.datasets())}\n"

# -- Project information -----------------------------------------------------

project = "PySAD"
copyright = "2020-2026, Selim Firat Yilmaz"
author = "Selim Firat Yilmaz"
version = __version__
release = __version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
# Single backticks in docstrings mean code.
default_role = "code"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_default_options = {"members": True, "inherited-members": True}
autodoc_typehints = "none"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

bibtex_bibfiles = ["refs.bib"]

# The ODDS site blocks automated clients by IP address through Cloudflare.
linkcheck_ignore = [r"https?://odds\.cs\.stonybrook\.edu/?"]

# Copy only the code from prompts such as ">>> " and "$ ".
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_title = f"PySAD {release}"
html_favicon = "favicon.svg"
html_static_path = ["_static"]
html_css_files = [
    "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap",
    "custom.css",
]

html_theme_options = {
    # Read the Docs builds "latest" from master, which can differ from the release on PyPI.
    "announcement": (
        "These pages describe the development version on GitHub. For the release on PyPI, read the "
        '<a href="https://pysad.readthedocs.io/en/stable/">stable documentation</a>.'
        if os.environ.get("READTHEDOCS_VERSION") == "latest"
        else None
    ),
    "light_logo": "logo.svg",
    "dark_logo": "logo-dark.svg",
    "sidebar_hide_name": True,
    "source_repository": "https://github.com/selimfirat/pysad/",
    "source_branch": "master",
    "source_directory": "docs/",
    # Brand colors: cobalt #0741ad, coral #fc3636, paper #fffef7, ivory #efebdf and ink #0e1a33.
    "light_css_variables": {
        "font-stack": 'Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif',
        "font-stack--monospace": '"IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
        "font-stack--headings": "var(--font-stack)",
        "color-foreground-primary": "#0e1a33",
        "color-background-primary": "#fffef7",
        "color-background-secondary": "#efebdf",
        "color-background-border": "#e2ddcd",
        "color-code-background": "#f6f3ea",
        "color-code-string": "#3f6fd0",
        "color-highlight-on-target": "#e6ecf8",
        "color-api-name": "#0e1a33",
        "color-api-pre-name": "#5a6478",
        "color-brand-primary": "#0741ad",
        "color-brand-content": "#0741ad",
        "color-brand-visited": "#0741ad",
    },
    "dark_css_variables": {
        "color-foreground-primary": "#eef1f8",
        "color-background-primary": "#0e1a33",
        "color-background-secondary": "#0a1428",
        "color-background-border": "#22304f",
        "color-code-background": "#0a1428",
        "color-code-string": "#b6c9f7",
        "color-highlight-on-target": "#1c2b4d",
        "color-api-name": "#eef1f8",
        "color-api-pre-name": "#a3adc2",
        "color-brand-primary": "#7c9ff0",
        "color-brand-content": "#7c9ff0",
        "color-brand-visited": "#7c9ff0",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/selimfirat/pysad",
            "html": '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>',
            "class": "",
        },
    ],
}
