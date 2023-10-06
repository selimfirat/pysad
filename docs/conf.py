from os.path import dirname, abspath

import sphinx_rtd_theme
import os
import sys
sys.path.insert(0, os.path.abspath('../'))

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------
sys.path.insert(0, abspath('..'))
pysad_dir = dirname(dirname(abspath(__file__)))

version_path = os.path.join(pysad_dir, 'pysad', 'version.py')
exec(open(version_path).read())

project = 'PySAD'
copyright = '2023, Selim Firat Yilmaz'
author = 'Selim Firat Yilmaz'

version = __version__
release = __version__

#version = "latest"

master_doc = 'index'
pygments_style = 'sphinx'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_rtd_theme",
    #"sphinx.ext.autodoc",
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    "sphinxcontrib.bibtex",
    'sphinx_copybutton',
    'sphinx.ext.napoleon'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', "conf.py", "examples/*", "pysad/models/kitnet_model/*", "tests/*"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'canonical_url': '',
    'analytics_id': 'UA-XXXXXXX-1',  #  Provided by Google in your dashboard
    'logo_only': True,
    'display_version': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    #'vcs_pageview_mode': '',
    #'style_nav_header_background': 'white',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 7,
    'includehidden': True,
    'titles_only': False,
}
html_logo = "logo.png"
html_favicon = "infinity.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
autosummary_generate = True

autodoc_default_options = {'members': True,
                           'inherited-members': True,
                           }
autodoc_typehints = "none"

intersphinx_mapping = {
    'python': ('https://docs.python.org/{.major}'.format(
        sys.version_info), None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    #'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    # 'matplotlib': ('https://matplotlib.org/', None),
    #'sklearn': ('https://scikit-learn.org/stable', None)
}
