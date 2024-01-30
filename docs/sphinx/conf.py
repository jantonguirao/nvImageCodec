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
import sys
import os
import sphinx_rtd_theme

# For build documentation for nvimgcodec from specif location (default is installed module), 
# uncomment below line to specify path to nvimgcodec module
# sys.path.insert(0, os.path.abspath("../../build/python"))

# Specify path to images used in jupyter notebooks samples
os.environ["PYNVIMGCODEC_EXAMPLES_RESOURCES_DIR"] = "../../../example/assets/images/"

# nbsphinx_allow_errors = True

# -- Project information -----------------------------------------------------

project = 'nvImageCodec'
copyright = '2023 - 2024, NVIDIA Corporation & Affiliates'
author = 'NVIDIA Corporation'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.

# Version and release are passed from CMake.
#version = None

# The full version, including alpha/beta/rc tags.
#release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinxcontrib.jquery",
    'sphinx.ext.ifconfig',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'breathe',
    'nbsphinx',
    'nbsphinx_link'
]
# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = 'cpp:any'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

highlight_language = 'c++'

primary_domain = 'cpp'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "collapse_navigation" : False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Output file base name for HTML help builder.
htmlhelp_basename = 'nvImageCodecDoc'

def setup(app):
    app.add_css_file('nvimagecodec_override.css')

# -- Options for BREATHE -------------------------------------------------

breathe_default_project = "nvImageCodec"
breathe_projects = { "nvImageCodec": "../doxygen/xml" }
