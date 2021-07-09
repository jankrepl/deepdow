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
import os
import pathlib
import sys

from sphinx_gallery.sorting import FileNameSortKey

some_path = pathlib.Path(os.path.abspath('.'))
parent_some_path = some_path.parent
sys.path.insert(0, str(some_path))
sys.path.insert(0, str(parent_some_path))

# -- Project information -----------------------------------------------------

project = 'DeepDow'
copyright = '2020, Jan Krepl'
author = 'Jan Krepl'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_gallery.gen_gallery'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# Set the welcome page for readthedocs
master_doc = 'index'

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',  # needs 'show_related': True theme option to display
        'searchbox.html',
        'donate.html',
    ]
}

# Disable prepending with package and module name
add_module_names = False


# Making sure __call__ shows up in the documentation
def skip(app, what, name, obj, would_skip, options):
    if name == "__call__":
        return False
    return would_skip


def setup(app):
    app.add_css_file('css/custom.css')  # adding custom styling
    app.connect("autodoc-skip-member", skip)  # making sure __call__ is shown when implemented in child class


# sphinx gallery
sphinx_gallery_conf = {
    'examples_dirs': '../examples',  # path to your example scripts
    'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
    'filename_pattern': '',  # include everything
    'within_subsection_order': FileNameSortKey
}
