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
import shutil
import sys
import time
import cryocat
import subprocess
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.abspath("sphinxext"))


for f in os.listdir("./generated/"):
    if f.endswith(".rst"):
        os.remove("./generated/" + f)

# -- Project information -----------------------------------------------------

project = "cryoCAT"
copyright = f'2022-{time.strftime("%Y")}'
author = "Beata Turonova"
version = release = "0.2.0"  # cryocat.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (amed 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "matplotlib.sphinxext.plot_directive",
    # "gallery_generator",
    # "tutorial_builder",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx_issues",
    "sphinx_design",
    "sphinx.ext.todo",
    "nbsphinx",
    # "IPython.sphinxext.ipython_directive",
    # "IPython.sphinxext.ipython_console_highlighting",
    # "sphinx.ext.doctest",
    # "sphinx.ext.extlinks",
    # "sphinx.ext.ifconfig",
    # "sphinx.ext.linkcode",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The root document.
root_doc = "index"

# The master toctree document.
# master_doc = "index"

# The suffix of source filenames.
# source_suffix = [".rst"]

# The encoding of source files.
# source_encoding = "utf-8"


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "docstrings", "nextgen", "Thumbs.db", ".DS_Store"]

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = "literal"

# Generate the API documentation when building
autosummary_generate = True
numpydoc_show_class_members = False

# Sphinx-issues configuration
issues_github_path = ""

# Include the example source for plots in API docs
plot_include_source = True
plot_formats = [("png", 90)]
plot_html_show_formats = False
plot_html_show_source_link = False
# plot_pre_code = """import numpy as np
# import pandas as pd"""

# nbsphinx do not use requirejs (breaks bootstrap)
nbsphinx_requirejs_path = ""

# https://sphinx-toggleprompt.readthedocs.io/en/stable/#offset
# toggleprompt_offset_right = 35

# Don't add a source link in the sidebar
html_show_sourcelink = False

# Control the appearance of type hints
autodoc_typehints = "none"
autodoc_typehints_format = "short"

# Allow shorthand references for main function interface
rst_prolog = """
.. currentmodule:: cryocat
"""


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named 'default.css' will overwrite the builtin 'default.css'.
html_static_path = ["_static"]
for path in html_static_path:
    if not os.path.exists(path):
        os.makedirs(path)

html_css_files = ["css/cryocat.css"]

html_logo = "_static/cryocat_logo_wide_light_bcg.png"
html_favicon = "_static/favicon.png"

html_theme_options = {
    "sidebarwidth": 100,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/turonova/cryocat",
            "icon": "fab fa-github",
            "type": "fontawesome",
        },
        # {
        #     "name": "PyPI",
        #     "url": "https://pypi.org/project/cryocat",
        #     "icon": "fab fa-pypi",
        #     "type": "fontawesome",
        # },
    ],
    "show_prev_next": False,
    "navbar_start": ["navbar-logo"],
    # "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    "navbar_end": ["searchbox.html", "version-switcher", "theme-switcher", "navbar-icon-links.html"],
    "header_links_before_dropdown": 8,
    "navbar_persistent": [],
    "secondary_sidebar_items": ["class-page-toc"],
    "logo": {
        "image_light": "_static/cryocat_logo_wide_light_bcg.png",
        "image_dark": "_static/cryocat_logo_wide_dark_bcg.png",
    },
    "switcher": {
        "version_match": version,
        "json_url": "https://numpy.org/doc/_static/versions.json",
    },
}

# html_context = {
#    "default_mode": "light",
# }

html_sidebars = {
    "index": [],
    "installing": [],
    "search": [],
    "examples/index": [],
    "**": ["sidebar_no_caption.html"],
}

toc_object_entries_show_parents = "hide"
numpydoc_show_inherited_class_members = False


# -- Create rst from notebooks ----------------------------------

# notebook_list = []
# for subdir, dirs, files in os.walk("../../notebooks/"):
#     for file in files:
#         notebook_list.append(os.path.join(subdir, file))

# for nb_file in notebook_list:
#     if nb_file.endswith(".ipynb"):
#         shutil.copy(nb_file, "./")


# -- Intersphinx ------------------------------------------------

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
}


def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    if not info["module"]:
        return None
    filename = info["module"].replace(".", "/")
    return "https://somesite/sourcerepo/%s.py" % filename


# Following piece of code is based on:
# https://github.com/pydata/pydata-sphinx-theme/issues/215#
# It goes over the class and list its members in the secondary side-bar
def _setup_navbar_side_toctree(app: Any):
    def add_class_toctree_function(app: Any, pagename: Any, templatename: Any, context: Any, doctree: Any):
        def get_class_toc() -> Any:
            if "body" not in context:
                return ""

            soup = BeautifulSoup(context["body"], "html.parser")

            matches = soup.find_all("dl")
            if matches is None or len(matches) == 0:
                return ""
            items = []
            deeper_depth = matches[0].find("dt").get("id").count(".")
            for match in matches:
                match_dt = match.find("dt")
                if match_dt is not None and match_dt.get("id") is not None:
                    current_title = match_dt.get("id")
                    current_depth = match_dt.get("id").count(".")
                    current_link = match.find(class_="headerlink")
                    if current_link is not None:
                        current_title = current_title.split(".")[-1]
                        if deeper_depth > current_depth:
                            deeper_depth = current_depth
                        if deeper_depth == current_depth:
                            items.append(
                                {"title": current_title, "link": current_link["href"], "attributes_and_methods": []}
                            )
                        if deeper_depth < current_depth:
                            items[-1]["attributes_and_methods"].append(
                                {
                                    "title": current_title,
                                    "link": current_link["href"],
                                }
                            )
            return items

        context["get_class_toc"] = get_class_toc

    app.connect("html-page-context", add_class_toctree_function)


def setup(app: Any):
    for setup_function in [
        _setup_navbar_side_toctree,
    ]:
        setup_function(app)
