from setuptools import setup

import deepdow

DESCRIPTION = "Portfolio optimization with deep learning"
LONG_DESCRIPTION = DESCRIPTION

INSTALL_REQUIRES = [
    "cvxpylayers",
    "pandas",
    "torch"
]

setup(
    name="deepdow",
    version=deepdow.__version__,
    author="Jan Krepl",
    author_email="kjan.official@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url="https://github.com/jankrepl/deepdow",
    packages=["deepdow"],
    license="MIT",
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "dev": ["codecov", "flake8", "pydocstyle", "pytest>=3.6", "pytest-cov", "tox"],
        "docs": ["sphinx", "sphinx_rtd_theme"],
    }
)
