from setuptools import find_packages, setup

import deepdow

DESCRIPTION = "Portfolio optimization with deep learning"
LONG_DESCRIPTION = DESCRIPTION

INSTALL_REQUIRES = [
    "cvxpylayers",
    "matplotlib",
    "mlflow",
    "numpy>=1.16",
    "pandas",
    "pillow",
    "torch>=1.5",
    "tensorboard",
    "tqdm"
]

setup(
    name="deepdow",
    version=deepdow.__version__,
    author="Jan Krepl",
    author_email="kjan.official@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url="https://github.com/jankrepl/deepdow",
    packages=find_packages(),
    license="Apache License 2.0",
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "dev": ["codecov", "flake8==3.7.9", "pydocstyle", "pytest>=3.6", "pytest-cov", "tox"],
        "docs": ["sphinx", "sphinx_rtd_theme"],
        "examples": ["sphinx_gallery", "statsmodels"]
    }
)
