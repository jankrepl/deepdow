from setuptools import find_packages, setup

import deepdow

DESCRIPTION = "Portfolio optimization with deep learning"
LONG_DESCRIPTION = DESCRIPTION

INSTALL_REQUIRES = [
    "cvxpylayers",
    "matplotlib",
    "mlflow",
    "numpy>=1.16.5",
    "pandas",
    "pillow",
    "seaborn",
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
    packages=find_packages(exclude=["tests"]),
    license="Apache License 2.0",
    install_requires=INSTALL_REQUIRES,
    python_requires='>=3.6',
    extras_require={
        "dev": ["codecov", "flake8==3.7.9", "pydocstyle", "pytest>=4.6", "pytest-cov", "tox"],
        "docs": ["sphinx", "sphinx_rtd_theme"],
        "examples": ["sphinx_gallery", "statsmodels"]
    }
)
