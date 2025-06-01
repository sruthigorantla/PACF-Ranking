import os
import subprocess

import setuptools
from setuptools import Command, setup

setup(
    name="ActiveFairRanking",
    version="0.1",
    author="Sruthi Gorantla",
    author_email="gorantlas@iisc.ac.in",
    description="Code for Active Fair Ranking",
    url="https://sites.google.com/view/sruthigorantla/home",
    packages=setuptools.find_packages(),
    package_dir={"": "src"},
    install_requires=[
        "matplotlib",
        "omegaconf",
        "pandas",
        "scipy",
        "seaborn",
        "tqdm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
