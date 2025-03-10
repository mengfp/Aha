#!/usr/bin/env python3
#
# Copyright 2025 Meng, Fanping. All rights reserved.
#
import sys
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import re

if sys.platform == "win32":
    CXX_OPTIONS = ["/std:c++17"]
else:
    CXX_OPTIONS = ["-std=c++17", "-O3", "-ffast-math"]
    # Native optimization
    CXX_OPTIONS += ["-march=native"]

def get_version():
    with open("include/version.h") as f:
        for line in f:
            tokens = [token for token in re.split("[ \t\r\n]", line) if token]
            if len(tokens) >= 3 and tokens[0] == "#define" and tokens[
                    1] == "VERSION":
                return tokens[2].strip('"')
    return ""


setup_info = dict(
    name="aha",
    version=get_version(),
    author="Fanping Meng",
    author_email="mengfpoliver@gmail.com",
    url="https://github.com/mengfp/Aha",
    description="Multivariate Multiscale Nonlinear Analysis and Modelling",
    python_requires=">=3.7",
    install_requires=[],
    packages=[],
    ext_modules=[
        Pybind11Extension(
            "aha",
            ["aha/aha.cpp", "src/py_aha.cpp"],
            include_dirs=["include", "eigen"],
            extra_compile_args=CXX_OPTIONS,
        ),
    ],
    scripts=[],
)

setup(**setup_info)
