#!/usr/bin/env python3
import sys
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import re

def get_version():
    with open("include/version.h") as f:
        for line in f:
            tokens = [token for token in re.split("[ \t\r\n]", line) if token]
            if len(tokens) >= 3 and tokens[0] == "#define" and tokens[1] == "VERSION":
                return tokens[2].strip('"')
    return ""

__version__ = get_version()

if sys.platform == "win32":
    library_dirs = ["x64/Release"]
    dll_files = ["x64/Release/aha.dll"]
else:
    library_dirs = ["x64/Release"]
    dll_files = ["x64/Release/aha.dll"]

setup_info = dict(
    name="aha",
    version=__version__,
    author="Fanping Meng",
    author_email="mengfpoliver@gmail.com",
    url="https://github.com/mengfp/Aha",
    description="Multivariate Multiscale Nonlinear Analysis and Modelling",
    python_requires=">=3.7",
    cmdclass={"build_ext": build_ext},
    install_requires=[],
    packages=[],
    ext_modules= [
        Pybind11Extension(
            "aha",
            ["src/py_aha.cpp"],
            # Example: passing in the version to the compiled code
            define_macros=[("VERSION", __version__)],
            include_dirs=["include", "eigen"],
            libraries=["aha"],
            library_dirs=library_dirs,
        ),
    ],
    data_files=dll_files,
    scripts=[],
)

setup(**setup_info)
