#!/usr/bin/env python

import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name="sahi_general",
        version="1.0",
        packages=setuptools.find_packages(),
        install_requires=["sahi"]
    )