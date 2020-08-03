#!/usr/bin/env bash
cd docs
sphinx-apidoc -f -o . ..
make clean
make html