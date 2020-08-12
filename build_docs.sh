#!/usr/bin/env bash
cd docs
sphinx-apidoc -f -o . .. ../tests ../setup.py ../pysad/models/kitnet_model ../pysad/version.py
make clean
make html