#!/usr/bin/env bash
cd docs
sphinx-apidoc -f -o . .. ../tests ../setup.py ../pysad/models/kitnet_model
make clean
make html