#!/usr/bin/env bash
rm -rf ./build/ ./dist/ ./pysad.egg-info/
python3 -m build
python3 -m twine upload dist/*
