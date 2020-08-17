#!/usr/bin/env bash
rm -rf ./build/ ./dist/ ./pysad.egg-info/
python3 setup.py sdist bdist_wheel
python3 -m twine upload dist/*