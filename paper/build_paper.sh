#!/bin/bash
# Build the StreamBench paper

set -e

cd "$(dirname "$0")"

# Check for required tools
for cmd in pdflatex biber; do
    if ! command -v $cmd &> /dev/null; then
        echo "Error: $cmd not found. Install texlive or equivalent."
        exit 1
    fi
done

# Build
pdflatex streambench.tex
biber streambench
pdflatex streambench.tex
pdflatex streambench.tex

echo "Paper built: streambench.pdf"
