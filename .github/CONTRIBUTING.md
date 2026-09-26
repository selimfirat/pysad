# Contributing to PySAD

Thanks for your interest in PySAD! Code is not the only way to help: bug reports, answering questions, improving the documentation and examples, and sharing what you built are all valuable contributions.

I aim to reply to new issues and pull requests within a few days.

## Where to go

- **Bugs:** open an [issue](https://github.com/selimfirat/pysad/issues/new/choose) with your PySAD version, Python version and a minimal example that reproduces the problem.
- **Feature requests:** open an [issue](https://github.com/selimfirat/pysad/issues/new/choose) describing the use case.
- **Usage questions and ideas:** start a thread in [Discussions](https://github.com/selimfirat/pysad/discussions).
- **First contribution?** Issues labelled [`good first issue`](https://github.com/selimfirat/pysad/labels/good%20first%20issue) are small and well scoped.

We follow the [Python Software Foundation Code of Conduct](https://policies.python.org/python.org/code-of-conduct/).

## Development setup

```bash
git clone https://github.com/selimfirat/pysad.git
cd pysad
pip install -r requirements-dev.txt
pip install -e .
```

Before opening a pull request, run:

```bash
bash lint.sh          # PEP8 style (flake8)
pytest --cov=pysad    # unit tests with coverage
pytest -m examples    # runs every script in examples/ end to end
bash build_docs.sh    # builds the documentation
```

## Pull request checklist

- The change fits the aim of the framework: anomaly detection on streaming data.
- Code passes `bash lint.sh` and all tests, including CI.
- You checked open [pull requests](https://github.com/selimfirat/pysad/pulls) and [issues](https://github.com/selimfirat/pysad/issues) so the work doesn't overlap.
- **New features** come with tests (aim for more than 95% coverage of the new code) and an example or docs showing how to use them.
- **New models** cite the original paper in the docstring.

The full guide is also in the [documentation](https://pysad.readthedocs.io/en/latest/contributing.html).
