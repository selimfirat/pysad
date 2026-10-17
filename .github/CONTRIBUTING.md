# Contributing to PySAD

The full guide is in the [documentation](https://pysad.readthedocs.io/en/latest/contributing.html). It explains the setup, the branch names, the review checklist and the rules for AI tools. We aim to reply to new issues and pull requests within a few days.

## Ask or report

- If you found a bug, open a [bug report](https://github.com/selimfirat/pysad/issues/new/choose).
- If you want a new feature or detector, open a [feature request](https://github.com/selimfirat/pysad/issues/new/choose).
- If you have a question, start a thread in [Discussions](https://github.com/selimfirat/pysad/discussions).
- If you found a security problem, follow the [security policy](https://github.com/selimfirat/pysad/security/policy).

## Check your change

Run these commands from the repository root before you open a pull request:

```console
pre-commit run --all-files
pytest
pytest -m examples
bash build_docs.sh
```

## Name your pull request

Write the title as a conventional commit, for example `fix: return finite LODA scores for constant features`. Use one of these types: `feat`, `fix`, `docs`, `build`, `ci`, `refactor`, `test`, `style` or `chore`. The release notes group pull requests by this type.

The project follows the [Python Software Foundation Code of Conduct](https://policies.python.org/python.org/code-of-conduct/).
