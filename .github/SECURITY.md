# Security Policy

## Supported versions

Only the latest release on PyPI receives security fixes. Older releases are not
patched; please upgrade to the latest version before reporting.

| Version | Supported |
| ------- | --------- |
| >= 0.6  | Yes       |
| < 0.6   | No        |

## Reporting a vulnerability

Please do not open a public issue for security problems.

Report them privately through GitHub:
https://github.com/selimfirat/pysad/security/advisories/new

If you cannot use GitHub, email the maintainer at the address listed on
https://github.com/selimfirat, with "pysad security" in the subject line.

Please include:

- the pysad version and Python version,
- a description of the issue and its impact,
- steps or a short script that reproduces it.

## What to expect

- An acknowledgement within 7 days.
- An assessment and, if the report is confirmed, a fix and a new release. Please
  allow up to 90 days before disclosing publicly, or sooner once a fix is released.
- Credit in the release notes and the GitHub advisory, unless you prefer to stay
  anonymous.

## Scope

pysad is a library for streaming anomaly detection. It does not handle
authentication, network traffic, or secrets. Issues in scope are things like
unsafe deserialization, code execution from untrusted inputs, or a dependency
pin that forces a known-vulnerable version. Vulnerabilities in third-party
dependencies should be reported to those projects; a pull request bumping the
pin here is welcome.
