Versioning
==========

`Semantic versioning <http://semver.org/>`_ is used for this project.

Releases are automated with GitHub Actions and Release Please.

Release flow
------------

* Merge feature and fix pull requests into ``develop`` using conventional commit
  messages such as ``feat:``, ``fix:``, ``docs:`` and ``chore:``.
* The ``Release Please`` workflow updates a release pull request on every push to
  ``develop``. That pull request owns the version bump, changelog update and
  release metadata.
* Merge the release pull request when the release contents are ready. Release
  Please creates the GitHub release and tag.
* The ``Publish`` workflow runs automatically when the GitHub release is
  published. It builds the package, checks the distribution metadata and
  publishes to PyPI using trusted publishing.

The Release Please workflow must use ``RELEASE_PLEASE_TOKEN`` rather than the
default ``GITHUB_TOKEN``. GitHub suppresses downstream workflow runs for events
created by ``GITHUB_TOKEN``, so a release created with the default token would
not trigger the PyPI publishing workflow.

Version source
--------------

The package version is stored in ``pysad/version.py``. ``pyproject.toml`` reads
that value dynamically so release automation only needs to update one version
source.

PyPI setup
----------

PyPI publishing uses OpenID Connect trusted publishing. Configure the PyPI
project with this GitHub repository, the ``Publish`` workflow and the ``pypi``
environment before the first automated release.

Release validation runs in CI for release-related changes. It builds the
package, runs ``twine check`` and verifies that the workflow's PyPI project URL
matches PyPI's JSON API for ``pysad``.
