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
