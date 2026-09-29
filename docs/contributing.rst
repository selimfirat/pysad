.. _contributing:

Contributing
============

You are welcome to contribute this project.

Code is not the only way to help the project. Reviewing pull
requests, answering questions to help others on mailing lists or
issues, organizing and teaching tutorials, working on the website,
improving the documentation, are all priceless contributions.

We abide by the principles of openness, respect, and consideration of
others of the Python Software Foundation:
https://www.python.org/psf/codeofconduct/

In case you experience issues using this package, do not hesitate to submit a
ticket to the
`GitHub issue tracker
<https://github.com/selimfirat/pysad/issues>`_. You are also
welcome to post feature requests or pull requests.

I aim to reply to new issues and pull requests within a few days.

For any questions, you may open issue on Github or drop me an email at `yilmazselimfirat (at)gmail(you know the rest)`.

Development Style
^^^^^^^^^^^^^^^^^

PySAD follows `Trunk-based development <https://trunkbaseddevelopment.com/>`_. All development is conducted using short-lived branches integrated into the trunk (``master``) via pull requests. Pull requests should be kept small and focused, and must pass automated testing and linting in continuous integration to ensure that the trunk remains continuously releasable.

Pull Request Checklist
^^^^^^^^^^^^^^^^^^^^^^

* Do the features/fixes in your pull request match the aim of this framework?
* Does your code pass the linters? You may check via ``pre-commit run --all-files``.
* Does your submission pass all tests (including CI)?
* Have you checked the active `pull requests <https://github.com/selimfirat/pysad/pulls>`_ and `issues <https://github.com/selimfirat/pysad/issues>`_ so that your contribution does not overlap significantly with these?
* **For new features** Have you implemented tests so that your new code has more than 95% test coverage and the tests are reasonable?
* **For new features** Have you implemented examples demonstrating the usage of your new feature?

Development Instructions
^^^^^^^^^^^^^^^^^^^^^^^^
To install requirements of development environment, run the following bash code:

.. code-block:: bash

    pip install -r requirements-dev.txt
    pre-commit install # Linting and formatting on every commit.

After you have done writing code, run the following bash code for checking:

.. code-block:: bash

    bash build_docs.sh # Building docs.
    pre-commit run --all-files # Linting and formatting with ruff.
    pytest --cov=pysad --cov-config=.coveragerc # Running tests.

`pre-commit.ci <https://pre-commit.ci>`_ also runs the hooks on every pull request and pushes a commit with the fixes it can make.
