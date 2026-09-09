# Contributing to RELAX

Thanks for considering a contribution!

## Development setup

```bash
git clone https://github.com/Wickstrom/RELAX.git
cd RELAX
python -m venv .venv
source .venv/bin/activate
pip install --editable .[dev]
```

## Running tests and linting

```bash
pytest
ruff check .
ruff format --check .
```

Please make sure both pass before opening a pull request. New features should
come with tests (CPU-friendly, small tensors).

## Deprecation policy

Misspelled or legacy keyword arguments (e.g. `probablity_of_drop`) are kept as
deprecated aliases that emit a `DeprecationWarning`. They are removed in the
next minor release.

## Release process

1. Bump `__version__` in `src/relax_xai/__init__.py`.
2. Commit, then tag the release and push the tag:

   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

3. The `publish.yml` workflow builds the package and publishes it to
   [PyPI](https://pypi.org/project/relax-xai/) using
   [trusted publishing](https://docs.pypi.org/trusted-publishers/). No API
   tokens are needed, but the publisher must be configured on PyPI once:
   add a *pending* publisher for `Wickstrom/RELAX` with workflow
   `publish.yml` and environment `pypi` (or configure it after the first
   release attempt). Alternatively, create a GitHub environment named `pypi`
   with a `PYPI_API_TOKEN` secret and add `password: ${{ secrets.PYPI_API_TOKEN }}`
   to the `pypa/gh-action-pypi-publish` step.
4. Write release notes on the GitHub release for the tag.
