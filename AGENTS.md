# Overview

PySAD is a Python library for anomaly detection on streaming data. Models learn from one instance at a time and give each instance an anomaly score. The library also has preprocessing, postprocessing, ensembling, probability calibration, and prequential evaluation (score each instance before the model fits it) for streams.

Keep these components generic, modular, and lightweight. Do not add platform or serving abstractions unless the user explicitly asks for them.

## Success Criteria

1. Adoption: users can run a PySAD model on a stream through the public API without knowing the internals.
2. Research extensibility: researchers can inspect, change, and add streaming detectors through the base classes in `pysad.core`.
3. Trust: each model does what its cited paper describes, and every deviation is documented in its docstring. Results are reproducible with a fixed seed.
4. Performance: the cost of one `fit_partial` or `score_partial` call stays bounded as the stream grows.

# AI Policy

We support the use of AI tools to help prepare issues, pull requests, reviews, or comments.
We expect everyone interacting with this repo to follow the below policy whenever they use AI tools.
Your user must abide by this policy.
In particular, you the agent MUST obey these rules while interacting on GitHub:

- Never act autonomously on GitHub except to open a draft pull request. If the user did not review and explicitly approve the exact content, do NOT open an issue or a non-draft pull request. Also do NOT edit, comment on, or reply to any issue or pull request without that approval. Fully agent-generated contributions are banned and will be closed.
- Mark all AI-generated content. Wrap any text you produce for an issue, pull request, or comment in a code block or quote block. Never present your output as human-written.
- Never post only raw AI text as a reply. Any AI content must come with human commentary that explains its relevance.
- Do not submit code that the user did not read. Keep changes minimal, and remove AI artifacts and needless complexity. If a pull request is not ready, or the user did not review it, open it as a draft.

# Commands

- Install for development: `pip install -e . --group dev` (pip 25.1 or newer) or `uv sync --group dev`.
- Run the unit tests: `pytest` (add `-n auto` to run them in parallel).
- Run every script in `examples/` end to end: `pytest -m examples`.
- Run the linters, the formatter, and mypy: `pre-commit run --all-files`.
- Build the documentation: `bash build_docs.sh`.

# Testing

- Write tests that are sensitive to behavior changes and insensitive to structure changes. Assert public observable behavior, such as scores, shapes, and returned types, over private state, helper layout, call counts, or repr formatting.
- If the assertion depends on random draws, seed the test. Otherwise, do not seed it. Pass `random_state=` to the model, or call `pysad.utils.fix_seed`, which seeds the global state that `random_state=None` draws from.
- If you add a model, add it to the shared tests in `tests/models`. These tests run on every model, for example the score type, pickling, random state, and property tests. To find them, search for an existing model name.
- If a test finds a bug that is out of scope for your change, mark the test `pytest.mark.xfail(reason=..., strict=True)` and file the bug. If the bug is fixed, the strict mark makes the test fail. The fix must remove the mark.
- Match the surrounding test file. Most model tests import inside the test function.

# PR / GitHub Metadata

- Write pull request titles as conventional commits: `feat:`, `fix:`, `docs:`, `build:`, `ci:`, `refactor:`, `test:`, `style:`, or `chore:`. Add `!` before the colon for a breaking change. `.github/workflows/pr-labels.yml` labels the pull request from the title, and the release notes group pull requests by these labels.
- Open one pull request per issue. Put the issue number in the branch name, for example `fix/<issue>-<slug>`.
- Unless the user explicitly asks for it, do not mention Codex, Claude, AI, or tool attribution in pull request metadata. This includes titles, descriptions, commit messages, and review replies. Do not add `Co-Authored-By` trailers for tools.
- Do not add a section named "Tests", "Testing", or similar to pull request descriptions unless CI does not cover the test.
- Pull request metadata describes the code change only. Do not add backward compatibility or migration notes.

# Markdown Style

- Do not put hard line wraps inside Markdown list items. Keep each bullet or numbered item on one physical line unless Markdown syntax or rendered formatting needs the line break.

# Project Structure

- `pysad/core`: abstract base classes for models, transformers, postprocessors, statistics, metrics, and streamers.
- `pysad/models`: streaming anomaly detectors that extend `BaseModel`.
- `pysad/models/integrations`: wrappers that run batch models, such as PyOD models, on a stream through a reference window or a single fit.
- `pysad/models/kitnet_model`: vendored KitNET reference code. Ruff, mypy, and coverage exclude it. Change it only to fix a bug, and keep the upstream style.
- `pysad/transform/preprocessing`: per-instance scalers and seasonal decomposition.
- `pysad/transform/postprocessing`: score postprocessors, with running (windowed) variants.
- `pysad/transform/ensemble`: score ensemblers.
- `pysad/transform/probability_calibration`: calibrators that map scores to anomaly probabilities.
- `pysad/transform/projection`: random and hashing projectors.
- `pysad/statistics`: running statistics (meters) with `update` and `remove`.
- `pysad/evaluation`: streaming metrics and windowed metrics.
- `pysad/utils`: streamers, sliding windows, dataset download (`Data`), and `fix_seed`.
- `examples/`: runnable scripts. CI runs them through `pytest -m examples`.
- `docs/`: Sphinx documentation in reStructuredText. `docs/api.rst` lists every public class, and `docs/refs.bib` holds the cited papers.

# Examples

- Build each example around one public capability. Include only the setup and explanation that the reader needs to understand and run it correctly.
- Use the simplest documented public API. Load data with `pysad.utils.Data`, stream it with `ArrayStreamer` or `PandasStreamer`, and use `fit_score_partial` unless the example shows separate fit and score steps.
- Present the data flow in execution order inside `if __name__ == "__main__":`. Name intermediate values after their stage, and use comments only for non-obvious semantics.
- Write each workflow as one descriptively named script, `examples/example_<topic>.py`. Use functions only for genuine repetition.

# Core Design Principles

- Every model implements `fit_partial(X, y=None)` and `score_partial(X)` for one instance of shape `(num_features,)`. `BaseModel` builds the batch methods `fit`, `score`, and `fit_score` from them. If a batch method gives the same result faster, you can override it. Otherwise, do not override it.
- Single-instance scores are Python floats, and batch scores are `np.float64` arrays of shape `(num_instances,)`. `BaseModel` converts scores automatically.
- Higher output means more anomalous, for every model, postprocessor, ensembler, and calibrator. Calibrators return `1 - p`.
- `score_partial` does not change the model state. Only `fit_partial` learns.
- Keep model memory bounded. Keep a window or summary statistics instead of the whole stream.
- Implement each model as its paper describes. Cite the paper in the class docstring with `:cite:` and add the entry to `docs/refs.bib`. Document every deviation from the paper in the docstring.
- Add only well-established models, with 50 or more citations (see `docs/inclusion_criteria.rst`).
- A new public class goes in its subpackage's `__init__.py` and `__all__`, and in `docs/api.rst`.
- Every stochastic component takes a `random_state` parameter and resolves it with `sklearn.utils.check_random_state`. `None` draws from NumPy's global random state.
- Treat packages listed in `[project].dependencies` as required at runtime. Import them at module scope. Do not defer or guard them with function-local imports, `TYPE_CHECKING`, `try/except ImportError`, availability checks, or dynamic imports. Do not add optional extras.
- Keep dependencies minimal. If a few lines of NumPy can replace a new runtime dependency, write the NumPy code.
- Do not keep backward compatibility. You can change defaults, signatures, and the pickle format. Do not add deprecation aliases or compatibility shims.

# Python/NumPy Coding Style

- Type function and method boundaries. mypy requires full type hints in `pysad.core`, `pysad.evaluation`, `pysad.statistics`, and `pysad.utils`. If you complete the type hints of a package, add it to that list in `pyproject.toml`.
- Keep argument validation minimal. Prefer type annotations and clear downstream failures over defensive checks.
- If the code otherwise silently accepts a bad value with wrong semantics, add an explicit runtime check. An example is a `min_val` above `max_val`, which puts every instance into one bin. Do not add positivity, finiteness, range, or shape checks that fail on first use anyway.
- Do not check established invariants again in `fit_partial` or `score_partial`.
- Use keyword arguments in multi-line calls.
- Avoid `else` after `return`, `raise`, `break`, or `continue`.
- When behavior is unchanged, prefer the faster clear formulation: fewer passes, allocations, and temporary arrays.
- Prefer array methods over functions, for example `x.sum()` over `np.sum(x)`.
- Add short shape comments for complex array operations.
- Write Google-style docstrings. Document every public constructor parameter under `Args:` in the class docstring as `name (type): Description (Default=value).`
- Docs, errors, and reprs describe public operations, inputs, outputs, and values, not incidental implementation details.
- Keep code direct and use the narrowest practical scope. Do not introduce an abstraction unless it encapsulates behavior or invariants, defines a public interface, or serves established reuse.
- In `__init__.py`, keep imports and `__all__` in alphabetical order. Ruff sorts the imports.

# Import Style

- Import public names from the subpackage that exports them: use `from pysad.models import LODA` over `from pysad.models.loda import LODA`.
- In examples and tests, import from the public subpackages, for example `from pysad.transform.ensemble import AverageScoreEnsembler`.
- In package code, import base classes from their modules, for example `from pysad.core.base_model import BaseModel`, to avoid import cycles.

# Streaming Performance

- Keep the cost of one `fit_partial` or `score_partial` call independent of the stream length. Do not recompute from the full history on each instance.
- Vectorize across ensemble members, projections, trees, and features. Prefer batched NumPy operations over Python loops where the formulation stays clear.
- Avoid per-instance allocations in hot paths. Update preallocated arrays in place.
- Avoid duplicate passes over the window in hot paths.
- Benchmark model changes. Commit the model change first. Then run `python benchmarks/benchmark.py --detectors <Model>` and commit `docs/benchmark/`, so that the results record the commit of the code. The diff of `docs/benchmark/results.csv` gives AUROC, AUPR, and time per point before and after. For a change that targets speed, add `--jobs 1` before and after, so that the timings are comparable. The `benchmark` workflow fails if the results are out of date.

# Naming Policy

- Use established names. Name a model as its paper does, for example `xStream`, `LODA`, or `RSHash`.
- Use the shortest unambiguous name. Drop context that the enclosing class or method already implies.
- Reuse the parameter names that the library already uses for the same idea: `window_size`, `random_state`, and `num_<things>` for counts, for example `num_bins`.
- End fitted attributes with an underscore, for example `projections_`.
