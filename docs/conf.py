# Configuration file for the Sphinx documentation builder.
# See https://www.sphinx-doc.org/en/master/usage/configuration.html

import html
import json
import os
import re
import sys
from pathlib import Path

from docutils import nodes
from sphinx.application import Sphinx
from sphinx.ext.autosummary import generate as autosummary_generate_mod
from sphinx.util.docutils import SphinxRole

DOCS_DIR = Path(__file__).resolve().parent
ROOT_DIR = DOCS_DIR.parent
sys.path.insert(0, str(DOCS_DIR))
sys.path.insert(0, str(ROOT_DIR))

from _autodiscover import (  # noqa: E402
    collect_bibtex,
    doc_counts,
    expand_api_autosummary,
    expand_detectors_tables,
    expand_examples_gallery,
)

from pysad.version import __version__  # noqa: E402


class StrongEmphasis(SphinxRole):
    """Render text in bold and italics."""

    def run(self) -> tuple[list[nodes.Node], list[nodes.system_message]]:
        return [nodes.strong("", "", nodes.emphasis("", self.text))], []


def _find_autosummary_in_files(
    filenames: list[str],
) -> list[autosummary_generate_mod.AutosummaryEntry]:
    documented: list[autosummary_generate_mod.AutosummaryEntry] = []
    for filename in filenames:
        raw = Path(filename).read_text(encoding="utf-8", errors="ignore")
        if "/api/" in Path(filename).as_posix():
            raw = expand_api_autosummary(raw)
        documented.extend(
            autosummary_generate_mod.find_autosummary_in_lines(raw.splitlines(), filename=filename)
        )
    return documented


def _on_source_read(app: Sphinx, docname: str, source: list[str]) -> None:
    if docname.startswith("api/"):
        source[0] = expand_api_autosummary(source[0])
    elif docname == "detectors":
        source[0] = expand_detectors_tables(source[0])
    elif docname == "examples":
        source[0] = expand_examples_gallery(source[0], ROOT_DIR)
    elif docname == "installation":
        source[0] = source[0].replace("|release|", str(app.config.release))


_DEFAULT_SEO_DESCRIPTION = (
    "PySAD is a Python library for streaming anomaly detection with 50+ online multivariate and "
    "univariate detectors, calibrated alerts (conformal and SPOT), drift monitoring, and "
    "prequential evaluation."
)
_META_TAG_RE = re.compile(r"<meta\b[^>]*>", re.I)
_META_NAME_DESC_RE = re.compile(r"""name\s*=\s*(["'])description\1""", re.I)
_META_CONTENT_RE = re.compile(r"""content\s*=\s*(["'])(.*?)\1""", re.I | re.S)


def _add_seo_context(
    app: Sphinx,
    pagename: str,
    templatename: str,
    context: dict[str, object],
    doctree: object,
) -> None:
    """Populate description, OpenGraph, Twitter Card, and SoftwareSourceCode JSON-LD metadata."""
    metatags = str(context.get("metatags", ""))
    extracted_description = ""
    has_meta_description = False
    for tag in _META_TAG_RE.findall(metatags):
        if _META_NAME_DESC_RE.search(tag):
            has_meta_description = True
            content_match = _META_CONTENT_RE.search(tag)
            if content_match:
                extracted_description = html.unescape(content_match.group(2)).strip()
            break

    context["seo_has_meta_description"] = has_meta_description
    context["seo_description"] = extracted_description or _DEFAULT_SEO_DESCRIPTION

    if pagename == "index":
        canonical_url = str(context.get("pageurl") or "https://pysad.readthedocs.io/en/stable/")
        context["seo_jsonld"] = json.dumps(
            {
                "@context": "https://schema.org",
                "@type": "SoftwareSourceCode",
                "name": "PySAD",
                "description": context["seo_description"],
                "url": canonical_url,
                "codeRepository": "https://github.com/selimfirat/pysad",
                "programmingLanguage": "Python",
                "license": "https://opensource.org/licenses/BSD-3-Clause",
                "softwareVersion": str(app.config.release),
                "author": {"@type": "Person", "name": "Selim Firat Yilmaz"},
                "keywords": [
                    "streaming anomaly detection",
                    "online outlier detection",
                    "concept drift",
                    "conformal prediction",
                    "prequential evaluation",
                    "Python",
                ],
            }
        )
    else:
        context["seo_jsonld"] = ""


def setup(app: Sphinx) -> None:
    autosummary_generate_mod.find_autosummary_in_files = _find_autosummary_in_files
    app.add_role("strong-emphasis", StrongEmphasis())
    app.connect("source-read", _on_source_read)
    app.connect("html-page-context", _add_seo_context)


# Live substitutions computed from Data.datasets(), pysad.models.__all__, and benchmark.py.
rst_epilog = (
    "\n"
    + "\n".join(f".. |{name}| replace:: {value}" for name, value in doc_counts(ROOT_DIR).items())
    + "\n"
)

_COLLECTED_BIB = DOCS_DIR / "_build" / "collected.bib"
_COLLECTED_BIB.parent.mkdir(parents=True, exist_ok=True)
_BIB_TEXT = collect_bibtex(ROOT_DIR)[0]
if not _COLLECTED_BIB.exists() or _COLLECTED_BIB.read_text(encoding="utf-8") != _BIB_TEXT:
    _COLLECTED_BIB.write_text(_BIB_TEXT, encoding="utf-8")

# -- Project information -----------------------------------------------------

project = "PySAD"
copyright = "2020-2026, Selim Firat Yilmaz"
author = "Selim Firat Yilmaz"
version = __version__
release = __version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_llms_txt",
]

templates_path = ["_templates"]
# Single backticks in docstrings mean code.
default_role = "code"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_default_options = {"members": True, "inherited-members": True}
autodoc_typehints = "none"
autodoc_mock_imports = ["river"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

bibtex_bibfiles = ["_build/collected.bib"]

# These services block automated clients.
linkcheck_ignore = [
    r"https?://odds\.cs\.stonybrook\.edu/?",
    r"https://scholar\.google\.com/",
    # DOI links redirect to bot challenges at these publishers.
    r"https://doi\.org/10\.1080/",  # Taylor & Francis
    r"https://doi\.org/10\.1093/",  # Oxford University Press
    r"https://doi\.org/10\.1137/",  # SIAM
    r"https://doi\.org/10\.1145/",  # ACM
    r"https://doi\.org/10\.14778/",  # VLDB papers on ACM
    r"https://doi\.org/10\.3390/",  # MDPI
]

# llms.txt lists the rendered pages, because the API pages get their text from docstrings.
llms_txt_summary = (
    "PySAD is a Python library for streaming anomaly detection on multivariate and univariate "
    "data streams. Detectors learn and score one instance at a time via fit_partial, "
    "score_partial, and fit_score_partial with bounded memory, returning higher scores for more "
    "anomalous points. Core patterns: compose pipelines with `|` (StreamPipeline, e.g. "
    "`StandardScaler() | LODA() | ConformalProbabilityCalibrator()`), ensemble models with `&` "
    "(ModelEnsemble), calibrate raw scores with ConformalProbabilityCalibrator or "
    "SPOTProbabilityCalibrator, attribute anomalous features with LODA.explain_partial(), monitor "
    "and reset on concept drift with DriftAdaptiveDetector and streaming drift meters, evaluate "
    "prequentially with evaluate_stream, and integrate PyOD (ReferenceWindowModel, OneFitModel), "
    "River (PySADToRiverAdapter, PredictiveAnomalyDetector), and scikit-learn "
    "(PySADToScikitLearnAdapter)."
)
llms_txt_uri_template = "{base_url}{docname}.html"

# Copy only the code from prompts such as ">>> " and "$ ".
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_title = f"PySAD {release}"
# Read the Docs gives the URL of the version that it builds. The links in llms.txt start with it.
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")
html_favicon = "favicon.svg"
html_static_path = ["_static"]
html_css_files = [
    "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap",
    "custom.css",
]
html_js_files = ["best.js"]  # bolds the best value of each metric column in the benchmark tables

html_theme_options = {
    # Read the Docs builds "latest" from master, which can differ from the release on PyPI.
    "announcement": (
        "These pages describe the development version on GitHub. For the release on PyPI, read the "
        '<a href="https://pysad.readthedocs.io/en/stable/">stable documentation</a>.'
        if os.environ.get("READTHEDOCS_VERSION") == "latest"
        else None
    ),
    "light_logo": "logo.svg",
    "dark_logo": "logo-dark.svg",
    "sidebar_hide_name": True,
    "source_repository": "https://github.com/selimfirat/pysad/",
    "source_branch": "master",
    "source_directory": "docs/",
    # Brand colors: abyssal teal #00757a (deep #00565a, dark #2dd4bf), signal coral #e63946, paper #fcfcf9, canvas #f2f4f2, ink #0d1b22.
    "light_css_variables": {
        "font-stack": 'Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif',
        "font-stack--monospace": '"IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
        "font-stack--headings": "var(--font-stack)",
        "color-foreground-primary": "#0d1b22",
        "color-background-primary": "#fcfcf9",
        "color-background-secondary": "#f2f4f2",
        "color-background-border": "#dce3e2",
        "color-code-background": "#f4f7f6",
        "color-code-string": "#2a9d9f",
        "color-highlight-on-target": "#e7f5f5",
        "color-api-name": "#0d1b22",
        "color-api-pre-name": "#5b6f78",
        "color-brand-primary": "#00757a",
        "color-brand-content": "#00565a",
        "color-brand-visited": "#00565a",
    },
    "dark_css_variables": {
        "color-foreground-primary": "#edf5f6",
        "color-background-primary": "#0b1519",
        "color-background-secondary": "#070f12",
        "color-background-border": "#1b2f37",
        "color-code-background": "#070f12",
        "color-code-string": "#99f6e4",
        "color-highlight-on-target": "#122a31",
        "color-api-name": "#edf5f6",
        "color-api-pre-name": "#99b0b6",
        "color-brand-primary": "#2dd4bf",
        "color-brand-content": "#2dd4bf",
        "color-brand-visited": "#2dd4bf",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/selimfirat/pysad",
            "html": '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>',
            "class": "",
        },
    ],
}
