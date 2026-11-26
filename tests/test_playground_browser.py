"""Test browser controls without downloading the WebAssembly runtime."""

from pathlib import Path

import pytest

from spaces.playground import engine


@pytest.fixture(scope="module")
def browser():
    """Use an existing browser when Playwright is available."""
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as runner:
        try:
            instance = runner.chromium.launch()
        except playwright.Error:
            try:
                instance = runner.chromium.launch(channel="chrome")
            except playwright.Error:
                pytest.skip("No installed Chromium browser")
        yield instance
        instance.close()


@pytest.fixture(scope="module")
def payload():
    """Use real engine output for worker responses."""
    return engine.run_playground_json(
        stream_name=engine.STREAM_SYNTHETIC,
        detectors=["LODA"],
        calibrator_name=engine.CALIBRATOR_CONFORMAL,
        max_points=300,
    )


@pytest.fixture
def page(browser, payload):
    """Load the real page with local worker and clipboard substitutes."""
    html = (Path(__file__).resolve().parents[1] / "spaces/playground/index.html").read_text()
    page = browser.new_page()
    page.set_default_timeout(3000)
    page.route(
        "**/*",
        lambda route: (
            route.fulfill(body=html)
            if route.request.is_navigation_request()
            else route.fulfill(body="")
        ),
    )
    page.add_init_script(
        """
        window.Plotly = {react(element) { element.on = () => {}; }};
        window.requests = [];
        window.copied = null;
        Object.defineProperty(navigator, 'clipboard', {
          value: {writeText: async text => { window.copied = text; }}
        });
        window.Worker = class {
          postMessage(request) {
            window.requests.push(request);
            setTimeout(() => this.onmessage({data: {
              id: request.id,
              result: request.method === 'evaluate' ? PAYLOAD : null
            }}), 0);
          }
        };
        """.replace("PAYLOAD", payload)
    )
    yield page
    page.close()


def test_shared_feature_reaches_worker_and_restores_control(page):
    """A shared feature controls scoring and remains selected after results load."""
    page.goto("https://playground.test/?feature=sensor_3&detectors=LODA&threshold=0.970")
    page.wait_for_function("window.__pysadPlaygroundReady")
    options = page.evaluate("window.requests.find(r => r.method === 'evaluate').options")
    assert options["primary_feature_name"] == "sensor_3"
    assert options["detectors"] == ["LODA"]
    assert options["threshold"] == 0.970
    assert page.locator("#featureSelect").input_value() == "sensor_3"


@pytest.mark.parametrize("injected", ["{bad", "null", "{}", "[null]", '[{"index": "bad"}]'])
def test_invalid_shared_injections_do_not_stop_startup(page, injected):
    """Invalid injection parameters leave the page ready with no injections."""
    from urllib.parse import urlencode

    page.goto("https://playground.test/?" + urlencode({"injected": injected}))
    page.wait_for_function("window.__pysadPlaygroundReady")
    options = page.evaluate("window.requests.find(r => r.method === 'evaluate').options")
    assert options["injected"] == []


def test_copy_pipeline_and_share_csv_settings(page):
    """CSV bytes stay local while the copy buttons expose code and settings."""
    from urllib.parse import parse_qs, urlparse

    page.goto("https://playground.test/?detectors=LODA")
    page.wait_for_function("window.__pysadPlaygroundReady")
    page.locator("#streamSelect").select_option("Upload CSV")
    csv = b"cpu,label\n" + b"0.2,0\n" * 40
    page.locator("#csvFileInput").set_input_files(
        {"name": "private.csv", "mimeType": "text/csv", "buffer": csv}
    )
    page.wait_for_function("window.requests.some(r => r.csvBytes?.length > 0)")
    uploaded = page.evaluate(
        "Array.from(window.requests.find(r => r.csvBytes?.length > 0).csvBytes)"
    )
    assert bytes(uploaded) == csv
    page.get_by_role("button", name="Copy Python pipeline", exact=True).click()
    assert page.evaluate("window.copied") == page.locator("#pipelineCode").text_content()
    assert "fit_score_partial" in page.evaluate("window.copied")
    page.get_by_role("button", name="Copy share link", exact=True).click()
    link = page.evaluate("window.copied")
    params = parse_qs(urlparse(link).query)
    assert params["stream"] == ["Upload CSV"]
    assert params["detectors"] == ["LODA"]
    assert "private.csv" not in link
    assert "cpu" not in link


def test_share_link_restores_experiment_settings(page):
    """Copying and reopening a link keeps the selected experiment settings."""
    from urllib.parse import urlencode

    settings = {
        "detectors": "LODA",
        "calibrator": engine.CALIBRATOR_SPOT,
        "threshold": "0.970",
        "calWindow": "400",
        "points": "800",
        "feature": "sensor_2",
        "injected": '[{"index":250,"feature_idx":1,"magnitude_sigma":7}]',
        "inspect": "250",
    }
    page.goto("https://playground.test/?" + urlencode(settings))
    page.wait_for_function("window.__pysadPlaygroundReady")
    original = page.evaluate("window.requests.find(r => r.method === 'evaluate').options")
    page.locator("#copyLinkBtn").click()
    page.wait_for_function("window.copied !== null")
    shared_url = page.evaluate("window.copied")
    page.goto(shared_url)
    page.wait_for_function("window.__pysadPlaygroundReady")
    restored = page.evaluate("window.requests.find(r => r.method === 'evaluate').options")
    assert restored == original
    assert restored["primary_feature_name"] == "sensor_2"
    assert restored["injected"] == [{"index": 250, "feature_idx": 1, "magnitude_sigma": 7}]
