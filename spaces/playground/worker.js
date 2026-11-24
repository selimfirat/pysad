/* Run all Python work away from the canvas thread. */
importScripts("https://cdn.jsdelivr.net/pyodide/v0.27.5/full/pyodide.js");
let pyodide;
const loadedDatasets = new Set();
async function initialize() {
  const [py, engineCode, manifest] = await Promise.all([
    loadPyodide({
      indexURL: "https://cdn.jsdelivr.net/pyodide/v0.27.5/full/",
      packages: ["numpy", "scipy", "scikit-learn"],
    }),
    fetch("./engine.py").then(r => {
      if (!r.ok) throw new Error(`Failed to fetch engine.py (${r.status})`);
      return r.text();
    }),
    fetch("./manifest.json").then(r => r.ok ? r.json() : null),
  ]);
  pyodide = py;
  if (manifest?.wheel) {
    const response = await fetch(`./${manifest.wheel}`);
    if (!response.ok) throw new Error(`Failed to fetch PySAD wheel (${response.status})`);
    py.unpackArchive(await response.arrayBuffer(), "zip", {extractDir: "/home/pyodide"});
  } else {
    await py.loadPackage("micropip");
    await py.runPythonAsync("import micropip; await micropip.install('pysad')");
  }
  py.FS.writeFile("/home/pyodide/engine.py", engineCode);
  py.runPython(`
import sys
sys.path.insert(0, "/home/pyodide")
import json
import engine
live_session = None
  `);
}
async function ensureStreamDataset(streamName) {
  const py = pyodide;
  if (!py) return;
  if (streamName === "NYC Taxi Demand (NAB, univariate)" && !loadedDatasets.has("nab")) {
    const [taxiRes, winRes] = await Promise.all([
      fetch("./data/nab/realKnownCause/nyc_taxi.csv"),
      fetch("./data/nab/combined_windows.json"),
    ]);
    if (!taxiRes.ok) throw new Error(`Failed to fetch NYC Taxi dataset (${taxiRes.status})`);
    py.FS.mkdirTree("/home/pyodide/data/nab/realKnownCause");
    py.FS.writeFile("/home/pyodide/data/nab/realKnownCause/nyc_taxi.csv", await taxiRes.text());
    if (winRes.ok) {
      py.FS.writeFile("/home/pyodide/data/nab/combined_windows.json", await winRes.text());
    }
    loadedDatasets.add("nab");
  } else if (streamName === "Arrhythmia (ODDS, 274 features)" && !loadedDatasets.has("arrhythmia")) {
    const res = await fetch("./data/arrhythmia.mat");
    if (!res.ok) throw new Error(`Failed to fetch Arrhythmia dataset (${res.status})`);
    py.FS.mkdirTree("/home/pyodide/data");
    py.FS.writeFile("/home/pyodide/data/arrhythmia.mat", new Uint8Array(await res.arrayBuffer()));
    loadedDatasets.add("arrhythmia");
  }
}

self.onmessage = async ({data: {id, method, options, csvBytes}}) => {
  try {
    let result = null;
    if (method === "init") {
      await initialize();
    } else if (method === "next") {
      result = JSON.parse(pyodide.runPython("json.dumps(live_session.next_batch(8))"));
    } else {
      await ensureStreamDataset(options.stream_name);
      pyodide.globals.set("_req_json", JSON.stringify(options));
      pyodide.globals.set("_csv_u8", csvBytes || null);
      pyodide.runPython(`
req = json.loads(_req_json)
req["csv_data"] = _csv_u8.to_bytes() if _csv_u8 is not None else None
      `);
      if (method === "start") {
        pyodide.runPython("live_session = engine.LiveSession(**req)");
      } else if (method === "evaluate") {
        result = JSON.parse(pyodide.runPython("engine.run_playground_json(**req)"));
      }
    }
    self.postMessage({id, result});
  } catch (error) {
    self.postMessage({id, error: error.message || String(error)});
  }
};
