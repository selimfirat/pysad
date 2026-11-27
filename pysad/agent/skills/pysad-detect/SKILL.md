---
name: pysad-detect
description: Score numeric data streams and produce calibrated anomaly alerts through the PySAD MCP server. Use for stream analysis, not detector implementation.
---

# Score a data stream

Use the connected PySAD MCP tools. If they are unavailable, explain that the server requires `pysad[mcp]` and `pysad-mcp`.
Do not install packages or change client configuration without the user's permission.

Read the connected tool schemas before choosing a detector or its parameters.
Use `list_detectors` to discover supported detectors. Client prefixes can change tool names.
Ask for missing data, feature meanings, and time order when they affect the analysis.
Keep rows in time order. Each row holds one numeric feature vector with the same feature count.
A flat list represents one multivariate instance. Use `[[1.0], [2.0]]` for two univariate instances.

For raw scores, call a detector tool such as `pysad_loda` with `fit=true`.
Each call starts a fresh model. Send the complete analysis window in one call.
Do not split a stream across calls and claim that training continues.
Use `random_state` when the detector supports it, and report the seed.
The fitting path scores each instance before learning from it.
Higher scores mean more anomalous instances. Raw scores are not probabilities.

If the user requests calibrated alerts, use `pysad_stream` with an explicit supported calibrator and threshold.
Read its schema for accepted parameters. Do not describe an arbitrary raw score cutoff as a probability.
If a tool returns an error, report it and correct the input before interpreting scores.

Report the detector, parameters, seed, input order, and score range.
Identify large scores by row or timestamp. Explain warmup limits and the chosen alert threshold.
If labels are absent, do not claim measured accuracy or a false alarm rate.
