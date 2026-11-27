---
name: pysad-explain
description: Explain feature contributions to anomaly scores through the PySAD MCP server. Use when a user asks which features affect an anomaly score.
---

# Explain anomaly scores

Use the connected `pysad_explain` tool. Client prefixes can change its name.
If it is unavailable, explain that the server requires `pysad[mcp]` and `pysad-mcp`.
Do not install packages or change client configuration without the user's permission.

Read the tool schema. Use the exact detector class name and supported constructor parameters.
Keep numeric rows in time order and preserve their feature order.
Ask for missing feature names when the user needs named contributions.
A flat list represents one multivariate instance. Nested lists represent successive instances.
Set `random_state` when the detector supports it, and report the seed.

The tool creates a fresh detector, fits all supplied rows, then explains those rows.
These explanations use the final fitted model. They do not explain earlier online scores from another tool call.
Do not claim that they provide prequential evaluation or independent held-out results.

If the result contains an error, report it before interpreting contributions.
If `has_explain` is false or an explanation is null, report that the detector lacks an explanation for that input.
Do not invent feature contributions or replace them with raw feature magnitudes.

Report the detector, parameters, training rows, feature mapping, and returned contributions.
Describe contributions as model explanations. Do not claim that they identify the cause of an anomaly.
