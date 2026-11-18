# Limitations

No real annotated egocentric dataset, CLIP weights, CUDA execution or scientific benchmark was available in this run. The CLIP adapter remains externally unverified. Dataset access and schema mapping are pending. Synthetic results are software tests only. No hypothesis has been accepted or rejected.

The graph is descriptive, not a learned/calibrated graph prior. Full history is a bounded statistical summary. The small head may underfit semantics, and sequential per-observation updates can bias training order. Confidence is uncalibrated. No background class is learned; unlabeled intervals are censored. Decoding comparisons and repeated-seed studies remain to be run on real data. Observation-grid boundaries limit timestamp resolution. Delay is conditional on tolerance matches and should always be read with missed counts.

OpenCV timestamp support depends on backend; unsupported videos fail rather than assume constant FPS. Offline renderer has been exercised with locally generated test footage only, not real dataset videos. Entire feature arrays load into RAM. Feature caches identify the model by name, not a pinned revision; archive a frozen environment/model snapshot for publication. No live AR interface, successful-completion verifier, error detector, or real-time capability is claimed.

Next: obtain licensed EgoPER access, inspect real annotations and participant splits, implement an audited adapter, extract public frozen features, run baselines and all conditions with predeclared validation settings and multiple seeds. Investigate deviations, memory error propagation and accuracy/cost tradeoffs using participant-level confidence intervals.
