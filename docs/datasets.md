# Dataset integration

Primary sources inspected on 2026-11-12:
- https://www.khoury.northeastern.edu/home/eelhami/egoper.htm
- https://github.com/robert80203/EgoPER_official

The official EgoPER README requires an access request to its authors with institutional and research-purpose information. Its listing describes annotation.json, per-task video archives and official split files. The actual dataset is absent locally. Annotation keys, participant IDs, time units, trimming offsets and licensing/redistribution permissions have not been verified from the distribution. A publicly readable repository is not evidence of an unrestricted dataset license. Request authorized access and review the supplied terms before use. No email has been sent and no private download links have been invented.

After obtaining access: run inspect_dataset.py on the actual directory and annotation JSON. Record release/version and hashes; verify video-task-subject identities; determine whether boundaries refer to original or trimmed footage; confirm seconds versus frame indices using native presentation timestamps; inspect error/background and overlapping intervals. Inspect official splits for participant leakage before choosing them. Do not infer participants from filenames without documentation.

DatasetAdapter contract: convert an authorized source into a list of normalized records documented in README and a separate procedure schema. The generic prepare command is the conversion destination/validator, not an EgoPER parser. Source-specific conversion code must be added after inspecting real samples. Keep an audit note of every source field mapping, ignored field, timestamp transform and unknown-label convention. Preserve official splits when appropriate by supplying a validated train/validation/test JSON mapping video IDs. One procedure per prepared root; no unrelated task label spaces are pooled.

Unknown/gap labels use -1. Overlap is rejected instead of silently selecting a label. Segments use [start,end), may have gaps, and must lie in positive finite duration. Step IDs are remapped to contiguous integers with a saved procedure mapping. Timestamp labels are sampled directly from segments. Never split frames from a video across data partitions.

Synthetic fixture: nine independently identified software sessions, three steps, eight-dimensional noisy features and known boundaries. Some histories repeat/backtrack. The IDs are fixture metadata, not human participants. No synthetic RGB footage is presented as egocentric evidence.
