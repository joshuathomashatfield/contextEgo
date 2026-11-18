# Architecture

Modules separate datasets/schema, video/reader, encoders/clip and cache, temporal/windows, procedural/memory, models/classifier, training/runner, evaluation/metrics and visualization/figures. CLI and thin scripts coordinate them. The encoder Protocol permits another backbone without downstream model changes. Cached arrays are timestamps (N) and features (N,D), plus JSON metadata, read without pickle. Feature extraction streams video in batches and experiment code loads caches in memory.

Prediction order: causal visual window ? pooled visual representation + prior predicted memory ? logits ? optional decoder ? saved prediction ? memory update. Procedure schema and historical predictions remain distinct. No future visual frame or annotation history enters forward inference.

The optional video export replays saved predictions; it cannot alter experiment metrics. Source annotations are converted before training through a normalized boundary contract, avoiding undocumented dataset assumptions.
