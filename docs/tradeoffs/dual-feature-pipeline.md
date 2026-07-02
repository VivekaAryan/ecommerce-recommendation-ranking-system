# Dual Feature Pipeline: Why Build Staleness From Day One

## Decision

Implement batch and online feature pipelines in parallel from Phase 0, with configurable staleness injection.

## Rationale

Train/serve skew is one of the most common silent production bugs. Retrofitting staleness simulation after the fact produces unrealistic comparisons. Building both paths against the same `FeatureRegistry` ensures feature definitions stay synchronized.

## Staleness Rules

- User behavioral features: 6-hour refresh lag
- Item popularity features: 24-hour refresh lag
- New users: zero interaction counts, default category affinity
- New items: content-only features, zero popularity

## Measurement

`scripts/measure_skew.py` compares batch vs online feature distributions and reports absolute/relative deltas per feature.
