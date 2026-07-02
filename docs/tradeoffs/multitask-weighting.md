# Multi-Task Head Weighting

## Decision

| Head | Weight | Rationale |
|------|--------|-----------|
| Click | 0.35 | Primary engagement signal; drives short-term metrics |
| Purchase | 0.30 | Conversion intent; closer to business outcome |
| Expected Value | 0.25 | Price-weighted; prevents surfacing cheap clickbait |
| Engagement | 0.10 | Long-term proxy (repeat interaction); low weight to avoid overfitting sparse signal |

## Product Framing

This is a product decision wearing an ML hat. In production, these weights would be tuned via A/B tests against revenue and retention OKRs. The click head alone would over-index on sensational items; the value head provides a corrective signal.

## Monitoring

SHAP analysis per head (`interpretability.py`) shows where click-maximizing and value-maximizing signals disagree — useful for explaining model behavior in review.
