# Cold-Start Retrieval

## Decision

Built alongside retrieval, not retrofitted.

## New Users

- No interaction history → content-based fallback using category-affinity items
- Item tower not used for user embedding; fallback ranks by category + price

## New Items

- Item tower produces embedding immediately from title/description metadata
- No retraining required for new item to enter ANN index

## Tradeoff

Cold-start quality is lower than warm-start collaborative retrieval. Acceptable for tail of distribution; monitored via separate cold-start recall metrics.
