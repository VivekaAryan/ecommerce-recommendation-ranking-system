# Interference-Aware Experimentation

## Problem

In marketplace settings, showing item X to user A depletes inventory and changes relevance for user B. Naive user-level A/B tests violate independence assumptions.

## Demonstration

1. Run naive A/B test in simulator with shared inventory
2. Observe biased treatment effect estimate
3. Run switchback test (6-hour windows) or cluster-randomized design
4. Compare corrected estimate

## Decision

Switchback for time-varying interference; cluster-randomized when geographic/user-cluster interference dominates.

## Expected Finding

Naive A/B overestimates positive treatment effect when treatment policy increases item consumption, reducing availability for control users.
