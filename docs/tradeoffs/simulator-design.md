# Simulator Design

## Decision

Lightweight marketplace simulator using real catalog items with synthetic user preference vectors, limited inventory, and attention decay.

## Rationale

Static offline datasets cannot demonstrate feedback loops, interference, or propensity-based evaluation. The simulator is the spine for Phases 4–5.

## Key Mechanics

- **Position bias** — Configurable decay curve over slate positions
- **Inventory** — Serving an item depletes stock, creating interference between users
- **Attention decay** — Overserved items lose attention score
- **Propensity logging** — Every impression logged with policy ID and propensity for IPS/DR

## Tradeoff

Synthetic user preferences approximate real behavior but enable controlled experiments. Real user IDs from the dataset are mixed with synthetic users for scale.
