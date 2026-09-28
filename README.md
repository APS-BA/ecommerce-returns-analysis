# E-Commerce Returns Analysis

Business Decision Models & Data Mining, IIM Nagpur · Team of 5 · [Your role: fill in]

## The problem
An online retailer's revenue stayed flat for 23 months while roughly 40% of orders
were cancelled or returned. We refocused the project on one question: why are so
many orders returned, and what would it be worth to fix?

## Data
Six linked tables (users, products, orders, order items, events, reviews). The
dataset is synthetic, which we treat as a finding in itself (see below).

## Approach
- Return-rate definitions made explicit (all orders vs non-cancelled orders)
- Chi-square and permutation tests for return "hotspots" by category, customer and SKU
- Predictive models for return risk, compared against naive baselines
- Watch-lists tested against what random chance would produce
- A cost-based decision model with low, base and high scenarios and a break-even point

## What we found
The predictive models came out near random (AUC ≈ 0.5), and flagged customers and
SKUs matched what chance alone would produce. Instead of presenting weak patterns
as insight, we reported the null result and changed the recommendation: fix the
return-reason data first, fund recovery now, and fund prevention only once reason
codes show where returns come from.

## Why this is here
It shows the part of analytics I care most about: telling a real signal from a
confident-looking wrong one.

## Files
[Add: final report PDF, workbook, notebooks]
