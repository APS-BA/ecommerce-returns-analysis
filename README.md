# E-Commerce Returns Analysis

**Business Data Mining & Decision Models (BDMDM) · IIM Nagpur · Team of 5**

Why is one in five orders returned, and what is it worth to fix? A data-mining
diagnosis of 20,000 orders that ended in a clear recommendation, even though the
data held no predictive signal.

![Python](https://img.shields.io/badge/Python-pandas%20%7C%20scikit--learn-3776AB)
![Methods](https://img.shields.io/badge/methods-classification%20%7C%20clustering%20%7C%20hypothesis%20tests-555)
![Excel](https://img.shields.io/badge/Excel-decision%20model-217346)

---

## The problem

An online retailer's revenue stayed flat for 23 months while roughly 40% of orders
were cancelled or returned. Returns alone were **4,066 of 20,000 orders (20.3%)**,
worth **$2.43M**, as many returns as completed orders. We refocused the project on
one question: *what drives returns, and which decisions would reduce them
cost-effectively?*

## Data

Six linked tables, Jan 2024 to Nov 2025, joined on `user_id`, `order_id` and `product_id`.

| Table | Rows | Used for |
|---|---:|---|
| orders | 20,000 | Target (returned) and order value |
| order_items | 43,525 | Basket size and composition |
| products | 2,000 | Category, brand, price, rating |
| users | 10,000 | Customer features and tenure |
| reviews | 15,000 | Experience signals |
| events | 80,000 | Engagement checks |

The dataset is synthetic, which we treat as a finding in itself: half the orders
predate the customer's signup date, and the order-status label fails a basic
validity check (see the report, Section 3).

## Approach (CRISP-DM)

- **Explicit definitions:** return rate on all orders vs non-cancelled orders
- **Hypothesis tests:** chi-square with Cramér's V and Bonferroni correction across 10 attributes
- **Six classifiers** (logistic regression, decision tree, kNN, random forest, gradient boosting, MLP), benchmarked against naive baselines and shuffled-label models
- **Unsupervised:** K-means on RFM, association rules on category baskets, Isolation Forest
- **Serial returners:** binomial test per customer vs what chance alone would produce
- **Forecasting:** naive, moving average, exponential smoothing and Holt, back-tested
- **Decision model:** expected-value cost-benefit with low/base/high scenarios and break-even
- **Independent cross-check:** the full pipeline re-run in an agentic AI analytics tool (Better Analyst) and reconciled number by number

## What we found

| Check | Result |
|---|---|
| Best classifier ROC-AUC | ≈ 0.5 (random) |
| Return-rate spread across category, brand, price, month | within ~2 pts of 20.3% |
| Serial returners | match the binomial expectation for chance |
| Returns over $1,000 | 18.6% of returns, **59.6% of returned value** |
| Returns + cancellations | **39.6%** of gross order value never becomes revenue |

The models came out near random, and flagged customers and SKUs matched what
chance alone would produce. Instead of presenting weak patterns as insight, we
reported the null result and changed the recommendation:

1. **Fix the record first:** reconcile order status and capture return-reason data
2. **Fund recovery now:** route returns by value (refund-without-collection under $25, inspect and exchange first over $1,000)
3. **Fund prevention later:** only once reason codes show where returns come from

A 15-metric *Returns Diagnostic Scorecard* and the **RETURN-SHIELD** programme turn
this into owners, decision rules and a pilot design (break-even at a 3.1-point cut
in the return rate).

## Repository contents

```
report/
  Returns_Report.pdf        Final report (10 pages)
  Returns_Report.docx       Editable version
  Project_Proposal.pdf      Original proposal deck (PDF, previews on GitHub)
  Project_Proposal.pptx     Original proposal deck (editable)
analysis/
  Returns_Analysis.xlsx     18-sheet workbook: KPIs, scorecard, tests, model
                            comparison, clusters, rules, forecast, decision model,
                            Better Analyst cross-check and order-level data
code/
  01_prepare_order_level_data.py          Join the tables, build the order-level returned flag
  02_return_drivers_tests_and_models.py   Data-quality checks, chi-square / Cramér's V by segment,
                                          review tests, 5 classifiers, binomial test, monthly trend
  03_category_chart_data.py               Corrected category return-rate table for charting
  04_monthly_forecast_partial_month.py    Linear-trend forecast with the partial final month excluded
  05_review_rating_distribution_test.py   Review-rating distribution, returned vs kept orders
  exploratory/
    retention_growth_analysis.py          Earlier retention/growth direction, before the refocus on returns
```

The scripts are the Python generated and run in the Better Analyst AI analytics
tool (the independent cross-check in Section 6 of the report), exported verbatim
with a header describing each one. Data paths point to that tool's sandbox; change
them to a local data folder to run. Requires `pandas`, `numpy`, `scipy`,
`scikit-learn` and `matplotlib` (the exploratory script also uses `statsmodels`
and `xgboost`).

## Why this is here

It shows the part of analytics I care most about: telling a real signal from a
confident-looking wrong one, and still giving the business a decision it can act on.

**Team:** Protik Das, Satyabrata Singh, Shekhar Yadav, Abhisek Samantaray,
Aishwarya Pratap Singh
