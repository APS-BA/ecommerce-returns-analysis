"""
Chi-square test of review-rating distribution for returned vs non-returned orders.

Source: generated and run in Better Analyst (app.betteranalyst.com), session
"E-Commerce Returns Analysis", 2026-09-24. Exported verbatim; only this header was added.
Input paths point to Better Analyst's sandbox: /home/user/workspace/1790268568295_orders.csv, /home/user/workspace/1790268570650_reviews.csv, /home/user/workspace/outputs.
Change them to your local data folder before running.
"""

import pandas as pd
from pathlib import Path
from scipy.stats import chi2_contingency
orders_path = "/home/user/workspace/1790268568295_orders.csv"
reviews_path = "/home/user/workspace/1790268570650_reviews.csv"
orders = pd.read_csv(orders_path)
reviews = pd.read_csv(reviews_path)
print("SOURCE_CHECK", {"orders": [orders_path, len(orders), list(orders.columns)], "reviews": [reviews_path, len(reviews), list(reviews.columns)]})
if orders["order_id"].duplicated().any():
    raise ValueError("order_id is not unique; refusing review status join")
orders["returned"] = orders["order_status"].eq("returned").astype(int)
review = reviews.merge(orders[["order_id", "returned"]], on="order_id", how="inner", validate="many_to_one")
review["rating_clean"] = pd.to_numeric(review["rating"], errors="coerce")
review = review.loc[review["rating_clean"].notna()].copy()
review["return_status"] = review["returned"].map({0: "Not returned", 1: "Returned"})
review["rating_label"] = review["rating_clean"].astype("string")
distribution = review.groupby(["rating_label", "return_status"], dropna=False).size().unstack(fill_value=0).reindex(columns=["Not returned", "Returned"], fill_value=0).reset_index()
distribution = distribution.rename(columns={"rating_label": "rating", "Not returned": "non_returned_review_rows", "Returned": "returned_review_rows"})
distribution["total_review_rows"] = distribution["non_returned_review_rows"] + distribution["returned_review_rows"]
distribution["return_rate"] = distribution["returned_review_rows"] / distribution["total_review_rows"]
ct = pd.crosstab(review["rating_label"], review["returned"]).reindex(columns=[0, 1], fill_value=0)
chi2, p_value, dof, expected = chi2_contingency(ct.to_numpy(), correction=False)
denominator = int(ct.to_numpy().sum()) * min(ct.shape[0] - 1, ct.shape[1] - 1)
cramers_v = (chi2 / denominator) ** 0.5 if denominator else float("nan")
test = pd.DataFrame([{"test": "Review rating distribution (rating category × return status)", "chi2": float(chi2), "p_value": float(p_value), "degrees_freedom": int(dof), "cramers_v": float(cramers_v), "review_rows": int(ct.to_numpy().sum()), "expected_cells_below_5": int((expected < 5).sum()), "expected_cell_count": int(expected.size)}])
outdir = Path("/home/user/workspace/outputs")
saved_paths = []
for filename, frame in [("return_review_rating_distribution.csv", distribution), ("return_review_rating_chi_square.csv", test)]:
    path = outdir / filename
    frame.to_csv(path, index=False)
    saved_paths.append(str(path))
    print("SAVED_TABLE", filename, "rows", len(frame), "columns", list(frame.columns))
print("RATING_TEST", test.to_dict("records"))
print("RATING_DISTRIBUTION", distribution.to_dict("records"))
print("OUTPUT_PATHS", saved_paths)
