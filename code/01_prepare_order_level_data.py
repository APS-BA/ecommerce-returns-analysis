"""
Load the 5 CSVs, join them and build the order-level table with the returned flag.

Source: generated and run in Better Analyst (app.betteranalyst.com), session
"E-Commerce Returns Analysis", 2026-09-24. Exported verbatim; only this header was added.
Input paths point to Better Analyst's sandbox: /home/user/workspace/1790268567916_products.csv, /home/user/workspace/1790268567919_order_items.csv, /home/user/workspace/1790268568295_orders.csv, /home/user/workspace/1790268568306_users.csv, /home/user/workspace/outputs/order_return_analysis_source.csv.
Change them to your local data folder before running.
"""

import pandas as pd
import numpy as np
from pathlib import Path
files = {
    "products.csv": "/home/user/workspace/1790268567916_products.csv",
    "users.csv": "/home/user/workspace/1790268568306_users.csv",
    "orders.csv": "/home/user/workspace/1790268568295_orders.csv",
    "order_items.csv": "/home/user/workspace/1790268567919_order_items.csv",
}
frames = {name: pd.read_csv(path) for name, path in files.items()}
for name, df in frames.items():
    print("SOURCE", name, "shape", df.shape, "columns", list(df.columns))
orders = frames["orders.csv"].copy()
items = frames["order_items.csv"].copy()
products = frames["products.csv"].copy()
users = frames["users.csv"].copy()
expected = {
    "orders.csv": ["order_id", "user_id", "order_date", "order_status", "total_amount"],
    "order_items.csv": ["order_item_id", "order_id", "product_id", "user_id", "quantity", "item_price", "item_total"],
    "products.csv": ["product_id", "category", "brand", "price", "rating"],
    "users.csv": ["user_id", "gender", "signup_date"],
}
for name, cols in expected.items():
    missing = [col for col in cols if col not in frames[name].columns]
    if missing:
        raise ValueError(f"Missing required columns in {name}: {missing}; found {list(frames[name].columns)}")
if orders["order_id"].duplicated().any():
    raise ValueError("order_id is not unique in orders; refusing an order-grain join that could inflate data")
if products["product_id"].duplicated().any():
    raise ValueError("product_id is not unique in products; refusing a product join that could inflate data")
if users["user_id"].duplicated().any():
    raise ValueError("user_id is not unique in users; refusing a customer join that could inflate data")
for col in ["order_date"]:
    orders[col + "_clean"] = pd.to_datetime(orders[col], errors="coerce")
for col in ["total_amount"]:
    orders[col + "_clean"] = pd.to_numeric(orders[col], errors="coerce")
for col in ["quantity", "item_price", "item_total"]:
    items[col + "_clean"] = pd.to_numeric(items[col], errors="coerce")
for col in ["price", "rating"]:
    products[col + "_clean"] = pd.to_numeric(products[col], errors="coerce")
users["signup_date_clean"] = pd.to_datetime(users["signup_date"], errors="coerce")
products_small = products[["product_id", "category", "brand", "rating_clean"]].copy()
item_product = items.merge(products_small, on="product_id", how="left", validate="many_to_one", indicator=True)
item_product["category_clean"] = item_product["category"].astype("string").str.strip().fillna("Missing")
item_product.loc[item_product["category_clean"].eq(""), "category_clean"] = "Missing"
item_product["brand_clean"] = item_product["brand"].astype("string").str.strip().fillna("Missing")
item_product.loc[item_product["brand_clean"].eq(""), "brand_clean"] = "Missing"
# Mode tie-break is alphabetical; dominance is by item-line count.
def dominant_value(values):
    counts = values.value_counts(dropna=False)
    if counts.empty:
        return "Missing"
    maximum = counts.max()
    return sorted([str(x) for x in counts[counts.eq(maximum).index]])[0]
item_rows = []
for order_id, group in item_product.groupby("order_id", sort=False, dropna=False):
    item_rows.append({
        "order_id": order_id,
        "item_count": int(len(group)),
        "units": float(group["quantity_clean"].sum(min_count=1)) if group["quantity_clean"].notna().any() else np.nan,
        "max_price": float(group["item_price_clean"].max()) if group["item_price_clean"].notna().any() else np.nan,
        "avg_product_rating": float(group["rating_clean"].mean()) if group["rating_clean"].notna().any() else np.nan,
        "dominant_category": dominant_value(group["category_clean"]),
        "dominant_brand": dominant_value(group["brand_clean"]),
        "unmatched_product_lines": int(group["_merge"].eq("left_only").sum()),
    })
item_agg = pd.DataFrame(item_rows)
order_level = orders[["order_id", "user_id", "order_status", "order_date_clean", "total_amount_clean"]].merge(item_agg, on="order_id", how="left", validate="one_to_one")
order_level = order_level.merge(users[["user_id", "gender", "signup_date_clean"]], on="user_id", how="left", validate="many_to_one", indicator="_user_merge")
order_level["returned"] = order_level["order_status"].eq("returned").astype(int)
order_level["return_status"] = np.where(order_level["returned"].eq(1), "Returned", "Not returned")
order_level["gender_clean"] = order_level["gender"].astype("string").str.strip().fillna("Missing")
order_level.loc[order_level["gender_clean"].eq(""), "gender_clean"] = "Missing"
order_level["order_month"] = order_level["order_date_clean"].dt.to_period("M").dt.to_timestamp()
order_level["day_of_week"] = order_level["order_date_clean"].dt.day_name()
order_level["tenure_days"] = (order_level["order_date_clean"] - order_level["signup_date_clean"]).dt.total_seconds() / 86400
order_level = order_level.sort_values(["user_id", "order_date_clean", "order_id"], kind="mergesort")
order_level["order_sequence"] = order_level.groupby("user_id", dropna=False).cumcount() + 1
order_level["order_sequence_group"] = np.select([order_level["order_sequence"].eq(1), order_level["order_sequence"].eq(2), order_level["order_sequence"].eq(3)], ["1st", "2nd", "3rd"], default="4+")
order_level["order_value_quintile"] = pd.qcut(order_level["total_amount_clean"].rank(method="first"), 5, labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"])
chart_cols = ["order_id", "user_id", "order_status", "return_status", "returned", "order_date_clean", "order_month", "total_amount_clean", "item_count", "units", "max_price", "avg_product_rating", "dominant_category", "dominant_brand", "gender_clean", "tenure_days", "order_sequence", "order_sequence_group", "order_value_quintile", "day_of_week"]
chart_data = order_level[chart_cols].copy()
for col in ["dominant_category", "dominant_brand", "gender_clean", "order_sequence_group", "order_value_quintile", "day_of_week"]:
    chart_data[col] = chart_data[col].astype("string").fillna("Missing")
chart_data["order_date_clean"] = chart_data["order_date_clean"].dt.strftime("%Y-%m-%dT%H:%M:%S")
chart_data["order_month"] = chart_data["order_month"].dt.strftime("%Y-%m-%d")
output_path = "/home/user/workspace/outputs/order_return_analysis_source.csv"
chart_data.to_csv(output_path, index=False)
print("PREPARED_SOURCE", output_path, "shape", chart_data.shape)
print("PREPARED_COLUMNS", list(chart_data.columns))
print("DATE_RANGE", order_level["order_date_clean"].min(), order_level["order_date_clean"].max())
print("RETURN_COUNTS", order_level["return_status"].value_counts(dropna=False).to_dict())
print("MERGE_AUDIT", {"unmatched_user_orders": int(order_level["_user_merge"].eq("left_only").sum()), "orders_without_items": int(order_level["item_count"].isna().sum()), "unmatched_product_lines": int(item_agg["unmatched_product_lines"].sum())})
print("SAMPLE", chart_data.head(3).to_dict("records"))
