"""
Corrected product-category return-rate table for charting.

Source: generated and run in Better Analyst (app.betteranalyst.com), session
"E-Commerce Returns Analysis", 2026-09-24. Exported verbatim; only this header was added.
Input paths point to Better Analyst's sandbox: /home/user/workspace/1790268567916_products.csv, /home/user/workspace/1790268567919_order_items.csv, /home/user/workspace/1790268568295_orders.csv, /home/user/workspace/outputs/return_category_chart_source.csv.
Change them to your local data folder before running.
"""

import pandas as pd
from pathlib import Path
orders_path = "/home/user/workspace/1790268568295_orders.csv"
items_path = "/home/user/workspace/1790268567919_order_items.csv"
products_path = "/home/user/workspace/1790268567916_products.csv"
orders = pd.read_csv(orders_path)
items = pd.read_csv(items_path)
products = pd.read_csv(products_path)
print("SOURCE_CHECK", {"orders": [orders_path, len(orders), list(orders.columns)], "items": [items_path, len(items), list(items.columns)], "products": [products_path, len(products), list(products.columns)]})
for frame, column, label in [(orders, "order_id", "orders"), (products, "product_id", "products")]:
    if frame[column].duplicated().any():
        raise ValueError(f"Duplicate {column} in {label}; cannot safely aggregate category per order")
joined = items[["order_id", "product_id"]].merge(products[["product_id", "category"]], on="product_id", how="left", validate="many_to_one")
joined["category_clean"] = joined["category"].astype("string").str.strip().fillna("Missing")
joined.loc[joined["category_clean"].eq(""), "category_clean"] = "Missing"
counts = joined.groupby(["order_id", "category_clean"], dropna=False).size().reset_index(name="item_lines")
counts = counts.sort_values(["order_id", "item_lines", "category_clean"], ascending=[True, False, True], kind="mergesort")
dominant = counts.drop_duplicates("order_id", keep="first")[["order_id", "category_clean"]].rename(columns={"category_clean": "dominant_category"})
chart_data = orders[["order_id", "order_status"]].merge(dominant, on="order_id", how="left", validate="one_to_one")
chart_data["dominant_category"] = chart_data["dominant_category"].astype("string").fillna("Missing")
chart_data["returned"] = chart_data["order_status"].eq("returned").astype(int)
chart_data["order_id"] = chart_data["order_id"].astype("string")
print("CATEGORY_VALIDATION", {"distinct": int(chart_data["dominant_category"].nunique()), "frequency": chart_data["dominant_category"].value_counts().to_dict(), "rows": len(chart_data)})
output_path = "/home/user/workspace/outputs/return_category_chart_source.csv"
chart_data[["order_id", "dominant_category", "returned"]].to_csv(output_path, index=False)
print("PREPARED_SOURCE", output_path, "shape", chart_data.shape, "sample", chart_data[["order_id", "dominant_category", "returned"]].head(5).to_dict("records"))
