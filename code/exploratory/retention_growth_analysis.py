"""
Earlier project direction (customer retention and revenue growth) before the group refocused on returns.

Source: generated and run in Better Analyst (app.betteranalyst.com), session
"Data Science Project Plan and Report", 2026-08-31. Exported verbatim; only this header was added.
Input paths point to Better Analyst's sandbox: /home/user/workspace/1788186802025_events.csv, /home/user/workspace/1788186802982_order_items.csv, /home/user/workspace/1788186802996_orders.csv, /home/user/workspace/1788186803017_products.csv, /home/user/workspace/1788186806776_reviews.csv, /home/user/workspace/1788186806813_users.csv.
Change them to your local data folder before running.
"""

import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import json, math, importlib, subprocess, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, roc_auc_score, f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeCV, LassoCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBClassifier

def ensure_packages(*pkgs):
    known_pip_names = {'sklearn':'scikit-learn','PIL':'Pillow','cv2':'opencv-python','yaml':'PyYAML'}
    for import_name in pkgs:
        pip_name = known_pip_names.get(import_name, import_name)
        try:
            importlib.import_module(import_name)
        except ImportError:
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', pip_name, '-q'], capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise ImportError(f'Could not install {pip_name}: {result.stderr[-500:]}')
            importlib.invalidate_caches()
            importlib.import_module(import_name)
ensure_packages('statsmodels')
from statsmodels.tsa.holtwinters import ExponentialSmoothing

OUT = Path('/home/user/workspace/outputs')
OUT.mkdir(parents=True, exist_ok=True)
paths = {
    'products':'/home/user/workspace/1788186803017_products.csv',
    'order_items':'/home/user/workspace/1788186802982_order_items.csv',
    'events':'/home/user/workspace/1788186802025_events.csv',
    'orders':'/home/user/workspace/1788186802996_orders.csv',
    'reviews':'/home/user/workspace/1788186806776_reviews.csv',
    'users':'/home/user/workspace/1788186806813_users.csv'
}
for path in paths.values():
    if not Path(path).is_file():
        raise FileNotFoundError(path)

# 1) Reload and validate the original sources; prior generated files are not used.
products = pd.read_csv(paths['products'])
order_items = pd.read_csv(paths['order_items'])
events = pd.read_csv(paths['events'])
orders = pd.read_csv(paths['orders'])
reviews = pd.read_csv(paths['reviews'])
users = pd.read_csv(paths['users'])
required = {
    'products':['product_id','product_name','category','brand','price','rating'],
    'order_items':['order_item_id','order_id','product_id','user_id','quantity','item_price','item_total'],
    'events':['event_id','user_id','product_id','event_type','event_timestamp'],
    'orders':['order_id','user_id','order_date','order_status','total_amount'],
    'reviews':['review_id','order_id','product_id','user_id','rating','review_text','review_date'],
    'users':['user_id','name','email','gender','city','signup_date']
}
frames = {'products':products,'order_items':order_items,'events':events,'orders':orders,'reviews':reviews,'users':users}
for name, cols in required.items():
    missing = [c for c in cols if c not in frames[name].columns]
    if missing:
        raise ValueError(f'{name} missing required columns: {missing}; available={list(frames[name].columns)}')

source_audit = []
for name, df in frames.items():
    before = len(df)
    exact_dups = int(df.duplicated().sum())
    df.drop_duplicates(inplace=True)
    key = required[name][0]
    duplicate_keys = int(df[key].duplicated().sum())
    source_audit.append({'table':name,'source_rows':before,'exact_duplicate_rows':exact_dups,'rows_after_exact_dedup':len(df),'duplicate_primary_keys_after_dedup':duplicate_keys})

for col in ['product_id','product_name','category','brand']:
    products[col] = products[col].astype('string').str.strip()
for col in ['order_item_id','order_id','product_id','user_id']:
    order_items[col] = order_items[col].astype('string').str.strip()
for col in ['event_id','user_id','product_id','event_type']:
    events[col] = events[col].astype('string').str.strip()
for col in ['order_id','user_id','order_status']:
    orders[col] = orders[col].astype('string').str.strip()
for col in ['review_id','order_id','product_id','user_id','review_text']:
    reviews[col] = reviews[col].astype('string').str.strip()
for col in ['user_id','gender']:
    users[col] = users[col].astype('string').str.strip()
products['price_clean'] = pd.to_numeric(products['price'], errors='coerce')
products['rating_clean'] = pd.to_numeric(products['rating'], errors='coerce')
products['price_log1p'] = np.log1p(products['price_clean'].clip(lower=0))
order_items['quantity_clean'] = pd.to_numeric(order_items['quantity'], errors='coerce')
order_items['item_price_clean'] = pd.to_numeric(order_items['item_price'], errors='coerce')
order_items['item_total_clean'] = pd.to_numeric(order_items['item_total'], errors='coerce')
orders['total_amount_clean'] = pd.to_numeric(orders['total_amount'], errors='coerce')
orders['total_amount_log1p'] = np.log1p(orders['total_amount_clean'].clip(lower=0))
orders['order_date_clean'] = pd.to_datetime(orders['order_date'], errors='coerce')
events['event_timestamp_clean'] = pd.to_datetime(events['event_timestamp'], errors='coerce')
reviews['rating_clean'] = pd.to_numeric(reviews['rating'], errors='coerce')
reviews['review_date_clean'] = pd.to_datetime(reviews['review_date'], errors='coerce')
users['signup_date_clean'] = pd.to_datetime(users['signup_date'], errors='coerce')
coercion_audit = {
    'products_price_invalid':int(products['price_clean'].isna().sum()),
    'orders_date_invalid':int(orders['order_date_clean'].isna().sum()),
    'orders_amount_invalid':int(orders['total_amount_clean'].isna().sum()),
    'order_items_quantity_invalid':int(order_items['quantity_clean'].isna().sum()),
    'order_items_price_invalid':int(order_items['item_price_clean'].isna().sum()),
    'order_items_total_invalid':int(order_items['item_total_clean'].isna().sum()),
    'events_timestamp_invalid':int(events['event_timestamp_clean'].isna().sum()),
    'reviews_date_invalid':int(reviews['review_date_clean'].isna().sum()),
    'users_signup_invalid':int(users['signup_date_clean'].isna().sum())
}
orders = orders[orders['order_id'].notna() & orders['user_id'].notna() & orders['order_date_clean'].notna() & orders['total_amount_clean'].notna()].copy()
order_items = order_items[order_items['order_item_id'].notna() & order_items['order_id'].notna() & order_items['product_id'].notna() & order_items['user_id'].notna() & order_items['quantity_clean'].notna() & order_items['item_price_clean'].notna() & order_items['item_total_clean'].notna()].copy()
products = products[products['product_id'].notna() & products['category'].notna() & products['brand'].notna() & products['price_clean'].notna()].copy()
events = events[events['user_id'].notna() & events['event_timestamp_clean'].notna() & events['event_type'].notna()].copy()
reviews = reviews[reviews['user_id'].notna() & reviews['review_date_clean'].notna() & reviews['rating_clean'].notna()].copy()
users = users[users['user_id'].notna() & users['signup_date_clean'].notna()].copy()

# 2) Build clean dimensions and transaction-level enriched tables.
product_dim_cols = ['product_id','product_name','category','brand','price_clean','rating_clean','price_log1p']
product_dim = products[product_dim_cols].drop_duplicates('product_id')
item_enriched = order_items.merge(product_dim, on='product_id', how='left', validate='many_to_one')
item_enriched['catalog_price_ratio'] = item_enriched['item_price_clean'] / item_enriched['price_clean'].replace(0, np.nan)
item_enriched['order_date_clean'] = item_enriched['order_id'].map(orders.set_index('order_id')['order_date_clean'])
item_enriched['item_total_reconciled'] = item_enriched['quantity_clean'] * item_enriched['item_price_clean']
item_enriched['item_total_delta'] = item_enriched['item_total_clean'] - item_enriched['item_total_reconciled']
order_rollup = item_enriched.groupby('order_id', dropna=False).agg(
    item_line_count=('order_item_id','nunique'), total_quantity=('quantity_clean','sum'), item_total_sum=('item_total_clean','sum'),
    unique_products=('product_id','nunique'), category_breadth=('category','nunique'), brand_breadth=('brand','nunique'),
    max_item_price=('item_price_clean','max'), mean_catalog_price_ratio=('catalog_price_ratio','mean'), max_catalog_price_ratio=('catalog_price_ratio','max')
).reset_index()
orders_enriched = orders.merge(order_rollup, on='order_id', how='left', validate='one_to_one')
orders_enriched['item_total_gap'] = orders_enriched['total_amount_clean'] - orders_enriched['item_total_sum']
orders_enriched['item_total_gap_abs'] = orders_enriched['item_total_gap'].abs()
order_item_match = int(item_enriched['category'].notna().sum())
order_item_unmatched = int(item_enriched['category'].isna().sum())
reconciliation = {
    'orders_with_item_rollup':int(orders_enriched['item_line_count'].notna().sum()),
    'orders_with_item_total_gap_over_1_cent':int((orders_enriched['item_total_gap_abs']>0.01).sum()),
    'item_total_gap_rate':float((orders_enriched['item_total_gap_abs']>0.01).mean()),
    'order_items_matching_product_dimension':order_item_match,
    'order_items_unmatched_product_dimension':order_item_unmatched
}
users_clean = users[['user_id','gender','signup_date_clean']].copy()
products_ready = products[product_dim_cols].copy()
orders_ready = orders_enriched.copy()
items_ready = item_enriched.copy()
users_clean.to_csv(OUT/'users_clean.csv', index=False)
products_ready.to_csv(OUT/'product_analysis_ready.csv', index=False)
orders_ready.to_csv(OUT/'order_analysis_ready.csv', index=False)
items_ready.to_csv(OUT/'order_item_analysis_ready.csv', index=False)
events.to_csv(OUT/'event_analysis_ready.csv', index=False)
reviews.to_csv(OUT/'review_analysis_ready.csv', index=False)

# 3) Define a leakage-safe monthly snapshot panel: history before snapshot, outcome in following 90 days.
max_day = orders['order_date_clean'].max().normalize()
min_day = orders['order_date_clean'].min().normalize()
final_snapshot = max_day - pd.Timedelta(days=90)
raw_snapshot_dates = list(pd.date_range(min_day + pd.Timedelta(days=90), final_snapshot, freq='30D'))
if final_snapshot not in raw_snapshot_dates:
    raw_snapshot_dates.append(final_snapshot)
snapshot_dates = sorted(set([d.normalize() for d in raw_snapshot_dates]))
feature_cols = ['recency_days','order_count','total_spend_log1p','avg_order_value_log1p','total_quantity','unique_products','category_breadth','brand_breadth','view_count','cart_count','wishlist_count','review_count','avg_review_rating','negative_review_share','tenure_days']
panel_parts = []
for snap in snapshot_dates:
    hist_o = orders[orders['order_date_clean'] < snap].copy()
    future_o = orders[(orders['order_date_clean'] >= snap) & (orders['order_date_clean'] < snap + pd.Timedelta(days=90))].copy()
    hist_i = item_enriched[item_enriched['order_date_clean'] < snap].copy()
    hist_e = events[(events['event_timestamp_clean'] < snap) & (events['event_type'].str.lower().isin(['view','cart','wishlist']))].copy()
    hist_r = reviews[reviews['review_date_clean'] < snap].copy()
    base = users_clean[users_clean['signup_date_clean'] <= snap].copy()
    of = hist_o.groupby('user_id').agg(last_order_date=('order_date_clean','max'), order_count=('order_id','nunique'), total_spend=('total_amount_clean','sum'), avg_order_value=('total_amount_clean','mean')).reset_index()
    of['recency_days'] = (snap - of['last_order_date']).dt.days.clip(lower=0)
    of['total_spend_log1p'] = np.log1p(of['total_spend'].clip(lower=0))
    of['avg_order_value_log1p'] = np.log1p(of['avg_order_value'].clip(lower=0))
    itf = hist_i.groupby('user_id').agg(total_quantity=('quantity_clean','sum'), unique_products=('product_id','nunique'), category_breadth=('category','nunique'), brand_breadth=('brand','nunique')).reset_index()
    if hist_e.empty:
        ef = pd.DataFrame(columns=['user_id','view_count','cart_count','wishlist_count'])
    else:
        ef = hist_e.pivot_table(index='user_id', columns='event_type', values='event_timestamp_clean', aggfunc='count', fill_value=0).reset_index()
        for ev_col in ['view','cart','wishlist']:
            if ev_col not in ef.columns:
                ef[ev_col] = 0
        ef = ef.rename(columns={'view':'view_count','cart':'cart_count','wishlist':'wishlist_count'})[['user_id','view_count','cart_count','wishlist_count']]
    if hist_r.empty:
        rf = pd.DataFrame(columns=['user_id','review_count','avg_review_rating','negative_review_share'])
    else:
        hist_r['is_negative'] = (hist_r['rating_clean'] <= 2).astype(int)
        rf = hist_r.groupby('user_id').agg(review_count=('review_id','nunique'), avg_review_rating=('rating_clean','mean'), negative_review_share=('is_negative','mean')).reset_index()
    panel = base.merge(of, on='user_id', how='inner', validate='one_to_one').merge(itf, on='user_id', how='left', validate='one_to_one').merge(ef, on='user_id', how='left', validate='one_to_one').merge(rf, on='user_id', how='left', validate='one_to_one')
    panel['tenure_days'] = (snap - panel['signup_date_clean']).dt.days.clip(lower=0)
    future_f = future_o.groupby('user_id').agg(future_order_count=('order_id','nunique'), future_spend=('total_amount_clean','sum')).reset_index()
    panel = panel.merge(future_f, on='user_id', how='left', validate='one_to_one')
    for c in ['total_quantity','unique_products','category_breadth','brand_breadth','view_count','cart_count','wishlist_count','review_count','negative_review_share','future_order_count','future_spend']:
        panel[c] = pd.to_numeric(panel[c], errors='coerce').fillna(0)
    panel['avg_review_rating'] = pd.to_numeric(panel['avg_review_rating'], errors='coerce').fillna(0)
    panel['churn'] = (panel['future_order_count'].eq(0)).astype(int)
    panel['snapshot_date'] = snap
    panel_parts.append(panel)
panel = pd.concat(panel_parts, ignore_index=True)
print('AUDIT: snapshot_panel_shape', panel.shape)
print('AUDIT: snapshot_dates', [d.strftime('%Y-%m-%d') for d in snapshot_dates])
print('AUDIT: panel_columns', list(panel.columns))
panel.to_csv(OUT/'customer_snapshot_panel.csv', index=False)

# 4) Final snapshot features, descriptive landscape, and operational RFM segments.
current = panel[panel['snapshot_date'].eq(final_snapshot)].copy()
current['risk_score_rule'] = 0.5*current['recency_days'].rank(pct=True) + 0.25*(1-current['order_count'].rank(pct=True)) + 0.25*(1-(current['view_count']+current['cart_count']+current['wishlist_count']).rank(pct=True))
current['cluster_input_spend'] = current['total_spend_log1p']
cluster_cols = ['recency_days','order_count','cluster_input_spend']
cluster_x = StandardScaler().fit_transform(current[cluster_cols].astype(float))
cluster_sils = []
for k in range(2,7):
    labels_k = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(cluster_x)
    cluster_sils.append({'k':k,'silhouette':float(silhouette_score(cluster_x, labels_k))})
selected_k = max(cluster_sils, key=lambda d:d['silhouette'])['k']
km = KMeans(n_clusters=selected_k, n_init=20, random_state=42).fit(cluster_x)
current['cluster'] = km.labels_
current['segment'] = 'Core'
current.loc[current['order_count'].eq(1), 'segment'] = 'One-time'
current.loc[current['recency_days'].gt(180), 'segment'] = 'Lost'
current.loc[current['recency_days'].gt(90) & current['order_count'].ge(2), 'segment'] = 'At-risk'
current.loc[current['recency_days'].le(30) & current['order_count'].ge(3), 'segment'] = 'Champions'
cluster_profiles = current.groupby('cluster')[['recency_days','order_count','total_spend','churn']].mean().reset_index()
hier = AgglomerativeClustering(n_clusters=selected_k).fit_predict(cluster_x)
hier_sil = float(silhouette_score(cluster_x, hier)) if len(set(hier))>1 else None
db = DBSCAN(eps=0.9, min_samples=15).fit_predict(cluster_x)
db_non_noise = db[db>=0]
db_sil = float(silhouette_score(cluster_x[db>=0], db_non_noise)) if len(set(db_non_noise))>1 and len(db_non_noise)>2 else None
segment_summary = current.groupby('segment').agg(customers=('user_id','nunique'), historical_spend=('total_spend','sum'), avg_historical_spend=('total_spend','mean'), observed_churn_rate=('churn','mean'), avg_recency_days=('recency_days','mean'), avg_order_count=('order_count','mean')).reset_index()
segment_summary.to_csv(OUT/'segment_summary.csv', index=False)
current.to_csv(OUT/'customer_analysis_ready_base.csv', index=False)

# 5) Category/brand landscape and cohort retention.
all_items = item_enriched[item_enriched['order_id'].isin(set(orders['order_id']))].copy()
category_revenue = all_items.groupby('category', dropna=False)['item_total_clean'].sum().sort_values(ascending=False).reset_index(name='revenue')
category_revenue['share'] = category_revenue['revenue'] / category_revenue['revenue'].sum()
brand_revenue = all_items.groupby('brand', dropna=False)['item_total_clean'].sum().sort_values(ascending=False).reset_index(name='revenue')
brand_revenue['share'] = brand_revenue['revenue'] / brand_revenue['revenue'].sum()
category_revenue.to_csv(OUT/'revenue_by_category.csv', index=False)
brand_revenue.to_csv(OUT/'revenue_by_brand.csv', index=False)
engagement_counts = events[events['event_type'].str.lower().isin(['view','cart','wishlist'])].groupby('event_type')['user_id'].nunique().reset_index(name='unique_users')
engagement_counts = pd.concat([engagement_counts, pd.DataFrame([{'event_type':'ordered_user','unique_users':int(orders['user_id'].nunique())}])], ignore_index=True)
engagement_counts.to_csv(OUT/'engagement_reach.csv', index=False)
user_cohort = users_clean[users_clean['signup_date_clean'] <= max_day].copy()
user_cohort['cohort_month'] = user_cohort['signup_date_clean'].dt.to_period('M').astype(str)
cohort_sizes = user_cohort.groupby('cohort_month')['user_id'].nunique().rename('cohort_size').reset_index()
cohort_orders = orders[['user_id','order_date_clean']].merge(user_cohort[['user_id','signup_date_clean','cohort_month']], on='user_id', how='inner', validate='many_to_one')
cohort_orders['months_since_signup'] = ((cohort_orders['order_date_clean'] - cohort_orders['signup_date_clean']).dt.days // 30).clip(lower=0)
cohort_orders = cohort_orders[cohort_orders['months_since_signup'].between(0,12)]
cohort_active = cohort_orders.groupby(['cohort_month','months_since_signup'])['user_id'].nunique().reset_index(name='active_users')
cohort_retention = cohort_active.merge(cohort_sizes, on='cohort_month', how='left', validate='many_to_one')
cohort_retention['retention'] = cohort_retention['active_users'] / cohort_retention['cohort_size']
cohort_retention.to_csv(OUT/'cohort_retention.csv', index=False)

# 6) Churn classification: one leakage-safe temporal holdout and imbalance-aware alternatives.
unique_snaps = sorted(panel['snapshot_date'].drop_duplicates().tolist())
split_idx = max(1, int(math.floor(len(unique_snaps)*0.8)))
train_dates = unique_snaps[:split_idx]
test_dates = unique_snaps[split_idx:]
train_panel = panel[panel['snapshot_date'].isin(train_dates)].copy()
test_panel = panel[panel['snapshot_date'].isin(test_dates)].copy()
X_train = train_panel[feature_cols].astype(float).fillna(0)
X_test = test_panel[feature_cols].astype(float).fillna(0)
y_train = train_panel['churn'].astype(int)
y_test = test_panel['churn'].astype(int)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
class_counts = y_train.value_counts().to_dict()
scale_pos = float(class_counts.get(0,1) / max(class_counts.get(1,1),1))
rng = np.random.default_rng(42)
def balance_rows(X, y):
    y_arr = np.asarray(y)
    idx0 = np.where(y_arr==0)[0]
    idx1 = np.where(y_arr==1)[0]
    if len(idx0)==0 or len(idx1)==0:
        return X, y_arr
    target = max(len(idx0), len(idx1))
    sel0 = rng.choice(idx0, size=target, replace=len(idx0)<target)
    sel1 = rng.choice(idx1, size=target, replace=len(idx1)<target)
    sel = np.concatenate([sel0, sel1])
    rng.shuffle(sel)
    return X[sel], y_arr[sel]
models = {
    'Logistic L2': (LogisticRegression(max_iter=1500, C=1.0, penalty='l2'), X_train_s, y_train),
    'Logistic L1 balanced': (LogisticRegression(max_iter=1500, C=1.0, penalty='l1', solver='liblinear', class_weight='balanced'), X_train_s, y_train),
    'kNN': (KNeighborsClassifier(n_neighbors=25, weights='distance'), X_train_s, y_train),
    'Decision tree balanced': (DecisionTreeClassifier(max_depth=5, min_samples_leaf=25, class_weight='balanced', random_state=42), X_train, y_train),
    'Random forest balanced': (RandomForestClassifier(n_estimators=250, max_depth=8, min_samples_leaf=10, class_weight='balanced_subsample', random_state=42, n_jobs=-1), X_train, y_train),
    'XGBoost balanced': (XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0, scale_pos_weight=scale_pos, eval_metric='logloss', random_state=42, n_jobs=2), X_train, y_train),
    'MLP balanced': (MLPClassifier(hidden_layer_sizes=(32,16), alpha=0.001, learning_rate_init=0.001, max_iter=400, early_stopping=True, random_state=42), balance_rows(X_train_s, y_train)[0], balance_rows(X_train_s, y_train)[1])
}
model_rows = []
trained_models = {}
for name, (model, xtr, ytr) in models.items():
    model.fit(xtr, ytr)
    trained_models[name] = model
    model_input = X_test_s if name in ['Logistic L2','Logistic L1 balanced','kNN','MLP balanced'] else X_test
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(model_input)[:,1]
    else:
        decision = model.decision_function(model_input)
        proba = 1/(1+np.exp(-decision))
    pred = (proba >= 0.5).astype(int)
    auc = float(roc_auc_score(y_test, proba)) if y_test.nunique()>1 else float('nan')
    model_rows.append({'model':name,'roc_auc':auc,'f1_at_0_5':float(f1_score(y_test,pred,zero_division=0)),'precision_at_0_5':float(precision_score(y_test,pred,zero_division=0)),'recall_at_0_5':float(recall_score(y_test,pred,zero_division=0)),'train_rows':len(ytr),'test_rows':len(y_test)})
model_results = pd.DataFrame(model_rows).sort_values('roc_auc', ascending=False)
model_results.to_csv(OUT/'classification_model_results.csv', index=False)
best_model_name = str(model_results.iloc[0]['model'])
best_model = trained_models[best_model_name]
best_model_input = scaler.transform(current[feature_cols].astype(float).fillna(0)) if best_model_name in ['Logistic L2','Logistic L1 balanced','kNN','MLP balanced'] else current[feature_cols].astype(float).fillna(0)
current['experimental_churn_probability'] = best_model.predict_proba(best_model_input)[:,1] if hasattr(best_model,'predict_proba') else 1/(1+np.exp(-best_model.decision_function(best_model_input)))
logit = trained_models['Logistic L2']
logit_coefficients = pd.DataFrame({'feature':feature_cols,'coefficient':logit.coef_[0]}).sort_values('coefficient', ascending=False)
logit_coefficients.to_csv(OUT/'logistic_driver_coefficients.csv', index=False)

# 7) Next-quarter customer value and product pricing.
value_train = panel[panel['snapshot_date'] < final_snapshot].copy()
value_test = panel[panel['snapshot_date'].isin(test_dates)].copy()
value_scaler = StandardScaler().fit(value_train[feature_cols].astype(float).fillna(0))
VX_train = value_scaler.transform(value_train[feature_cols].astype(float).fillna(0))
VX_test = value_scaler.transform(value_test[feature_cols].astype(float).fillna(0))
vy_train = np.log1p(value_train['future_spend'].astype(float).clip(lower=0))
vy_test_raw = value_test['future_spend'].astype(float).clip(lower=0).to_numpy()
value_models = {
    'OLS':LinearRegression(),
    'Ridge':RidgeCV(alphas=[0.1,1,10,100]),
    'Lasso':LassoCV(alphas=[0.0001,0.001,0.01,0.1,1.0], max_iter=5000, random_state=42)
}
value_rows=[]; fitted_value={}
for name, model in value_models.items():
    model.fit(VX_train, vy_train)
    fitted_value[name]=model
    pred=np.maximum(0,np.expm1(model.predict(VX_test)))
    value_rows.append({'model':name,'mae_raw_spend':float(mean_absolute_error(vy_test_raw,pred)),'rmse_raw_spend':float(mean_squared_error(vy_test_raw,pred)**0.5),'r2_raw_spend':float(r2_score(vy_test_raw,pred))})
value_results=pd.DataFrame(value_rows).sort_values('mae_raw_spend')
value_results.to_csv(OUT/'customer_value_model_results.csv', index=False)
primary_value=fitted_value['Ridge']
prior_panel = panel[panel['snapshot_date'] < final_snapshot].copy()
if prior_panel.empty:
    prior_panel = panel[panel['snapshot_date'].isin(train_dates)].copy()
primary_value.fit(value_scaler.transform(prior_panel[feature_cols].astype(float).fillna(0)), np.log1p(prior_panel['future_spend'].astype(float).clip(lower=0)))
current['predicted_next_quarter_spend'] = np.maximum(0, np.expm1(primary_value.predict(value_scaler.transform(current[feature_cols].astype(float).fillna(0)))))
value_coefficients=pd.DataFrame({'feature':feature_cols,'ridge_coefficient':primary_value.coef_}).sort_values('ridge_coefficient', ascending=False)
value_coefficients.to_csv(OUT/'ridge_value_coefficients.csv', index=False)
# Product price benchmark; this estimates price from catalogue attributes, not elasticity.
price_df = products[['price_clean','rating_clean','category','brand']].dropna().copy()
price_x = pd.get_dummies(price_df[['rating_clean','category','brand']], columns=['category','brand'], dtype=float)
price_y = price_df['price_clean'].astype(float).to_numpy()
price_rng = np.random.default_rng(42)
price_idx = price_rng.permutation(len(price_df))
price_cut = int(len(price_df)*0.8)
price_train_idx, price_test_idx = price_idx[:price_cut], price_idx[price_cut:]
price_models={'OLS':LinearRegression(),'Ridge':RidgeCV(alphas=[0.1,1,10,100]),'Lasso':LassoCV(alphas=[0.001,0.01,0.1,1.0],max_iter=5000,random_state=42)}
price_rows=[]
for name, model in price_models.items():
    model.fit(price_x.iloc[price_train_idx], price_y[price_train_idx])
    pp=np.maximum(0,model.predict(price_x.iloc[price_test_idx]))
    price_rows.append({'model':name,'mae_price':float(mean_absolute_error(price_y[price_test_idx],pp)),'rmse_price':float(mean_squared_error(price_y[price_test_idx],pp)**0.5),'r2_price':float(r2_score(price_y[price_test_idx],pp))})
price_results=pd.DataFrame(price_rows).sort_values('mae_price')
price_results.to_csv(OUT/'product_price_model_results.csv', index=False)

# 8) Association rules at category/brand level and content similarity recommendations.
def make_rules(baskets, level, min_support=0.005):
    baskets=[set(x) for x in baskets if len(set(x))>=2]
    n=len(baskets)
    min_count=max(20,int(math.ceil(n*min_support))) if n else 0
    freq={}
    for basket in baskets:
        for item in basket:
            freq[item]=freq.get(item,0)+1
    names=sorted(freq)
    rows=[]
    for i, a in enumerate(names):
        for b in names[i+1:]:
            both=sum((a in z and b in z) for z in baskets)
            if both>=min_count:
                for antecedent, consequent in [(a,b),(b,a)]:
                    confidence=both/max(freq[antecedent],1)
                    lift=confidence/(freq[consequent]/max(n,1))
                    rows.append({'rule_level':level,'antecedent':antecedent,'consequent':consequent,'support':both/max(n,1),'confidence':confidence,'lift':lift,'co_orders':both,'min_co_orders':min_count})
    cols=['rule_level','antecedent','consequent','support','confidence','lift','co_orders','min_co_orders']
    return pd.DataFrame(rows,columns=cols).sort_values(['lift','confidence'],ascending=False) if rows else pd.DataFrame(columns=cols)
category_baskets=all_items.groupby('order_id')['category'].apply(lambda s:set(s.dropna().astype(str))).tolist()
brand_baskets=all_items.groupby('order_id')['brand'].apply(lambda s:set(s.dropna().astype(str))).tolist()
rules=pd.concat([make_rules(category_baskets,'category'),make_rules(brand_baskets,'brand')], ignore_index=True)
rules.to_csv(OUT/'association_rules_category_brand.csv', index=False)
# Product-level sparsity diagnostic; no product rules are forced into the action plan.
product_baskets=all_items.groupby('order_id')['product_id'].apply(lambda s:set(s.dropna().astype(str))).tolist()
product_rule_diag=make_rules(product_baskets,'product')
product_diag={'product_baskets':len(product_baskets),'product_unique_in_baskets':len(set(x for z in product_baskets for x in z)),'product_rules_at_0_5pct':int(len(product_rule_diag))}
vectorizer=TfidfVectorizer(ngram_range=(1,2), min_df=1)
products['content_text']=(products['category'].astype(str)+' '+products['brand'].astype(str)+' '+products['product_name'].astype(str)).str.lower()
tfidf=vectorizer.fit_transform(products['content_text'])
sim=cosine_similarity(tfidf)
product_index={pid:i for i,pid in enumerate(products['product_id'])}
hist_item_for_rec=item_enriched[item_enriched['order_date_clean'] < final_snapshot].sort_values(['user_id','order_date_clean'])
last_products=hist_item_for_rec.drop_duplicates('user_id',keep='last')[['user_id','product_id']]
user_bought=hist_item_for_rec.groupby('user_id')['product_id'].apply(lambda s:set(s.dropna().astype(str))).to_dict()
rec_rows=[]
for rec in current[['user_id','segment']].to_dict('records'):
    uid=str(rec['user_id'])
    seed=last_products[last_products['user_id'].eq(uid)]['product_id']
    if seed.empty or str(seed.iloc[0]) not in product_index:
        continue
    seed_pid=str(seed.iloc[0]); seed_idx=product_index[seed_pid]
    purchased=user_bought.get(uid,set())
    candidate_idx=np.argsort(-sim[seed_idx])
    rank=0
    for idx in candidate_idx:
        pid=str(products.iloc[int(idx)]['product_id'])
        if pid==seed_pid or pid in purchased:
            continue
        rank += 1
        rec_rows.append({'user_id':uid,'segment':rec['segment'],'seed_product_id':seed_pid,'recommended_product_id':pid,'similarity':float(sim[seed_idx,int(idx)]),'recommendation_rank':rank})
        if rank>=3:
            break
content_recs=pd.DataFrame(rec_rows)
content_recs.to_csv(OUT/'content_recommendations.csv', index=False)

# 9) Daily revenue forecast: daily series, 30-day holdout, weekly seasonal naive, Holt-Winters, simple RNN candidate.
daily=orders.groupby(orders['order_date_clean'].dt.normalize())['total_amount_clean'].sum().sort_index()
daily=daily.reindex(pd.date_range(daily.index.min(),daily.index.max(),freq='D'),fill_value=0.0)
holdout_days=30
forecast_holdout_start=daily.index.max()-pd.Timedelta(days=holdout_days-1)
train_series=daily[daily.index<forecast_holdout_start].astype(float)
test_series=daily[daily.index>=forecast_holdout_start].astype(float)
def seasonal_naive(train_values, horizon, season=7):
    history=list(np.asarray(train_values,dtype=float))
    preds=[]
    for _ in range(horizon):
        pred=history[-season] if len(history)>=season else history[-1]
        preds.append(float(pred)); history.append(float(pred))
    return np.array(preds)
def rnn_forecast(train_values, horizon, seq_len=14, hidden=12, epochs=100, lr=0.004):
    y=np.asarray(train_values,dtype=float)
    mean=float(y.mean()); std=float(y.std()) if float(y.std())>1e-9 else 1.0
    z=(y-mean)/std
    if len(z)<=seq_len+2:
        return seasonal_naive(y,horizon,7)
    X=np.array([z[i-seq_len:i] for i in range(seq_len,len(z))])
    Y=np.array([z[i] for i in range(seq_len,len(z))])
    rng_local=np.random.default_rng(42)
    Wxh=rng_local.normal(0,0.08,size=(hidden,1)); Whh=rng_local.normal(0,0.08,size=(hidden,hidden)); Why=rng_local.normal(0,0.08,size=(1,hidden))
    bh=np.zeros(hidden); by=np.zeros(1)
    for _ in range(epochs):
        for x_seq, target in zip(X,Y):
            hs=[np.zeros(hidden)]
            for x_val in x_seq:
                hs.append(np.tanh(Wxh[:,0]*x_val + Whh.dot(hs[-1]) + bh))
            pred=float(Why.dot(hs[-1])+by[0]); dy=pred-target
            dWhy=dy*hs[-1][None,:]; dby=np.array([dy]); dWxh=np.zeros_like(Wxh); dWhh=np.zeros_like(Whh); dbh=np.zeros_like(bh); dh=Why.ravel()*dy
            for t in range(seq_len-1,-1,-1):
                h=hs[t+1]; hprev=hs[t]; da=dh*(1-h*h); dWxh[:,0]+=da*x_seq[t]; dWhh+=np.outer(da,hprev); dbh+=da; dh=Whh.T.dot(da)
            for grad in [dWxh,dWhh,dWhy,dbh,dby]:
                np.clip(grad,-1.0,1.0,out=grad)
            Wxh-=lr*dWxh; Whh-=lr*dWhh; Why-=lr*dWhy; bh-=lr*dbh; by-=lr*dby
    seq=list(z[-seq_len:]); preds=[]
    for _ in range(horizon):
        h=np.zeros(hidden)
        for x_val in seq:
            h=np.tanh(Wxh[:,0]*x_val + Whh.dot(h) + bh)
        pred=float(Why.dot(h)+by[0]); preds.append(pred*std+mean); seq=seq[1:]+[pred]
    return np.maximum(0,np.array(preds))
naive_pred=seasonal_naive(train_series.to_numpy(),len(test_series),7)
hw_model=ExponentialSmoothing(train_series, trend='add', seasonal='add', seasonal_periods=7, initialization_method='estimated').fit(optimized=True,use_brute=True)
hw_pred=np.maximum(0,np.asarray(hw_model.forecast(len(test_series)),dtype=float))
rnn_pred=rnn_forecast(train_series.to_numpy(),len(test_series))
forecast_results=pd.DataFrame([
    {'model':'7-day seasonal naive','mae':float(mean_absolute_error(test_series.to_numpy(),naive_pred)),'rmse':float(mean_squared_error(test_series.to_numpy(),naive_pred)**0.5)},
    {'model':'Holt-Winters weekly','mae':float(mean_absolute_error(test_series.to_numpy(),hw_pred)),'rmse':float(mean_squared_error(test_series.to_numpy(),hw_pred)**0.5)},
    {'model':'Simple RNN candidate','mae':float(mean_absolute_error(test_series.to_numpy(),rnn_pred)),'rmse':float(mean_squared_error(test_series.to_numpy(),rnn_pred)**0.5)}
]).sort_values('mae')
forecast_results.to_csv(OUT/'forecast_model_results.csv', index=False)
best_forecast_name=str(forecast_results.iloc[0]['model'])
if best_forecast_name=='7-day seasonal naive':
    future_pred=rnn_forecast(daily.to_numpy(),90) if False else seasonal_naive(daily.to_numpy(),90,7)
elif best_forecast_name=='Holt-Winters weekly':
    full_hw=ExponentialSmoothing(daily,trend='add',seasonal='add',seasonal_periods=7,initialization_method='estimated').fit(optimized=True,use_brute=True)
    future_pred=np.maximum(0,np.asarray(full_hw.forecast(90),dtype=float))
else:
    future_pred=rnn_forecast(daily.to_numpy(),90)
future_dates=pd.date_range(daily.index.max()+pd.Timedelta(days=1), periods=90, freq='D')
forecast_future=pd.DataFrame({'date':future_dates,'forecast_revenue':future_pred,'forecast_model':best_forecast_name})
forecast_future.to_csv(OUT/'next_quarter_revenue_forecast.csv', index=False)

# 10) Outlier and fraud triage: order-level anomaly queue, not confirmed fraud.
item_for_anomaly=item_enriched.copy()
cat_stats=item_for_anomaly.groupby('category')['item_price_clean'].agg(cat_median='median',cat_mad=lambda s:float(np.median(np.abs(s-np.median(s)))))
item_for_anomaly=item_for_anomaly.merge(cat_stats,on='category',how='left',validate='many_to_one')
item_for_anomaly['price_robust_z']=0.6745*(item_for_anomaly['item_price_clean']-item_for_anomaly['cat_median'])/item_for_anomaly['cat_mad'].replace(0,np.nan)
item_for_anomaly['price_robust_z']=item_for_anomaly['price_robust_z'].replace([np.inf,-np.inf],np.nan).fillna(0)
anom_rollup=item_for_anomaly.groupby('order_id').agg(total_amount=('item_total_clean','sum'),total_quantity=('quantity_clean','sum'),item_line_count=('order_item_id','nunique'),max_item_price=('item_price_clean','max'),max_price_robust_z=('price_robust_z',lambda s:float(np.max(np.abs(s))) if len(s) else 0),mean_price_robust_z=('price_robust_z',lambda s:float(np.mean(np.abs(s))) if len(s) else 0)).reset_index()
anom_orders=orders[['order_id','user_id','order_date_clean','total_amount_clean']].merge(anom_rollup,on='order_id',how='left',validate='one_to_one')
for col in ['total_amount','total_quantity','item_line_count','max_item_price','max_price_robust_z','mean_price_robust_z']:
    med=float(anom_orders[col].median()); mad=float(np.median(np.abs(anom_orders[col]-med)))
    anom_orders[col+'_robust_z']=0.6745*(anom_orders[col]-med)/(mad if mad>0 else 1.0)
robust_cols=['total_amount_robust_z','total_quantity_robust_z','item_line_count_robust_z','max_item_price_robust_z','max_price_robust_z']
anom_orders['robust_z_flag']=(anom_orders[robust_cols].abs().max(axis=1)>3.5).astype(int)
iforest_cols=['total_amount','total_quantity','item_line_count','max_item_price','max_price_robust_z','mean_price_robust_z']
anom_x=anom_orders[iforest_cols].replace([np.inf,-np.inf],np.nan).fillna(anom_orders[iforest_cols].median())
iforest=IsolationForest(n_estimators=300, contamination=0.02, random_state=42)
anom_orders['isolation_forest_flag']=(iforest.fit_predict(anom_x)==-1).astype(int)
anom_orders['fraud_review_flag']=((anom_orders['robust_z_flag']==1)|(anom_orders['isolation_forest_flag']==1)).astype(int)
anom_orders['flag_reason']=np.where((anom_orders['robust_z_flag']==1)&(anom_orders['isolation_forest_flag']==1),'both_methods',np.where(anom_orders['robust_z_flag']==1,'robust_z_only','isolation_forest_only'))
anom_orders.to_csv(OUT/'fraud_order_review_queue.csv',index=False)
user_flags=anom_orders.groupby('user_id').agg(flagged_orders=('fraud_review_flag','sum'),orders_reviewed=('order_id','nunique'),flagged_order_value=('total_amount','sum')).reset_index()
user_flags['flagged_order_rate']=user_flags['flagged_orders']/user_flags['orders_reviewed'].replace(0,np.nan)
user_flags.to_csv(OUT/'fraud_user_summary.csv',index=False)
status_rates=orders.assign(is_cancelled=orders['order_status'].str.lower().eq('cancelled').astype(int),is_returned=orders['order_status'].str.lower().eq('returned').astype(int)).groupby('user_id').agg(cancelled_rate=('is_cancelled','mean'),returned_rate=('is_returned','mean')).reset_index()
status_rates.to_csv(OUT/'status_rate_sensitivity_only.csv',index=False)

# 11) Synthesis: transparent action plan, value-tiered, with fraud review hold.
current=current.merge(user_flags[['user_id','flagged_orders','flagged_order_value']],on='user_id',how='left',validate='one_to_one')
current[['flagged_orders','flagged_order_value']]=current[['flagged_orders','flagged_order_value']].fillna(0)
current['value_rank']=current['predicted_next_quarter_spend'].rank(pct=True)
current['priority_score']=100*(0.55*current['value_rank']+0.45*current['risk_score_rule'])
current['risk_band']=np.select([current['segment'].isin(['At-risk','Lost']),current['segment'].isin(['One-time','Core'])],['High','Medium'],'Low')
current['recommended_action']=np.select([
    current['segment'].eq('At-risk') & current['predicted_next_quarter_spend'].ge(current['predicted_next_quarter_spend'].quantile(0.75)),
    current['segment'].eq('At-risk'),
    current['segment'].eq('Lost'),
    current['segment'].eq('One-time'),
    current['segment'].eq('Champions')
],[
    'Personalized category/brand bundle plus time-limited incentive',
    'Personalized reminder and low-cost incentive',
    'Win-back test with strict spend cap',
    'Second-purchase journey using content recommendations',
    'VIP protection and cross-sell; avoid blanket discounting'
],default='Cross-sell and category-expansion test')
current['retention_budget_cap_10pct']=0.10*current['predicted_next_quarter_spend']
current['fraud_review_status']=np.where(current['flagged_orders']>0,'Review flagged orders before crediting recovered revenue','No anomaly flag in order history')
plan_cols=['user_id','segment','risk_band','churn','experimental_churn_probability','recency_days','order_count','total_spend','predicted_next_quarter_spend','retention_budget_cap_10pct','priority_score','recommended_action','flagged_orders','flagged_order_value','fraud_review_status']
priority_plan=current[plan_cols].sort_values(['risk_band','priority_score'],ascending=[True,False])
priority_plan.to_csv(OUT/'prioritized_retention_action_plan.csv',index=False)
segment_action_summary=current.groupby('segment').agg(customers=('user_id','nunique'),historical_spend=('total_spend','sum'),predicted_next_quarter_spend=('predicted_next_quarter_spend','sum'),retention_budget_cap_10pct=('retention_budget_cap_10pct','sum'),observed_churn_rate=('churn','mean'),flagged_customers=('flagged_orders',lambda s:int((s>0).sum()))).reset_index().sort_values('predicted_next_quarter_spend',ascending=False)
segment_action_summary.to_csv(OUT/'segment_action_summary.csv',index=False)
# Attach final current data with all decision fields.
current.to_csv(OUT/'customer_analysis_ready.csv',index=False)

# 12) Infographics and report-supporting visuals.
plt.rcParams.update({'figure.dpi':120,'axes.titlesize':12,'axes.labelsize':10})
def money_fmt(x,pos): return f'${x/1e6:.1f}M' if abs(x)>=1e6 else f'${x/1e3:.0f}K'
# Revenue by category.
fig,ax=plt.subplots(figsize=(9,5)); top=category_revenue.head(10).sort_values('revenue'); ax.barh(top['category'].astype(str),top['revenue'],color='#2563eb'); ax.xaxis.set_major_formatter(money_fmt); ax.set_title('Revenue concentration by category'); ax.set_xlabel('Item revenue across valid orders'); ax.grid(axis='x',alpha=.2); fig.tight_layout(); fig.savefig(OUT/'infographic_revenue_by_category.png',bbox_inches='tight'); plt.close(fig)
# Segment value and risk.
seg_plot=segment_action_summary.sort_values('predicted_next_quarter_spend',ascending=True); fig,ax=plt.subplots(figsize=(9,5)); bars=ax.barh(seg_plot['segment'],seg_plot['predicted_next_quarter_spend'],color=['#94a3b8' if x!='At-risk' else '#dc2626' for x in seg_plot['segment']]); ax.xaxis.set_major_formatter(money_fmt); ax.set_title('Segment value at the decision snapshot'); ax.set_xlabel('Predicted next-quarter spend'); ax.grid(axis='x',alpha=.2); fig.tight_layout(); fig.savefig(OUT/'infographic_segment_value_risk.png',bbox_inches='tight'); plt.close(fig)
# Model comparison.
fig,ax=plt.subplots(figsize=(10,5)); mp=model_results.sort_values('roc_auc'); ax.barh(mp['model'],mp['roc_auc'],color='#0f766e'); ax.axvline(.5,color='#dc2626',linestyle='--',label='Random ranking'); ax.set_xlim(0,1); ax.set_xlabel('ROC-AUC on temporal holdout'); ax.set_title('Churn models: no reliable predictive winner'); ax.legend(); ax.grid(axis='x',alpha=.2); fig.tight_layout(); fig.savefig(OUT/'infographic_churn_model_comparison.png',bbox_inches='tight'); plt.close(fig)
# Forecast validation.
fig,ax=plt.subplots(figsize=(10,5)); dates=test_series.index; ax.plot(dates,test_series.values,label='Actual',color='#111827',linewidth=2); ax.plot(dates,naive_pred,label='7-day seasonal naive'); ax.plot(dates,hw_pred,label='Holt-Winters'); ax.plot(dates,rnn_pred,label='Simple RNN candidate'); ax.yaxis.set_major_formatter(money_fmt); ax.set_title('Daily revenue forecast validation: final 30-day holdout'); ax.set_ylabel('Daily gross order value'); ax.legend(); ax.grid(alpha=.2); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(OUT/'infographic_forecast_validation.png',bbox_inches='tight'); plt.close(fig)
# Fraud review queue.
fraud_counts=pd.DataFrame({'method':['Isolation Forest','Robust z-score','Union review queue'],'orders':[int(anom_orders['isolation_forest_flag'].sum()),int(anom_orders['robust_z_flag'].sum()),int(anom_orders['fraud_review_flag'].sum())],'value':[float(anom_orders.loc[anom_orders['isolation_forest_flag'].eq(1),'total_amount'].sum()),float(anom_orders.loc[anom_orders['robust_z_flag'].eq(1),'total_amount'].sum()),float(anom_orders.loc[anom_orders['fraud_review_flag'].eq(1),'total_amount'].sum())]}); fig,ax=plt.subplots(figsize=(8,5)); ax.bar(fraud_counts['method'],fraud_counts['orders'],color=['#f59e0b','#7c3aed','#dc2626']); ax.set_ylabel('Orders in review queue'); ax.set_title('Anomaly triage is an investigation queue, not a fraud verdict'); ax.tick_params(axis='x',rotation=15); ax.grid(axis='y',alpha=.2); fig.tight_layout(); fig.savefig(OUT/'infographic_fraud_triage.png',bbox_inches='tight'); plt.close(fig)
# Cohort retention heatmap.
if not cohort_retention.empty:
    piv=cohort_retention.pivot(index='cohort_month',columns='months_since_signup',values='retention').sort_index(); fig,ax=plt.subplots(figsize=(10,6)); im=ax.imshow(piv.fillna(np.nan),aspect='auto',cmap='Blues',vmin=0,vmax=max(1,float(np.nanmax(piv.values)))); ax.set_title('Cohort retention: share of users with an order by month since signup'); ax.set_xlabel('Months since signup'); ax.set_ylabel('Signup cohort'); ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([str(c) for c in piv.columns]); ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index); fig.colorbar(im,ax=ax,label='Retention'); fig.tight_layout(); fig.savefig(OUT/'infographic_cohort_retention.png',bbox_inches='tight'); plt.close(fig)
# Executive story board.
fig,axs=plt.subplots(2,2,figsize=(13,9));
axs[0,0].bar(['Active next 90d','Inactive next 90d'],[int((current['churn']==0).sum()),int(current['churn'].sum())],color=['#16a34a','#dc2626']); axs[0,0].set_title(f'Customer inactivity at decision snapshot ({current["churn"].mean():.1%} inactive)'); axs[0,0].grid(axis='y',alpha=.2)
cat_top=category_revenue.head(5).sort_values('revenue'); axs[0,1].barh(cat_top['category'].astype(str),cat_top['revenue'],color='#2563eb'); axs[0,1].xaxis.set_major_formatter(money_fmt); axs[0,1].set_title(f'Top 2 categories = {category_revenue.head(2)["share"].sum():.1%} of item revenue')
seg2=segment_action_summary.sort_values('predicted_next_quarter_spend',ascending=True); axs[1,0].barh(seg2['segment'],seg2['predicted_next_quarter_spend'],color='#0f766e'); axs[1,0].xaxis.set_major_formatter(money_fmt); axs[1,0].set_title('Predicted next-quarter spend by segment')
axs[1,1].bar(forecast_results['model'],forecast_results['mae'],color='#7c3aed'); axs[1,1].set_title(f'Forecast winner: {best_forecast_name}'); axs[1,1].set_ylabel('30-day holdout MAE'); axs[1,1].tick_params(axis='x',rotation=25); axs[1,1].yaxis.set_major_formatter(money_fmt); axs[1,1].grid(axis='y',alpha=.2)
fig.suptitle('E-commerce retention and revenue-growth decision story',fontsize=16,fontweight='bold'); fig.tight_layout(rect=[0,0,1,.96]); fig.savefig(OUT/'infographic_executive_story.png',bbox_inches='tight'); plt.close(fig)

# 13) Verified report markdown with definitions, evidence, and decisions.
valid_order_count=int(len(orders)); valid_order_value=float(orders['total_amount_clean'].sum()); buyer_count=int(current['user_id'].nunique()); churn_count=int(current['churn'].sum()); churn_rate=float(current['churn'].mean()); active_count=buyer_count-churn_count
top2_share=float(category_revenue.head(2)['share'].sum()); top2_names=', '.join(category_revenue.head(2)['category'].astype(str).tolist()); at_risk=current[current['segment'].eq('At-risk')]; at_risk_pred=float(at_risk['predicted_next_quarter_spend'].sum()); at_risk_cap=float(at_risk['retention_budget_cap_10pct'].sum()); flagged_count=int(anom_orders['fraud_review_flag'].sum()); flagged_value=float(anom_orders.loc[anom_orders['fraud_review_flag'].eq(1),'total_amount'].sum()); forecast_total=float(future_pred.sum()); forecast_daily=float(future_pred.mean())
report=[]
report.append('# E-Commerce Retention and Revenue Growth Business Report')
report.append('')
report.append('## Executive decision')
report.append(f'At the final decision snapshot ({final_snapshot.date()}), the valid order history contains {valid_order_count:,} orders and ${valid_order_value:,.0f} of gross order value. Among {buyer_count:,} customers with at least one order before the snapshot, {churn_count:,} ({churn_rate:.1%}) placed no order in the following 90 days, leaving {active_count:,} active buyers. The recommended decision is a value-tiered retention program: protect high-value at-risk buyers with personalized category/brand bundles and controlled incentives, use low-cost journeys for lower-value customers, and do not automate churn suppression until the predictive signal improves.')
report.append(f'The business is concentrated: {top2_names} together account for {top2_share:.1%} of item revenue. At-risk customers represent ${at_risk_pred:,.0f} of model-estimated next-quarter spend; a provisional 10% retention-spend cap would be ${at_risk_cap:,.0f}. This cap is a governance heuristic, not a measured profit optimum.')
report.append(f'For planning, the selected 90-day revenue forecast is {best_forecast_name}, implying ${forecast_total:,.0f} of future gross order value at about ${forecast_daily:,.0f} per day. Separately, {flagged_count:,} orders worth ${flagged_value:,.0f} enter an anomaly review queue; they are not confirmed fraud.')
report.append('')
report.append('## 1. Business question, scope, and definitions')
report.append('The analysis answers one integrated question: where is value and churn risk concentrated, which customers should be prioritized, what intervention should they receive, what revenue baseline should fund the plan, and which transactions require review before recovered revenue is credited?')
report.append('')
report.append('Definitions used:')
report.append(f'- **Decision snapshot:** {final_snapshot.date()}, 90 days before the last valid order date ({max_day.date()}).')
report.append('- **Historical buyer:** a user with at least one valid order before the snapshot and a signup date on or before the snapshot.')
report.append('- **Churn/inactivity outcome:** no order in the 90-day window beginning at the snapshot. This is an inactivity proxy, not a confirmed cancellation intent.')
report.append('- **Gross order value:** the sum of valid `total_amount` values. `order_status` was not used as a target or feature because the stated business scope excludes predicting it and the field was not validated as reliable transaction truth.')
report.append('- **Engagement:** counts of `view`, `cart`, and `wishlist` events before each snapshot. The `purchase` event tag was not used as transaction truth; orders and order items are the transaction sources.')
report.append('- **Next-quarter customer value:** model-estimated spend in the following 90 days, trained only on earlier snapshots and scored at the final snapshot.')
report.append('')
report.append('## 2. Data foundation and cleaning')
report.append(f'The six original tables were reloaded and validated: products ({len(products):,}), order_items ({len(order_items):,}), events ({len(events):,}), orders ({len(orders):,} valid after required-field filtering), reviews ({len(reviews):,}), and users ({len(users):,}). Exact duplicate rows were removed; duplicate primary-key counts were audited rather than silently collapsed.')
report.append('')
report.append('Cleaning decisions:')
report.append('- Dropped `name`, `email`, and `city` from the customer analysis table to avoid using personal identifiers and high-cardinality location noise.')
report.append('- Preserved raw numeric fields and added typed clean fields; `price` and `total_amount` received `log1p` versions for skewed modeling inputs.')
report.append('- Parsed all date/time fields with coercion checks and removed only rows missing fields required for the specific analysis.')
report.append('- Joined order items to products with a many-to-one key validation. Product coverage was {order_item_match/len(item_enriched):.1%}; {order_item_unmatched:,} item rows lacked a product-dimension match.')
report.append(f'- Order totals and item totals were reconciled. {reconciliation["orders_with_item_total_gap_over_1_cent"]:,} orders ({reconciliation["item_total_gap_rate"]:.1%}) differed by more than one cent, so order-level `total_amount` remains the source for gross order value while item totals are used for category/brand mix.')
report.append('')
report.append('The analysis-ready files created are `users_clean.csv`, `product_analysis_ready.csv`, `order_item_analysis_ready.csv`, `order_analysis_ready.csv`, `event_analysis_ready.csv`, `review_analysis_ready.csv`, `customer_snapshot_panel.csv`, and `customer_analysis_ready.csv`.')
report.append('')
report.append('## 3. Pillar 1 — Where value and risk sit today')
report.append('### Revenue concentration')
report.append(category_revenue.head(10).to_string(index=False))
report.append('')
report.append(f'The top two categories ({top2_names}) contribute {top2_share:.1%} of item revenue. This makes them both a retention priority and a concentration risk: protecting their repeat buyers matters, but growth should also test whether adjacent categories can diversify the base.')
report.append('')
report.append('### Customer segments')
report.append(segment_summary.sort_values('historical_spend',ascending=False).to_string(index=False))
report.append('')
report.append(f'Four-cluster RFM/K-Means was compared with hierarchical clustering and DBSCAN. The silhouette-selected K-Means k was {selected_k}; K-Means silhouette values by k were {cluster_sils}; hierarchical silhouette at k={selected_k} was {hier_sil}; DBSCAN silhouette among non-noise clusters was {db_sil}. The named operating segments are deterministic RFM rules applied to the validated customer snapshot, not arbitrary cluster names.')
report.append('')
report.append('### Visual evidence')
report.append('![Executive story](infographic_executive_story.png)')
report.append('![Revenue concentration](infographic_revenue_by_category.png)')
report.append('![Segment value and risk](infographic_segment_value_risk.png)')
report.append('')
report.append('## 4. Pillar 2 — Which customers are being lost, and why?')
report.append('A monthly snapshot panel was used so that every training row uses only behavior before its snapshot and every label uses the following 90 days. The final 20% of snapshot dates was held out chronologically; no random row split was used.')
report.append('')
report.append(model_results.to_string(index=False))
report.append('')
report.append('The imbalance-aware methods used class weighting or balanced resampling rather than raw accuracy. ROC-AUC and F1 are reported because accuracy would reward predicting the majority class. The model results do not show a reliable automated churn ranker if ROC-AUC remains near 0.50. Therefore, model probabilities are retained as experimental diagnostics, while the operational target list is based on transparent RFM/value rules.')
report.append('')
report.append('The interpretable logistic model points to the following directional drivers, but they are associations rather than causal effects:')
report.append(logit_coefficients.head(6).to_string(index=False))
report.append('')
report.append(logit_coefficients.tail(6).sort_values('coefficient').to_string(index=False))
report.append('')
report.append('### Decision')
report.append('Use recency, repeat-order history, historical value, category breadth, and engagement as a transparent targeting baseline. Do not automatically suppress or discount customers based on a near-random classifier. The key missing ingredient is intervention history: offers, exposures, redemptions, margin, and reactivation outcomes.')
report.append('')
report.append('## 5. Pillar 3 — What is each customer worth, and what should we charge?')
report.append(value_results.to_string(index=False))
report.append('')
report.append('Ridge is the primary customer-value model because correlated spend and order-history variables make unregularized coefficients unstable. Lasso is retained to identify a sparse set of candidate drivers. The target is next-90-day spend, not a fully observed lifetime value. The product-price model uses category, brand, and rating as a benchmark; it cannot estimate price elasticity because the data has no controlled price variation, margin, or demand response.')
report.append('')
report.append(price_results.to_string(index=False))
report.append('')
report.append('### Decision')
report.append('Use predicted next-quarter spend to size retention attention and set an initial governance cap, not to claim an optimal discount. Any charge or incentive change must be tested with margin and redemption measurement.')
report.append('')
report.append('## 6. Pillar 4 — What is the re-engagement lever?')
report.append(f'Category and brand association rules were computed on unique baskets with a 0.5% support floor and at least 20 co-orders. {len(rules):,} category/brand rules cleared the floor. Product-level rules were not forced into the decision because the basket contains {product_diag["product_unique_in_baskets"]:,} unique products and only {product_diag["product_rules_at_0_5pct"]:,} product rules clear the same floor.')
report.append('')
report.append(rules.head(15).to_string(index=False) if len(rules) else 'No category/brand rules cleared the support floor; use content similarity and controlled merchandising tests.')
report.append('')
report.append(f'The content recommender generated {len(content_recs):,} recommendation rows for historical buyers with a known last product. It uses product name, category, and brand TF-IDF similarity, excludes products already purchased by the customer, and returns three candidates per eligible customer. Similarity is a relevance heuristic, not incremental lift.')
report.append('')
report.append('### Decision')
report.append('Use the association rules for bundle and cross-sell experiments in the concentrated categories, and use content similarity for cold-start or sparse-history placements. Hold out a control group and measure incremental orders, margin, and 90-day reactivation.')
report.append('')
report.append('## 7. Pillar 5 — How much should be planned for?')
report.append(f'Daily gross order value was modeled from {len(daily):,} calendar days, with missing days filled as zero. The final {len(test_series):,}-day holdout compared a 7-day seasonal naive baseline, weekly Holt-Winters, and a simple recurrent neural candidate.')
report.append(forecast_results.to_string(index=False))
report.append('')
report.append('The selected model is the lowest-MAE holdout model, not the most complex model. The 90-day planning estimate is the selected model forecast and should be treated as a baseline scenario, not a guaranteed budget.')
report.append('')
report.append('![Forecast validation](infographic_forecast_validation.png)')
report.append('')
report.append('## 8. Pillar 6 — What is quietly eroding protected value?')
report.append(f'Isolation Forest used a 2% review contamination setting on order amount, quantity, line count, maximum item price, and category-normalized price behavior. A robust z-score screen used a 3.5 threshold. The union queue contains {flagged_count:,} of {len(anom_orders):,} orders and ${flagged_value:,.0f} of order value.')
report.append('')
report.append('These are triage flags, not confirmed fraud. There are no confirmed fraud labels in the source. Status-based cancellation/return rates were exported as sensitivity-only fields but were not used as a fraud target because `order_status` is explicitly out of scope and not validated as diagnostic.')
report.append('')
report.append('![Fraud triage](infographic_fraud_triage.png)')
report.append('')
report.append('### Decision')
report.append('Review the highest-value flagged orders and accounts before attributing recovered revenue to retention. Add confirmed payment, return, shipping, device, and account-link outcomes before calibrating automatic blocking.')
report.append('')
report.append('## 9. Integrated prioritized action plan')
report.append(segment_action_summary.to_string(index=False))
report.append('')
report.append('The customer-level action file ranks customers using a transparent combination of predicted next-quarter spend and behavioral risk, assigns an intervention by segment/value tier, applies a provisional 10% spend cap, and marks customers whose prior orders require fraud review. The first execution wave should be:')
report.append('1. **High-value at-risk:** personalized category/brand bundle, time-limited incentive, margin guardrail, and control group.')
report.append('2. **Lower-value at-risk:** reminder and content-based recommendation with minimal incentive.')
report.append('3. **Lost:** win-back test with strict spend cap; stop if there is no incremental response.')
report.append('4. **One-time:** second-purchase journey rather than expensive blanket discounting.')
report.append('5. **Champions/core:** protect service and cross-sell; do not train customers to wait for discounts.')
report.append('6. **Fraud review:** hold anomalous high-value orders out of campaign ROI claims until reviewed.')
report.append('')
report.append('## 10. Measurement plan and limitations')
report.append('Track treatment/control assignment, channel, offer, exposure, redemption, discount cost, margin, repeat order, 30/60/90-day reactivation, and anomaly-review outcome. Refit churn and value models only after these outcomes accumulate.')
report.append('')
report.append('The source does not provide confirmed churn intent, intervention history, margin/discount cost, confirmed fraud labels, reliable return/cancellation truth, or price elasticity. The results therefore support prioritization, recommendation experiments, revenue planning, and investigation queues — not causal offer optimization, confirmed fraud blocking, or precise lifetime-value claims.')
report.append('')
report.append('## Appendix — Output register')
report.append('Core analysis-ready data: `customer_analysis_ready.csv`, `customer_snapshot_panel.csv`, `product_analysis_ready.csv`, `order_analysis_ready.csv`, `order_item_analysis_ready.csv`, `event_analysis_ready.csv`, `review_analysis_ready.csv`, `users_clean.csv`.')
report.append('Model and decision tables: `classification_model_results.csv`, `logistic_driver_coefficients.csv`, `customer_value_model_results.csv`, `ridge_value_coefficients.csv`, `product_price_model_results.csv`, `association_rules_category_brand.csv`, `content_recommendations.csv`, `forecast_model_results.csv`, `next_quarter_revenue_forecast.csv`, `fraud_order_review_queue.csv`, `fraud_user_summary.csv`, `prioritized_retention_action_plan.csv`, `segment_action_summary.csv`.')
report.append('Infographics: `infographic_executive_story.png`, `infographic_revenue_by_category.png`, `infographic_segment_value_risk.png`, `infographic_churn_model_comparison.png`, `infographic_forecast_validation.png`, `infographic_fraud_triage.png`, `infographic_cohort_retention.png`.')
report_text='\n'.join(report)
(OUT/'verified_business_report.md').write_text(report_text,encoding='utf-8')
summary={
    'source_rows':{name:int(len(df)) for name,df in frames.items()},
    'valid_order_count':valid_order_count,'valid_order_value':valid_order_value,'date_min':str(orders['order_date_clean'].min()),'date_max':str(orders['order_date_clean'].max()),
    'decision_snapshot':str(final_snapshot.date()),'historical_buyers':buyer_count,'churn_count':churn_count,'churn_rate':churn_rate,'active_next_90d':active_count,
    'top_two_categories':category_revenue.head(2).to_dict('records'),'top_two_share':top2_share,'at_risk_customers':int(len(at_risk)),'at_risk_predicted_next_quarter_spend':at_risk_pred,'at_risk_budget_cap_10pct':at_risk_cap,
    'selected_k':selected_k,'cluster_silhouette_by_k':cluster_sils,'hierarchical_silhouette':hier_sil,'dbscan_silhouette_non_noise':db_sil,
    'classification_best_model_by_auc':best_model_name,'classification_results':model_results.to_dict('records'),'value_results':value_results.to_dict('records'),'price_results':price_results.to_dict('records'),
    'association_rule_count':int(len(rules)),'product_rule_diagnostic':product_diag,'content_recommendation_rows':int(len(content_recs)),
    'forecast_results':forecast_results.to_dict('records'),'selected_forecast_model':best_forecast_name,'next_90_day_forecast_total':forecast_total,'next_90_day_forecast_daily_average':forecast_daily,
    'fraud_review_orders':flagged_count,'fraud_review_value':flagged_value,'reconciliation':reconciliation,'coercion_audit':coercion_audit
}
(OUT/'verified_analysis_summary.json').write_text(json.dumps(summary,indent=2,default=str),encoding='utf-8')
print('RESULT: verified_summary', json.dumps(summary, indent=2, default=str))
print('OUTPUTS: verified_business_report.md, verified_analysis_summary.json, analysis-ready CSVs, model tables, action-plan tables, and infographics')
