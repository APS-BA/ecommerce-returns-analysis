"""
Data quality checks, return rates by segment with chi-square / Cramer's V, review tests, 5 classification models (logistic regression, decision tree, kNN, random forest, gradient boosting), customer-level binomial test, monthly trend.

Source: generated and run in Better Analyst (app.betteranalyst.com), session
"E-Commerce Returns Analysis", 2026-09-24. Exported verbatim; only this header was added.
Input paths point to Better Analyst's sandbox: /home/user/workspace/1790268567916_products.csv, /home/user/workspace/1790268567919_order_items.csv, /home/user/workspace/1790268568295_orders.csv, /home/user/workspace/1790268568306_users.csv, /home/user/workspace/1790268570650_reviews.csv, /home/user/workspace/outputs.
Change them to your local data folder before running.
"""

import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import chi2_contingency, binomtest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
paths={"products":"/home/user/workspace/1790268567916_products.csv","users":"/home/user/workspace/1790268568306_users.csv","orders":"/home/user/workspace/1790268568295_orders.csv","order_items":"/home/user/workspace/1790268567919_order_items.csv","reviews":"/home/user/workspace/1790268570650_reviews.csv"}
src={k:pd.read_csv(v) for k,v in paths.items()}
for k,d in src.items(): print("SOURCE",k,paths[k],"rows",len(d),"columns",list(d.columns),"sample",d.head(2).to_dict("records"))
req={"products":["product_id","category","brand","price","rating"],"users":["user_id","gender","signup_date"],"orders":["order_id","user_id","order_date","order_status","total_amount"],"order_items":["order_item_id","order_id","product_id","user_id","quantity","item_price","item_total"],"reviews":["review_id","order_id","product_id","user_id","rating","review_text","review_date"]}
for k,cols in req.items():
 miss=[c for c in cols if c not in src[k].columns]
 if miss: raise ValueError(f"Missing required columns in {k}: {miss}; available {list(src[k].columns)}")
orders=src["orders"].copy(); users=src["users"].copy(); products=src["products"].copy(); items=src["order_items"].copy(); reviews=src["reviews"].copy()
orders["returned"]=orders["order_status"].eq("returned").astype(int)
keys={"products":"product_id","users":"user_id","orders":"order_id","order_items":"order_item_id","reviews":"review_id"}
fq=[]; nq=[]; dq=[]
for name,d in src.items():
 key=keys[name]; fq.append({"file":name+".csv","rows":len(d),"columns":d.shape[1],"null_cells":int(d.isna().sum().sum()),"rows_with_any_null":int(d.isna().any(axis=1).sum()),"exact_duplicate_extra_rows":int(d.duplicated().sum()),"primary_key":key,"primary_key_null_rows":int(d[key].isna().sum()),"duplicate_primary_key_extra_rows":int(d[key].duplicated().sum()),"duplicate_primary_key_values":int(d.loc[d[key].duplicated(keep=False),key].nunique())})
 for c in d.columns:nq.append({"file":name+".csv","column":c,"null_count":int(d[c].isna().sum()),"null_pct":float(d[c].isna().mean()*100),"blank_string_count":int((d[c].astype("string").str.strip().eq("").fillna(False)).sum()) if pd.api.types.is_object_dtype(d[c]) else 0})
 dq.append({"file":name+".csv","key":key,"rows_repeated_beyond_first":int(d[key].duplicated().sum()),"distinct_repeated_key_values":int(d.loc[d[key].duplicated(keep=False),key].nunique())})
fq=pd.DataFrame(fq); nq=pd.DataFrame(nq); dq=pd.DataFrame(dq)
for d,k,n in [(orders,"order_id","orders"),(users,"user_id","users"),(products,"product_id","products")]:
 if d[k].duplicated().any():raise ValueError(f"Duplicate {k} in {n}; refusing join inflation")
orders["date_clean"]=pd.to_datetime(orders["order_date"],errors="coerce"); orders["amount_clean"]=pd.to_numeric(orders["total_amount"],errors="coerce"); users["signup_clean"]=pd.to_datetime(users["signup_date"],errors="coerce"); products["rating_clean"]=pd.to_numeric(products["rating"],errors="coerce")
for c in ["quantity","item_price","item_total"]:items[c+"_clean"]=pd.to_numeric(items[c],errors="coerce")
ip=items.merge(products[["product_id","category","brand","rating_clean"]],on="product_id",how="left",validate="many_to_one",indicator="product_join")
ip["cat"]=ip["category"].astype("string").str.strip().fillna("Missing");ip.loc[ip["cat"].eq(""),"cat"]="Missing";ip["brand_clean"]=ip["brand"].astype("string").str.strip().fillna("Missing");ip.loc[ip["brand_clean"].eq(""),"brand_clean"]="Missing"
def modelex(s):
 vc=s.value_counts(dropna=False);return sorted(str(v) for v in vc.index[vc.eq(vc.max())])[0] if len(vc) else "Missing"
agg=[]
for oid,g in ip.groupby("order_id",sort=False,dropna=False):agg.append({"order_id":oid,"item_count":len(g),"units":g["quantity_clean"].sum(min_count=1),"max_price":g["item_price_clean"].max(),"avg_product_rating":g["rating_clean"].mean(),"dominant_category":modelex(g["cat"]),"dominant_brand":modelex(g["brand_clean"]),"unmatched_product_lines":int(g["product_join"].eq("left_only").sum())})
itemagg=pd.DataFrame(agg)
od=orders[["order_id","user_id","order_status","date_clean","amount_clean","returned"]].merge(itemagg,on="order_id",how="left",validate="one_to_one",indicator="item_join")
od=od.merge(users[["user_id","gender","signup_clean"]],on="user_id",how="left",validate="many_to_one",indicator="user_join")
od["return_status"]=np.where(od["returned"].eq(1),"Returned","Not returned");od["gender_clean"]=od["gender"].astype("string").str.strip().fillna("Missing");od.loc[od["gender_clean"].eq(""),"gender_clean"]="Missing"
od["month_date"]=od["date_clean"].dt.to_period("M").dt.to_timestamp();od["month_name"]=od["date_clean"].dt.month_name().astype("string").fillna("Missing");od["day_name"]=od["date_clean"].dt.day_name().astype("string").fillna("Missing");od["tenure_days"]=(od["date_clean"]-od["signup_clean"]).dt.total_seconds()/86400
od=od.sort_values(["user_id","date_clean","order_id"],kind="mergesort");od["order_sequence"]=od.groupby("user_id",dropna=True).cumcount()+1;od.loc[od["user_id"].isna(),"order_sequence"]=np.nan
od["sequence_group"]=np.select([od["order_sequence"].eq(1),od["order_sequence"].eq(2),od["order_sequence"].eq(3)],["1st","2nd","3rd"],default="4+");od.loc[od["order_sequence"].isna(),"sequence_group"]="Missing";od["value_quintile"]=pd.qcut(od["amount_clean"].rank(method="first"),5,labels=["Q1 lowest","Q2","Q3","Q4","Q5 highest"])
ou=orders[["order_id","user_id","date_clean"]].merge(users[["user_id","signup_clean"]],on="user_id",how="left",validate="many_to_one",indicator=True)
pre=int((ou["date_clean"].notna()&ou["signup_clean"].notna()&(ou["date_clean"]<ou["signup_clean"])).sum())
io=items.merge(orders[["order_id","user_id"]],on="order_id",how="left",validate="many_to_one",suffixes=("_item","_order"),indicator=True)
item_um=int((io["_merge"].eq("both")&io["user_id_item"].notna()&io["user_id_order"].notna()&io["user_id_item"].ne(io["user_id_order"])).sum())
ro=reviews.merge(orders[["order_id","user_id","returned"]],on="order_id",how="left",validate="many_to_one",suffixes=("_review","_order"),indicator=True)
review_um=int((ro["_merge"].eq("both")&ro["user_id_review"].notna()&ro["user_id_order"].notna()&ro["user_id_review"].ne(ro["user_id_order"])).sum())
quality=pd.DataFrame([
 {"check":"Orders before matched user's signup_date","count":pre,"denominator":int((ou["date_clean"].notna()&ou["signup_clean"].notna()).sum()),"detail":"Timestamp vs signup date at midnight; matched parseable pairs."},
 {"check":"Orders with unmatched user_id","count":int(od["user_join"].eq("left_only").sum()),"denominator":len(orders),"detail":"Orders retained."},{"check":"Orders without order_items","count":int(od["item_join"].eq("left_only").sum()),"denominator":len(orders),"detail":"Orders retained at order grain."},
 {"check":"Order items with unmatched order_id","count":int(io["_merge"].eq("left_only").sum()),"denominator":len(items),"detail":"Item to order linkage."},{"check":"Order items with unmatched product_id","count":int(ip["product_join"].eq("left_only").sum()),"denominator":len(items),"detail":"Item to product linkage."},{"check":"Order item user_id mismatches order","count":item_um,"denominator":int(io["_merge"].eq("both").sum()),"detail":"Both IDs present."},{"check":"Reviews with unmatched order_id","count":int(ro["_merge"].eq("left_only").sum()),"denominator":len(reviews),"detail":"Review to order linkage."},{"check":"Review user_id mismatches order","count":review_um,"denominator":int(ro["_merge"].eq("both").sum()),"detail":"Both IDs present."},{"check":"Null/unparseable order amount","count":int(od["amount_clean"].isna().sum()),"denominator":len(od),"detail":"orders.total_amount."},{"check":"Null/unparseable order date","count":int(od["date_clean"].isna().sum()),"denominator":len(od),"detail":"Parsed datetime."}])
N=len(od);rN=int(od["returned"].sum());total=float(od["amount_clean"].sum(min_count=1));retval=float(od.loc[od["returned"].eq(1),"amount_clean"].sum(min_count=1));nonval=float(od.loc[od["returned"].eq(0),"amount_clean"].sum(min_count=1));p0=rN/N
overall=pd.DataFrame([{"metric":"Orders","value":N,"definition":"All orders; order_id unique."},{"metric":"Returned orders","value":rN,"definition":"Status exactly 'returned'."},{"metric":"Overall return rate","value":p0,"definition":"Returned / all orders."},{"metric":"Total order value","value":total,"definition":"Sum of numeric orders.total_amount."},{"metric":"Returned order value","value":retval,"definition":"Sum total_amount on returned orders."},{"metric":"Returned value share","value":retval/total,"definition":"Returned value / total order value."},{"metric":"Non-returned order value","value":nonval,"definition":"Sum of all other statuses."},{"metric":"Orders with usable total_amount","value":int(od["amount_clean"].notna().sum()),"definition":"Valid numeric values."}])
def chisq(tab):
 if tab.shape[0]<2 or tab.shape[1]<2:return np.nan,np.nan,0,np.nan,0,0
 c,p,d,e=chi2_contingency(tab.to_numpy(),correction=False);den=int(tab.to_numpy().sum())*min(tab.shape[0]-1,tab.shape[1]-1);return float(c),float(p),int(d),float(np.sqrt(c/den)) if den else np.nan,int((e<5).sum()),int(e.size)
def one_segment(label,s,order=None):
 tmp=pd.DataFrame({"level":s.astype("string").fillna("Missing"),"returned":od["returned"]});res=tmp.groupby("level")["returned"].agg(orders="count",returned_orders="sum",return_rate="mean").reset_index();tab=pd.crosstab(tmp["level"],tmp["returned"]).reindex(columns=[0,1],fill_value=0);tab=tab.loc[tab.sum(axis=1)>0];c,p,d,v,low,cells=chisq(tab);res["dimension"]=label;res["chi2"]=c;res["p_value"]=p;res["cramers_v"]=v
 if order is not None:res["_s"]=res["level"].map({str(a):i for i,a in enumerate(order)});res=res.sort_values(["_s","level"],na_position="last").drop(columns="_s")
 else:res["_s"]=pd.to_numeric(res["level"],errors="coerce");res=res.sort_values(["_s","level"]).drop(columns="_s")
 test={"dimension":label,"chi2":c,"p_value":p,"degrees_freedom":d,"cramers_v":v,"orders_tested":int(tab.to_numpy().sum()),"expected_cells_below_5":low,"expected_cell_count":cells,"pct_expected_cells_below_5":low/cells if cells else np.nan};return res,test
seg_inputs=[("Dominant product category",od["dominant_category"],None),("Dominant brand",od["dominant_brand"],None),("Order-value quintile",od["value_quintile"],["Q1 lowest","Q2","Q3","Q4","Q5 highest","Missing"]),("Items per order",od["item_count"].astype("Int64").astype("string"),None),("Gender",od["gender_clean"],None),("Month of year",od["month_name"],["January","February","March","April","May","June","July","August","September","October","November","December","Missing"]),("Day of week",od["day_name"],["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday","Missing"]),("Customer order sequence",od["sequence_group"],["1st","2nd","3rd","4+","Missing"])]
segments=[];tests=[]
for a,b,c in seg_inputs:x,y=one_segment(a,b,c);segments.append(x);tests.append(y)
segments=pd.concat(segments,ignore_index=True);tests=pd.DataFrame(tests)
rv=reviews.merge(orders[["order_id","returned"]],on="order_id",how="left",validate="many_to_one",indicator="order_join");rv=rv.loc[rv["order_join"].eq("both")].copy();rv["rating_clean"]=pd.to_numeric(rv["rating"],errors="coerce");rv["text_clean"]=rv["review_text"].astype("string").str.strip().fillna("Missing");rv.loc[rv["text_clean"].eq(""),"text_clean"]="Missing";rv["return_status"]=np.where(rv["returned"].eq(1),"Returned","Not returned")
review_summary=rv.groupby("return_status").agg(review_rows=("review_id","size"),orders_with_review=("order_id","nunique"),avg_rating=("rating_clean","mean"),rating_count=("rating_clean","count"),median_rating=("rating_clean","median")).reset_index()
rt=pd.crosstab(rv["text_clean"],rv["returned"]).reindex(columns=[0,1],fill_value=0);rt=rt.loc[rt.sum(axis=1)>0];review_dist=rt.reset_index().rename(columns={"text_clean":"review_text",0:"non_returned_review_rows",1:"returned_review_rows"});review_dist["total_review_rows"]=review_dist["non_returned_review_rows"]+review_dist["returned_review_rows"];review_dist["return_rate"]=review_dist["returned_review_rows"]/review_dist["total_review_rows"]
c,p,d,v,low,cells=chisq(rt);tests=pd.concat([tests,pd.DataFrame([{"dimension":"Review text (exact normalized labels)","chi2":c,"p_value":p,"degrees_freedom":d,"cramers_v":v,"orders_tested":int(rt.to_numpy().sum()),"expected_cells_below_5":low,"expected_cell_count":cells,"pct_expected_cells_below_5":low/cells if cells else np.nan}])],ignore_index=True)
num=["amount_clean","item_count","units","max_price","avg_product_rating","tenure_days","order_sequence"];cat=["dominant_category","dominant_brand","gender_clean"];X=od[num+cat].copy();y=od["returned"].astype(int)
for c in num:X[c]=pd.to_numeric(X[c],errors="coerce")
for c in cat:X[c]=X[c].astype("string").fillna("Missing").replace("","Missing")
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
prep=ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=True))]),cat)])
A=prep.fit_transform(Xtr);B=prep.transform(Xte);tnames=list(prep.get_feature_names_out())
models={"Logistic regression":LogisticRegression(class_weight="balanced",max_iter=1000,solver="liblinear",random_state=42),"Decision tree":DecisionTreeClassifier(class_weight="balanced",min_samples_leaf=10,random_state=42),"kNN":KNeighborsClassifier(n_neighbors=15,weights="distance",n_jobs=-1),"Random forest":RandomForestClassifier(n_estimators=300,class_weight="balanced",min_samples_leaf=3,n_jobs=-1,random_state=42),"Gradient boosting":GradientBoostingClassifier(n_estimators=100,learning_rate=.05,max_depth=2,random_state=42)}
def basefeat(n):
 if n.startswith("num__"):return n.split("num__",1)[1]
 t=n.split("cat__",1)[-1]
 for c in cat:
  if t==c or t.startswith(c+"_"):return c
 return t
mr=[];ir=[]
for name,m in models.items():
 trainn=len(ytr)
 if name=="kNN":
  yy=np.asarray(ytr);i0=np.flatnonzero(yy==0);i1=np.flatnonzero(yy==1);rng=np.random.default_rng(42);q=min(len(i0),len(i1));idx=np.r_[rng.choice(i0,q,False),rng.choice(i1,q,False)];rng.shuffle(idx);m.fit(A[idx],yy[idx]);trainn=len(idx)
 elif name=="Gradient boosting":m.fit(A,ytr,sample_weight=compute_sample_weight(class_weight="balanced",y=ytr))
 else:m.fit(A,ytr)
 pred=m.predict(B);prob=m.predict_proba(B)[:,1];balance="balanced down-sampling (kNN; no class_weight parameter)" if name=="kNN" else ("balanced sample_weight" if name=="Gradient boosting" else "class_weight='balanced'")
 mr.append({"model":name,"roc_auc":roc_auc_score(yte,prob),"precision":precision_score(yte,pred,zero_division=0),"recall":recall_score(yte,pred,zero_division=0),"f1":f1_score(yte,pred,zero_division=0),"train_rows_used":trainn,"test_rows":len(yte),"class_balance_method":balance})
 imp=np.abs(np.asarray(m.coef_)).mean(axis=0) if name=="Logistic regression" else (m.feature_importances_ if name in ["Decision tree","Random forest","Gradient boosting"] else None)
 if imp is not None:
  group={}
  for col,val in zip(tnames,imp):f=basefeat(col);group[f]=group.get(f,0)+float(val)
  for rank,(f,val) in enumerate(sorted(group.items(),key=lambda q:(-q[1],q[0]))[:10],1):ir.append({"model":name,"rank":rank,"feature":f,"importance":val,"importance_definition":"sum absolute standardized logistic coefficients" if name=="Logistic regression" else "sum tree impurity importance across encoded levels"})
metrics=pd.DataFrame(mr);importances=pd.DataFrame(ir)
methodology=pd.DataFrame([{"item":"Target","definition":"1 iff order_status == returned; all other statuses 0."},{"item":"Split","definition":"Order-level stratified 80/20 random holdout, random_state=42; default decision thresholds."},{"item":"Features","definition":"Amount from orders.total_amount; item count counts item rows; units sum quantity; max price is max order_items.item_price; avg rating is mean products.rating per item line; dominant category/brand is mode across item lines with alphabetical tie-break; tenure in days; sequence chronological per user across all statuses."},{"item":"Balancing","definition":"Logistic/tree/random forest use class_weight balanced; gradient boosting balanced sample_weight; kNN majority down-sampling because no class_weight parameter."},{"item":"Importance","definition":"Aggregate encoded variables to source feature; logistic absolute standardized coefficients, tree impurity importance; kNN no native importance."},{"item":"Limit","definition":"Random order holdout may place one customer's orders in train and test; predictive association, not causation."},{"item":"Forecast","definition":"Linear least-squares trend on observed months, next three months, clipped at zero; no seasonality or interval."}])
co=orders[["order_id","user_id","order_status","amount_clean"]].copy();co["returned"]=co["order_status"].eq("returned").astype(int);cc=co.loc[co["user_id"].notna()].groupby("user_id").agg(customer_orders=("order_id","size"),customer_returns=("returned","sum"));cr=co.loc[co["user_id"].notna()&co["returned"].eq(1)];cv=cr.groupby("user_id")["amount_clean"].sum(min_count=1);cvn=cr.groupby("user_id")["amount_clean"].count()
ct=users[["user_id"]].drop_duplicates().set_index("user_id").join(cc,how="left");ct[["customer_orders","customer_returns"]]=ct[["customer_orders","customer_returns"]].fillna(0).astype(int);ct["expected_returns"]=ct["customer_orders"]*p0;ct["returned_order_value"]=cv;ct.loc[ct["customer_returns"].eq(0),"returned_order_value"]=0.0;ct["returned_value_order_count"]=cvn.reindex(ct.index).fillna(0).astype(int)
ct["binomial_p_value"]=[float(binomtest(int(k),int(n),p=p0).pvalue) if n else 1.0 for k,n in zip(ct["customer_returns"],ct["customer_orders"])]
ct["statistically_unusual_p_lt_0_01"]=ct["binomial_p_value"]<.01;ct["direction"]=np.select([ct["customer_returns"]>ct["expected_returns"],ct["customer_returns"]<ct["expected_returns"]],["Above expectation","Below expectation"],default="At expectation");ct=ct.reset_index();u=ct.loc[ct["statistically_unusual_p_lt_0_01"]];hi=u.loc[u["direction"].eq("Above expectation")];lo=u.loc[u["direction"].eq("Below expectation")]
cs=pd.DataFrame([{"metric":"Customers in users.csv","value":int(users["user_id"].nunique()),"definition":"Distinct user_id."},{"metric":"Customers with linked orders","value":int(ct["customer_orders"].gt(0).sum()),"definition":"Listed users with at least one order."},{"metric":"Orders linked to listed users","value":int(ct["customer_orders"].sum()),"definition":"Orders belonging to listed users."},{"metric":"Pooled return probability","value":p0,"definition":"Overall return rate."},{"metric":"Expected returns among linked customers","value":float(ct["expected_returns"].sum()),"definition":"Sum customer order counts * pooled rate."},{"metric":"Observed returns among linked customers","value":int(ct["customer_returns"].sum()),"definition":"Sum for listed users."},{"metric":"Unusual customers (two-sided exact p<0.01, unadjusted)","value":len(u),"definition":"Binomial(n=customer orders,p=overall return rate)."},{"metric":"Unusual customers above expectation","value":len(hi),"definition":"Significant customers observed above expected."},{"metric":"Returned value for unusual customers","value":float(u["returned_order_value"].sum(min_count=1)) if len(u) else 0.0,"definition":"Returned order amount for p<.01 customers."},{"metric":"Returned value for above-expectation unusual customers","value":float(hi["returned_order_value"].sum(min_count=1)) if len(hi) else 0.0,"definition":"Value for high anomalies."},{"metric":"Unusual customers below expectation","value":len(lo),"definition":"Significant customers below expected."}])
ma=od.loc[od["month_date"].notna()].groupby("month_date").agg(returned_order_count=("returned","sum"),returned_order_value=("amount_clean",lambda s:float(s[od.loc[s.index,"returned"].eq(1)].sum(min_count=1)))).sort_index()
if ma.empty:raise ValueError("No parseable order dates for monthly analysis")
months=pd.date_range(ma.index.min(),ma.index.max(),freq="MS");ma=ma.reindex(months,fill_value=0);xx=np.arange(len(ma),dtype=float);fx=np.arange(len(ma),len(ma)+3,dtype=float);fm=pd.date_range(ma.index.max()+pd.offsets.MonthBegin(1),periods=3,freq="MS")
monthly=pd.DataFrame({"month":ma.index.strftime("%Y-%m-%d"),"returned_order_count":ma["returned_order_count"].astype(int).to_numpy(),"returned_order_value":ma["returned_order_value"].to_numpy(),"forecast_order_count":np.nan,"forecast_order_value":np.nan,"row_type":"Actual"})
fc=np.maximum(0,np.polyval(np.polyfit(xx,ma["returned_order_count"].to_numpy(float),1),fx));fv=np.maximum(0,np.polyval(np.polyfit(xx,ma["returned_order_value"].to_numpy(float),1),fx));monthly=pd.concat([monthly,pd.DataFrame({"month":fm.strftime("%Y-%m-%d"),"returned_order_count":np.nan,"returned_order_value":np.nan,"forecast_order_count":fc,"forecast_order_value":fv,"row_type":"Forecast"})],ignore_index=True)
# Combined overview chart for parts (2), (3), and (7).
plt.rcParams.update({"figure.figsize":(13,9),"figure.dpi":120,"savefig.dpi":200,"savefig.bbox":"tight","font.family":"sans-serif","font.sans-serif":["DejaVu Sans"],"font.size":9,"axes.titlesize":12,"axes.titleweight":"bold","axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"axes.grid.axis":"y","grid.color":"#E6E6E6","axes.axisbelow":True,"legend.frameon":False})
fig,axs=plt.subplots(2,2,constrained_layout=True);axs[0,0].bar(["All order value","Returned order value"],[total,retval],color=["#9AA6B2","#356B8C"]);axs[0,0].set_title("Order value: total vs returned");axs[0,0].set_ylabel("Value ($)");axs[0,0].yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f"${v:,.0f}"))
cat_rates=segments.loc[segments["dimension"].eq("Dominant product category")].sort_values("return_rate");axs[0,1].barh(cat_rates["level"].astype(str),cat_rates["return_rate"]*100,color="#356B8C");axs[0,1].set_title("Return rate by dominant category");axs[0,1].set_xlabel("Returned orders (%)");axs[0,1].xaxis.set_major_formatter(FuncFormatter(lambda v,pos:f"{v:.0f}%"))
ad=monthly.loc[monthly["row_type"].eq("Actual")];fd=monthly.loc[monthly["row_type"].eq("Forecast")];adates=pd.to_datetime(ad["month"]);fdates=pd.to_datetime(fd["month"]);ac=ad["returned_order_count"].to_numpy(float);fcv=fd["forecast_order_count"].to_numpy(float);av=ad["returned_order_value"].to_numpy(float);fvv=fd["forecast_order_value"].to_numpy(float)
forecast_dates=[adates.iloc[-1]]+fdates.to_list()
axs[1,0].plot(adates,ac,color="#356B8C",label="Actual",marker="o",markersize=3);axs[1,0].plot(forecast_dates,np.r_[ac[-1],fcv],color="#C36B37",linestyle="--",marker="o",markersize=3,label="Linear forecast");axs[1,0].set_title("Monthly returned orders");axs[1,0].set_ylabel("Orders");axs[1,0].legend()
axs[1,1].plot(adates,av,color="#356B8C",label="Actual",marker="o",markersize=3);axs[1,1].plot(forecast_dates,np.r_[av[-1],fvv],color="#C36B37",linestyle="--",marker="o",markersize=3,label="Linear forecast");axs[1,1].set_title("Monthly returned value");axs[1,1].set_ylabel("Value ($)");axs[1,1].yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f"${v:,.0f}"));axs[1,1].legend()
for ax in [axs[1,0],axs[1,1]]:ax.tick_params(axis="x",rotation=35)
fig.suptitle("E-commerce returns: value, category and monthly trend",fontsize=15,fontweight="bold")
_plot_values=np.concatenate([np.asarray([total,retval]),cat_rates["return_rate"].to_numpy(float),ac,fcv,av,fvv]);print("PLOT_INPUT shape",_plot_values.shape,"preview",_plot_values[:8])
if _plot_values.size==0 or pd.isna(_plot_values).all():raise SystemExit("CHART_DATA_EMPTY: plotted values empty/all-NaN")
chart_path="/home/user/workspace/outputs/return_analysis_overview.png";fig.savefig(chart_path,dpi=200,bbox_inches="tight");plt.close(fig);print("SAVED_CHART",chart_path)
out={"return_data_quality_files.csv":fq,"return_data_quality_nulls.csv":nq,"return_data_quality_duplicates.csv":dq,"return_data_quality_checks.csv":quality,"return_overall_summary.csv":overall,"return_rates_by_segment.csv":segments,"return_association_tests.csv":tests,"return_review_summary.csv":review_summary,"return_review_text_distribution.csv":review_dist,"return_customer_summary.csv":cs,"return_customer_binomial_detail.csv":ct,"return_model_metrics.csv":metrics,"return_model_feature_importance.csv":importances,"return_model_methodology.csv":methodology,"return_monthly_forecast.csv":monthly}
outdir=Path("/home/user/workspace/outputs");outdir.mkdir(parents=True,exist_ok=True)
for filename,df in out.items():df.to_csv(outdir/filename,index=False);print("SAVED_TABLE",filename,"rows",len(df))
print("SUMMARY",{"file_rows":{k:len(v) for k,v in src.items()},"orders":N,"returned":rN,"return_rate":p0,"total_value":total,"returned_value":retval,"pre_signup":pre,"date_range":[str(od["date_clean"].min()),str(od["date_clean"].max())]})
print("FILE_QUALITY",fq.to_dict("records"));print("QUALITY_CHECKS",quality.to_dict("records"));print("ASSOCIATION_TESTS",tests.to_dict("records"));print("REVIEW_SUMMARY",review_summary.to_dict("records"));print("MODEL_METRICS",metrics.to_dict("records"));print("TOP_FEATURES",importances.groupby("model",sort=False).head(5).to_dict("records"));print("CUSTOMER_SUMMARY",cs.to_dict("records"));print("MONTHLY_LAST_6_AND_FORECAST",monthly.tail(9).to_dict("records"))
