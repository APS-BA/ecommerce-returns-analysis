"""
Monthly returned orders/value with a linear-trend 3-month forecast; the partial final month is excluded from the fit.

Source: generated and run in Better Analyst (app.betteranalyst.com), session
"E-Commerce Returns Analysis", 2026-09-24. Exported verbatim; only this header was added.
Input paths point to Better Analyst's sandbox: /home/user/workspace/1790268567916_products.csv, /home/user/workspace/1790268567919_order_items.csv, /home/user/workspace/1790268568295_orders.csv, /home/user/workspace/outputs, /home/user/workspace/outputs/return_analysis_overview.png.
Change them to your local data folder before running.
"""

import pandas as pd, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
base={"orders":"/home/user/workspace/1790268568295_orders.csv","items":"/home/user/workspace/1790268567919_order_items.csv","products":"/home/user/workspace/1790268567916_products.csv"}
orders=pd.read_csv(base["orders"]);items=pd.read_csv(base["items"]);products=pd.read_csv(base["products"])
print("SOURCE_CHECK",{k:[v,len(pd.read_csv(v,nrows=0).columns)] for k,v in base.items()},"order_rows",len(orders),"sample",orders.head(2).to_dict("records"))
for col in ["order_id","product_id"]:
 d=orders if col=="order_id" else products
 if d[col].duplicated().any():raise ValueError(f"Duplicate key {col}; cannot safely create corrected trend chart")
orders["date_clean"]=pd.to_datetime(orders["order_date"],errors="coerce");orders["amount_clean"]=pd.to_numeric(orders["total_amount"],errors="coerce");orders["returned"]=orders["order_status"].eq("returned").astype(int);orders["month"]=orders["date_clean"].dt.to_period("M").dt.to_timestamp()
monthly=orders.loc[orders["month"].notna()].groupby("month").agg(returned_order_count=("returned","sum"),returned_order_value=("amount_clean",lambda s:float(s[orders.loc[s.index,"returned"].eq(1)].sum(min_count=1)))).sort_index()
months=pd.date_range(monthly.index.min(),monthly.index.max(),freq="MS");monthly=monthly.reindex(months,fill_value=0);last_date=orders["date_clean"].max();last_month=monthly.index.max();partial=bool(last_date.day<last_date.days_in_month)
fit=monthly.loc[monthly.index<last_month] if partial else monthly.copy()
if len(fit)<2:raise ValueError("Insufficient complete months to fit a monthly trend")
fit_x=np.arange(len(fit),dtype=float);future_months=pd.date_range(last_month+pd.offsets.MonthBegin(1),periods=3,freq="MS");start=monthly.index.min();future_x=np.array([(d.year-start.year)*12+d.month-start.month for d in future_months],dtype=float)
forecast_count=np.maximum(0,np.polyval(np.polyfit(fit_x,fit["returned_order_count"].to_numpy(float),1),future_x));forecast_value=np.maximum(0,np.polyval(np.polyfit(fit_x,fit["returned_order_value"].to_numpy(float),1),future_x))
monthly_out=pd.DataFrame({"month":monthly.index.strftime("%Y-%m-%d"),"period_status":["Actual (partial month)" if partial and d==last_month else "Actual (full month)" for d in monthly.index],"returned_order_count":monthly["returned_order_count"].astype(int).to_numpy(),"returned_order_value":monthly["returned_order_value"].to_numpy(),"forecast_order_count":np.nan,"forecast_order_value":np.nan,"included_in_trend_fit":[bool(d in fit.index) for d in monthly.index]})
future_out=pd.DataFrame({"month":future_months.strftime("%Y-%m-%d"),"period_status":"Forecast","returned_order_count":np.nan,"returned_order_value":np.nan,"forecast_order_count":forecast_count,"forecast_order_value":forecast_value,"included_in_trend_fit":False})
monthly_out=pd.concat([monthly_out,future_out],ignore_index=True)
# Recompute total/returned values and product category return rates from source for a corrected overview image.
amount_total=float(orders["amount_clean"].sum(min_count=1));amount_returned=float(orders.loc[orders["returned"].eq(1),"amount_clean"].sum(min_count=1))
joined=items[["order_id","product_id"]].merge(products[["product_id","category"]],on="product_id",how="left",validate="many_to_one")
joined["category_clean"]=joined["category"].astype("string").str.strip().fillna("Missing");joined.loc[joined["category_clean"].eq(""),"category_clean"]="Missing"
cat_counts=joined.groupby(["order_id","category_clean"],dropna=False).size().reset_index(name="n").sort_values(["order_id","n","category_clean"],ascending=[True,False,True],kind="mergesort").drop_duplicates("order_id")
cat_orders=orders[["order_id","returned"]].merge(cat_counts[["order_id","category_clean"]],on="order_id",how="left",validate="one_to_one")
cat_rates=cat_orders.groupby("category_clean").agg(orders=("returned","size"),returned_orders=("returned","sum"),return_rate=("returned","mean")).reset_index().sort_values("return_rate")
plt.rcParams.update({"figure.figsize":(13,9),"figure.dpi":120,"savefig.dpi":200,"savefig.bbox":"tight","font.family":"sans-serif","font.sans-serif":["DejaVu Sans"],"font.size":9,"axes.titlesize":12,"axes.titleweight":"bold","axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"axes.grid.axis":"y","grid.color":"#E6E6E6","axes.axisbelow":True,"legend.frameon":False})
fig,axs=plt.subplots(2,2,constrained_layout=True)
axs[0,0].bar(["All order value","Returned order value"],[amount_total,amount_returned],color=["#9AA6B2","#356B8C"]);axs[0,0].set_title("Order value: total vs returned");axs[0,0].set_ylabel("Value ($)");axs[0,0].yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f"${v:,.0f}"))
axs[0,1].barh(cat_rates["category_clean"].astype(str),cat_rates["return_rate"]*100,color="#356B8C");axs[0,1].set_title("Return rate by dominant category");axs[0,1].set_xlabel("Returned orders (%)");axs[0,1].xaxis.set_major_formatter(FuncFormatter(lambda v,pos:f"{v:.0f}%"))
actual_dates=monthly.index;actual_counts=monthly["returned_order_count"].to_numpy(float);actual_values=monthly["returned_order_value"].to_numpy(float)
future_plot_dates=future_months;forecast_plot_counts=forecast_count;forecast_plot_values=forecast_value
axs[1,0].plot(actual_dates,actual_counts,color="#356B8C",label="Actual",marker="o",markersize=3);axs[1,0].plot(future_plot_dates,forecast_plot_counts,color="#C36B37",linestyle="--",marker="o",markersize=3,label="Linear forecast");axs[1,0].set_title("Monthly returned orders");axs[1,0].set_ylabel("Orders");axs[1,0].legend()
axs[1,1].plot(actual_dates,actual_values,color="#356B8C",label="Actual",marker="o",markersize=3);axs[1,1].plot(future_plot_dates,forecast_plot_values,color="#C36B37",linestyle="--",marker="o",markersize=3,label="Linear forecast");axs[1,1].set_title("Monthly returned value");axs[1,1].set_ylabel("Value ($)");axs[1,1].yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f"${v:,.0f}"));axs[1,1].legend()
if partial:
 for ax,yval in [(axs[1,0],actual_counts[-1]),(axs[1,1],actual_values[-1])]:ax.annotate("Partial month",(actual_dates[-1],yval),xytext=(-58,10),textcoords="offset points",fontsize=8,arrowprops={"arrowstyle":"-","color":"#666666"})
for ax in [axs[1,0],axs[1,1]]:ax.tick_params(axis="x",rotation=35)
fig.suptitle("E-commerce returns: value, category and monthly trend",fontsize=15,fontweight="bold")
plot_values=np.concatenate([np.asarray([amount_total,amount_returned]),cat_rates["return_rate"].to_numpy(float),actual_counts,forecast_plot_counts,actual_values,forecast_plot_values]);print("PLOT_INPUT shape",plot_values.shape,"preview",plot_values[:8])
if plot_values.size==0 or pd.isna(plot_values).all():raise SystemExit("CHART_DATA_EMPTY: plotted values are empty or all-NaN")
chart_path="/home/user/workspace/outputs/return_analysis_overview.png";fig.savefig(chart_path,dpi=200,bbox_inches="tight");plt.close(fig)
methodology=pd.DataFrame([{"item":"Target","definition":"1 iff order_status == returned; all other statuses 0."},{"item":"Split","definition":"Order-level stratified random 80/20 holdout, random_state=42; default decision thresholds."},{"item":"Features","definition":"Amount from orders.total_amount; item count counts item rows; units sum quantity; max price is max order_items.item_price; avg rating is mean products.rating per item line; dominant category/brand are item-line modes, alphabetical tie-break; tenure in days; sequence chronological per user across all statuses."},{"item":"Balancing","definition":"Logistic/tree/random forest use class_weight balanced; gradient boosting balanced sample_weight; kNN majority down-sampling because no class_weight parameter."},{"item":"Importance","definition":"Aggregate encoded variables to source feature; logistic absolute standardized coefficients, tree impurity importance; kNN has no native importance."},{"item":"Limit","definition":"Random order holdout may place one customer's orders in train and test; results are predictive association, not causation."},{"item":"Forecast","definition":"Linear least-squares trend fitted to complete months only; latest month is partial and excluded; forecasts cover the next three months after that partial month, clipped at zero, without seasonality or interval."}])
outdir=Path("/home/user/workspace/outputs");saved_paths=[]
for filename,frame in [("return_monthly_forecast.csv",monthly_out),("return_model_methodology.csv",methodology)]:
 p=outdir/filename;frame.to_csv(p,index=False);saved_paths.append(str(p));print("SAVED_TABLE",filename,"rows",len(frame))
saved_paths.append(chart_path)
print("SAVED_CHART",chart_path,"SOURCE_DATE_RANGE",str(orders["date_clean"].min()),str(last_date),"FINAL_MONTH_PARTIAL",partial,"TREND_FIT_THROUGH",str(fit.index.max().date()),"FORECAST",monthly_out.tail(3).to_dict("records"))
print("OUTPUT_PATHS",saved_paths)
