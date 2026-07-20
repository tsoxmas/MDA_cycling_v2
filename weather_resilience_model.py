import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import paths as p

# load the daily panel
df = pd.read_parquet(p.res_dir / "site_day_panel.parquet")
df["date"] = pd.to_datetime(df["date"])
df["site_id"] = df["site_id"].astype(str)

df["y"] = np.log1p(df["total_count"]) # using log transformation for the dependent variable (log daily cycling count)
# so very busy sites do not overpower everything
df["log_rain"] = np.log1p(df["precip_mm"].fillna(0))
df["site_rain"] = np.where(df["rain_day"] == 1, df["site_id"], "dry_day")

# numeric weather/calendar/context variables
num = ["log_rain", "temp_mean", "wind_mean", "hum_mean", "pressure_mean", "radiation_mean",
       "station_km", "dow", "month", "commute_share", "n_slots"]

# categorical variables, site_rain is for site-specific rain response
cat = ["site_id", "site_rain"]

# binary variables
binary = ["rain_day", "heavy_rain_day", "weekend", "holiday", "covid",]

# pipeline for imputing missing numeric weather values and scaling them for ridge
num_part = Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())])

binary_part = Pipeline([("fill", SimpleImputer(strategy="most_frequent"))])

# different preprocessing to numeric binary and categorical columns
prep = ColumnTransformer([("num", num_part, num), ("binary", binary_part, binary),
                          ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])

dates = sorted(df["date"].unique())

# time split so future days are not used in training
valid_day = dates[int(len(dates) * 0.70)]
test_day = dates[int(len(dates) * 0.80)]

# train/validation/test by date
train = df[df["date"] < valid_day]
valid = df[(df["date"] >= valid_day) & (df["date"] < test_day)]
test = df[df["date"] >= test_day]

# model inputs and logged target
x_train = train[num + cat + binary]
y_train = train["y"]
x_valid = valid[num + cat + binary]
y_valid = valid["y"]
x_test = test[num + cat + binary]
y_test = test["y"]
alpha_rows = []

# trying a few ridge penalties and picking the best one on validation
for alpha in [0.1, 1, 10, 100]:
    model = Pipeline([("prep", prep), ("ridge", Ridge(alpha=alpha))])
    model.fit(x_train, y_train)
    valid_pred = model.predict(x_valid)
    alpha_rows.append({"alpha": alpha, "valid_rmse_log": np.sqrt(mean_squared_error(y_valid, valid_pred)),
        "valid_mae_log": mean_absolute_error(y_valid, valid_pred), "valid_r2_log": r2_score(y_valid, valid_pred)})

alpha_scores = pd.DataFrame(alpha_rows)
print(alpha_scores)
alpha_scores = alpha_scores.sort_values("valid_rmse_log")
best_alpha = alpha_scores["alpha"].values[0]
print(best_alpha)

# refit with train + validation, then test on the last period
train_full = df[df["date"] < test_day]

model = Pipeline([("prep", prep), ("ridge", Ridge(alpha=best_alpha))])

model.fit(train_full[num + cat + binary], train_full["y"])
test_pred = model.predict(x_test)


names = model.named_steps["prep"].get_feature_names_out()
coef = model.named_steps["ridge"].coef_
coef_table = pd.DataFrame({"feature": names, "coef": coef,})
coef_table["feature"] = coef_table["feature"].str.replace("num__", "", regex=False)
coef_table["feature"] = coef_table["feature"].str.replace("binary__", "", regex=False)
coef_table["feature"] = coef_table["feature"].str.replace("cat__", "", regex=False)
coef_table["abs_coef"] = coef_table["coef"].abs()
coef_table["pct_change"] = (np.exp(coef_table["coef"]) - 1) * 100
coef_table = coef_table.sort_values("abs_coef", ascending=False)
coef_table.to_csv(p.res_dir / "ridge_coefficients.csv", index=False)
print(coef_table.head(25))


# final model scores on the test set
metrics = pd.DataFrame([{"best_alpha": best_alpha, "test_start": str(pd.Timestamp(test_day).date()),
    "rmse_log": np.sqrt(mean_squared_error(y_test, test_pred)), "mae_log": mean_absolute_error(y_test, test_pred),
    "r2_log": r2_score(y_test, test_pred)}])

metrics.to_csv(p.res_dir / "model_metrics.csv", index=False)
main_coef = coef_table[~coef_table["feature"].str.startswith("site_id_")
                       & ~coef_table["feature"].str.startswith("site_rain_")].copy()

main_coef = main_coef.sort_values("abs_coef", ascending=False)
main_coef.to_csv(p.res_dir / "ridge_main_coefficients.csv", index=False)

print(main_coef)

#plot
plot_coef = main_coef.head(15).copy()
plot_coef = plot_coef.sort_values("pct_change")
colors = np.where(plot_coef["coef"] >= 0, "royalblue", "firebrick")
plt.figure(figsize=(9, 6))
plt.barh(plot_coef["feature"], plot_coef["pct_change"], color=colors)
plt.axvline(0, color="black", linewidth=1)
plt.title("Ridge coefficients — approximate effect on daily cycling count")
plt.xlabel("Approximate percent change")
plt.ylabel("")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(p.fig_dir / "fig6_ridge_main_coefficients_percent.png", dpi=300)
plt.close()


wet = df[df["rain_day"] == 1].copy() # only rainy days because this is where the rain penalty can be estimated

x_wet = wet[num + cat + binary].copy()
x_dry = x_wet.copy()
rain_cols = ["log_rain", "rain_day", "heavy_rain_day"]
x_dry[rain_cols] = 0
x_dry["site_rain"] = "dry_day" # same rows but pretending there was no rain

pred_wet = np.expm1(model.predict(x_wet))
pred_dry = np.expm1(model.predict(x_dry))

rain_loss = np.clip(pred_dry - pred_wet, 0, None) # predicted cyclists lost because of rain
wet["rain_loss"] = rain_loss
wet["rain_loss_pct"] = rain_loss / np.maximum(pred_dry, 1)

# site-level rain penalty from rainy days
site = wet.groupby("site_id").agg(
    site_lat=("site_lat", "first"),
    site_lon=("site_lon", "first"),
    rain_penalty_pct=("rain_loss_pct", "median"),
    rain_penalty_count=("rain_loss", "median"),
).reset_index()

base = df.groupby("site_id").agg(
    median_daily_count=("total_count", "median"),
    rain_share=("rain_day", "mean"),
    commute_share=("commute_share", "median"),
    station_km=("station_km", "median"),
).reset_index()

site = site.merge(base, on="site_id", how="left")
site["weather_resilience_index"] = 1 - site["rain_penalty_pct"]
site.to_csv(p.res_dir / "site_weather_resilience.csv", index=False)
