import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

# numeric weather/calendar/context variables
num = ["log_rain", "rainy_slots", "heavy_slots", "temp_mean", "wind_mean", "gust_max", "hum_mean","pressure_mean",
       "sun_min", "radiation_mean", "station_km", "dow", "month", "commute_share", "n_slots"]

# categorical variable
cat = ["site_id"]

# binary variables
binary = ["rain_day", "heavy_rain_day", "weekend", "holiday", "covid",]

# pipeline for imputing missing numeric weather values
num_part = Pipeline([("fill", SimpleImputer(strategy="median"))])

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

# a few random forest settings to compare on validation
settings = [{"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 1},
            {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 1},
            {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 5},
            {"n_estimators": 300, "max_depth": 20, "min_samples_leaf": 5}]

rows = []

# fitting each setting and saving validation scores
for s in settings:
    model = Pipeline([("prep", prep), ("rf", RandomForestRegressor(min_samples_leaf=s["min_samples_leaf"],
            n_estimators=s["n_estimators"], max_depth=s["max_depth"], random_state=p.seed, n_jobs=-1))])
    model.fit(x_train, y_train)
    pred = model.predict(x_valid)

    rows.append({"n_estimators": s["n_estimators"], "max_depth": s["max_depth"],
                 "valid_rmse_log": np.sqrt(mean_squared_error(y_valid, pred)),
                 "valid_mae_log": mean_absolute_error(y_valid, pred),
                 "valid_r2_log": r2_score(y_valid, pred), "min_samples_leaf": s["min_samples_leaf"],})

rf_scores = pd.DataFrame(rows)
rf_scores.to_csv(p.res_dir / "rf_validation_scores.csv", index=False)
print(rf_scores)

# best rf setup based on validation rmse
rf_scores = rf_scores.sort_values("valid_rmse_log")
best_n_estimators = rf_scores["n_estimators"].values[0]
best_max_depth = rf_scores["max_depth"].values[0]
best_min_samples_leaf = rf_scores["min_samples_leaf"].values[0]

train_full = df[df["date"] < test_day]

# final model trained on train + validation
model = Pipeline([("prep", prep),("rf", RandomForestRegressor(min_samples_leaf=int(best_min_samples_leaf),
                                                              n_estimators=int(best_n_estimators),
                                                              max_depth=int(best_max_depth),
                                                              random_state=p.seed,
                                                              n_jobs=-1))])

model.fit(train_full[cat + num + binary], train_full["y"])
test_pred = model.predict(x_test)

# final score on the test period
metrics = pd.DataFrame([{"n_estimators": int(best_n_estimators),
                         "max_depth": int(best_max_depth),
                         "min_samples_leaf": int(best_min_samples_leaf),
                         "test_start": str(pd.Timestamp(test_day).date()),
                         "rmse_log": np.sqrt(mean_squared_error(y_test, test_pred)),
                         "mae_log": mean_absolute_error(y_test, test_pred),
                         "r2_log": r2_score(y_test, test_pred)}])

metrics.to_csv(p.res_dir / "rf_model_metrics.csv", index=False)
print(metrics)

names = model.named_steps["prep"].get_feature_names_out() # getting feature importance from the fitted random forest
importance = model.named_steps["rf"].feature_importances_

imp = pd.DataFrame({"feature": names, "importance": importance})

imp["feature"] = imp["feature"].str.replace("num__", "", regex=False)
imp["feature"] = imp["feature"].str.replace("binary__", "", regex=False)
imp["feature"] = imp["feature"].str.replace("cat__", "", regex=False)

imp["main_feature"] = imp["feature"]
imp.loc[imp["feature"].str.startswith("site_id_"), "main_feature"] = "site_id"

# combining all site dummies into one site_id importance
imp = imp.groupby("main_feature", as_index=False)["importance"].sum()
imp["importance_pct"] = 100 * imp["importance"] / imp["importance"].sum()
imp = imp.sort_values("importance", ascending=False)

imp.to_csv(p.res_dir / "rf_feature_importance.csv", index=False)
print(imp.head(20))

# plotting top features
plot_imp = imp.head(15).sort_values("importance_pct")

plt.figure(figsize=(9, 6))
plt.barh(plot_imp["main_feature"], plot_imp["importance_pct"])
plt.title("Random forest feature importance")
plt.xlabel("Importance (%)")
plt.ylabel("")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(p.fig_dir / "fig7_rf_feature_importance.png", dpi=200)
plt.close()
