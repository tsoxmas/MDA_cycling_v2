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
num = ["log_rain", "rain_day", "heavy_rain_day", "rainy_slots", "heavy_slots", "temp_mean", "wind_mean", "gust_max",
       "hum_mean", "pressure_mean", "sun_min", "radiation_mean", "station_km", "dow", "month", "weekend",
    "commute_share", "holiday", "covid", "n_slots"]

# categorical variables, site_rain is for site-specific rain response
cat = ["site_id", "site_rain"]

# pipeline for imputing missing numeric weather values and scaling them for ridge
num_part = Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())])

# different preprocessing to numeric and categorical columns
prep = ColumnTransformer([("num", num_part, num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])

dates = sorted(df["date"].unique())

# time split so future days are not used in training
valid_day = dates[int(len(dates) * 0.70)]
test_day = dates[int(len(dates) * 0.80)]

# train/validation/test by date
train = df[df["date"] < valid_day]
valid = df[(df["date"] >= valid_day) & (df["date"] < test_day)]
test = df[df["date"] >= test_day]

# model inputs and logged target
x_train = train[num + cat]
y_train = train["y"]
x_valid = valid[num + cat]
y_valid = valid["y"]
x_test = test[num + cat]
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

