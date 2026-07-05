# one site-day table from the 15-minute cycling/weather data.

import numpy as np
import pandas as pd
import holidays
import paths as p
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

bike = pd.read_parquet(p.data_file, columns=p.cols)

bike["ts"] = pd.to_datetime(bike["ts"], utc=True, errors="coerce")
bike = bike.dropna(subset=["ts", "site_id", "count"])  # removing rows w/ missing data
local_time = bike["ts"].dt.tz_convert(p.local_tz)  # converting time to BElgian time (tz = "Europe/Brussels")
bike["date"] = local_time.dt.tz_localize(None).dt.normalize()  # calendar day
bike["hour"] = local_time.dt.hour  # local hour
bike["dow"] = local_time.dt.dayofweek  # Monday = 0, Sunday = 6
bike["month"] = local_time.dt.month  # month
bike["weekend"] = bike["dow"].isin([5, 6]).astype(int)  # Saturday/Sunday dummy
bike["commute"] = bike["hour"].isin([7, 8, 16, 17, 18]).astype(int)  # commute-hour dummy
bike["is_rain"] = (bike["precip_quantity"].fillna(0) > 0).astype(int)  # rain in a 15 min interval dummy
bike["is_heavy_rain"] = (bike["precip_quantity"].fillna(0) >= 1).astype(int)  # heavy rain dummy
bike["commute_count"] = bike["count"] * bike["commute"]  # count cyclists during commute hours

# adding one row per day for each site
daily = bike.groupby(["site_id", "date"]).agg(
    total_count=("count", "sum"),  # daily cycling volume
    n_slots=("count", "size"),  # how many 15-minute rows were present
    commute_count=("commute_count", "sum"),  # cyclists count during commute hours
    precip_mm=("precip_quantity", "sum"),  # daily rain amount
    rainy_slots=("is_rain", "sum"),  # number of rain intervals
    heavy_slots=("is_heavy_rain", "sum"),  # number of heavy rain intervals
    temp_mean=("temp_dry_shelter_avg", "mean"),  # daily average temperature
    wind_mean=("wind_speed_10m", "mean"),  # daily average wind speed
    gust_max=("wind_gusts_speed", "max"),  # maximum gust that day
    hum_mean=("humidity_rel_shelter_avg", "mean"),  # daily average humidity
    pressure_mean=("pressure", "mean"),  # daily average atmosphetic pressure
    sun_min=("sun_duration", "sum"),  # number sunshine minutes
    radiation_mean=("short_wave_from_sky_avg", "mean"),  # daily average solar radiation
    station_km=("nearest_station_km", "mean"),  # distance to nearest weather station
    site_lat=("site_lat", "first"),  # coordinates
    site_lon=("site_lon", "first"),
    dow=("dow", "first"),  # weekday
    month=("month", "first"),  # ьщтер
    weekend=("weekend", "first"),  # weekend flag
).reset_index()

print(daily["wind_mean"].isna().mean())
print(daily["wind_mean"].describe())
numeric_cols = ["precip_mm", "temp_mean", "wind_mean", "gust_max", "hum_mean", "pressure_mean", "sun_min",
                "radiation_mean", "station_km"]
imputation = Pipeline([("imputer", SimpleImputer(strategy="median")), ])
daily[numeric_cols] = imputation.fit_transform(daily[numeric_cols])

daily["rain_day"] = (daily["precip_mm"].fillna(0) > 0).astype(int)  # day with any amount of rain dummy
daily["heavy_rain_day"] = (daily["precip_mm"].fillna(0) >= 5).astype(int)  # day with a heavy-ish amount of rain dummy
daily["commute_share"] = daily["commute_count"] / daily["total_count"].replace(0, np.nan)  # commuters share
years = sorted(daily["date"].dt.year.unique())
be_holidays = holidays.country_holidays("BE", years=years)  # Belgian holidays
daily["holiday"] = daily["date"].dt.date.isin(be_holidays).astype(int)  # holiday dummy
daily["summer_break"] = daily["month"].isin([7, 8]).astype(int)  # school break dummy
daily["covid"] = ((daily["date"] >= "2020-03-13") & (daily["date"] <= "2021-05-09")).astype(int)  # roughly covid period

daily.to_parquet(p.res_dir / "site_day_panel.parquet", index=False)
daily.head(50).to_csv(p.res_dir / "site_day_panel_preview.csv", index=False)
