# exploratory data analysis figures

import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import seaborn as sns
import geopandas as gpd
import contextily as ctx
from matplotlib.colors import LogNorm
import paths as p
import numpy as np

sample_frac = 0.05


day = pd.read_parquet(p.res_dir / "site_day_panel.parquet")
day["log_rain"] = np.log1p(day["precip_mm"].fillna(0))

model_cols = ["total_count", "log_rain", "rain_day", "heavy_rain_day", "temp_mean", "wind_mean", "hum_mean",
              "pressure_mean", "radiation_mean", "station_km", "dow", "month", "commute_share", "n_slots", "weekend",
              "holiday", "covid"]

month_avg = day.groupby("month")["total_count"].mean()
plt.figure(figsize=(7, 4))
plt.plot(month_avg.index, month_avg.values, marker="o")
plt.title("Average daily cycling volume by month")
plt.xlabel("Month")
plt.ylabel("Average daily cyclists per site")
plt.xticks(range(1, 13))
plt.tight_layout()
plt.savefig(p.fig_dir / "fig0_monthly_volume.png", dpi=300)
plt.close()

rng = np.random.default_rng(p.seed)
parts = []
parquet_file = pq.ParquetFile(p.data_file)
for batch in parquet_file.iter_batches(batch_size=200_000, columns=p.cols):
    batch_df = batch.to_pandas()
    parts.append(batch_df.loc[rng.random(len(batch_df)) < sample_frac])

df = pd.concat(parts, ignore_index=True)
df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
df = df.dropna(subset=["ts"])
df["ts_local"] = df["ts"].dt.tz_convert(p.local_tz)
df["hour"] = df["ts_local"].dt.hour
df["dayofweek"] = df["ts_local"].dt.dayofweek
df["month"] = df["ts_local"].dt.month
df["is_weekend"] = df["dayofweek"] >= 5

summary = day[model_cols].describe(percentiles=[0.5, 0.95]).T.round(2)
summary["pct_missing"] = (day[model_cols].isna().mean() * 100).round(2).values
summary.to_csv(p.res_dir / "eda_variable_summary.csv")

missing = pd.DataFrame(
    {"n_missing": day[model_cols].isna().sum(),
     "pct_missing": (day[model_cols].isna().mean() * 100).round(2)}).sort_values("pct_missing", ascending=False)
missing.to_csv(p.res_dir / "eda_missing_summary.csv")

hourly = df.groupby(["hour", "is_weekend"], observed=True)["count"].mean().reset_index()

plt.figure(figsize=(13, 4.5))
plt.subplot(1, 2, 1)
for is_weekend, label, color in [
    (False, "Weekday", "#1f77b4"),
    (True, "Weekend", "#d62728"),
]:
    sub = hourly[hourly["is_weekend"] == is_weekend]
    plt.plot(sub["hour"], sub["count"], "o-", label=label, color=color, linewidth=2)
plt.xticks(range(0, 24, 2))
plt.xlabel("Hour (Brussels local time)")
plt.ylabel("Mean cyclists per 15 min")
plt.title("Daily rhythm")
plt.legend()

plt.subplot(1, 2, 2)
monthly = df.groupby("month")["count"].mean()
plt.bar(monthly.index, monthly.values, color="#dd8452")
plt.xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
plt.xlabel("Month")
plt.ylabel("Mean cyclists per 15 min")
plt.title("Seasonal rhythm")
plt.tight_layout()
plt.savefig(p.fig_dir / "fig1_temporal_patterns.png", bbox_inches="tight", dpi=180)
plt.close()

rain_by_hour = df.dropna(subset=["count", "precip_quantity"]).copy()
rain_by_hour["wet"] = rain_by_hour["precip_quantity"] > 0
rain_by_hour = rain_by_hour.groupby(["hour", "wet"])["count"].mean().reset_index()

plt.figure(figsize=(7, 4.5))
for wet, label, color in [(False, "Dry", "#1f77b4"), (True, "Any rain", "#d62728")]:
    sub = rain_by_hour[rain_by_hour["wet"] == wet]
    plt.plot(sub["hour"], sub["count"], "o-", label=label, color=color, linewidth=2)
plt.xticks(range(0, 24, 2))
plt.xlabel("Hour (Brussels local time)")
plt.ylabel("Mean cyclists per 15 min")
plt.title("Rain effect by hour")
plt.legend()
plt.tight_layout()
plt.savefig(p.fig_dir / "fig2_rain_by_hour.png", bbox_inches="tight", dpi=180)
plt.close()

site_stats = df.dropna(subset=["count", "precip_quantity"]).copy()
site_stats["wet"] = site_stats["precip_quantity"] > 0.1
site_stats = site_stats.groupby(["site_id", "wet"])["count"].mean().unstack("wet").dropna()
site_stats.columns = ["dry_mean", "wet_mean"]
site_stats["rain_drop_pct"] = (
    100 * (site_stats["wet_mean"] - site_stats["dry_mean"]) / site_stats["dry_mean"]
)
print("\nPer-site rain sensitivity:")
print(site_stats.round(2))

plt.figure(figsize=(13, 4.5))
plt.subplot(1, 2, 1)
plt.hist(site_stats["dry_mean"], bins=40, color="#4c72b0", edgecolor="white")
plt.xlabel("Mean cyclists per 15 min (dry intervals)")
plt.ylabel("Number of sites")
plt.title("Baseline volume across sites")

plt.subplot(1, 2, 2)
plt.hist(
    site_stats["rain_drop_pct"].clip(-100, 50),
    bins=40,
    color="#dd8452",
    edgecolor="white",
)
plt.axvline(
    site_stats["rain_drop_pct"].median(),
    color="black",
    linestyle="--",
    label=f"median: {site_stats['rain_drop_pct'].median():.1f}%",
)
plt.xlabel("Percent change in cycling, wet vs dry")
plt.ylabel("Number of sites")
plt.title("Rain sensitivity across sites")
plt.legend()
plt.tight_layout()
plt.savefig(p.fig_dir / "fig3_site_heterogeneity.png", bbox_inches="tight", dpi=180)
plt.close()

corr_cols = ["log_rain", "rain_day", "heavy_rain_day", "temp_mean", "wind_mean", "hum_mean", "pressure_mean",
             "radiation_mean"]
corr_matrix = day[corr_cols].corr()
plt.figure(figsize=(8, 6.5))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
)
plt.title("Correlations between weather features")
plt.tight_layout()
plt.savefig(p.fig_dir / "fig4_weather_correlations.png", bbox_inches="tight", dpi=180)
plt.close()

site_geo = (
    df.groupby("site_id")
    .agg(
        site_lat=("site_lat", "first"),
        site_lon=("site_lon", "first"),
        mean_count=("count", "mean"),
        nearest_station_km=("nearest_station_km", "first"),
    )
    .reset_index()
)
site_geo["mean_daily"] = site_geo["mean_count"] * 96

gdf = gpd.GeoDataFrame(
    site_geo,
    geometry=gpd.points_from_xy(site_geo["site_lon"], site_geo["site_lat"]),
    crs="EPSG:4326",
).to_crs(epsg=3857)

fig, axes = plt.subplots(1, 2, figsize=(18, 9))
gdf.plot(
    ax=axes[0],
    column="nearest_station_km",
    cmap="Reds",
    markersize=80,
    edgecolor="black",
    linewidth=0.7,
    legend=True,
    legend_kwds={
        "label": "km to nearest weather station",
        "shrink": 0.5,
        "orientation": "horizontal",
        "pad": 0.02,
    },
)
ctx.add_basemap(axes[0], source=ctx.providers.CartoDB.Positron, zoom=9)
axes[0].set_title(f"Distance to nearest weather station ({len(gdf)} sites)", fontsize=14)
axes[0].axis("off")

gdf.plot(
    ax=axes[1],
    column="mean_daily",
    cmap="Blues",
    markersize=80,
    edgecolor="black",
    linewidth=0.7,
    legend=True,
    norm=LogNorm(vmin=max(gdf["mean_daily"].min(), 1), vmax=gdf["mean_daily"].max()),
    legend_kwds={
        "label": "mean cyclists per day (log scale)",
        "shrink": 0.5,
        "orientation": "horizontal",
        "pad": 0.02,
    },
)
ctx.add_basemap(axes[1], source=ctx.providers.CartoDB.Positron, zoom=9)
axes[1].set_title("Mean daily cycling volume", fontsize=14)
axes[1].axis("off")

plt.tight_layout()
plt.savefig(p.fig_dir / "fig5_site_map.png", bbox_inches="tight", dpi=150)
plt.close()
