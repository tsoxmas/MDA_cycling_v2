# MDA cycling weather resilience

This project uses cycling counter data from Flanders to look at cycling volume, weather, and rain.

The raw data is 15-minute cycling counts joined with weather data. I turn it into a daily site-level table and then use it to estimate rain loss, compare sites, group them into clusters, and show the results on a map.

![Dashboard preview](200_d.gif)

## running the project

Run the scripts from the project folder.

```bash
python make_panel.py
python exploratory_data_analysis.py
python weather_resilience_model.py
python random_forest_model.py
python clustering_model.py
```

The dashboard comes after the clustering step:

```bash
shiny run --reload dashboard.py
```

## setup

Install the packages:

```bash
pip install -r requirements.txt
```

The raw joined data should be here:

```text
data/cycling_weather_full.parquet
```

Most scripts use paths from `paths.py`, so the file names and folders should stay the same unless `paths.py` is changed too.

## what each script does

`paths.py`

Shared paths, selected raw columns, timezone, and random seed. Other scripts import this file so the folders are not repeated everywhere.

`make_panel.py`

Makes the daily panel from the original 15-minute data. It converts UTC timestamps to Belgian local time, creates date/hour/month/weekend variables, adds rain indicators, and aggregates everything to one row per site per day.

It saves:

```text
results/site_day_panel.parquet
results/site_day_panel_preview.csv
```

`exploratory_data_analysis.py`

Creates the first summary tables and plots: missing values, cycling by hour/month, rain by hour, site differences, weather correlations, and the site map.

It saves EDA summaries in `results/` and plots in `figures/`.

`weather_resilience_model.py`

Fits the ridge regression model. The target is log daily cycling count, so very busy sites do not dominate the model too much.

The script uses a time split, tests a few ridge alpha values, chooses the best one on validation, then evaluates on the test period. It also estimates rain loss for each site by comparing predicted rainy days with the same rows treated as dry days.

This creates the main weather resilience output:

```text
results/site_weather_resilience.csv
```

and also saves ridge metrics and coefficients.

`random_forest_model.py`

Fits a random forest model on the same daily panel. It tests a few settings, picks the best one on validation, then saves final test metrics and feature importance.

`clustering_model.py`

Uses the site-level resilience table from the ridge model. It clusters sites based on resilience, rain penalty, volume, rain exposure, commute share, and distance to the nearest weather station.

The script removes some outlier sites first, uses PCA before k-means, compares different numbers of clusters, then saves the final cluster labels and cluster profiles.

`dashboard.py`

Small Shiny dashboard for looking at the final clusters. It shows the sites on a map, some summary boxes, cluster descriptions, and the sites with the highest estimated rainy-day loss.

## main results

Important result files:

```text
results/site_day_panel.parquet
results/eda_variable_summary.csv
results/eda_missing_summary.csv
results/model_metrics.csv
results/ridge_coefficients.csv
results/ridge_main_coefficients.csv
results/site_weather_resilience.csv
results/rf_validation_scores.csv
results/rf_model_metrics.csv
results/rf_feature_importance.csv
results/site_clusters.csv
results/cluster_profiles.csv
```

Important figures:

```text
figures/fig1_temporal_patterns.png
figures/fig2_rain_by_hour.png
figures/fig3_site_heterogeneity.png
figures/fig4_weather_correlations.png
figures/fig5_site_map.png
figures/fig6_ridge_main_coefficients_percent.png
figures/fig7_rf_feature_importance.png
figures/fig8_cluster_silhouette_scores.png
figures/fig9_cluster_pca_scatter.png
figures/fig10_cluster_profiles_heatmap.png
figures/fig11_cluster_map_flanders.png
```

## model idea

The ridge model is the main model for the weather resilience part. It includes weather, calendar variables, site effects, and site-specific rain response.

The random forest is used as a comparison model and for feature importance.

The clustering part uses the site-level outputs to make groups for the map and dashboard.
