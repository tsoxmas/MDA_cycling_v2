import geopandas
import contextily
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import paths as p


site = pd.read_csv(p.res_dir / "site_weather_resilience.csv")
site["site_id"] = site["site_id"].astype(str)

site["log_median_daily_count"] = np.log1p(site["median_daily_count"])

# variables used to compare the sites
cluster_vars = ["weather_resilience_index", "rain_penalty_pct", "rain_penalty_count", "log_median_daily_count",
                "rain_share", "commute_share", "station_km"]

x = site[cluster_vars].copy()


# fill missing values and scale everything before clustering
prep = Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())])

x_scaled = prep.fit_transform(x)

# remove a few unusual sites so they do not affect the clusters too much and compromises silhouette width
outlier_model = IsolationForest(contamination=0.05, random_state=p.seed)
keep_site = outlier_model.fit_predict(x_scaled) == 1
site["used_in_clustering"] = keep_site
site["cluster"] = np.nan
x_keep = x_scaled[keep_site]
site_keep = site[keep_site].copy()
print("sites before outlier removal:", len(site))
print("sites used for clustering:", len(site_keep))
print("sites treated as outliers:", len(site) - len(site_keep))


# reduce the scaled variables to four principal components (4 because they explain >80% of the variance and
# improve silhouette scores significantly as opposed to doing k-means on the raw data)
pca = PCA(n_components=4, random_state=p.seed)
x_pca = pca.fit_transform(x_keep)

print("PCA explained variance:")
for i, val in enumerate(pca.explained_variance_ratio_, start=1):
    print(f"PC{i}: {val:.3f}")

print("cumulative:", round(pca.explained_variance_ratio_.sum(), 3))

rows = []

# comparing different numbers of clusters
for k in range(2, 8):
    km = KMeans(n_clusters=k, n_init=50, random_state=p.seed)
    labels = km.fit_predict(x_pca)
    rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(x_pca, labels),
        "smallest_cluster": pd.Series(labels).value_counts().min(),
        "largest_cluster": pd.Series(labels).value_counts().max()})

k_table = pd.DataFrame(rows)
print(k_table)

# three clusters gave the most useful final groups with acceptable silhouette width (almost as good as k=2 solution)
km = KMeans(n_clusters=3, n_init=50, random_state=p.seed)
labels = km.fit_predict(x_pca)
site_keep["cluster"] = labels + 1
site.loc[keep_site, "cluster"] = site_keep["cluster"].values
site["cluster"] = site["cluster"].astype("Int64")

profile = site_keep.groupby("cluster")[cluster_vars].mean().reset_index()
profile["n_sites"] = site_keep.groupby("cluster")["site_id"].size().values

# save cluster labels and average profile for each cluster
site.to_csv(p.res_dir / "site_clusters.csv", index=False)
profile.to_csv(p.res_dir / "cluster_profiles.csv", index=False)

print("cluster profiles:")
print(profile)

# plot for silhouette scores for each tested k
plt.figure(figsize=(7, 4))
plt.plot(k_table["k"], k_table["silhouette"], marker="o")
plt.title("K-means silhouette scores")
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette score")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(p.fig_dir / "fig8_cluster_silhouette_scores.png", dpi=300)
plt.close()


# plot for the final clusters using the first two principal components
pc_plot = pd.DataFrame({"PC1": x_pca[:, 0],"PC2": x_pca[:, 1],"cluster": site_keep["cluster"].values})

plt.figure(figsize=(7, 5))
for cl in sorted(pc_plot["cluster"].unique()):
    part = pc_plot[pc_plot["cluster"] == cl]
    plt.scatter(part["PC1"], part["PC2"], label=f"Cluster {cl}", s=35)

plt.title("Final k-means clusters on first two PCs")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(p.fig_dir / "fig9_cluster_pca_scatter.png", dpi=300)
plt.close()

# plot for the final clusters on the map
heat = profile.set_index("cluster")[cluster_vars]
heat_z = (heat - heat.mean()) / heat.std(ddof=0)
plt.figure(figsize=(9, 4))
plt.imshow(heat_z, aspect="auto")
plt.xticks(range(len(cluster_vars)), cluster_vars, rotation=35, ha="right")
plt.yticks(range(len(heat_z.index)), [f"Cluster {i}" for i in heat_z.index])
plt.colorbar(label="standardized cluster mean")
plt.title("Cluster profiles")
plt.tight_layout()
plt.savefig(p.fig_dir / "fig10_cluster_profiles_heatmap.png", dpi=200)
plt.close()
map_df = site.dropna(subset=["cluster", "site_lat", "site_lon"]).copy()

gdf = geopandas.GeoDataFrame(map_df, geometry=geopandas.points_from_xy(map_df["site_lon"], map_df["site_lat"]),
                             crs="EPSG:4326").to_crs(epsg=3857)
colors = {1: "royalblue", 2: "salmon", 3: "lightgreen"}
fig, ax = plt.subplots(figsize=(13, 8))

xmin, ymin, xmax, ymax = gdf.total_bounds
pad_x = (xmax - xmin) * 0.08
pad_y = (ymax - ymin) * 0.12
ax.set_xlim(xmin - pad_x, xmax + pad_x)
ax.set_ylim(ymin - pad_y, ymax + pad_y)

contextily.add_basemap(ax, source=contextily.providers.CartoDB.Positron, zoom=10, attribution=False)

for cl in [1, 2, 3]:
    part = gdf[gdf["cluster"] == cl]
    part.plot(
        ax=ax,
        color=colors[cl],
        markersize=45,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.4,
        label=f"Cluster {cl}"
    )

ax.legend(loc="lower left", frameon=True)
ax.set_title("Cycling site clusters in Flanders", fontsize=18)
ax.set_axis_off()

plt.tight_layout()
plt.savefig(p.fig_dir / "fig11_cluster_map_flanders.png", dpi=400, bbox_inches="tight")
plt.close()
