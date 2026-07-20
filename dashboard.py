import contextily
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from shiny import App, reactive, render, ui
import paths as p


site = pd.read_csv(p.res_dir / "site_clusters.csv")
profile = pd.read_csv(p.res_dir / "cluster_profiles.csv")
site = site.dropna(subset=["site_lat", "site_lon", "cluster"]).copy()
site["cluster"] = site["cluster"].astype(int)

cluster_name = {1: "Highest average rain penalty", 2: "Lowest average rain penalty",
                3: "High rain exposure, moderate average penalty"}
cluster_focus = {1: "Main group to check first for drainage, surface quality, lighting and route continuity.",
                 2: "High cycling use and low estimated rain loss. Keep capacity and monitor as the stable group.",
                 3: "Small high-rain-exposure group. Check weather matching and local exposure before reading it too strongly."}
colors = {1: "royalblue", 2: "salmon", 3: "lightgreen"}

site["cluster_name"] = site["cluster"].map(cluster_name)
site["rain_loss_pct"] = 100 * site["rain_penalty_pct"].abs()
site["estimated_rain_loss"] = site["median_daily_count"] * site["rain_penalty_pct"].abs()

# map needs belgian coordinates projected for the basemap
site_gdf = geopandas.GeoDataFrame(site, geometry=geopandas.points_from_xy(site["site_lon"], site["site_lat"]),
                                  crs="EPSG:4326").to_crs(epsg=3857)

xmin, ymin, xmax, ymax = site_gdf.total_bounds
pad_x = (xmax - xmin) * 0.08
pad_y = (ymax - ymin) * 0.12


profile = profile.copy()
profile["cluster"] = profile["cluster"].astype(int)
profile["cluster_name"] = profile["cluster"].map(cluster_name)
profile["planning_focus"] = profile["cluster"].map(cluster_focus)
profile["rain_loss_pct"] = 100 * profile["rain_penalty_pct"].abs()
profile["typical_daily_count"] = np.expm1(profile["log_median_daily_count"])

cluster_table = profile[["cluster", "cluster_name", "planning_focus", "n_sites", "weather_resilience_index",
                         "rain_loss_pct", "typical_daily_count", "commute_share"]].copy()

cluster_table["commute_share"] = 100 * cluster_table["commute_share"]

cluster_table = cluster_table.rename(columns={"cluster": "Cluster", "cluster_name": "Meaning",
                                              "planning_focus": "Planning use", "n_sites": "Sites",
                                              "weather_resilience_index": "Weather resilience index",
                                              "rain_loss_pct": "Rain loss (%)",
                                              "typical_daily_count": "Typical daily cyclists",
                                              "commute_share": "Commute share (%)"})


def clean_table(df):
    html = df.to_html(index=False, classes="clean-table", border=0, justify="left", escape=False)
    return ui.HTML(html)


app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("Filters"),
        ui.input_checkbox_group("picked_clusters", "Clusters",
                                choices={"1": "1 — Highest average rain penalty",
                                         "2": "2 — Lowest average rain penalty",
                                         "3": "3 — High rain exposure, moderate average penalty"},
                                selected=["1", "2", "3"]),
        ui.input_slider("min_count", "Minimum typical daily cyclists", min=int(site["median_daily_count"].min()),
                        max=int(site["median_daily_count"].max()), value=int(site["median_daily_count"].min())),
        ui.hr(),
        ui.p("Weather resilience index: higher means the site keeps more cycling activity during rainy weather.",
             class_="metric-note"),
        ui.p("Rain loss: estimated percentage drop in cycling activity on rainy days.", class_="metric-note"),
        width=310
    ),

    ui.tags.style(
        """
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .small-note {
            color: #666;
            font-size: 0.95rem;
            margin-top: -8px;
            margin-bottom: 18px;
        }

        .clean-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.92rem;
        }

        .clean-table th {
            text-align: left;
            border-bottom: 1px solid #ddd;
            padding: 8px;
            background: #f7f7f7;
        }

        .clean-table td {
            border-bottom: 1px solid #eee;
            padding: 8px;
            vertical-align: top;
        }

        .metric-note {
            color: #666;
            font-size: 0.9rem;
        }
        """
    ),

    ui.h2("Cycling weather-resilience dashboard"),
    ui.p("The clusters describe average site profiles. The review table is sorted by absolute rainy-day loss, so high-volume sites can still appear high there.", class_="small-note"),

    ui.layout_columns(
        ui.value_box("Sites shown", ui.output_text("n_sites")),
        ui.value_box("Median resilience index", ui.output_text("median_resilience")),
        ui.value_box("Median rain loss", ui.output_text("median_rain_loss")),
        ui.value_box("Estimated rainy-day loss", ui.output_text("total_rain_loss")),
        col_widths=[3, 3, 3, 3]
    ),

    ui.card(ui.card_header("Cluster map"), ui.output_plot("cluster_map", height="620px")),

    ui.layout_columns(
        ui.card(ui.card_header("Cluster overview"), ui.output_ui("cluster_overview")),
        ui.card(ui.card_header("Sites to review first"), ui.output_ui("site_list")),
        col_widths=[6, 6]
    )
)


def server(input, output, session):

    @reactive.Calc
    def shown_sites():
        picked = [int(x) for x in input.picked_clusters()]

        # filtered data used by all cards and the map
        data = site_gdf[site_gdf["cluster"].isin(picked) & (site_gdf["median_daily_count"] >= input.min_count())].copy()

        return data

    @output
    @render.text
    def n_sites():
        return str(len(shown_sites()))

    @output
    @render.text
    def median_resilience():
        data = shown_sites()
        if len(data) == 0:
            return "-"
        return f"{data['weather_resilience_index'].median():.2f}"

    @output
    @render.text
    def median_rain_loss():
        data = shown_sites()
        if len(data) == 0:
            return "-"
        return f"{data['rain_loss_pct'].median():.1f}%"

    @output
    @render.text
    def total_rain_loss():
        data = shown_sites()
        if len(data) == 0:
            return "-"
        return f"{data['estimated_rain_loss'].sum():,.0f} cyclists"

    @output
    @render.plot
    def cluster_map():
        data = shown_sites()

        fig, ax = plt.subplots(figsize=(13, 8))

        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)

        # background map
        contextily.add_basemap(ax, source=contextily.providers.CartoDB.Positron, zoom=10, attribution=False)

        if len(data) == 0:
            ax.text(0.5, 0.5, "No sites selected", ha="center", va="center")
            ax.set_axis_off()
            return fig

        for cl in [1, 2, 3]:
            part = data[data["cluster"] == cl]

            if len(part) == 0:
                continue

            part.plot(
                ax=ax,
                color=colors[cl],
                markersize=50,
                alpha=0.9,
                edgecolor="black",
                linewidth=0.4
            )

        legend_items = [
            Line2D([0], [0], marker="o", color="w", label=f"Cluster {cl}", markerfacecolor=colors[cl],
                   markeredgecolor="black", markersize=8)
            for cl in [1, 2, 3]
            if cl in set(data["cluster"])
        ]

        ax.legend(handles=legend_items, loc="lower left", frameon=True)
        ax.set_title("Cycling site clusters in Flanders", fontsize=16)
        ax.set_axis_off()

        plt.tight_layout()
        return fig

    @output
    @render.ui
    def cluster_overview():
        picked = [int(x) for x in input.picked_clusters()]

        data = cluster_table[cluster_table["Cluster"].isin(picked)].copy()
        data["Weather resilience index"] = data["Weather resilience index"].round(2)
        data["Rain loss (%)"] = data["Rain loss (%)"].round(1)
        data["Typical daily cyclists"] = data["Typical daily cyclists"].round(0)
        data["Commute share (%)"] = data["Commute share (%)"].round(1)

        return clean_table(data)

    @output
    @render.ui
    def site_list():
        data = shown_sites().copy()
        data = data.sort_values("estimated_rain_loss", ascending=False).head(12)

        # these are the sites that probably need checking first
        site_cols = ["site_id", "cluster", "cluster_name", "median_daily_count", "rain_loss_pct",
                     "estimated_rain_loss", "weather_resilience_index", "commute_share"]
        data = data[site_cols].copy()

        data["commute_share"] = 100 * data["commute_share"]

        data = data.rename(columns={"site_id": "Site", "cluster": "Cluster", "cluster_name": "Cluster profile",
                                    "median_daily_count": "Typical daily cyclists",
                                    "rain_loss_pct": "Rain loss (%)",
                                    "estimated_rain_loss": "Estimated rainy-day loss",
                                    "weather_resilience_index": "Weather resilience index",
                                    "commute_share": "Commute share (%)"})

        data["Typical daily cyclists"] = data["Typical daily cyclists"].round(0)
        data["Rain loss (%)"] = data["Rain loss (%)"].round(1)
        data["Estimated rainy-day loss"] = data["Estimated rainy-day loss"].round(0)
        data["Weather resilience index"] = data["Weather resilience index"].round(2)
        data["Commute share (%)"] = data["Commute share (%)"].round(1)

        return clean_table(data)


app = App(app_ui, server)
