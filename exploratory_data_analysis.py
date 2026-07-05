# exploratory data analysis figures

import pandas as pd
import matplotlib.pyplot as plt
import paths as p

day = pd.read_parquet(p.res_dir / "site_day_panel.parquet")
month_avg = day.groupby("month")["total_count"].mean()
plt.figure(figsize=(7, 4))
plt.plot(month_avg.index, month_avg.values, marker="o")
plt.title("Average daily cycling volume by month")
plt.xlabel("Month")
plt.ylabel("Average daily cyclists per site")
plt.xticks(range(1, 13))
plt.tight_layout()
plt.savefig(p.fig_dir / "eda_monthly_volume.png", dpi=300)
plt.close()
