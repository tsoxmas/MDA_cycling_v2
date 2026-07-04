# file with paths used by the other scripts
from pathlib import Path

here = Path(__file__).resolve().parent
data_dir = here / "data"
res_dir = here / "results"
fig_dir = here / "figures"
model_dir = here / "models"
data_file = data_dir / "cycling_weather_full.parquet"
local_tz = "Europe/Brussels"
seed = 67

for folder in [data_dir, res_dir, fig_dir, model_dir]: folder.mkdir(exist_ok=True)

