# one site-day table from the 15-minute cycling/weather data.

import numpy as np
import pandas as pd
import holidays
import paths as p

bike = pd.read_parquet(p.data_file, columns=p.cols)

bike["ts"] = pd.to_datetime(bike["ts"], utc=True, errors="coerce")
bike = bike.dropna(subset=["ts", "site_id", "count"]) # removing rows w/ missing data
local_time = bike["ts"].dt.tz_convert(p.tz) # converting time to BElgian time (tz = "Europe/Brussels")
bike["date"] = local_time.dt.tz_localize(None).dt.normalize()  #  calendar day
bike["hour"] = local_time.dt.hour  # local hour
bike["dow"] = local_time.dt.dayofweek  # Monday = 0, Sunday = 6
bike["month"] = local_time.dt.month  # month
bike["weekend"] = bike["dow"].isin([5, 6]).astype(int)  # Saturday/Sunday dummy
bike["commute"] = bike["hour"].isin([7, 8, 16, 17, 18]).astype(int)  # commute-hour dummy
bike["is_rain"] = (bike["precip_quantity"].fillna(0) > 0).astype(int)  # rain in a 15 min interval dummy
bike["is_heavy_rain"] = (bike["precip_quantity"].fillna(0) >= 1).astype(int)  # heavy rain dummy
bike["commute_count"] = bike["count"] * bike["commute"]  # count cyclists during commute hours

