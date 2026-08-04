#!/bin/sh
set -eu

DATA_FILE="/app/data/cycling_weather_full.parquet"

if [ ! -f "$DATA_FILE" ]; then
    echo "Missing input file: $DATA_FILE" >&2
    echo "Mount data/cycling_weather_full.parquet at that path when starting the container." >&2
    exit 1
fi

run_step() {
    script="$1"
    echo "==> Running $script"
    python "$script"
}

run_step make_panel.py
run_step exploratory_data_analysis.py
run_step weather_resilience_model.py
# run_step random_forest_model.py
run_step clustering_model.py

for output in results/site_clusters.csv results/cluster_profiles.csv; do
    if [ ! -s "$output" ]; then
        echo "Pipeline did not create required dashboard input: $output" >&2
        exit 1
    fi
done

echo "==> Starting dashboard on http://0.0.0.0:${PORT}"
exec shiny run --host 0.0.0.0 --port "$PORT" dashboard.py
