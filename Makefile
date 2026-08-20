.PHONY: all extract streaming run run-forecast ms check-prose lint clean

all: extract run

# Extract accident.csv files from FARS zips
extract:
	@mkdir -p data/fars
	@for zip in data/raw/FARS*NationalCSV.zip; do \
		year=$$(echo $$zip | grep -o '[0-9]\{4\}'); \
		if [ ! -f "data/fars/accident_$$year.csv" ]; then \
			echo "Extracting $$year..."; \
			unzip -p "$$zip" "*/accident.csv" > "data/fars/accident_$$year.csv" 2>/dev/null || \
			unzip -p "$$zip" "*/ACCIDENT.CSV" > "data/fars/accident_$$year.csv" 2>/dev/null || \
			unzip -p "$$zip" "accident.csv" > "data/fars/accident_$$year.csv" 2>/dev/null || \
			echo "  Warning: No accident file found in $$zip"; \
		fi; \
		if [ ! -s "data/fars/miacc_$$year.csv" ]; then \
			unzip -C -p "$$zip" "*miacc.csv" > "data/fars/miacc_$$year.csv" 2>/dev/null; \
			if [ ! -s "data/fars/miacc_$$year.csv" ]; then \
				echo "  Warning: No MIACC file found in $$zip"; \
			fi; \
		fi; \
	done
	@echo "Extraction complete. Files in data/fars/:"
	@ls data/fars/*.csv 2>/dev/null | wc -l | xargs echo "  "

# Spotify Charts US top 200. Optional: the pipeline skips the streaming
# analyses when data/spotify/us_daily.csv is absent. Needs ~/.kaggle/kaggle.json.
streaming:
	@mkdir -p data/spotify
	@command -v kaggle >/dev/null 2>&1 || command -v uvx >/dev/null 2>&1 || { \
	  echo "Need the Kaggle CLI: pipx install kaggle  (or install uv)"; exit 1; }
	@test -f data/spotify/us_daily.csv || ( \
	  KG=$$(command -v kaggle || echo "uvx --from kaggle kaggle"); \
	  cd data/spotify && \
	  $$KG datasets download gonzalopezgil/spotify-charts-daily-updated \
	    -f charts_songs_daily.csv --unzip -q && \
	  python3 -c "import pandas as pd; \
d = pd.read_csv('charts_songs_daily.csv', low_memory=False); \
d[d['country'] == 'us'].to_csv('us_daily.csv', index=False)" && \
	  rm -f charts_songs_daily.csv )
	@wc -l < data/spotify/us_daily.csv | xargs echo "  us_daily.csv rows:"

# Run analysis (includes placebo tests)
run: extract
	python3 -m src.pipeline

# Run forecast-based estimator
run-forecast: extract
	python3 -c "from src.s01_load import load_local_fars; \
	from src.s02_preprocess import build_daily_series; \
	from src.s04_estimate import residualize; \
	from src.s06_specification import forecast_estimate, print_forecast_results, save_forecast_tables; \
	r = forecast_estimate(residualize(build_daily_series(load_local_fars('data/fars/')))); \
	print_forecast_results(r); save_forecast_tables(r)"

# Compile the manuscript (tables and figures come from tabs/ and figs/)
ms:
	cd ms && latexmk -pdf -interaction=nonstopmode ms.tex && latexmk -c

# Report prose numbers and tables that disagree with tabs/
check-prose:
	python3 -m src.check_prose

# Linting
lint:
	black --check src
	isort --check-only src
	flake8 src

format:
	black src
	isort src

# Clean extracted CSVs (keeps raw zips)
clean:
	rm -f data/fars/*.csv
	rm -f album_release_fatality_prediction.png
	rm -f fars_accident_cache.parquet
