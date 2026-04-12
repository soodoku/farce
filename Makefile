.PHONY: all extract run run-forecast clean

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
	done
	@echo "Extraction complete. Files in data/fars/:"
	@ls data/fars/*.csv 2>/dev/null | wc -l | xargs echo "  "

# Run analysis (includes placebo tests)
run: extract
	python3 -m src.pipeline

# Run forecast-based estimator
run-forecast: extract
	python3 -m src.s06_forecast

# Clean extracted CSVs (keeps raw zips)
clean:
	rm -f data/fars/*.csv
	rm -f album_release_fatality_prediction.png
	rm -f fars_accident_cache.parquet
