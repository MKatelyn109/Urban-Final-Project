# Pedestrian–Bikeshare Nexus & Repositioning Engine

This project explores the relationship between **pedestrian traffic intensity** and **bikeshare usage** across space and time, and develops a **data-adaptive repositioning engine** to identify supply mismatches and accessibility gaps.

The work uses two deliberately contrasting datasets:

* **Washington, DC**: continuous, high-quality hourly pedestrian counters paired with Capital Bikeshare trip data
* **New York City**: sparse, bi-annual pedestrian count snapshots paired with Citi Bike data

NYC was intentionally included as a *challenging case* to test whether meaningful insights and a robust repositioning strategy can still be extracted from limited observational data.

---

## Project Goals

The project addresses three research questions:

1. **Correlation**
   How does pedestrian traffic intensity correlate with bikeshare usage across time and space?

2. **Latent Demand**
   Which high-pedestrian traffic zones exhibit disproportionately low bikeshare activity, suggesting latent or unmet demand?

3. **Accessibility Gaps**
   Which pedestrian-dense areas are weakly connected to the bikeshare network, indicating spatial accessibility gaps?

Building on these analyses, the project also develops a **repositioning engine** that recommends how bikeshare supply could be reallocated to better match pedestrian demand patterns, even under data sparsity.

---

## Repository Structure

```text
src/
├── preprocessing.py     # NYC + DC preprocessing and spatial joins
├── analysis.py          # Research question analysis + supporting plots
├── engine.py            # RepositioningMasterEngine
├── exploration/         # Rough notebooks and exploratory work (kept intentionally)
└── README.md
```

### A note on the `exploration/` folder

The `exploration/` directory contains **rough, exploratory notebooks** used during development.
These are intentionally kept to be transparent about the analytical process, intermediate checks, and design iterations.

The **core logic** (preprocessing, analysis, and the repositioning engine) lives in the `.py` files at the top level of `src/`.

---

## How the Code Is Organized

* **`preprocessing.py`**

  * Dataset-specific preprocessing for NYC and DC
  * Spatial and temporal joins between pedestrian counters and bikeshare stations
  * Identification of unmatched pedestrian locations (no nearby bikeshare)

* **`analysis.py`**

  * Functions directly tied to the three research questions
  * Additional diagnostic and descriptive analysis (distributions, segmented correlations, time-lag analysis, network inequality)
  * These functions are modular and can be run independently

* **`engine.py`**

  * A unified, data-adaptive repositioning engine
  * Designed to work with both continuous (DC) and sparse (NYC) datasets
  * Supports undersupply / oversupply ranking and coverage improvement evaluation

---

## Rough Instructions to Run the Code

This project is not packaged as an installable library.
The intended workflow is **script- or notebook-driven**.

The most honest work output it still in the `exploration/` folder, where you may see the plots. A nicer, cleaner version exists in `src/`, and can be used to showcase *how* we approached the work.

### 1. Set up the environment

```bash
pip install pandas numpy matplotlib seaborn scipy geopandas contextily scikit-learn
```

(Exact dependencies may vary depending on which plots are run.)

---

### 2. Preprocess a dataset (example: DC)

From a Python session or notebook:

```python
from preprocessing import preprocess_dc
from analysis import analyze_rq1_correlation, analyze_rq2_latent_demand
from engine import RepositioningMasterEngine

# Load raw data
dc_ped = ...
dc_bike = ...

# Run preprocessing
merged, unmatched = preprocess_dc(dc_ped, dc_bike, ped_coords=DC_PED_COORDS)
```

---

### 3. Run research question analyses

```python
# RQ1
r, p = analyze_rq1_correlation(merged)

# RQ2
latent = analyze_rq2_latent_demand(merged)

# RQ3 and supporting plots are available in analysis.py
```

---

### 4. Run the repositioning engine

```python
engine = RepositioningMasterEngine(
    merged,
    city_name="DC",
    unmatched_ped=unmatched
)

profiles = engine.build_location_profiles()
undersupplied = engine.rank_undersupplied()
oversupplied = engine.rank_oversupplied()
```

The engine is designed to work with **either NYC or DC outputs**, despite their very different data structures.

---

## Methodological Notes

* Spatial joins use **distance-based proximity** (meters approximated via latitude/longitude) rather than exact station matching.
* NYC analyses are intentionally constrained by data availability; conclusions are framed accordingly.
* DC enables richer temporal analyses (hourly patterns, time-lag correlations) that are not possible in NYC.
* The repositioning engine prioritizes **robustness over precision**, making conservative assumptions when data is sparse.

---

## The Report

Exists in this repo, uploaded to the root.  **[Open the full written report (PDF)](report.pdf)**

