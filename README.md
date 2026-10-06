# Multimodal Healthcare Accessibility for Alaska and Hawaii

This repository contains a Python pipeline for computing travel times from census block groups (BGs) to healthcare points of interest (POIs) in Alaska (AK) and Hawaii (HI). Unlike contiguous U.S. states, AK and HI require multimodal routing — road travel alone leaves a substantial share of block groups with no reachable destination. This pipeline combines ArcGIS Network Analyst with commercial flight data (BTS T-100) to produce a three-layer accessibility model.

## Algorithm Overview

Travel time is computed in three layers, applied in order:

- **Layer 1 — Road only**: ArcGIS OD Cost Matrix using Esri StreetMap Premium. Block groups that reach a POI within the cutoff (180 min) are assigned here.
- **Layer 2 — Road + flight**: For block groups on road-connected islands or components that have access to a commercial airport, travel time is computed as drive-to-airport + flight time + drive-from-airport-to-POI. Flight times come from BTS T-100 segment data; missing AK airport pairs are filled by OLS interpolation on distance.
- **Layer 3 — Snap routing (AK only)**: For AK block groups unreachable by Layers 1–2 (e.g., remote bush communities), the nearest airport with commercial service is identified via Haversine distance, and a straight-line travel time estimate (`distance_miles / 70 × 60`) is used as the access leg to that airport.

Each block group is assigned the minimum travel time across whichever layer first resolves it.

## Requirements

- ArcGIS Pro with Network Analyst extension
- Esri StreetMap Premium routing network dataset
- Python packages: `arcpy`, `pandas`, `pyarrow`, `scikit-learn`, `numpy`, `openpyxl`

`arcpy` is bundled with ArcGIS Pro and cannot be installed via pip. All other packages can be installed with:

```
pip install pandas pyarrow scikit-learn numpy openpyxl
```

## Data Inputs

The following inputs are required but **not included** in this repository:

| File | Description | Source |
|------|-------------|--------|
| `us_bg` | U.S. census block group polygons and centroids | U.S. Census Bureau TIGER/Line |
| `aviation_facilities` | Airport point locations | FAA Aviation Facilities dataset (HIFLD) |
| `poi_advan_<NAICS>` | Healthcare POI points by NAICS code | Advan Research (licensed) |
| `T_T100D_SEGMENT_US_CARRIER_ONLY.csv` | BTS T-100 domestic segment data | [Bureau of Transportation Statistics](https://www.transtats.bts.gov/) |
| `arp_cy2024_commercial_service_enplanements.xlsx` | FAA commercial service enplanements | [FAA Airport Data](https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger/) |
| StreetMap Premium NDS | ArcGIS routing network | Esri (licensed) |

## NAICS Codes Processed

The pipeline runs the following 13 healthcare service types by default:

| NAICS Code | Description |
|------------|-------------|
| 446110 | Pharmacies and Drug Stores |
| 621111 | Offices of Physicians (except Mental Health) |
| 621320 | Offices of Optometrists |
| 621340 | Offices of Physical Therapists |
| 621420 | Outpatient Mental Health Centers |
| 621492 | Kidney Dialysis Centers |
| 621493 | Freestanding Ambulatory Surgical Centers |
| 621498 | Other Outpatient Care Centers |
| 621511 | Medical Laboratories |
| 621512 | Diagnostic Imaging Centers |
| 622110 | General Medical and Surgical Hospitals |
| 622310 | Specialty Hospitals |
| 812191 | Diet and Weight Reducing Centers |

## Configuration

Open `01_ak_hi_accessibility.py` and set the following variables before running:

```python
WD     = r"C:\path\to\your\project"   # project root directory
NDS    = r"C:\path\to\Routing_ND"     # StreetMap Premium network dataset
```

All other paths are derived from `WD`. Adjust `NAICS_CODES`, `IMPEDANCE_CUTOFF` (default 180 min), and `AIRPORT_DRIVE_CUTOFF` (default 60 min) as needed.

## Output

Results are saved as Parquet files under `output/<NAICS_CODE>/parquet/`:

| File | Description |
|------|-------------|
| `AK_result_<NAICS>.parquet` | Per-BG minimum travel time, AK, one NAICS code |
| `HI_result_<NAICS>.parquet` | Per-BG minimum travel time, HI, one NAICS code |
| `AK_result_all.parquet` | Combined AK results, all NAICS codes |
| `HI_result_all.parquet` | Combined HI results, all NAICS codes |

Each file contains: `OriginName` (12-digit BG FIPS), `DestinationName` (POI PLACEKEY), `travel_time_min`, `status`, `method` (road / multimodal / snap), `layer` (1 / 2 / 3), `NAICS_CODE`.

## Acknowledgement
This work used analytical resources developed and disseminated by the Gateway Exposome Coordinating Center (GECC) (https://gatewayexposome.org/). GECC resources are available through the GECC GitHub organization (https://github.com/GatewayExposomeCoordinatingCenter). The GECC is funded by the National Institute on Aging (NIA), under award U24AG088894.

## Citation

If you use this code in your work, please cite it:

> Kan, W., DeJohn, A., Kim, K.(2026). A Three-Layer Multimodal Accessibility Framework for Island and Remote Settings. Zenodo. https://doi.org/10.5281/zenodo.23191711

## License

MIT
