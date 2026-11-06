# Crop-type classification: Sentinel-1 + Sentinel-2 temporal signatures

## What the data says (read this first)

`farm_boundaries_FINAL_Squares_27Oct_updated.shp` holds 5,910 labelled fields, but they come from **9 different seasons**.
Imagery has to match the season each crop was observed in, so the pipeline trains on one season at a time:

| season | fields | main crops |
|---|---|---|
| **Rabi 2022-23** (default) | 5,279 | Wheat 1258, Gram 1065, Onion 776, Sorghum 549, Maize 474, Cotton 325, Sugarcane 275 |
| Kharif 2026 | 376 | Maize, Sugarcane, Bajra (enough for a separate kharif model later) |
| others | <210 each | too few to train on |

After QC filters (`train_ok`, mono-crop only, non-crop labels removed), Rabi 2022-23 has **5,188 fields in 10 classes**: Wheat, Gram, Onion, Sorghum, Maize, Cotton, Sugarcane, Soybean, Brinjal, and Other (rare crops grouped).
The surveys ran 10 Jan to 16 Feb 2023, and sowing was mostly Oct to Nov 2022, so the time window is **1 Sep 2022 to 30 Apr 2023 in 10-day steps (24 dates)**. Starting in September captures the preceding kharif crop and land preparation.

Sentinel-2 L2A has **12 bands** (B1 to B9, B8A, B11, B12). There is no B13 to B15, and B10 (cirrus) isn't delivered in L2A.

## Why a *temporal* signature

On any single date, wheat, gram and onion look almost the same; all are green in January.
What sets them apart is the **shape of the season**: when each crop greens up, how high it peaks, and when it dries down (gram peaks earlier and lower, onion later, sugarcane stays green the whole time).
So each crop's "unique signature" here is its full time series of every feature:

```
long table (inspect / plot)                    wide table (model input)
field | date  | NDVI EVI NDMI ... VV VH B1..B12  ->  field | NDVI_t00..t23 | EVI_t00..t23 | ... | phenology metrics
```

The phenology metrics are peak NDVI, peak date, green-up and senescence rates, season length, and NDVI area under the curve.
Sentinel-1 (VV, VH, VH/VV, RVI) still sees through cloud and responds to crop structure, which helps tell sorghum and maize apart from wheat.

## The two sampling approaches

1. **Polygon**: shrink the field by 10 m to remove mixed edge pixels. The default (`polygon_mode: inner`) keeps the whole shrunk field, about 17 pixels. `square` keeps only the largest square inside it, about 5 pixels, because many fields are narrow strips.
2. **Point**: a 30 x 30 m square (3x3 pixels) around the GPS point, clipped to the farm polygon.

Both approaches are compared in `summary.txt`. The final model uses the polygon, because new fields come with boundaries rather than GPS points.

## Run it

One-time setup: `pip install -r requirements.txt`, then put your Google Cloud project id in `config.yaml` (`gee.project`).
The farm shapefile is not in git (it holds survey photos and surveyor names): unzip
`farm_boundaries_FINAL_Squares_27Oct_updated.zip` into `data/raw/`. Downloaded data and
`outputs/` are also kept out of git; `python run.py` recreates them.

**Train everything:**

```bash
python run.py
```

This one command builds the labels and both sampling geometries, downloads the Sentinel-1/2 data from Earth Engine (field averages for both approaches, plus every pixel inside each polygon), writes the crop signatures, trains and compares all models, trains the final ensemble, writes the label-review list, and prints a summary. A fresh run takes about 20 minutes; a rerun with nothing to download takes about 15.
Downloads are cached: if it stops partway, run it again and it continues. Re-running after a config change only re-downloads what the change affects.

**Classify new fields** (any polygon shapefile):

```bash
python run.py predict new_fields.shp
```

All settings (season, dates, buffer sizes, model parameters) live in `config.yaml`.

## Outputs (`outputs/`)

| file | what it is |
|---|---|
| `summary.txt` | Final accuracy, accuracy by confidence level, and every model side by side |
| `final/model.joblib` | **The model `predict` uses**: field + pixel ensemble |
| `final/test_predictions.csv` | Every test field: surveyed crop, predicted crop, confidence |
| `label_review.csv` | Fields whose satellite signature strongly contradicts the surveyed crop, with the survey photo link, for re-checking |
| `polygon/`, `point/` | One folder per sampling approach, holding the files below |
| `signatures.csv` | **The signature table**: one row per (crop, date), columns NDVI, EVI, NDMI, NDRE, SAVI, GNDVI, VV, VH, VH_VV, RVI, B1 to B12 (median) |
| `signatures.png` | Temporal profile of each crop, with the IQR shaded |
| `separability.csv` | Jeffries-Matusita distance for each feature and crop pair (2 = fully separable) |
| `report.txt`, `confusion.png` | Per-crop precision, recall and F1, and the confusion matrix on the test fields |
| `importance_by_feature.csv` | Which bands and indices the model relies on |
| `model.joblib` | Best single field-level model for that sampling |
| `predictions/` | Output of `predict`: `predictions.gpkg` (open in QGIS) and `predictions.csv`, each with `pred_crop` and `confidence` |

## Models

All models are scored on the same 1,604 test fields (the shapefile's `split` column).

**The final model is an ensemble of three, whose probabilities are averaged:**

- **field**: XGBoost on each field's median time series plus phenology metrics.
- **cnn**: TempCNN, a 1-D convolutional network that reads every pixel's 24-date curve. A field's pixels then vote.
- **pxgb**: XGBoost on every pixel's curve, with pixels voting per field.

Training uses up to 64 pixels per field, weighted so that each field, and each crop, counts equally.

**For comparison:** nearest-signature matching (your concept, implemented directly), Random Forest and XGBoost, on each sampling approach.

Set `model.spatial_cv: true` to add a cross-validation grouped by village, which estimates accuracy in villages the model has never seen.

## Accuracy, and why 90% is not reachable with the current labels

The ensemble reaches about **63% on all fields**, and **83-86% on the 30-40% of fields it is most confident about** (see `summary.txt`).
Every model we tried lands between 61% and 63%:

- 10-day and 15-day composites;
- a September or October start;
- square or whole-polygon sampling;
- field or pixel level;
- trees or a neural network;
- an ensemble of all three.

That pattern means the limit is the labels, not the model:

- **None of the Rabi 2022-23 labels are verified.** All are status "Pending" and quality "Non-QC".
- **About 17% of fields have a satellite curve that strongly matches a different crop** than the surveyed one. An example is a "wheat" field whose curve peaks in March like onion.
- **Even the fields the model is most sure about top out at about 85%.** The label is likely wrong for roughly 15% of fields, so no model can score above about 85-90% against these labels.

**Path to 90%:**
1. Re-check the fields in `label_review.csv` against their survey photos (`image_url`), starting from the top of the file, where the disagreement is strongest.
2. Fix or remove the wrong labels.
3. Rerun `python run.py`.

Getting the test fields right matters most, because they define the score.

Until then, use the `confidence` column: predictions with confidence of 0.8 or more are about 83% correct and cover about 40% of fields.

## Layout

```
config.yaml               all parameters
src/croptype/labels.py    shapefile -> clean labels (season, QC, class grouping)
src/croptype/geometry.py  inner-square and point-square sampling
src/croptype/gee_extract.py  Earth Engine S1/S2 composites -> long signature table
src/croptype/dataset.py   gap-fill, pivot to wide, phenology metrics
src/croptype/signatures.py  crop signatures, plots, JM separability, signature classifier
src/croptype/train.py     model comparison, metrics, importance
src/croptype/pixel_model.py  pixel features, gap filling, TempCNN
src/croptype/ensemble.py  final ensemble, confidence, label review
src/croptype/cli.py       the two commands (run, predict)
run.py                    entry point
tests/test_pipeline.py    offline end-to-end test (synthetic imagery)
```
