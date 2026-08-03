"""
Village Forecast — Combined Per-Grid Train + Forecast (light on render)
=======================================================================
Single-shot pipeline that replaces the two-step workflow:

    pergrid_train.py            -> ./models_pergrid/{var}/grid_*.joblib
    village_forecast_pergrid.py -> reads ./models_pergrid, forecasts

Instead of training ALL grids and saving them to disk, this script:

  1. Takes a single location from CLI (--lat / --lon)
  2. Finds the 4 surrounding 0.25 deg grid points
  3. Loads ONLY those 4 grids' rows from the (huge) parquets, using a
     pyarrow predicate-pushdown filter so ~58M-row files are never fully
     read into memory  ("light on render")
  4. Trains the 4 per-grid models IN MEMORY (nothing written to ./models_pergrid)
  5. Fetches the Open-Meteo forecast for the location, with the same
     paid -> free API fallback on an expired / invalid / over-quota key
  6. Applies inverse-distance-weighted (IDW) correction from the 4 models
  7. Writes the corrected village forecast CSV (+ optional parquet)

Because models are never saved and only 4 grids are ever touched, memory
and CPU stay tiny regardless of how large the source parquets are.

Usage
-----
    python village_forecast_combined.py \
        --lat 22.3 --lon 72.6 \
        --era5_dir ./training_data \
        --forecast ./forecast_data/om_forecast_all.parquet \
        --imd_dir ./imd_processed \
        --elev ./ml_ready/grid_elevation.parquet \
        --apikey XLv82nM2BPBe6qVe \
        --output_dir ./out

The --forecast parquet is only used for TRAINING (past forecast-vs-IMD
pairs). The live forecast for the target location is fetched from
Open-Meteo at run time.
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import requests
from datetime import date, timedelta
import xgboost as xgb

warnings.filterwarnings("ignore")


# -- Config --------------------------------------------------------------------

RAIN_THRESHOLD = 1.0
GRID_STEP = 0.25
GRID_OFFSET = 0.125

# -- Cross-validation (replaces the single 2025+ chronological holdout) --------
# Skill is now reported via BLOCKED-BY-YEAR K-fold CV: each fold holds out one
# or more whole calendar years, trains on the rest, and evaluates on the held-
# out year(s). Blocking by year (not random rows) prevents leakage from temporal
# autocorrelation and from rolling/climatology features that would otherwise let
# near-adjacent days sit in both train and test. The FINAL production model that
# generates the live forecast is trained on the WHOLE timeseries — CV is only
# for honest skill estimation.
CV_N_FOLDS = 5                 # target number of year-blocks (<= available years)

# -- Per-variable training source ---------------------------------------------
# The global --train_source flag sets the default, but ERA5 helps some variables
# and hurts others: ERA5's tmin (and tmax) bias vs IMD drags an already-accurate
# raw temperature the wrong way, while precip benefits from the extra rows. This
# map overrides the flag per variable. Set an entry to None to fall back to the
# CLI flag. Values: "om", "era5", or "both".
PER_VAR_TRAIN_SOURCE = {"tmax": "om", "tmin": "om", "pcp": "both"}

# -- Tail blend for precip (ratio model + QM on heavy days) --------------------
# The anchored ratio model is the per-day corrector, but across all wet days it
# mean-reverts and UNDER-corrects the heavy tail (corrected P90 < IMD P90). QM
# fixes the tail magnitude but misplaces it day-to-day. So on heavy raw days we
# blend the two: below PCP_BLEND_LO_MM use the ratio model alone; above
# PCP_BLEND_HI_MM use PCP_BLEND_QM_WEIGHT of QM; linear ramp in between. This
# lifts the tail toward IMD's P90 while keeping ratio-model per-day skill on
# ordinary days. Set PCP_BLEND_QM_WEIGHT = 0 to disable the blend entirely.
PCP_BLEND_LO_MM = 15.0
PCP_BLEND_HI_MM = 30.0
PCP_BLEND_QM_WEIGHT = 0.5

# -- Empirical (piecewise) quantile mapping ------------------------------------
# Instead of one smooth OM->IMD quantile curve, the empirical map is built band
# by band so the heavy tail is mapped on its own resolution. Bands are given as
# quantile edges over WET days (om>=1 & imd>=1): deciles up to the 80th, then a
# finer split in the tail (80-90-95-100) where the extreme bias lives.
QM_BAND_EDGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]

# --- Precipitation correction (anchored, multiplicative) ---------------------
# The precip corrector no longer predicts an absolute value that can collapse
# on extremes. It learns a log correction ratio log(imd / om) and multiplies
# the RAW forecast by exp(pred). The ratio is clipped so the corrected value
# can never wander too far from the raw forecast — only nudged by the learned
# historical bias. Bounds are in multiplicative space:
#   PCP_RATIO_MIN = 0.5  -> corrected can drop to at most 50% of raw
#   PCP_RATIO_MAX = 2.0  -> corrected can rise to at most 200% of raw
# --- Precipitation correction (anchored, multiplicative) ---------------------
# The precip corrector no longer predicts an absolute value that can collapse
# on extremes. It learns a log correction ratio log(imd / om) and multiplies
# the RAW forecast by exp(pred). The ratio is clipped so the corrected value
# can never wander too far from the raw forecast — only nudged by the learned
# historical bias.
#
# The MAX multiplier is REGIME-DEPENDENT, not flat. Analysis of extreme days
# shows OM often under-forecasts heavy rain by 3-6x while ordinary/drizzle days
# need only small tweaks. A single flat cap forces a bad trade-off: tight enough
# for drizzle means it strangles genuine extremes; loose enough for extremes
# means drizzle days get over-inflated. So the cap scales with the raw forecast:
#   - near RAIN_THRESHOLD  -> cap = PCP_RATIO_MAX_LOW  (stay anchored)
#   - at/above PCP_HEAVY_MM -> cap = PCP_RATIO_MAX_HIGH (let extremes correct)
# interpolated linearly in between. The floor (min multiplier) stays flat.
PCP_RATIO_MIN = 0.5          # corrected can drop to at most 50% of raw
PCP_RATIO_MAX_LOW = 2.0      # cap when raw forecast is light (~RAIN_THRESHOLD)
PCP_RATIO_MAX_HIGH = 6.0     # cap when raw forecast is already heavy
PCP_HEAVY_MM = 20.0          # raw mm at which the cap reaches PCP_RATIO_MAX_HIGH
PCP_LOG_RATIO_MIN = float(np.log(PCP_RATIO_MIN))
# Training-target clip uses the HIGH cap so the model is free to learn large
# ratios; the regime cap is then applied at apply/eval time based on raw mm.
PCP_LOG_RATIO_MAX = float(np.log(PCP_RATIO_MAX_HIGH))


def _pcp_max_log_ratio(om):
    """
    Regime-dependent upper log-ratio cap, as a function of the raw forecast (mm).

    Ramps linearly from PCP_RATIO_MAX_LOW at om=RAIN_THRESHOLD to
    PCP_RATIO_MAX_HIGH at om>=PCP_HEAVY_MM. Below RAIN_THRESHOLD the low cap is
    used (those rows aren't scaled anyway). Returned in log space, elementwise.
    """
    om = np.asarray(om, dtype=float)
    span = max(PCP_HEAVY_MM - RAIN_THRESHOLD, 1e-6)
    frac = np.clip((om - RAIN_THRESHOLD) / span, 0.0, 1.0)
    max_ratio = PCP_RATIO_MAX_LOW + frac * (PCP_RATIO_MAX_HIGH - PCP_RATIO_MAX_LOW)
    return np.log(max_ratio)

# Extreme-precip skill reporting: MAE is also reported on days where IMD is at
# or above this percentile (computed over the wet test days), so the skill line
# reflects performance in the heavy-rain regime that matters most.
PCP_EXTREME_PCTL = 90.0

OUTPUT_PAST_DAYS = 85           # past days to include in the corrected output
ROLL_BUFFER = 7                 # extra past days fetched only to warm up om_roll7
MAX_FORECAST_FILES = 100        # maximum number of forecast files to keep in output_dir

HIST_FORECAST_URL = "https://customer-historical-forecast-api.open-meteo.com/v1/forecast"
LIVE_FORECAST_URL = "https://customer-api.open-meteo.com/v1/forecast"
FREE_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FREE_ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
ELEVATION_URL     = "https://api.opentopodata.org/v1/srtm90m"
DAILY_PARAMS      = "temperature_2m_max,temperature_2m_min,precipitation_sum"

# Lighter XGBoost params for per-grid models (~14k samples each)
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "random_state": 42,
    "verbosity": 0,
}

# Features (no lat/lon — each model is grid-specific)
TEMP_FEATURES = [
    "om", "elevation",
    "doy_sin", "doy_cos", "month",
    "season_monsoon", "season_premonsoon", "season_postmonsoon", "season_winter",
    "om_roll7", "om_clim_anom",
    "source",  # 0=ERA5, 1=forecast
]
RAIN_CLS_FEATURES = TEMP_FEATURES + ["rain_om"]
RAIN_REG_FEATURES = ["om_log"] + TEMP_FEATURES


# ==============================================================================
# GRID GEOMETRY
# ==============================================================================

def surrounding_grid_points(vlat, vlon):
    """Return the 4 surrounding 0.25-deg grid centres for a location."""
    base_lat = np.floor((vlat - GRID_OFFSET) / GRID_STEP) * GRID_STEP + GRID_OFFSET
    base_lon = np.floor((vlon - GRID_OFFSET) / GRID_STEP) * GRID_STEP + GRID_OFFSET
    pts = []
    for dlat in (0, GRID_STEP):
        for dlon in (0, GRID_STEP):
            glat = round(base_lat + dlat, 4)
            glon = round(base_lon + dlon, 4)
            dist = float(np.sqrt((vlat - glat) ** 2 + (vlon - glon) ** 2))
            pts.append((glat, glon, dist))
    pts.sort(key=lambda x: x[2])
    return pts  # list of (glat, glon, dist) nearest-first


# ==============================================================================
# FILTERED PARQUET READ  (the "light on render" core)
# ==============================================================================

def _read_parquet_grids(path, grid_pairs, columns=None):
    """
    Read only the rows matching the requested (lat, lon) grid pairs.

    Uses pyarrow predicate pushdown when available so that a ~58M-row parquet
    is never fully materialised. Falls back to a chunked pandas scan if
    pyarrow filtering is unavailable.

    grid_pairs : iterable of (lat, lon) floats (already rounded to grid centres)
    """
    if not os.path.isfile(path):
        return None

    lats = sorted({round(float(la), 4) for la, _ in grid_pairs})
    lons = sorted({round(float(lo), 4) for _, lo in grid_pairs})
    # Bounding box with a small pad so float32/float64 storage drift can't
    # exclude a grid centre that is nominally in range.
    pad = 0.01
    lat_lo, lat_hi = min(lats) - pad, max(lats) + pad
    lon_lo, lon_hi = min(lons) - pad, max(lons) + pad

    # --- Preferred path: pyarrow dataset with a pushdown RANGE filter ---
    try:
        import pyarrow.dataset as ds
        import pyarrow.compute as pc

        dataset = ds.dataset(path, format="parquet")
        flt = ((pc.field("lat") >= lat_lo) & (pc.field("lat") <= lat_hi) &
               (pc.field("lon") >= lon_lo) & (pc.field("lon") <= lon_hi))
        table = dataset.to_table(filter=flt, columns=columns)
        df = table.to_pandas()
    except Exception:
        # --- Fallback: chunked read with range filtering ---
        df = _read_parquet_grids_fallback(path, lat_lo, lat_hi, lon_lo, lon_hi,
                                          columns)
        if df is None:
            return None

    if df.empty:
        return df

    # Snap each row to the nearest requested grid centre; keep only rows within
    # half a grid step of a requested centre (handles precision drift and any
    # sub-grid offset between the OM / IMD / ERA products).
    tol = GRID_STEP / 2.0 + 1e-6
    df = df.copy()
    lat_arr = df["lat"].to_numpy(dtype=float)
    lon_arr = df["lon"].to_numpy(dtype=float)
    lats_np = np.array(lats)
    lons_np = np.array(lons)
    nearest_lat = lats_np[np.abs(lat_arr[:, None] - lats_np[None, :]).argmin(axis=1)]
    nearest_lon = lons_np[np.abs(lon_arr[:, None] - lons_np[None, :]).argmin(axis=1)]
    keep = (np.abs(lat_arr - nearest_lat) <= tol) & \
           (np.abs(lon_arr - nearest_lon) <= tol)
    df["lat"] = np.round(nearest_lat, 4)
    df["lon"] = np.round(nearest_lon, 4)
    df = df.loc[keep].reset_index(drop=True)

    # Final refine to the exact requested (lat, lon) pairs.
    wanted = {(round(float(la), 4), round(float(lo), 4)) for la, lo in grid_pairs}
    mask = [(la, lo) in wanted for la, lo in zip(df["lat"], df["lon"])]
    return df.loc[mask].reset_index(drop=True)


def _read_parquet_grids_fallback(path, lat_lo, lat_hi, lon_lo, lon_hi, columns):
    """Chunked fallback when pyarrow dataset filtering is unavailable."""
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path, memory_map=False)
        frames = []
        for batch in pf.iter_batches(batch_size=1_000_000, columns=columns):
            chunk = batch.to_pandas()
            chunk = chunk[(chunk["lat"] >= lat_lo) & (chunk["lat"] <= lat_hi) &
                          (chunk["lon"] >= lon_lo) & (chunk["lon"] <= lon_hi)]
            if not chunk.empty:
                frames.append(chunk)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    except Exception:
        # Last resort: full pandas read (heavy, but correct)
        df = pd.read_parquet(path, columns=columns, memory_map=False)
        return df[(df["lat"] >= lat_lo) & (df["lat"] <= lat_hi) &
                  (df["lon"] >= lon_lo) & (df["lon"] <= lon_hi)]


# ==============================================================================
# TRAINING DATA (filtered to nearby grids only)
# ==============================================================================

def build_grid_training(era5_dir, forecast_path, imd_dir, elev_path,
                        var_key, grid_pairs, train_source="om"):
    """
    Build per-grid training frames for ONLY the requested grid points.
    Returns {(lat, lon): {"train", "test"}}.

    train_source controls which sources feed the model:
      "om"   -> OM forecast-history vs IMD only (source-matched to apply target)
      "era5" -> ERA5 vs IMD only
      "both" -> ERA5 + OM (original behaviour; can bias low-error vars like tmin)
    Note: the 2025+ TEST split always uses OM rows, since OM is the apply target.
    """
    # -- Elevation (small file, read whole then filter) --
    elev_dict = {}
    if os.path.isfile(elev_path):
        elev_df = pd.read_parquet(elev_path, memory_map=False)
        elev_dict = dict(zip(
            zip(elev_df["lat"].round(4), elev_df["lon"].round(4)),
            elev_df["elevation"],
        ))

    # -- ERA5 training (filtered) — only if requested --
    era5 = pd.DataFrame(columns=["lat", "lon", "date", "om", "imd", "source"])
    if train_source in ("era5", "both"):
        era5_file = os.path.join(era5_dir, f"training_{var_key}_daily.parquet")
        era5 = _read_parquet_grids(era5_file, grid_pairs,
                                   columns=["lat", "lon", "date", "imd", "om"])
        if era5 is None or era5.empty:
            era5 = pd.DataFrame(columns=["lat", "lon", "date", "om", "imd", "source"])
        else:
            era5["date"] = pd.to_datetime(era5["date"]).dt.normalize()
            era5 = era5.dropna(subset=["imd", "om"])
            era5["source"] = 0

    # -- Forecast history (filtered), paired with IMD --
    fc = _read_parquet_grids(
        forecast_path, grid_pairs,
        columns=["lat", "lon", "date", var_key, "is_forecast"])
    fc_imd = pd.DataFrame(columns=["lat", "lon", "date", "om", "imd", "source"])
    if fc is not None and not fc.empty:
        fc["date"] = pd.to_datetime(fc["date"]).dt.normalize()
        fc["is_forecast"] = pd.to_numeric(fc["is_forecast"], errors="coerce")
        n_fc_all = len(fc)
        fc = fc[fc["is_forecast"] == 0]
        n_fc_past = len(fc)
        fc = fc.rename(columns={var_key: "om_fc"})
        fc = fc[["lat", "lon", "date", "om_fc"]].copy()
        fc["om_fc"] = pd.to_numeric(fc["om_fc"], errors="coerce")
        fc = fc.dropna(subset=["om_fc"])
        n_fc_valid = len(fc)

        imd_file = os.path.join(imd_dir, f"imd_{var_key}_daily.parquet")
        imd = _read_parquet_grids(imd_file, grid_pairs,
                                  columns=["lat", "lon", "date", "value"])
        n_imd = 0 if (imd is None or imd.empty) else len(imd)
        if imd is not None and not imd.empty:
            imd["date"] = pd.to_datetime(imd["date"]).dt.normalize()
            imd = imd.rename(columns={"value": "imd_fc"})
            imd["imd_fc"] = pd.to_numeric(imd["imd_fc"], errors="coerce")
            fc["lat"] = fc["lat"].round(4); fc["lon"] = fc["lon"].round(4)
            imd["lat"] = imd["lat"].round(4); imd["lon"] = imd["lon"].round(4)
            merged = fc.merge(imd[["lat", "lon", "date", "imd_fc"]],
                              on=["lat", "lon", "date"], how="inner")
            merged = merged.dropna(subset=["om_fc", "imd_fc"])
            merged = merged.rename(columns={"om_fc": "om", "imd_fc": "imd"})
            merged["source"] = 1
            fc_imd = merged[["lat", "lon", "date", "om", "imd", "source"]]

    # -- Combine --
    combined = pd.concat(
        [era5[["lat", "lon", "date", "om", "imd", "source"]], fc_imd],
        ignore_index=True)
    if combined.empty:
        return {}
    combined = combined.sort_values(["lat", "lon", "date"]).reset_index(drop=True)
    combined["date"] = pd.to_datetime(combined["date"])
    # Force numeric dtype — filtered/concat reads can yield object-dtype columns
    combined["om"] = pd.to_numeric(combined["om"], errors="coerce")
    combined["imd"] = pd.to_numeric(combined["imd"], errors="coerce")
    combined = combined.dropna(subset=["om", "imd"]).reset_index(drop=True)

    # -- Features (identical to pergrid_train.py) --
    doy = combined["date"].dt.dayofyear
    combined["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    combined["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    combined["month"] = combined["date"].dt.month
    combined["season_monsoon"]    = combined["month"].isin([6, 7, 8, 9]).astype(np.int8)
    combined["season_premonsoon"] = combined["month"].isin([3, 4, 5]).astype(np.int8)
    combined["season_postmonsoon"]= combined["month"].isin([10, 11]).astype(np.int8)
    combined["season_winter"]     = combined["month"].isin([12, 1, 2]).astype(np.int8)

    combined["lat_r"] = combined["lat"].round(4)
    combined["lon_r"] = combined["lon"].round(4)
    combined["elevation"] = combined.apply(
        lambda r: elev_dict.get((r["lat_r"], r["lon_r"]), 0.0), axis=1)

    combined["om_roll7"] = (
        combined.groupby(["lat", "lon"])["om"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean()))
    om_clim = combined.groupby(["lat", "lon", "month"])["om"].transform("mean")
    combined["om_clim_anom"] = combined["om"] - om_clim

    if var_key == "pcp":
        combined["rain_om"] = (combined["om"] >= RAIN_THRESHOLD).astype(np.int8)
        combined["om_log"] = np.log1p(combined["om"].clip(lower=0))

    # -- Split into per-grid dict: FULL timeseries per grid --
    # No chronological holdout anymore. The whole OM+ERA5 timeseries is kept per
    # grid; blocked-by-year CV (see cv_evaluate_pcp) carves out folds for skill,
    # and the final production model trains on everything. A `year` column is
    # attached to drive year-blocked folding.
    grid_data = {}
    for (lat, lon), gdf in combined.groupby(["lat", "lon"]):
        gdf = gdf.copy()
        gdf["year"] = gdf["date"].dt.year
        grid_data[(round(lat, 4), round(lon, 4))] = {"full": gdf}
    return grid_data


# ==============================================================================
# IN-MEMORY MODEL TRAINING  (no disk writes)
# ==============================================================================

def _apply_pcp_correction(om, cls_pred, log_ratio):
    """
    Anchored multiplicative precip correction.

    Starts from the RAW forecast and only adjusts it:
      - classifier can zero out a raw wet day it judges to be spurious drizzle
      - regressor supplies a log correction ratio; corrected = om * exp(ratio),
        with the ratio clipped between PCP_RATIO_MIN and a REGIME-DEPENDENT max
        (bigger for heavier raw forecasts) so genuine extremes can correct hard
        while drizzle days stay tightly anchored.
    """
    pred = np.array(om, dtype=float, copy=True)
    above = om >= RAIN_THRESHOLD
    # classifier only allowed to *remove* a raw wet day (spurious drizzle),
    # never to invent rain the raw forecast didn't have.
    pred[above & (cls_pred == 0)] = 0.0
    keep = above & (cls_pred == 1)
    if keep.any():
        # Regime-dependent cap: heavy raw days may correct up to
        # PCP_RATIO_MAX_HIGH, light days only up to PCP_RATIO_MAX_LOW.
        hi_cap = _pcp_max_log_ratio(om[keep])
        r = np.clip(log_ratio[keep], PCP_LOG_RATIO_MIN, hi_cap)
        pred[keep] = om[keep] * np.exp(r)
    return pred


# ------------------------------------------------------------------------------
# Empirical (piecewise) quantile mapping
# ------------------------------------------------------------------------------

def build_empirical_qm(om_train, imd_train, band_edges=QM_BAND_EDGES):
    """
    Build a piecewise OM->IMD quantile map from WET training pairs.

    For each quantile band (e.g. 80-90th, 90-95th, 95-100th) we record the OM
    edge values and the matching IMD quantile values. Applying the map means:
    locate a new OM value's quantile within OM's wet distribution, then read the
    IMD value at that same quantile. Building it band-by-band (deciles + a finer
    tail split) lets the heavy tail be mapped on its own resolution instead of
    being smoothed away by a single global interpolation.

    Returns a dict {om_q, imd_q, edges} or None if too few wet pairs.
    NOTE: quantile mapping corrects the *distribution*, not individual days —
    it cannot know which specific day is heavy, only that a given fraction of
    days should be. Judge it on distributional fidelity, not per-day MAE.
    """
    om_train = np.asarray(om_train, dtype=float)
    imd_train = np.asarray(imd_train, dtype=float)
    wet = (om_train >= RAIN_THRESHOLD) & (imd_train >= RAIN_THRESHOLD)
    if wet.sum() < 50:
        return None
    edges = np.asarray(band_edges, dtype=float)
    om_q = np.quantile(om_train[wet], edges)
    imd_q = np.quantile(imd_train[wet], edges)
    # enforce monotonic non-decreasing edges so interpolation is well-defined
    om_q = np.maximum.accumulate(om_q)
    imd_q = np.maximum.accumulate(imd_q)
    return {"om_q": om_q, "imd_q": imd_q, "edges": edges}


def apply_empirical_qm(qm, om):
    """
    Apply a piecewise quantile map to raw OM values.

    Dry raw days (om < RAIN_THRESHOLD) are passed through unchanged — QM built on
    wet pairs says nothing about them, and a dry forecast shouldn't be inflated.
    Wet raw days are mapped OM-quantile -> IMD-quantile via the banded curve.
    OM values beyond the training max are extrapolated by holding the top band's
    OM:IMD ratio (so a record-breaking raw day isn't clipped to the training max).
    """
    om = np.asarray(om, dtype=float)
    out = om.copy()
    if qm is None:
        return out
    wet = om >= RAIN_THRESHOLD
    if not wet.any():
        return out
    om_q, imd_q, edges = qm["om_q"], qm["imd_q"], qm["edges"]
    # OM value -> its quantile position -> IMD value at that quantile
    p = np.interp(om[wet], om_q, edges)
    mapped = np.interp(p, edges, imd_q)
    # extrapolate above the training max by the top-band multiplicative ratio
    top_ratio = (imd_q[-1] / om_q[-1]) if om_q[-1] > 0 else 1.0
    above_max = om[wet] > om_q[-1]
    if above_max.any():
        mapped[above_max] = om[wet][above_max] * top_ratio
    out[wet] = mapped
    return out


def _pcp_metrics(pred, om, imd, ext_pctl=None):
    """
    MAE metrics for a precip prediction vs IMD truth. Returns all-day MAE plus,
    if ext_pctl given and enough wet days exist, extreme-day MAE over days with
    IMD >= that percentile (computed on wet days). Also returns a simple
    distributional score: |P90(pred_wet) - P90(imd_wet)| so QM can be judged on
    tail fidelity rather than only per-day MAE.
    """
    om = np.asarray(om, float); imd = np.asarray(imd, float); pred = np.asarray(pred, float)
    raw_mae = float(np.abs(om - imd).mean())
    corr_mae = float(np.abs(pred - imd).mean())
    m = {"raw_mae": raw_mae, "corr_mae": corr_mae, "n": len(imd),
         "raw_mae_ext": None, "corr_mae_ext": None, "ext_n": 0, "ext_thresh": None,
         "p90_imd": None, "p90_raw": None, "p90_corr": None}
    wet = imd >= RAIN_THRESHOLD
    if wet.sum() >= 10:
        wet_om = om >= RAIN_THRESHOLD
        m["p90_imd"] = float(np.percentile(imd[wet], 90))
        if wet_om.sum() >= 10:
            m["p90_raw"] = float(np.percentile(om[wet_om], 90))
            pw = pred[pred >= RAIN_THRESHOLD]
            if len(pw) >= 10:
                m["p90_corr"] = float(np.percentile(pw, 90))
        if ext_pctl is not None:
            thr = float(np.percentile(imd[wet], ext_pctl))
            ext = imd >= thr
            if ext.sum() > 0:
                m["ext_thresh"] = thr; m["ext_n"] = int(ext.sum())
                m["raw_mae_ext"] = float(np.abs(om[ext] - imd[ext]).mean())
                m["corr_mae_ext"] = float(np.abs(pred[ext] - imd[ext]).mean())
    return m


def _fit_pcp(train):
    """
    Fit the precip correctors on a training frame. Returns a dict with the
    classifier, ratio-regressor and empirical QM map, or None if insufficient
    data. Shared by the CV folds and the final production model so the two are
    guaranteed identical.
    """
    if train is None or len(train) < 100:
        return None
    cls_feat = [f for f in RAIN_CLS_FEATURES if f in train.columns]
    tc = train.copy()
    tc["rain_obs"] = (tc["imd"] >= RAIN_THRESHOLD).astype(int)
    y_cls = tc["rain_obs"].values
    if y_cls.sum() < 20 or (1 - y_cls.mean()) < 0.01:
        return None

    cls_params = XGB_PARAMS.copy()
    cls_params["n_estimators"] = 100
    cls_params["max_depth"] = 4
    spw = (1 - y_cls.mean()) / max(y_cls.mean(), 1e-6)
    cls_model = xgb.XGBClassifier(**cls_params, scale_pos_weight=spw, n_jobs=1)
    cls_model.fit(tc[cls_feat].values, y_cls)

    rain_train = tc[(tc["rain_obs"] == 1) & (tc["om"] >= RAIN_THRESHOLD)].copy()
    reg_feat = [f for f in RAIN_REG_FEATURES if f in rain_train.columns]
    if len(rain_train) < 20:
        return None
    log_ratio = np.log(
        rain_train["imd"].clip(lower=RAIN_THRESHOLD).values
        / rain_train["om"].clip(lower=RAIN_THRESHOLD).values)
    log_ratio = np.clip(log_ratio, PCP_LOG_RATIO_MIN, PCP_LOG_RATIO_MAX)
    reg_model = xgb.XGBRegressor(**XGB_PARAMS, n_jobs=1)
    reg_model.fit(rain_train[reg_feat].values, log_ratio)

    # Empirical piecewise quantile map from all wet pairs in the training frame.
    qm = build_empirical_qm(tc["om"].values, tc["imd"].values)

    return {"cls_model": cls_model, "cls_features": cls_feat,
            "reg_model": reg_model, "reg_features": reg_feat, "qm": qm}


def _predict_pcp_ratio(fit, df):
    """Anchored ratio-model prediction for a frame (uses cls + reg)."""
    om = df["om"].values.astype(float)
    cls_feat = [f for f in fit["cls_features"] if f in df.columns]
    reg_feat = [f for f in fit["reg_features"] if f in df.columns]
    cls_pred = fit["cls_model"].predict(df[cls_feat].values)
    log_ratio = np.zeros_like(om)
    above = om >= RAIN_THRESHOLD
    if above.any():
        log_ratio[above] = fit["reg_model"].predict(df[above][reg_feat].values)
    return _apply_pcp_correction(om, cls_pred, log_ratio)


def _tail_blend_weight(om):
    """
    QM weight as a function of raw mm: 0 below PCP_BLEND_LO_MM, ramping linearly
    to PCP_BLEND_QM_WEIGHT at/above PCP_BLEND_HI_MM. Elementwise.
    """
    om = np.asarray(om, dtype=float)
    span = max(PCP_BLEND_HI_MM - PCP_BLEND_LO_MM, 1e-6)
    frac = np.clip((om - PCP_BLEND_LO_MM) / span, 0.0, 1.0)
    return frac * PCP_BLEND_QM_WEIGHT


def _blend_ratio_qm(ratio_pred, qm_pred, om):
    """
    Blend the ratio-model and QM predictions by raw mm. Ordinary days stay on the
    ratio model (per-day skill); heavy raw days are pulled toward QM to restore
    the tail magnitude the ratio model under-corrects. If QM is unavailable
    (qm_pred is None) or the weight is zero, returns the ratio prediction.
    """
    if qm_pred is None or PCP_BLEND_QM_WEIGHT <= 0:
        return ratio_pred
    w = _tail_blend_weight(om)
    return (1.0 - w) * ratio_pred + w * qm_pred


def _predict_pcp_blend(fit, df):
    """Tail-blended prediction: ratio model, pulled toward QM on heavy raw days."""
    ratio_pred = _predict_pcp_ratio(fit, df)
    qm_pred = apply_empirical_qm(fit.get("qm"), df["om"].values.astype(float))
    return _blend_ratio_qm(ratio_pred, qm_pred, df["om"].values.astype(float))


def _year_folds(years_available, n_folds):
    """
    Partition sorted unique years into up to n_folds contiguous blocks.
    Returns a list of arrays of held-out years. Contiguous (not interleaved)
    so each fold is a coherent time block.
    """
    yrs = np.array(sorted(set(int(y) for y in years_available)))
    if len(yrs) < 2:
        return []
    k = int(min(n_folds, len(yrs)))
    return [b for b in np.array_split(yrs, k) if len(b) > 0]


def cv_evaluate_pcp(full, ext_pctl=PCP_EXTREME_PCTL, n_folds=CV_N_FOLDS):
    """
    Blocked-by-year CV for precipitation. For each year-block fold: fit on the
    other years, evaluate on the held-out year for BOTH the ratio model and the
    empirical QM. Test rows are OM-only (source==1) so skill reflects what we
    actually correct at apply time. Pooled predictions across folds give the
    reported MAE (all-day + extreme) and P90 distributional check per method.
    """
    if full is None or len(full) < 200 or "year" not in full.columns:
        return None
    folds = _year_folds(full["year"].values, n_folds)
    if not folds:
        return None

    # accumulate pooled test-fold predictions
    om_all, imd_all, ratio_all, qm_all, blend_all = [], [], [], [], []
    for held in folds:
        tr = full[~full["year"].isin(held)]
        te = full[(full["year"].isin(held)) & (full.get("source", 1) == 1)]
        te = te.dropna(subset=["imd"])
        if len(te) == 0:
            continue
        fit = _fit_pcp(tr)
        if fit is None:
            continue
        rp = _predict_pcp_ratio(fit, te)
        qp = apply_empirical_qm(fit["qm"], te["om"].values)
        om_all.append(te["om"].values.astype(float))
        imd_all.append(te["imd"].values.astype(float))
        ratio_all.append(rp)
        qm_all.append(qp)
        blend_all.append(_blend_ratio_qm(rp, qp, te["om"].values.astype(float)))

    if not imd_all:
        return None
    om = np.concatenate(om_all); imd = np.concatenate(imd_all)
    ratio_pred = np.concatenate(ratio_all); qm_pred = np.concatenate(qm_all)
    blend_pred = np.concatenate(blend_all)
    return {
        "ratio": _pcp_metrics(ratio_pred, om, imd, ext_pctl),
        "qm": _pcp_metrics(qm_pred, om, imd, ext_pctl),
        "blend": _pcp_metrics(blend_pred, om, imd, ext_pctl),
        "n_folds": len(imd_all),
    }


def train_grid_model(gd, var_key):
    """
    Fit the production model on a grid's FULL timeseries and attach blocked-by-
    year CV skill. `gd` is the per-grid dict from build_grid_training, expected
    to contain 'full'. Returns a model-data dict or None.
    """
    if gd is None:
        return None
    full = gd.get("full")
    if full is None or len(full) < 100:
        return None

    if var_key in ("tmax", "tmin"):
        feat = [f for f in TEMP_FEATURES if f in full.columns]
        model = xgb.XGBRegressor(**XGB_PARAMS, n_jobs=1)
        model.fit(full[feat].values, full["imd"].values)
        cvm = _cv_evaluate_temp(full, feat, var_key)
        return {"type": "regressor", "model": model, "features": feat,
                "raw_mae": cvm["raw_mae"] if cvm else None,
                "corr_mae": cvm["corr_mae"] if cvm else None,
                "test_n": cvm["n"] if cvm else 0}

    elif var_key == "pcp":
        fit = _fit_pcp(full)
        if fit is None:
            return None
        cv = cv_evaluate_pcp(full)
        md = {"type": "two_stage",
              "cls_model": fit["cls_model"], "cls_features": fit["cls_features"],
              "reg_model": fit["reg_model"], "reg_features": fit["reg_features"],
              "qm": fit["qm"], "cv": cv}
        return md
    return None


def _cv_evaluate_temp(full, feat, var_key, n_folds=CV_N_FOLDS):
    """Blocked-by-year CV for temperature (single regressor)."""
    if full is None or "year" not in full.columns:
        return None
    folds = _year_folds(full["year"].values, n_folds)
    if not folds:
        return None
    om_all, imd_all, pred_all = [], [], []
    for held in folds:
        tr = full[~full["year"].isin(held)]
        te = full[(full["year"].isin(held)) & (full.get("source", 1) == 1)]
        te = te.dropna(subset=["imd"])
        if len(tr) < 100 or len(te) == 0:
            continue
        m = xgb.XGBRegressor(**XGB_PARAMS, n_jobs=1)
        m.fit(tr[feat].values, tr["imd"].values)
        om_all.append(te["om"].values.astype(float))
        imd_all.append(te["imd"].values.astype(float))
        pred_all.append(m.predict(te[feat].values))
    if not imd_all:
        return None
    om = np.concatenate(om_all); imd = np.concatenate(imd_all); pred = np.concatenate(pred_all)
    return {"raw_mae": float(np.abs(om - imd).mean()),
            "corr_mae": float(np.abs(pred - imd).mean()), "n": len(imd)}


def train_surrounding_models(era5_dir, forecast_path, imd_dir, elev_path,
                             var_key, grid_pts, train_source="om"):
    """
    Train models for the surrounding grids (in memory).
    Returns list of (glat, glon, dist, model_data) for grids that trained OK.
    """
    grid_pairs = [(g[0], g[1]) for g in grid_pts]
    # Per-variable override: ERA5 helps precip but hurts temperature here.
    effective_source = PER_VAR_TRAIN_SOURCE.get(var_key) or train_source
    if effective_source != train_source:
        print(f"    [{var_key}] train_source override: "
              f"{train_source} -> {effective_source}")
    grid_data = build_grid_training(
        era5_dir, forecast_path, imd_dir, elev_path, var_key, grid_pairs,
        train_source=effective_source)

    trained = []
    for glat, glon, dist in grid_pts:
        gd = grid_data.get((glat, glon))
        if gd is None:
            print(f"    skipped grid ({glat}, {glon}) — no data")
            continue
        md = train_grid_model(gd, var_key)
        if md is not None:
            trained.append((glat, glon, dist, md))
    return trained


# ==============================================================================
# ELEVATION + FORECAST FETCH  (with paid -> free fallback)
# ==============================================================================

def fetch_elevation(lat, lon):
    try:
        resp = requests.get(ELEVATION_URL,
                            params={"locations": f"{lat:.6f},{lon:.6f}"},
                            timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "OK" and data["results"]:
            elev = data["results"][0].get("elevation")
            return float(elev) if elev is not None else 0.0
    except Exception:
        pass
    return 0.0


def _parse_daily(data):
    if "daily" not in data:
        return None
    d = data["daily"]
    return pd.DataFrame({
        "date": pd.to_datetime(d["time"]).normalize(),
        "tmax": pd.to_numeric(pd.Series(d.get("temperature_2m_max", [])), errors="coerce"),
        "tmin": pd.to_numeric(pd.Series(d.get("temperature_2m_min", [])), errors="coerce"),
        "pcp":  pd.to_numeric(pd.Series(d.get("precipitation_sum", [])), errors="coerce"),
    })


def fetch_forecast(lat, lon, apikey, past_days=OUTPUT_PAST_DAYS + ROLL_BUFFER):
    """
    Fetch past_days of recent history + the 16-day live forecast from the
    Open-Meteo forecast API (no historical-forecast leg — that product isn't
    part of the standard package). On an expired / invalid / over-quota key
    (HTTP 400/401/402/403) the paid endpoint falls back to the free API.

    `past_days` includes both the days we want in the output (OUTPUT_PAST_DAYS)
    and a small ROLL_BUFFER used only to warm up the 7-day rolling feature;
    the buffer days are dropped before output.
    """
    live_use_free = False
    live_url = LIVE_FORECAST_URL
    live_df = None

    for attempt in range(5):
        try:
            tz = "Asia/Kolkata" if (6.0 <= lat <= 38.0 and 68.0 <= lon <= 98.0) else "GMT"
            params = {
                "latitude": f"{lat:.6f}", "longitude": f"{lon:.6f}",
                "forecast_days": 16, "past_days": past_days,
                "daily": DAILY_PARAMS, "timezone": tz,
            }
            if not live_use_free:
                params["apikey"] = apikey
            resp = requests.get(live_url, params=params, timeout=60)
            resp.raise_for_status()
            live_df = _parse_daily(resp.json())
            break
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else None
            if not live_use_free and code in (400, 401, 402, 403):
                live_use_free = True
                live_url = FREE_FORECAST_URL
                continue
            if code == 429:
                time.sleep(10 * (attempt + 1))
            else:
                raise
        except Exception:
            time.sleep(5 * (attempt + 1))

    if live_df is None:
        return None
    live_df = live_df.drop_duplicates(subset=["date"], keep="last")
    live_df = live_df.sort_values("date").reset_index(drop=True)
    live_df["is_forecast"] = (live_df["date"] >= pd.Timestamp(date.today())).astype(int)
    return live_df


# ==============================================================================
# FEATURE ENGINEERING (forecast side)
# ==============================================================================

def add_features(df, elevation):
    df = df.copy()
    df["elevation"] = elevation
    df["source"] = 1  # forecast
    doy = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["month"] = df["date"].dt.month
    df["season_monsoon"]    = df["month"].isin([6, 7, 8, 9]).astype(np.int8)
    df["season_premonsoon"] = df["month"].isin([3, 4, 5]).astype(np.int8)
    df["season_postmonsoon"]= df["month"].isin([10, 11]).astype(np.int8)
    df["season_winter"]     = df["month"].isin([12, 1, 2]).astype(np.int8)
    return df


def add_var_features(df, var_col):
    df = df.copy()
    df["om"] = pd.to_numeric(df[var_col], errors="coerce")
    df["om_roll7"] = df["om"].rolling(7, min_periods=1).mean()
    om_clim = df.groupby("month")["om"].transform("mean")
    df["om_clim_anom"] = df["om"] - om_clim
    df["rain_om"] = (df["om"] >= RAIN_THRESHOLD).astype(np.int8)
    df["om_log"] = np.log1p(df["om"].clip(lower=0))
    return df


# ==============================================================================
# IDW PREDICTION FROM IN-MEMORY MODELS
# ==============================================================================

def predict_idw(df_var, var_key, trained_models, method="blend"):
    """
    trained_models : list of (glat, glon, dist, model_data)
    Applies each grid's model then inverse-distance-weights the results.
    For pcp, `method` selects the corrector: 'ratio' (anchored ratio model,
    the default production column) or 'qm' (empirical quantile mapping).
    """
    if not trained_models:
        return df_var["om"].values

    predictions, weights = [], []
    for glat, glon, dist, md in trained_models:
        om = df_var["om"].values
        if var_key in ("tmax", "tmin"):
            feat = [f for f in md["features"] if f in df_var.columns]
            pred = md["model"].predict(df_var[feat].values)
        elif var_key == "pcp":
            if method == "qm":
                pred = apply_empirical_qm(md.get("qm"), om.astype(float))
            elif method == "ratio":
                pred = _predict_pcp_ratio(md, df_var)
            else:  # "blend" (default production corrector)
                pred = _predict_pcp_blend(md, df_var)
        else:
            pred = om
        predictions.append(pred)
        weights.append(1.0 / max(dist, 0.001))

    weights = np.array(weights)
    weights = weights / weights.sum()
    result = np.zeros_like(predictions[0], dtype=float)
    for pred, w in zip(predictions, weights):
        result += w * pred
    return result


# ==============================================================================
# NEAREST IMD (filtered read)
# ==============================================================================

def find_nearest_imd(vlat, vlon, imd_dir):
    grid_lat = round((vlat - GRID_OFFSET) / GRID_STEP) * GRID_STEP + GRID_OFFSET
    grid_lon = round((vlon - GRID_OFFSET) / GRID_STEP) * GRID_STEP + GRID_OFFSET
    grid_lat = round(grid_lat, 4)
    grid_lon = round(grid_lon, 4)

    imd_data = {}
    for var_key, fname in [("pcp", "imd_pcp_daily.parquet"),
                           ("tmax", "imd_tmax_daily.parquet"),
                           ("tmin", "imd_tmin_daily.parquet")]:
        path = os.path.join(imd_dir, fname)
        df = _read_parquet_grids(path, [(grid_lat, grid_lon)],
                                 columns=["lat", "lon", "date", "value"])
        if df is None or df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        nearest = df[["date", "value"]].rename(columns={"value": f"imd_{var_key}"})
        imd_data[var_key] = nearest
    return imd_data


# ==============================================================================
# MAIN LOCATION PIPELINE
# ==============================================================================

def process_location(
    lat,
    lon,
    era5_dir="./data/training_data",
    forecast_path="./data/om_forecast_all.parquet",
    imd_dir="./data/IMD_parquets",
    elev="./data/grid_elevation.parquet",
    apikey="xxx",
    train_source="om",
    output_dir="./out"
):
    grid_pts = surrounding_grid_points(lat, lon)
    elevation = fetch_elevation(lat, lon)
    forecast = fetch_forecast(lat, lon, apikey)
    if forecast is None:
        print("    ERROR: No forecast data!")
        return None
    forecast = add_features(forecast, elevation)
    results = forecast[["date", "is_forecast"]].copy()

    for var_key, var_col in [("tmax", "tmax"), ("tmin", "tmin"), ("pcp", "pcp")]:
        trained = train_surrounding_models(
            era5_dir, forecast_path, imd_dir, elev,
            var_key, grid_pts, train_source=train_source)

        df_var = add_var_features(forecast, var_col)
        if not trained:
            results[f"{var_key}_forecast_raw"] = df_var["om"].values
            results[f"{var_key}_forecast_corrected"] = df_var["om"].values
            continue

        corrected = predict_idw(df_var, var_key, trained)
        results[f"{var_key}_forecast_raw"] = df_var["om"].values
        results[f"{var_key}_forecast_corrected"] = corrected
        if var_key == "pcp":
            # corrected column above is the tail-blend (production). Also expose
            # the pure ratio and pure QM columns for comparison/inspection.
            results["pcp_forecast_ratio"] = predict_idw(
                df_var, var_key, trained, method="ratio")
            results["pcp_forecast_qm"] = predict_idw(
                df_var, var_key, trained, method="qm")

        # Blocked-by-year CV skill, IDW-weighted across the surrounding grids.
        if var_key in ("tmax", "tmin"):
            w, raw_acc, corr_acc = 0.0, 0.0, 0.0
            for _, _, dist, md in trained:
                if md.get("raw_mae") is not None and md.get("corr_mae") is not None:
                    wi = 1.0 / max(dist, 0.001)
                    raw_acc += wi * md["raw_mae"]; corr_acc += wi * md["corr_mae"]; w += wi
            if w > 0:
                raw_s, corr_s = raw_acc / w, corr_acc / w
                imp = (raw_s - corr_s) / raw_s * 100 if raw_s else 0.0
                print(f"    {var_key:4s} CV skill (blocked-year, IDW): "
                      f"raw MAE={raw_s:.3f}  corr MAE={corr_s:.3f}  improvement={imp:+.1f}%")
            else:
                print(f"    {var_key:4s} CV skill: no test data")

        elif var_key == "pcp":
            # IDW-accumulate each method's all-day and extreme MAE + P90.
            methods = ("ratio", "qm", "blend")
            acc = {m: {"w": 0.0, "raw": 0.0, "corr": 0.0,
                       "we": 0.0, "raw_e": 0.0, "corr_e": 0.0, "ext_n": 0,
                       "wp": 0.0, "p90_imd": 0.0, "p90_raw": 0.0, "p90_corr": 0.0}
                   for m in methods}
            for _, _, dist, md in trained:
                cv = md.get("cv")
                if not cv:
                    continue
                wi = 1.0 / max(dist, 0.001)
                for m in methods:
                    e = cv.get(m)
                    if not e or e.get("raw_mae") is None:
                        continue
                    a = acc[m]
                    a["w"] += wi; a["raw"] += wi * e["raw_mae"]; a["corr"] += wi * e["corr_mae"]
                    if e.get("corr_mae_ext") is not None:
                        a["we"] += wi; a["raw_e"] += wi * e["raw_mae_ext"]
                        a["corr_e"] += wi * e["corr_mae_ext"]; a["ext_n"] += e.get("ext_n", 0)
                    if e.get("p90_corr") is not None:
                        a["wp"] += wi; a["p90_imd"] += wi * e["p90_imd"]
                        a["p90_raw"] += wi * e["p90_raw"]; a["p90_corr"] += wi * e["p90_corr"]

            for m, label in (("ratio", "ratio-model"), ("qm", "empirical-QM"),
                             ("blend", "tail-blend*")):
                a = acc[m]
                if a["w"] <= 0:
                    print(f"    pcp  CV [{label:12s}]: no test data"); continue
                raw_s, corr_s = a["raw"] / a["w"], a["corr"] / a["w"]
                imp = (raw_s - corr_s) / raw_s * 100 if raw_s else 0.0
                line = (f"    pcp  CV [{label:12s}]: all-day raw={raw_s:.2f} "
                        f"corr={corr_s:.2f} ({imp:+.1f}%)")
                if a["we"] > 0:
                    raw_e, corr_e = a["raw_e"] / a["we"], a["corr_e"] / a["we"]
                    imp_e = (raw_e - corr_e) / raw_e * 100 if raw_e else 0.0
                    line += (f" | EXTREME(P{PCP_EXTREME_PCTL:.0f},n~{a['ext_n']}) "
                             f"raw={raw_e:.1f} corr={corr_e:.1f} ({imp_e:+.1f}%)")
                print(line)
                if a["wp"] > 0:
                    print(f"                        P90 wet-day mm: IMD={a['p90_imd']/a['wp']:.1f} "
                          f"raw={a['p90_raw']/a['wp']:.1f} corrected={a['p90_corr']/a['wp']:.1f}"
                          f"   (QM judged on this, not MAE)")
            print("    * tail-blend = production corrected column "
                  "(ratio model, pulled toward QM on heavy raw days)")

    # Keep the last OUTPUT_PAST_DAYS of history + all forecast rows;
    # drop only the oldest ROLL_BUFFER days used to warm up om_roll7.
    cutoff = pd.Timestamp(date.today()) - pd.Timedelta(days=OUTPUT_PAST_DAYS)
    results = results[results["date"] >= cutoff].reset_index(drop=True)

    # Metadata
    results["lat"] = lat
    results["lon"] = lon
    results["elevation"] = elevation

    # Merge nearest IMD observations (available for the past-days portion)
    if imd_dir and os.path.isdir(imd_dir):
        imd_data = find_nearest_imd(lat, lon, imd_dir)
        for _, imd_df in imd_data.items():
            results = results.merge(imd_df, on="date", how="left")

    col_order = [
        "lat", "lon", "elevation", "date", "is_forecast",
        "tmax_forecast_raw", "tmax_forecast_corrected", "imd_tmax",
        "tmin_forecast_raw", "tmin_forecast_corrected", "imd_tmin",
        "pcp_forecast_raw", "pcp_forecast_corrected",
        "pcp_forecast_ratio", "pcp_forecast_qm", "imd_pcp",
    ]
    results = results[[c for c in col_order if c in results.columns]]

    os.makedirs(output_dir, exist_ok=True)
    tag = f"{lat:.4f}_{lon:.4f}".replace("-", "m")
    csv_path = os.path.join(output_dir, f"forecast_{tag}.csv")
    results.to_csv(csv_path, index=False, float_format="%.2f")

    # Keep only up to MAX_FORECAST_FILES in output_dir (act like a queue)
    import glob
    files = glob.glob(os.path.join(output_dir, "forecast_*.csv"))
    files.sort(key=os.path.getmtime)
    if len(files) > MAX_FORECAST_FILES:
        for f in files[:-MAX_FORECAST_FILES]:
            os.remove(f)

    # Live-window check vs IMD (past-days portion). NOTE: this uses short-window
    # features (om_roll7 / om_clim_anom computed over the fetched window only),
    # so it is a rough diagnostic, NOT the model's true skill. The trustworthy
    # number is the "2025+ skill (IDW)" line printed during training above.
    past = results[results["is_forecast"] == 0]
    printed_header = False
    for var_key in ["tmax", "tmin", "pcp"]:
        imd_col = f"imd_{var_key}"
        if imd_col in past.columns:
            valid = past[past[imd_col].notna()]
            if len(valid) > 1:
                if not printed_header:
                    printed_header = True

    # Filter to 16-day forecast and format for JSON return
    forecast_16 = results[results["is_forecast"] == 1].copy()
    forecast_16 = forecast_16[["date", "tmax_forecast_corrected", "tmin_forecast_corrected", "pcp_forecast_corrected"]]
    forecast_16.columns = ["date", "tmax", "tmin", "pcp"]
    
    # Format date as YYYY-MM-DD
    if pd.api.types.is_datetime64_any_dtype(forecast_16["date"]):
        forecast_16["date"] = forecast_16["date"].dt.strftime("%Y-%m-%d")
    else:
        forecast_16["date"] = pd.to_datetime(forecast_16["date"]).dt.strftime("%Y-%m-%d")
        
    return forecast_16.to_json(orient="records")


def run_forecast_pipeline(
    lat,
    lon,
    era5_dir="./data/training_data",
    forecast="./data/om_forecast_all.parquet",
    imd_dir="./data/IMD_parquets",
    elev="./data/grid_elevation.parquet",
    apikey="xxx",
    train_source="om",
    output_dir="./out"
):
    import time
    t0 = time.time()
    res = process_location(
        lat=lat,
        lon=lon,
        era5_dir=era5_dir,
        forecast_path=forecast,
        imd_dir=imd_dir,
        elev=elev,
        apikey=apikey,
        train_source=train_source,
        output_dir=output_dir
    )
    duration = time.time() - t0
    try:
        from app.stats import update_stats
        update_stats("forecast", duration)
    except Exception as e:
        print(f"[STATS WARNING] Error logging stats: {e}")
    return res


# ==============================================================================
# CLI
# ==============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Combined per-grid train + village forecast (light on render)")
    p.add_argument("--lat", type=float, required=True, help="Target latitude")
    p.add_argument("--lon", type=float, required=True, help="Target longitude")
    p.add_argument("--era5_dir", default="./data/training_data",
                   help="Dir with training_*_daily.parquet")
    p.add_argument("--forecast", default="./data/om_forecast_all.parquet",
                   help="Forecast parquet (used for training pairs only)")
    p.add_argument("--imd_dir", default="./data/IMD_parquets",
                   help="Dir with imd_*_daily.parquet")
    p.add_argument("--elev", default="./data/grid_elevation.parquet",
                   help="Grid elevation parquet")
    p.add_argument("--apikey", required=False, default="xxx", help="Open-Meteo customer API key")
    p.add_argument("--train_source", choices=["om", "era5", "both"],
                   default="om",
                   help="Which sources train the model. 'om' (default) is "
                        "source-matched to the OM apply target; 'both' adds "
                        "ERA5 (can bias low-error vars like tmin).")
    p.add_argument("--output_dir", default="./out", help="Output CSV directory")
    args = p.parse_args()

    res_json = run_forecast_pipeline(
        lat=args.lat,
        lon=args.lon,
        era5_dir=args.era5_dir,
        forecast=args.forecast,
        imd_dir=args.imd_dir,
        elev=args.elev,
        apikey=args.apikey,
        train_source=args.train_source,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
