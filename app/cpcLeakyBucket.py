#!/usr/bin/env python3
"""
CPC one-layer "leaky bucket" soil moisture model (Huang et al. 1996;
van den Dool et al. 2003; Fan & van den Dool 2004) with a simple
temperature-threshold snow module (after Fan 2019).

Forcing: daily Precip, Tmax, Tmin point files in the IMD format
    columns: [year month day value]   (whitespace separated)
    Precip in mm, Tmax/Tmin in degC.

Water balance (daily, single 1.6 m layer, wmax = 760 mm):
    w_{t+1} = w_t + Peff - E - R - G
where
    Peff = rain + snowmelt
    E    = beta(w) * PE        (PE: FAO-56 Penman-Monteith ETo, reduced-set)
    R    = surface + base runoff (CPC Bm-style, function of w/wmax)
    G    = linear groundwater loss = gamma * (w/wmax)

Irrigation: an optional module adds irrigation as a water input sized by
the demand formulation of VIC-WUR (Droppers et al. 2020), H08 (Hanasaki
et al. 2008) and FAO-56 (Allen et al. 1998) rather than by refilling the
column to saturation. Events are triggered on root-zone depletion
relative to readily available water, capped by application efficiency
and by a physical supply ceiling. See the IRRIGATION MODULE header for
the full derivation, equation numbering and citations.

Output: CSV with daily date, forcing, snowpack, w, E, R, G, w as a
fraction of wmax, and (when irrigation is active) gross and net applied
depths, root-zone storage w_rz, capacity TAW and depletion Dr.

Usage:
    python cpc_leaky_bucket.py \
        --pcp  /media/urmin/data1/.../RT_pcp/data_36.875_74.625 \
        --tmax /media/urmin/data1/.../RT_tmax/data_36.875_74.625 \
        --tmin /media/urmin/data1/.../RT_tmin/data_36.875_74.625 \
        --lat  36.875 \
        --out  sm_36.875_74.625.csv

If --pcp etc. are not given, the script derives the three paths from a
single --base point file path by swapping the RT_pcp/RT_tmax/RT_tmin
folder names, and reads --lat from the filename if not supplied.
"""

import argparse
import os
import re
import sys
import math
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Model parameters (CPC operational defaults; see Huang et al. 1996,
# Table-style tuned values, van den Dool et al. 2003). Tunable.
# ----------------------------------------------------------------------
WMAX = 760.0        # max water holding capacity (mm)  -> 1.6 m, porosity 0.47

# Runoff parameters (surface + base). CPC parameterizes runoff as a
# nonlinear function of the storage ratio. Bm controls surface runoff
# curvature; alpha_base the linear baseflow.
PARAMS = dict(
    Bm=2.0,          # surface-runoff exponent (storage-ratio sensitivity)
    alpha_surf=1.0,  # surface-runoff scaling on instantaneous input
    alpha_base=0.005,# base-runoff (slow drainage) linear coefficient
    gamma=0.0,       # groundwater loss coefficient (set 0 -> folded into base)
    # Thornthwaite uses long-term monthly normals for the heat index; we
    # build the heat index from the record itself.
)

# Snow module (temperature-threshold, Fan 2019 style)
SNOW = dict(
    t_snow=0.0,      # below this mean-T (degC) precip falls as snow
    t_melt=0.0,      # above this mean-T melting occurs
    melt_factor=2.5, # degree-day melt factor (mm / degC / day)
)


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------
def read_point_file(path):
    """Read a [year month day value] whitespace file -> indexed Series."""
    if not os.path.exists(path):
        sys.exit(f"ERROR: file not found: {path}")
    df = pd.read_csv(
        path, sep=r"\s+", header=None,
        names=["year", "month", "day", "value"],
        engine="python",
    )
    # Build a proper datetime index, dropping any malformed dates.
    dt = pd.to_datetime(
        dict(year=df.year, month=df.month, day=df.day),
        errors="coerce",
    )
    s = pd.Series(df.value.values, index=dt, name=os.path.basename(path))
    s = s[~s.index.isna()]
    # Some IMD point files contain repeated dates; keep the last value per
    # date so every downstream consumer gets a unique DatetimeIndex
    # (duplicate labels break pandas reindex/align operations).
    s = s[~s.index.duplicated(keep="last")]
    return s


def derive_paths(base):
    """Given one RT_* point file path, derive the other two by folder swap."""
    paths = {}
    for var, folder in (("pcp", "RT_pcp"), ("tmax", "RT_tmax"),
                        ("tmin", "RT_tmin")):
        if "RT_pcp" in base:
            paths[var] = base.replace("RT_pcp", folder)
        elif "RT_tmax" in base:
            paths[var] = base.replace("RT_tmax", folder)
        elif "RT_tmin" in base:
            paths[var] = base.replace("RT_tmin", folder)
        else:
            sys.exit("ERROR: --base must contain RT_pcp / RT_tmax / RT_tmin")
    return paths["pcp"], paths["tmax"], paths["tmin"]


def lat_from_filename(path):
    """data_<lat>_<lon> -> lat as float, or None."""
    m = re.search(r"data_(-?\d+\.?\d*)_(-?\d+\.?\d*)", os.path.basename(path))
    return float(m.group(1)) if m else None


def latlon_from_filename(path):
    """data_<lat>_<lon> -> (lat, lon) as floats, or (None, None)."""
    m = re.search(r"data_(-?\d+\.?\d*)_(-?\d+\.?\d*)", os.path.basename(path))
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


def load_elevation_file(path):
    """
    Read a whitespace 'lon lat elev' grid file OR a parquet file
    -> dict keyed by rounded (lat, lon) for fast nearest-cell lookup.
    Returns (dict, grid_res).
    """
    if not os.path.exists(path):
        sys.exit(f"ERROR: elevation file not found: {path}")
    if path.endswith(".parquet"):
        df = pd.read_parquet(path, memory_map=False)
        lats = df["lat"].to_numpy()
        lons = df["lon"].to_numpy()
        elevs = df["elevation"].to_numpy()
    else:
        arr = np.loadtxt(path)
        if arr.ndim != 2 or arr.shape[1] < 3:
            sys.exit("ERROR: elevation file must have columns: lon lat elev")
        lons, lats, elevs = arr[:, 0], arr[:, 1], arr[:, 2]
    table = {(round(la, 3), round(lo, 3)): el
             for lo, la, el in zip(lons, lats, elevs)}
    return table, lats, lons, elevs


def lookup_elevation(lat, lon, table, lats, lons, elevs):
    """
    Look up elevation for (lat, lon). Exact match on 3-dp rounding first;
    otherwise nearest grid cell by Euclidean distance in lat/lon.
    """
    key = (round(lat, 3), round(lon, 3))
    if key in table:
        return table[key]
    d2 = (lats - lat) ** 2 + (lons - lon) ** 2
    return float(elevs[int(np.argmin(d2))])


# ----------------------------------------------------------------------
# Reference evapotranspiration: FAO-56 Penman-Monteith (Allen et al. 1998)
#
# Full FAO-56 grass-reference ETo (Eq. 6) using a temperature-only
# ("reduced-set") input stream, exactly as FAO-56 prescribes when the
# station reports only Tmax/Tmin (Chapter 3 / Annex 6). This is the same
# strategy VIC's MTCLIM preprocessor uses: solar radiation is derived from
# the diurnal temperature range (Hargreaves radiation, Eq. 50), actual
# vapour pressure from Tmin as a dewpoint proxy (Eq. 48), net radiation
# from the estimated Rs and ea, wind defaulted to 2 m/s, and G = 0 daily.
#
# Equation numbers below refer to FAO Irrigation & Drainage Paper 56.
# ----------------------------------------------------------------------
PM = dict(
    kRs=0.16,        # Hargreaves radiation coefficient (0.16 interior, 0.19 coastal)
    u2=2.0,          # default wind speed at 2 m (m/s) when unavailable
    albedo=0.23,     # grass reference albedo
    G=0.0,           # soil heat flux, ~0 at daily step
)
GSC = 0.0820         # solar constant (MJ m-2 min-1)
SIGMA = 4.903e-9     # Stefan-Boltzmann (MJ K-4 m-2 day-1)


def sat_vapour_pressure(t):
    """Saturation vapour pressure e0(T) (kPa), FAO-56 Eq. 11."""
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def penman_monteith_eto(tmax, tmin, dates, lat_deg, elev_m, pm=PM):
    """
    Daily FAO-56 grass-reference ETo (mm/day) from Tmax, Tmin, latitude
    and elevation only. Returns a Series aligned to `dates`.
    """
    tmx = np.asarray(tmax, dtype=float)
    tmn = np.asarray(tmin, dtype=float)
    tmean = (tmx + tmn) / 2.0

    # --- Atmospheric pressure (Eq. 7) and psychrometric constant (Eq. 8) ---
    P = 101.3 * ((293.0 - 0.0065 * elev_m) / 293.0) ** 5.26      # kPa
    gamma = 0.000665 * P                                          # kPa/degC

    # --- Vapour pressure terms ---
    es = (sat_vapour_pressure(tmx) + sat_vapour_pressure(tmn)) / 2.0  # Eq. 12
    ea = sat_vapour_pressure(tmn)                                     # Eq. 48 (Tdew~Tmin)
    # Slope of saturation vapour pressure curve (Eq. 13)
    delta = (4098.0 * sat_vapour_pressure(tmean)
             / (tmean + 237.3) ** 2)

    # --- Extraterrestrial radiation Ra (Eq. 21-25) ---
    lat = math.radians(lat_deg)
    doy = np.asarray(dates.dayofyear, dtype=float)
    dr = 1.0 + 0.033 * np.cos(2.0 * math.pi / 365.0 * doy)        # Eq. 23
    decl = 0.409 * np.sin(2.0 * math.pi / 365.0 * doy - 1.39)     # Eq. 24
    ws_arg = np.clip(-np.tan(lat) * np.tan(decl), -1.0, 1.0)
    ws = np.arccos(ws_arg)                                        # sunset hour angle (Eq. 25)
    Ra = (24.0 * 60.0 / math.pi) * GSC * dr * (
        ws * math.sin(lat) * np.sin(decl)
        + math.cos(lat) * np.cos(decl) * np.sin(ws))             # Eq. 21, MJ m-2 day-1

    # --- Solar radiation from temperature range (Hargreaves, Eq. 50) ---
    dt_range = np.clip(tmx - tmn, 0.0, None)
    Rs = pm["kRs"] * np.sqrt(dt_range) * Ra
    # Clear-sky radiation (Eq. 37) for the longwave cloudiness term
    Rso = (0.75 + 2e-5 * elev_m) * Ra
    Rs = np.minimum(Rs, Rso)                                      # Rs <= Rso

    # --- Net shortwave (Eq. 38) and net longwave (Eq. 39) ---
    Rns = (1.0 - pm["albedo"]) * Rs
    with np.errstate(divide="ignore", invalid="ignore"):
        rs_rso = np.where(Rso > 0, np.clip(Rs / Rso, 0.0, 1.0), 0.0)
    tmaxK = tmx + 273.16
    tminK = tmn + 273.16
    Rnl = (SIGMA * (tmaxK ** 4 + tminK ** 4) / 2.0
           * (0.34 - 0.14 * np.sqrt(np.clip(ea, 0.0, None)))
           * (1.35 * rs_rso - 0.35))                             # MJ m-2 day-1
    Rn = Rns - Rnl                                               # Eq. 40

    # --- FAO-56 Penman-Monteith reference ETo (Eq. 6) ---
    u2 = pm["u2"]
    num = (0.408 * delta * (Rn - pm["G"])
           + gamma * (900.0 / (tmean + 273.0)) * u2 * (es - ea))
    den = delta + gamma * (1.0 + 0.34 * u2)
    eto = num / den
    eto = np.maximum(eto, 0.0)
    return pd.Series(eto, index=dates)


# ----------------------------------------------------------------------
# Crop coefficient: FAO-56 single crop coefficient, Kc (Chapter 6)
#
# ETc = Kc * ETo. Kc follows the FAO-56 seasonal curve (Fig. 34, Eq. 66):
#   - initial stage   : Kc = Kc_ini            (constant)
#   - development     : linear ramp Kc_ini -> Kc_mid
#   - mid-season      : Kc = Kc_mid            (constant)
#   - late season     : linear ramp Kc_mid -> Kc_end
#   - off-season      : Kc = Kc_off (bare soil / fallow)
#
# Kc_mid and Kc_end are climate-adjusted (Eq. 62 / 65) from the tabulated
# values for RHmin != 45% or u2 != 2 m/s. RHmin is estimated from Tmin as a
# dewpoint proxy: RHmin = 100 * e0(Tmin)/e0(Tmax)  (consistent with the PM
# ea = e0(Tmin) assumption used for ETo above).
#
# Table 11 (stage lengths, days) and Table 12 (Kc_ini/mid/end + crop
# height h, m) values below are FAO-56 defaults; Indian-relevant crops.
# Keys match the crop names in the project's Unique_Crops list (uppercase).
# Every row is traceable to FAO-56 Table 11 (stage lengths) and Table 12
# (Kc_ini / Kc_mid / Kc_end and max crop height h); India-relevant Table 11
# rows are preferred where listed. Aggregate categories (MINORPULSES,
# OILSEEDS, FRUITS, VEGETABLES, etc.) use a representative crop and are
# flagged as approximate.
# ----------------------------------------------------------------------
# crop -> (L_ini, L_dev, L_mid, L_late, Kc_ini, Kc_mid, Kc_end, h_m)
CROP_TABLE = {
    # --- Cereals (Table 12 group i: Kc_ini 0.30, Kc_mid 1.15, Kc_end 0.25-0.4) ---
    "rice":          (30,  30,  60,  30,  1.05, 1.20, 0.75, 1.0),  # paddy; T12 rice
    "wheat":         (15,  25,  50,  30,  0.30, 1.15, 0.30, 1.0),  # T11 Central India
    "kharifsorghum": (20,  35,  40,  30,  0.30, 1.10, 0.55, 1.5),  # sorghum grain; kharif
    "rabisorghum":   (20,  35,  45,  30,  0.30, 1.10, 0.55, 1.5),  # sorghum grain; rabi
    "sorghum":       (20,  35,  40,  30,  0.30, 1.10, 0.55, 1.5),  # T11 May/Jun; T12 grain
    "pearlmillet":   (15,  25,  40,  25,  0.30, 1.00, 0.30, 1.5),  # T11 millet Pakistan
    "maize":         (20,  35,  40,  30,  0.30, 1.20, 0.60, 2.0),  # T11 India (grain)
    "fingermillet":  (15,  25,  40,  25,  0.30, 1.00, 0.30, 1.0),  # ragi; millet proxy
    "barley":        (15,  25,  50,  30,  0.30, 1.15, 0.25, 1.0),  # T12 barley
    # --- Legumes / pulses (Table 12 group e: Kc_ini 0.40, Kc_mid 1.15, Kc_end 0.35) ---
    "chickpea":      (20,  30,  40,  20,  0.40, 1.00, 0.35, 0.4),  # T12 chick pea
    "pigeonpea":     (20,  30,  40,  30,  0.40, 1.15, 0.35, 1.0),  # arhar/tur; pulse proxy
    "minorpulses":   (20,  30,  30,  20,  0.40, 1.05, 0.35, 0.4),  # ~green gram/cowpea (approx)
    # --- Oil crops (Table 12 group h: Kc_ini 0.35, Kc_mid 1.15, Kc_end 0.35) ---
    "groundnut":     (25,  35,  45,  25,  0.40, 1.15, 0.60, 0.4),  # T11 dry W Africa; T12 peanut
    "sesamum":       (20,  30,  40,  20,  0.35, 1.10, 0.25, 1.0),  # T11/T12 sesame
    "rapeseedandmustard": (20, 35, 45, 25, 0.35, 1.10, 0.35, 0.6), # T12 rapeseed/canola
    "safflower":     (20,  35,  45,  25,  0.35, 1.10, 0.25, 0.8),  # T11/T12 safflower
    "castor":        (25,  40,  65,  50,  0.35, 1.15, 0.55, 0.3),  # T11/T12 castorbean
    "linseed":       (25,  35,  50,  40,  0.35, 1.10, 0.25, 1.2),  # T11/T12 flax
    "sunflower":     (25,  35,  45,  25,  0.35, 1.10, 0.35, 2.0),  # T11/T12 sunflower
    "soyabean":      (20,  25,  75,  30,  0.40, 1.15, 0.50, 0.75), # T11 Japan; T12 soybean
    "oilseeds":      (25,  35,  45,  25,  0.35, 1.10, 0.35, 0.8),  # aggregate ~ oil crops (approx)
    # --- Fibre / sugar / roots / veg ---
    "sugarcane":     (35,  60,  190, 120, 0.40, 1.25, 0.75, 3.0),  # T11/T12 virgin cane
    "cotton":        (30,  50,  60,  55,  0.35, 1.18, 0.60, 1.35), # T11/T12 cotton
    "potatoes":      (25,  30,  45,  30,  0.50, 1.15, 0.75, 0.6),  # T12 potato
    "onion":         (15,  25,  70,  40,  0.70, 1.05, 0.75, 0.4),  # T11/T12 onion (dry)
    # --- Aggregate horticulture (approximate representatives) ---
    "fruits":        (20,  70,  120, 60,  0.50, 0.95, 0.70, 4.0),  # ~deciduous orchard (approx)
    "vegetables":    (25,  35,  40,  20,  0.60, 1.05, 0.90, 0.5),  # ~small vegetables (approx)
    "fruitsandvegetables": (25, 40, 60, 30, 0.55, 1.00, 0.80, 1.0),# mixed hort. (approx)
    # --- Forage ---
    "fodder":        (10,  20,  60,  30,  0.30, 0.85, 0.85, 0.5),  # ~grazing pasture (approx)
}
KC_OFF = 0.20   # off-season / bare-soil-fallow Kc


def _clip01(x):
    return max(0.0, min(1.0, x))


def climate_adjust_kc(kc_tab, h_m, rhmin_mean, u2, stage):
    """
    FAO-56 Eq. 62 (Kc_mid) / Eq. 65 (Kc_end) climate correction.
    Only applied for Kc >= 0.45 (per FAO-56, and for Kc_end only when the
    tabulated value is > 0.45). rhmin_mean is the mean RHmin (%) over the
    stage; h_m the mean plant height for that stage.
    """
    if stage == "mid" or (stage == "end" and kc_tab > 0.45):
        return (kc_tab
                + (0.04 * (u2 - 2.0) - 0.004 * (rhmin_mean - 45.0))
                * (h_m / 3.0) ** 0.3)
    return kc_tab


def build_kc_series(dates, tmax, tmin, crop, plant_doy, u2,
                    stage_lengths=None, kc_values=None, height=None):
    """
    Build a daily Kc series over the full record for a repeating annual
    growing season starting at plant_doy each year. Returns a Series.
    """
    if crop is not None:
        (Li, Ld, Lm, Ll, kc_ini, kc_mid, kc_end, h) = CROP_TABLE[crop]
    else:
        Li = Ld = Lm = Ll = None
        kc_ini = kc_mid = kc_end = None
        h = 1.0
    # CLI overrides
    if stage_lengths is not None:
        Li, Ld, Lm, Ll = stage_lengths
    if kc_values is not None:
        kc_ini, kc_mid, kc_end = kc_values
    if height is not None:
        h = height
    if None in (Li, Ld, Lm, Ll, kc_ini, kc_mid, kc_end):
        sys.exit("ERROR: crop undefined; pass --crop or "
                 "--kc-stages and --kc-values")

    season_len = Li + Ld + Lm + Ll

    # RHmin(%) from Tmin dewpoint proxy: 100 * e0(Tmin)/e0(Tmax).
    tmx = np.asarray(tmax, dtype=float)
    tmn = np.asarray(tmin, dtype=float)
    rhmin = 100.0 * sat_vapour_pressure(tmn) / sat_vapour_pressure(tmx)
    rhmin = np.clip(rhmin, 5.0, 100.0)
    rhmin_ser = pd.Series(rhmin, index=dates)

    doy = np.asarray(dates.dayofyear, dtype=float)
    # Day-into-season for each date (0 at plant_doy), wrapping the year.
    dis = (doy - plant_doy) % 365.0

    # Mean RHmin over mid and late windows (for Eq. 62/65), approximated
    # by the record-wide mean RHmin on days that fall in each stage.
    in_mid = (dis >= Li + Ld) & (dis < Li + Ld + Lm)
    in_late = (dis >= Li + Ld + Lm) & (dis < season_len)
    rhmin_mid = float(rhmin_ser[in_mid].mean()) if in_mid.any() else 45.0
    rhmin_late = float(rhmin_ser[in_late].mean()) if in_late.any() else 45.0

    kc_mid_adj = climate_adjust_kc(kc_mid, h, rhmin_mid, u2, "mid")
    kc_end_adj = climate_adjust_kc(kc_end, h, rhmin_late, u2, "end")

    kc = np.full(len(dates), KC_OFF, dtype=float)
    for k, d in enumerate(dis):
        if d < Li:                                   # initial
            kc[k] = kc_ini
        elif d < Li + Ld:                            # development ramp
            f = (d - Li) / Ld
            kc[k] = kc_ini + f * (kc_mid_adj - kc_ini)
        elif d < Li + Ld + Lm:                       # mid-season
            kc[k] = kc_mid_adj
        elif d < season_len:                         # late-season ramp
            f = (d - (Li + Ld + Lm)) / Ll
            kc[k] = kc_mid_adj + f * (kc_end_adj - kc_mid_adj)
        # else: off-season -> KC_OFF (already set)
    return pd.Series(kc, index=dates)


# ----------------------------------------------------------------------
# Snow module (temperature-threshold + degree-day melt)
# ----------------------------------------------------------------------
def snow_step(precip, tmean, snowpack, snow_params=None):
    """Return (effective_water_to_soil, new_snowpack)."""
    if snow_params is None:
        snow_params = SNOW
    if tmean <= snow_params["t_snow"]:
        snowpack += precip            # all precip accumulates as snow
        rain = 0.0
    else:
        rain = precip                 # all precip is rain
    melt = 0.0
    if tmean > snow_params["t_melt"] and snowpack > 0.0:
        melt = min(snowpack, snow_params["melt_factor"] * (tmean - snow_params["t_melt"]))
        snowpack -= melt
    return rain + melt, snowpack


# ----------------------------------------------------------------------
# Core leaky-bucket update
# ----------------------------------------------------------------------
def beta(w, wmax=WMAX):
    """Evaporation efficiency = w / wmax (linearly water-limited)."""
    return w / wmax


def runoff(w, peff, p, wmax=WMAX):
    """
    CPC-style runoff: surface runoff scales with the storage ratio raised
    to Bm times the incoming water, plus a linear base runoff.
    """
    ratio = max(0.0, min(1.0, w / wmax))
    r_surf = p["alpha_surf"] * (ratio ** p["Bm"]) * peff
    r_base = p["alpha_base"] * w * ratio
    return r_surf + r_base


# ======================================================================
# IRRIGATION MODULE
# ======================================================================
#
# METHODOLOGY AND PROVENANCE
# --------------------------
# This module replaces the earlier "refill the bucket to WMAX" treatment
# with the demand formulation standard in macroscale hydrological models
# that carry an explicit irrigation scheme. Three literature strands are
# combined; each is cited at the equation it supplies.
#
#   [1] Droppers, B., Franssen, W.H.P., van Vliet, M.T.H., Nijssen, B.,
#       and Ludwig, F. (2020). Simulating human impacts on global water
#       resources using VIC-5. Geoscientific Model Development 13,
#       5029-5052. doi:10.5194/gmd-13-5029-2020.   [VIC-WUR]
#   [2] Hanasaki, N., Kanae, S., Oki, T., Masuda, K., Motoya, K.,
#       Shirakawa, N., Shen, Y., and Tanaka, K. (2008). An integrated
#       model for the assessment of global water resources - Part 1 and
#       Part 2. Hydrology and Earth System Sciences 12, 1007-1025 and
#       1027-1037.                                  [H08]
#   [3] Allen, R.G., Pereira, L.S., Raes, D., and Smith, M. (1998). Crop
#       Evapotranspiration - Guidelines for Computing Crop Water
#       Requirements. FAO Irrigation and Drainage Paper 56, Rome.
#                                                    [FAO-56]
#   [4] Huang, J., van den Dool, H.M., and Georgakakos, K.G. (1996).
#       Analysis of model-calculated soil moisture over the US
#       (1931-1993). Journal of Climate 9, 1350-1362.  [CPC bucket]
#
# ---------------------------------------------------------------------
# 1. WHY THE PREVIOUS TREATMENT WAS PHYSICALLY WRONG
# ---------------------------------------------------------------------
# The prior code set  I = WMAX - w(t), i.e. it refilled the column to
# saturation at 100% efficiency. Three independent errors:
#
#   (a) WRONG TARGET. In VIC-WUR [1, Eq. D1] conventional irrigation
#       demand is triggered when soil moisture falls below the CRITICAL
#       content at which evapotranspiration becomes limited, and the
#       demand relieves exactly that stress:
#           ID'_conventional = (Wcr,1 + Wcr,2) - (W1 + W2),
#                              for W1 + W2 < Wcr,1 + Wcr,2
#       The target is Wcr, NOT Wmax. Saturation is reserved in [1] for
#       rice paddy alone [1, Eq. D2: ID'_paddy = Wmax,1 - W1], following
#       [2]. Rainfed/well-irrigated rabi and kharif systems in semi-arid
#       Marathwada are not paddy, so the paddy branch does not apply.
#
#   (b) NO EFFICIENCY. VIC-WUR [1, Eq. D3] converts net demand to gross
#       withdrawal by the irrigation efficiency,  ID = ID' * IE, and
#       notes that transport and application losses are "not lost but
#       rather returned to the soil column without being used by the
#       crop". A 100%-efficient application misstates both the depth
#       withdrawn and the percolation signal.
#
#   (c) NO SUPPLY LIMIT. A cultivator abstracting from a 20-25 m dug
#       well or an 80 m borewell cannot deliver unbounded depth in one
#       day; pump discharge, plot size and labour bound the event.
#
# ---------------------------------------------------------------------
# 2. ROOT-ZONE MAPPING  (the critical scaling step)
# ---------------------------------------------------------------------
# The CPC bucket's WMAX = 760 mm is NOT a plant-available water holding
# capacity. It is a parameter tuned to reproduce observed runoff in
# small eastern-Oklahoma basins; with an assumed porosity of 0.47 it
# corresponds to a 1.6 m column [4; CPC Soil Moisture documentation].
# It therefore represents TOTAL column storage over a depth far exceeding
# the rooting depth of the crops of interest.
#
# FAO-56 quantities are defined over the ROOT ZONE and over PLANT-
# AVAILABLE water only [3, Eqs. 82-83]:
#       TAW = 1000 * (theta_FC - theta_WP) * Zr        [3, Eq. 82]
#       RAW = p * TAW                                  [3, Eq. 83]
# Applying "p * WMAX" directly to the CPC bucket would overstate RAW by
# roughly a factor of three and defeat the purpose of the correction.
#
# We therefore introduce an explicit mapping. Let
#       f_AW = (theta_FC - theta_WP) / porosity
# be the plant-available fraction of total pore space, and
#       f_Zr = Zr / Z_col
# the fraction of the modelled column occupied by roots. Then
#       TAW = WMAX * f_AW * f_Zr                              (Eq. I1)
# and the root-zone storage tracked against that capacity is the same
# fraction of the bucket:
#       w_rz(t) = w(t) * f_AW * f_Zr                          (Eq. I2)
# so that the storage RATIO is preserved, w_rz/TAW = w/WMAX. Depletion
# in FAO-56 terms is then
#       Dr(t) = TAW - w_rz(t)                                 (Eq. I3)
#
# IMPORTANT - direction of use. Eqs. I1-I3 map the model STATE into
# FAO-56 units so that the trigger and depth can be evaluated against
# agronomically meaningful thresholds. They are NOT a unit conversion
# for the applied FLUX. Irrigation depth is an absolute quantity of
# water per unit field area and enters the column balance unscaled,
# exactly as precipitation does. Dividing the depth by f to "return it
# to column units" would inject roughly 1/f times the water actually
# applied and would violate mass conservation. Because the CPC column
# is deeper than the root zone, a given application therefore raises
# w/WMAX by less than it raises the notional root-zone fraction; this
# is the physically correct behaviour, since water applied at the
# surface is distributed through, and drains from, the whole column.
#
# Defaults: theta_FC = 0.32, theta_WP = 0.17 for the medium-to-deep
# clayey and gravelly clay-loam soils typical of the Osmanabad/Tuljapur
# cluster; porosity 0.47 to stay consistent with the CPC column
# definition; Z_col = 1.6 m. This yields f_AW = 0.319. With Zr = 1.0 m,
# f_Zr = 0.625 and TAW = 760 * 0.319 * 0.625 = 151.6 mm, which is the
# physically expected order for a 1 m root zone and consistent with the
# H08 bucket, where Smax = SD * (fFC - fWP) with defaults SD = 1 m,
# fFC = 0.30, fWP = 0.15 giving 150 mm [2].
#
# ---------------------------------------------------------------------
# 3. TRIGGER  (management-allowed depletion)
# ---------------------------------------------------------------------
# An event fires when root-zone depletion reaches readily available
# water [3, Eqs. 83, 84] - the operational form of the VIC-WUR critical
# threshold [1, Eq. D1]:
#       irrigate when  Dr(t) >= RAW = p_adj * TAW             (Eq. I4)
# The tabulated depletion fraction is adjusted for atmospheric demand
# [3, Table 22 note; p typically 0.30-0.65]:
#       p_adj = p_table + 0.04 * (5 - ETc),  clipped to [0.1, 0.8]
#                                                             (Eq. I5)
# so that under high evaporative demand the crop stresses at a smaller
# depletion and the cultivator irrigates sooner.
#
# ---------------------------------------------------------------------
# 4. DEPTH  (net, gross, and the three binding constraints)
# ---------------------------------------------------------------------
# Net depth targets field capacity, optionally under deficit irrigation
# with refill fraction f_r <= 1:
#       I_net = min( f_r * Dr(t),  I_net_cap )                (Eq. I6)
# Gross withdrawal follows VIC-WUR Eq. D3:
#       I_gross = I_net / Ea                                  (Eq. I7)
# subject to the supply ceiling, with net back-computed if it binds:
#       I_gross = min(I_gross, I_supply_cap);  I_net = I_gross * Ea
#                                                             (Eq. I8)
# Following [1], application losses are returned to the soil column
# rather than discarded, so the depth entering the water balance is the
# GROSS depth; the loss fraction I_gross*(1-Ea) is then free to leave
# through the model's existing runoff and drainage terms. The gross
# depth is expressed back in bucket units by inverting Eq. I2.
#
# Application efficiencies Ea are the FAO/field-study consensus values:
# subsurface drip ~0.90, centre pivot / linear ~0.85, sprinkler ~0.75,
# furrow and surface flood ~0.55-0.65. Surface flood is the dominant
# method for all crops except groundnut in semi-arid Indian smallholder
# catchments, so `flood` is the default here.
#
# ---------------------------------------------------------------------
# 5. TWO OPERATING MODES
# ---------------------------------------------------------------------
#   SCHEDULED (--daysbefore N): a single counterfactual event N days
#       before the end of record. Answers "what does the profile look
#       like if the cultivator irrigated N days ago?" Depth is sized by
#       Eqs. I6-I8 rather than by saturation. This preserves the
#       existing CLI contract.
#   AUTOMATIC (--irrigate): the trigger of Eq. I4 is evaluated every
#       day, subject to a minimum inter-event interval representing
#       pump/labour availability. Answers "what irrigation does this
#       forcing imply?" and yields a full season schedule.
# The two are mutually exclusive.
# ======================================================================


# Application efficiency Ea (root-zone delivery per unit withdrawn) and
# per-event depth ceilings (mm) by method. Ea after FAO/field practice
# (see Sec. 4 above); caps reflect single-event depths feasible for
# smallholder abstraction in the pilot cluster.
IRRIG_METHODS = {
    "flood":     {"Ea": 0.60, "I_net_cap": 50.0, "supply_cap": 60.0},
    "furrow":    {"Ea": 0.60, "I_net_cap": 45.0, "supply_cap": 60.0},
    "border":    {"Ea": 0.65, "I_net_cap": 50.0, "supply_cap": 65.0},
    "sprinkler": {"Ea": 0.75, "I_net_cap": 35.0, "supply_cap": 45.0},
    "pivot":     {"Ea": 0.85, "I_net_cap": 30.0, "supply_cap": 35.0},
    "drip":      {"Ea": 0.90, "I_net_cap": 20.0, "supply_cap": 25.0},
}

# FAO-56 Table 22 depletion fractions p for crops of the pilot cluster.
# Fallback 0.50 where a crop is absent.
CROP_P = {
    "soybean": 0.50, "cotton": 0.65, "sorghum": 0.55, "wheat": 0.55,
    "maize": 0.55, "chickpea": 0.50, "pigeonpea": 0.50, "groundnut": 0.50,
    "greengram": 0.45, "onion": 0.30, "tomato": 0.40, "potato": 0.35,
    "sugarcane": 0.65, "grapes": 0.35, "sunflower": 0.45, "millet": 0.55,
}

# Root-zone mapping defaults (Sec. 2). Soil water contents are volumetric.
ROOTZONE = dict(
    theta_fc=0.32,   # field capacity, medium/deep clayey soils
    theta_wp=0.17,   # wilting point
    porosity=0.47,   # CPC column porosity, keeps WMAX self-consistent
    z_col=1.6,       # CPC modelled column depth (m)
    zr=1.0,          # crop rooting depth (m)
)


def rootzone_scale(rz=ROOTZONE):
    """Fraction of bucket storage that is plant-available root-zone water.

    Returns f = f_AW * f_Zr such that TAW = WMAX * f and w_rz = w * f
    (Eqs. I1, I2). Preserves the storage ratio w/WMAX.
    """
    f_aw = (rz["theta_fc"] - rz["theta_wp"]) / rz["porosity"]
    f_zr = min(rz["zr"] / rz["z_col"], 1.0)
    return float(np.clip(f_aw * f_zr, 1e-3, 1.0))


def adjusted_p(p_table, etc_mm):
    """FAO-56 demand adjustment, Eq. I5.  p_adj = p + 0.04*(5 - ETc)."""
    return float(np.clip(p_table + 0.04 * (5.0 - etc_mm), 0.1, 0.8))


def irrigation_demand(w, wmax, etc_mm, cfg):
    """Depth for one potential irrigation event. All depths in mm.

    Implements Eqs. I3-I8. Operates on the CPC bucket state `w` but
    evaluates the trigger in FAO-56 root-zone terms.

    Parameters
    ----------
    w      : current bucket storage (mm, CPC column units)
    wmax   : bucket capacity (mm, CPC column units)
    etc_mm : crop water demand for the day (mm), for the p adjustment
    cfg    : dict with keys p, Ea, I_net_cap, supply_cap, refill_frac,
             rz_scale, adjust_p

    Returns
    -------
    dict: applied, I_gross_bucket (mm, to add to the water balance),
          I_gross, I_net, I_loss, Dr, TAW, RAW, p_adj, limited_by
    """
    f = cfg["rz_scale"]
    taw = wmax * f                       # Eq. I1
    w_rz = w * f                         # Eq. I2
    dr = max(taw - w_rz, 0.0)            # Eq. I3

    p_adj = adjusted_p(cfg["p"], etc_mm) if cfg["adjust_p"] else cfg["p"]
    raw = p_adj * taw

    res = dict(applied=False, I_gross_bucket=0.0, I_gross=0.0, I_net=0.0,
               I_loss=0.0, Dr=dr, TAW=taw, RAW=raw, p_adj=p_adj,
               limited_by="no_trigger")

    if dr < raw:                         # Eq. I4 not met: no stress
        return res

    i_net = dr * cfg["refill_frac"]      # Eq. I6
    limited = "demand"
    if i_net > cfg["I_net_cap"]:
        i_net = cfg["I_net_cap"]
        limited = "I_net_cap"

    i_gross = i_net / cfg["Ea"]          # Eq. I7
    if i_gross > cfg["supply_cap"]:      # Eq. I8
        i_gross = cfg["supply_cap"]
        i_net = i_gross * cfg["Ea"]
        limited = "supply_cap"

    # The applied depth is an ABSOLUTE quantity of water (mm over the
    # field) and enters the column water balance unscaled, exactly like
    # precipitation. The root-zone mapping (Eqs. I1-I3) is a diagnostic
    # for sizing the event, not a unit conversion for the flux: scaling
    # the depth by 1/f would inject several times the water the
    # cultivator actually applied and would not conserve mass.
    res.update(applied=True,
               I_gross_bucket=i_gross,
               I_gross=i_gross, I_net=i_net, I_loss=i_gross - i_net,
               limited_by=limited)
    return res


def build_irrig_config(method="flood", p=None, crop=None, Ea=None,
                       net_cap=None, supply_cap=None, refill_frac=1.0,
                       min_interval=5, adjust_p=True, rz=ROOTZONE):
    """Assemble the irrigation parameter set, applying method presets."""
    preset = IRRIG_METHODS.get(method, IRRIG_METHODS["flood"])
    if p is None:
        p = CROP_P.get((crop or "").lower(), 0.50)
    cfg = dict(
        method=method,
        p=float(p),
        Ea=float(Ea if Ea is not None else preset["Ea"]),
        I_net_cap=float(net_cap if net_cap is not None
                        else preset["I_net_cap"]),
        supply_cap=float(supply_cap if supply_cap is not None
                         else preset["supply_cap"]),
        refill_frac=float(refill_frac),
        min_interval=int(min_interval),
        adjust_p=bool(adjust_p),
        rz_scale=rootzone_scale(rz),
    )
    if not 0.05 < cfg["Ea"] <= 1.0:
        sys.exit(f"ERROR: irrigation efficiency must be in (0.05, 1.0], "
                 f"got {cfg['Ea']}")
    if not 0.05 < cfg["p"] <= 0.9:
        sys.exit(f"ERROR: depletion fraction p must be in (0.05, 0.9], "
                 f"got {cfg['p']}")
    if not 0.0 < cfg["refill_frac"] <= 1.0:
        sys.exit(f"ERROR: refill fraction must be in (0, 1], "
                 f"got {cfg['refill_frac']}")
    return cfg


def run_model(precip, tmean, pe, lat, params=PARAMS, snow=True,
              w0_frac=0.5, spinup_years=1, irrig_daysbefore=None,
              wmax=WMAX, snow_params=None,
              irrig_cfg=None, irrig_auto=False):
    """Run the daily water balance. Returns a DataFrame of states/fluxes.

    Irrigation (see IRRIGATION MODULE above) operates in one of two
    mutually exclusive modes:
      * scheduled: `irrig_daysbefore` sets a single event that many days
        before the end of record, sized by Eqs. I6-I8.
      * automatic: `irrig_auto=True` evaluates the FAO-56 trigger
        (Eq. I4) every day, honouring cfg['min_interval'].
    `irrig_cfg` comes from build_irrig_config(); if None while a mode is
    requested, defaults (surface flood, p from crop table) are used.

    `wmax` and `snow_params` are passed per-point so that concurrent
    callers (e.g. the API server driving many villages) never mutate
    shared module state.
    """
    if snow_params is None:
        snow_params = SNOW
    idx = precip.index
    n = len(idx)
    w = w0_frac * wmax
    snowpack = 0.0

    if irrig_daysbefore is not None and irrig_auto:
        sys.exit("ERROR: --daysbefore and --irrigate are mutually exclusive")

    # Single scheduled event: index of the day it was applied.
    irrig_k = None
    if irrig_daysbefore is not None:
        if irrig_daysbefore < 0 or irrig_daysbefore >= n:
            sys.exit(f"ERROR: --daysbefore must be in [0, {n-1}]")
        irrig_k = n - 1 - irrig_daysbefore

    if (irrig_k is not None or irrig_auto) and irrig_cfg is None:
        irrig_cfg = build_irrig_config()
    last_event = -10 ** 9

    out = np.zeros((n, 9))  # peff, snow, w, E, R, G, irrig, irrig_net, Dr
    pv = precip.values
    tv = tmean.values
    ev = pe.values

    for k in range(n):
        p = pv[k]
        t = tv[k]
        if snow:
            peff, snowpack = snow_step(p if not np.isnan(p) else 0.0, t,
                                       snowpack, snow_params=snow_params)
        else:
            peff = p if not np.isnan(p) else 0.0

        pe_k = ev[k] if not np.isnan(ev[k]) else 0.0

        # --- Irrigation (Eqs. I3-I8) ---------------------------------
        # Gross depth enters the balance exactly like precipitation
        # (FAO-56 Ch. 8); application losses stay in the column and
        # leave via the existing runoff/drainage terms, after VIC-WUR.
        irrig = 0.0
        irrig_net = 0.0
        dr_k = np.nan
        if irrig_cfg is not None:
            due = (irrig_k is not None and k == irrig_k) or (
                irrig_auto and (k - last_event) >= irrig_cfg["min_interval"])
            if due:
                evt = irrigation_demand(w, wmax, pe_k, irrig_cfg)
                dr_k = evt["Dr"]
                if evt["applied"]:
                    irrig = evt["I_gross_bucket"]
                    irrig_net = evt["I_net"]
                    peff += irrig
                    last_event = k

        # Fluxes evaluated on the storage at start of step (explicit).
        E = beta(w, wmax=wmax) * pe_k
        R = runoff(w, peff, params, wmax=wmax)
        G = params["gamma"] * (w / wmax)

        w_new = w + peff - E - R - G

        # Enforce bounds; spill overflow into runoff, deficit into reduced E.
        if w_new > wmax:
            R += (w_new - wmax)
            w_new = wmax
        if w_new < 0.0:
            # scale losses back so storage hits exactly zero
            deficit = -w_new
            total_loss = E + R + G
            if total_loss > 0:
                E -= E / total_loss * deficit
                R -= R / total_loss * deficit
                G -= G / total_loss * deficit
            w_new = 0.0

        out[k] = (peff, snowpack, w_new, E, R, G, irrig, irrig_net, dr_k)
        w = w_new

    df = pd.DataFrame(
        out, index=idx,
        columns=["P_eff", "snowpack", "w", "E", "R", "G", "irrig",
                 "irrig_net", "Dr"],
    )
    df["P_obs"] = precip.values
    df["Tmean"] = tmean.values
    df["PE"] = pe.values
    df["w_frac"] = df["w"] / wmax

    # Root-zone storage in FAO-56 terms (Eqs. I1-I2). Reported so that
    # irrigation depths and the state they act on share one unit system.
    f = irrig_cfg["rz_scale"] if irrig_cfg is not None else rootzone_scale()
    df["w_rz"] = df["w"] * f
    df["TAW"] = wmax * f

    # Drop spin-up.
    if spinup_years > 0:
        cutoff = idx[0] + pd.DateOffset(years=spinup_years)
        df = df[df.index >= cutoff]
    return df[["P_obs", "Tmean", "PE", "P_eff", "irrig", "irrig_net", "Dr",
               "snowpack", "w", "w_rz", "TAW", "E", "R", "G", "w_frac"]]


# ----------------------------------------------------------------------
def cpc_leaky_bucket_pipeline(
    base=None,
    pcp=None,
    tmax=None,
    tmin=None,
    lat=None,
    elev=None,
    out="soil_moisture.csv",
    no_snow=False,
    w0=0.5,
    spinup=1,
    crop=None,
    plant_doy=1,
    kc_stages=None,
    kc_values=None,
    crop_height=None,
    elev_file=None,
    lon=None,
    daysbefore=None,
    irrigate=False,
    irr_method="flood",
    irr_p=None,
    irr_efficiency=None,
    irr_net_cap=None,
    irr_supply_cap=None,
    irr_refill_frac=1.0,
    irr_min_interval=5,
    irr_no_et_adjust=False,
    irr_root_depth=ROOTZONE["zr"],
    irr_theta_fc=ROOTZONE["theta_fc"],
    irr_theta_wp=ROOTZONE["theta_wp"],
):
    if base and not (pcp and tmax and tmin):
        pcp, tmax, tmin = derive_paths(base)
    if not (pcp and tmax and tmin):
        sys.exit("ERROR: supply --base, or all of --pcp --tmax --tmin")

    if lat is None:
        lat = lat_from_filename(pcp)
        if lat is None:
            sys.exit("ERROR: could not parse latitude; pass --lat")

    # Resolve elevation: explicit --elev wins; else look up in --elev-file
    # by the point's lat/lon; else default to 0 m.
    if elev is None:
        if elev_file:
            if lon is None:
                _, lon = latlon_from_filename(pcp)
                if lon is None:
                    sys.exit("ERROR: --elev-file needs longitude; pass --lon")
            table, elat, elon, eelev = load_elevation_file(elev_file)
            elev = lookup_elevation(lat, lon,
                                    table, elat, elon, eelev)
        else:
            elev = 0.0

    pcp_series = read_point_file(pcp)
    tmax_series = read_point_file(tmax)
    tmin_series = read_point_file(tmin)

    # Align on common dates.
    df = pd.concat({"pcp": pcp_series, "tmax": tmax_series, "tmin": tmin_series}, axis=1).dropna(
        subset=["tmax", "tmin"])
    df["pcp"] = df["pcp"].fillna(0.0)
    tmean = (df["tmax"] + df["tmin"]) / 2.0

    pe = penman_monteith_eto(df["tmax"], df["tmin"], df.index,
                             lat, elev)

    # FAO-56 crop coefficient: ETc = Kc * ETo. Applied only if requested
    # via --crop or explicit --kc-values overrides; otherwise Kc = 1
    # (bare reference ET, unchanged from before).
    if crop is not None or kc_values is not None:
        kc = build_kc_series(
            df.index, df["tmax"], df["tmin"],
            crop=crop, plant_doy=plant_doy, u2=PM["u2"],
            stage_lengths=kc_stages, kc_values=kc_values,
            height=crop_height,
        )
        pe = pe * kc          # ETc = Kc * ETo

    # Irrigation configuration (see IRRIGATION MODULE, Secs. 2-4).
    irrig_cfg = None
    if daysbefore is not None or irrigate:
        rz = dict(ROOTZONE, zr=irr_root_depth,
                  theta_fc=irr_theta_fc, theta_wp=irr_theta_wp)
        if rz["theta_fc"] <= rz["theta_wp"]:
            sys.exit("ERROR: irr_theta_fc must exceed irr_theta_wp")
        irrig_cfg = build_irrig_config(
            method=irr_method, p=irr_p, crop=crop,
            Ea=irr_efficiency, net_cap=irr_net_cap,
            supply_cap=irr_supply_cap,
            refill_frac=irr_refill_frac,
            min_interval=irr_min_interval,
            adjust_p=not irr_no_et_adjust, rz=rz,
        )

    result = run_model(
        df["pcp"], tmean, pe, lat,
        snow=not no_snow, w0_frac=w0, spinup_years=spinup,
        irrig_daysbefore=daysbefore,
        irrig_cfg=irrig_cfg, irrig_auto=irrigate,
    )
    if out:
        result.to_csv(out, float_format="%.3f",
                      index_label="date")
    return result


def main():
    ap = argparse.ArgumentParser(description="CPC leaky-bucket soil moisture")
    ap.add_argument("--base", help="one RT_* point file; others derived")
    ap.add_argument("--pcp")
    ap.add_argument("--tmax")
    ap.add_argument("--tmin")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--elev", type=float, default=None,
                    help="site elevation (m) for FAO-56 PM pressure/gamma; "
                         "overrides --elev-file. Default 0 m if neither given")
    ap.add_argument("--out", default="soil_moisture.csv")
    ap.add_argument("--no-snow", action="store_true")
    ap.add_argument("--w0", type=float, default=0.5,
                    help="initial storage as fraction of wmax")
    ap.add_argument("--spinup", type=int, default=1,
                    help="spin-up years to discard")
    # --- FAO-56 crop coefficient (ETc = Kc * ETo) ---
    ap.add_argument("--crop", choices=sorted(CROP_TABLE.keys()),
                    help="apply FAO-56 seasonal Kc for this crop; "
                         "omit for bare reference ET (Kc=1)")
    ap.add_argument("--plant-doy", type=int, default=1,
                    help="growing-season start day-of-year (repeats yearly)")
    ap.add_argument("--kc-stages", type=int, nargs=4,
                    metavar=("LINI", "LDEV", "LMID", "LLATE"),
                    help="override stage lengths (days)")
    ap.add_argument("--kc-values", type=float, nargs=3,
                    metavar=("KCINI", "KCMID", "KCEND"),
                    help="override tabulated Kc_ini/mid/end")
    ap.add_argument("--crop-height", type=float,
                    help="override crop height (m) for Kc climate adjustment")
    ap.add_argument("--elev-file",
                    help="'lon lat elev' grid file; elevation looked up by "
                         "the point's lat/lon (overridden by explicit --elev)")
    ap.add_argument("--lon", type=float,
                    help="point longitude for --elev-file lookup "
                         "(parsed from filename if omitted)")
    # --- Irrigation (VIC-WUR / H08 / FAO-56; see IRRIGATION MODULE) ---
    g = ap.add_argument_group("irrigation")
    g.add_argument("--daysbefore", type=int,
                   help="scheduled mode: one irrigation event this many "
                        "days before the end of record (0 = last day). "
                        "Depth is demand-based, not saturating.")
    g.add_argument("--irrigate", action="store_true",
                   help="automatic mode: fire the FAO-56 depletion trigger "
                        "(Dr >= RAW) whenever it is met")
    g.add_argument("--irr-method", default="flood",
                   choices=sorted(IRRIG_METHODS),
                   help="application method; sets efficiency and depth caps "
                        "(default: flood, dominant in the pilot cluster)")
    g.add_argument("--irr-p", type=float, default=None,
                   help="FAO-56 depletion fraction p (MAD). Default: from "
                        "the crop table, else 0.50")
    g.add_argument("--irr-efficiency", type=float, default=None,
                   help="application efficiency Ea; overrides method preset")
    g.add_argument("--irr-net-cap", type=float, default=None,
                   help="maximum NET depth per event (mm)")
    g.add_argument("--irr-supply-cap", type=float, default=None,
                   help="maximum GROSS depth per event (mm): well/pump limit")
    g.add_argument("--irr-refill-frac", type=float, default=1.0,
                   help="1.0 refills to field capacity; <1.0 gives deficit "
                        "irrigation")
    g.add_argument("--irr-min-interval", type=int, default=5,
                   help="minimum days between events in automatic mode")
    g.add_argument("--irr-no-et-adjust", action="store_true",
                   help="disable the FAO-56 p adjustment for daily ETc")
    g.add_argument("--irr-root-depth", type=float, default=ROOTZONE["zr"],
                   help="crop rooting depth Zr (m) for the root-zone "
                        "mapping (Eq. I1)")
    g.add_argument("--irr-theta-fc", type=float, default=ROOTZONE["theta_fc"],
                   help="volumetric field capacity for the root-zone mapping")
    g.add_argument("--irr-theta-wp", type=float, default=ROOTZONE["theta_wp"],
                   help="volumetric wilting point for the root-zone mapping")
    args = ap.parse_args()

    cpc_leaky_bucket_pipeline(
        base=args.base,
        pcp=args.pcp,
        tmax=args.tmax,
        tmin=args.tmin,
        lat=args.lat,
        elev=args.elev,
        out=args.out,
        no_snow=args.no_snow,
        w0=args.w0,
        spinup=args.spinup,
        crop=args.crop,
        plant_doy=args.plant_doy,
        kc_stages=args.kc_stages,
        kc_values=args.kc_values,
        crop_height=args.crop_height,
        elev_file=args.elev_file,
        lon=args.lon,
        daysbefore=args.daysbefore,
        irrigate=args.irrigate,
        irr_method=args.irr_method,
        irr_p=args.irr_p,
        irr_efficiency=args.irr_efficiency,
        irr_net_cap=args.irr_net_cap,
        irr_supply_cap=args.irr_supply_cap,
        irr_refill_frac=args.irr_refill_frac,
        irr_min_interval=args.irr_min_interval,
        irr_no_et_adjust=args.irr_no_et_adjust,
        irr_root_depth=args.irr_root_depth,
        irr_theta_fc=args.irr_theta_fc,
        irr_theta_wp=args.irr_theta_wp,
    )


if __name__ == "__main__":
    main()
