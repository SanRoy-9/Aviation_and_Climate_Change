import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    USE_CARTOPY = True
except ImportError:
    USE_CARTOPY = False

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
ERA5_DIR = RAW_DIR / "era5"
FIG_DIR = BASE_DIR / "reports" / "figures"
TAB_DIR = BASE_DIR / "reports" / "tables"
ROUTE_FILE = RAW_DIR / "eurostat_corridor_pax_2008_2023.csv"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

USE_EXACT_SAC = True
T_THRESHOLD_C = -40.0
RHI_THRESHOLD = 100.0

ETA_PROP = 0.30
EI_H2O = 1.223
Q_FUEL = 43.2e6

ALL_TARGET_LEVELS = [200, 250, 300, 400]
CORE_LEVELS = [250, 300]

LEVEL_LABELS = {
    200: "200 hPa (~39,000 ft / 11.8 km — Lower Stratosphere)",
    250: "250 hPa (~34,000 ft / 10.4 km — High Cruise)",
    300: "300 hPa (~30,000 ft / 9.1 km — Low Cruise)",
    400: "400 hPa (~23,500 ft / 7.2 km — Mid Troposphere)",
}

SEASON_LABELS = {
    "DJF": "Winter\n(Dec–Feb)",
    "MAM": "Spring\n(Mar–May)",
    "JJA": "Summer\n(Jun–Aug)",
    "SON": "Autumn\n(Sep–Nov)",
}

AIRPORTS = {
    "London (LHR)": (-0.46, 51.47),
    "Paris (CDG)": (2.55, 49.01),
    "Frankfurt (FRA)": (8.57, 50.03),
    "Amsterdam (AMS)": (4.76, 52.31),
    "Madrid (MAD)": (-3.56, 40.48),
    "Rome (FCO)": (12.24, 41.80),
    "Dublin (DUB)": (-6.27, 53.42),
    "Vienna (VIE)": (16.57, 48.11),
    "Lisbon (LIS)": (-9.13, 38.77),
    "Stockholm (ARN)": (17.92, 59.65),
    "Warsaw (WAW)": (20.97, 52.17),
}

CORRIDORS = [
    ("London (LHR)", "Frankfurt (FRA)"),
    ("Paris (CDG)", "Frankfurt (FRA)"),
    ("London (LHR)", "Amsterdam (AMS)"),
    ("Paris (CDG)", "Madrid (MAD)"),
    ("Frankfurt (FRA)", "Rome (FCO)"),
    ("Frankfurt (FRA)", "Vienna (VIE)"),
    ("London (LHR)", "Dublin (DUB)"),
    ("Madrid (MAD)", "Lisbon (LIS)"),
    ("Frankfurt (FRA)", "Warsaw (WAW)"),
    ("Frankfurt (FRA)", "Stockholm (ARN)"),
]

EUROPE_COASTLINES = [
    [(-5.7, 50.1), (-5.0, 50.2), (-3.5, 50.4), (-1.9, 50.7), (0.2, 50.8), (1.4, 51.3),
     (0.8, 51.5), (1.7, 52.5), (0.5, 52.9), (0.1, 53.5), (-0.2, 54.1), (-1.5, 55.0),
     (-2.0, 55.8), (-3.1, 56.0), (-2.1, 57.0), (-1.8, 57.7), (-4.0, 57.8), (-3.1, 58.6),
     (-5.0, 58.6), (-5.8, 57.3), (-5.6, 56.0), (-4.8, 55.0), (-3.5, 54.5), (-3.1, 53.4),
     (-4.5, 53.3), (-4.1, 52.5), (-5.3, 51.9), (-4.1, 51.5), (-5.7, 50.1)],
    [(-10.0, 51.6), (-8.5, 51.9), (-6.4, 52.2), (-6.0, 53.3), (-6.2, 54.0), (-5.5, 54.6),
     (-6.1, 55.2), (-7.5, 55.3), (-8.5, 54.9), (-8.5, 54.3), (-10.0, 54.2), (-9.9, 53.5),
     (-9.0, 53.1), (-9.9, 52.6), (-10.0, 51.6)],
    [(-9.0, 37.0), (-9.5, 38.8), (-8.9, 41.9), (-9.3, 43.0), (-7.8, 43.7), (-1.8, 43.4),
     (-1.2, 44.6), (-1.1, 46.0), (-2.5, 47.5), (-4.7, 48.4), (-1.6, 48.6), (-1.9, 49.7),
     (0.1, 49.5), (1.6, 50.9), (3.4, 51.4), (4.2, 52.0), (4.8, 53.0), (7.1, 53.4),
     (8.6, 53.9), (8.1, 55.5), (8.6, 57.1), (10.6, 57.7), (10.5, 56.0), (9.5, 54.8),
     (10.9, 54.0), (12.1, 54.2), (14.2, 54.0), (16.5, 54.5), (18.6, 54.4), (19.6, 54.5),
     (21.0, 55.8), (21.8, 57.6), (24.0, 57.0), (24.4, 58.0), (23.5, 59.4), (28.0, 59.5),
     (30.2, 60.0), (27.5, 60.5), (24.0, 60.0), (21.5, 60.8), (21.5, 63.0), (25.0, 65.5),
     (22.2, 65.8), (20.0, 63.8), (17.5, 62.5), (17.2, 60.6), (18.8, 59.8), (18.0, 58.9),
     (16.5, 57.0), (15.8, 56.1), (14.2, 55.4), (12.9, 55.4), (12.7, 56.1), (11.9, 57.7),
     (11.1, 58.9), (10.7, 59.9), (9.6, 59.0), (8.0, 58.0), (5.7, 58.8), (5.0, 60.4),
     (5.1, 62.2), (7.5, 63.1), (10.5, 64.5), (12.5, 66.0), (14.5, 67.3), (17.5, 68.5),
     (20.0, 70.0)],
    [(-5.6, 36.0), (-4.4, 36.7), (-2.2, 36.8), (-0.7, 37.6), (-0.1, 38.8), (-0.3, 39.5),
     (0.9, 41.0), (3.2, 42.4), (3.1, 43.3), (4.8, 43.4), (6.6, 43.2), (7.5, 43.8),
     (8.9, 44.4), (10.3, 43.7), (11.7, 42.1), (12.9, 41.2), (14.2, 40.8), (15.2, 40.0),
     (15.8, 38.0), (16.6, 38.8), (17.1, 40.4), (18.4, 40.0), (18.0, 40.7), (16.2, 41.9),
     (14.0, 42.5), (13.5, 43.6), (12.3, 44.5), (12.3, 45.4), (13.8, 45.6), (13.6, 45.1),
     (15.2, 44.1), (16.4, 43.5), (18.1, 42.6), (19.5, 41.8), (19.3, 40.5), (20.8, 38.9),
     (21.4, 37.7), (22.5, 36.4), (23.0, 36.5), (23.2, 37.8), (24.0, 37.7), (23.0, 39.3),
     (22.6, 40.5), (24.5, 40.8), (26.5, 40.9), (29.0, 41.0)],
    [(12.5, 37.7), (15.2, 36.7), (15.6, 38.2), (13.3, 38.2), (12.5, 37.7)],
    [(8.2, 39.0), (9.6, 39.2), (9.7, 41.2), (8.2, 41.0), (8.2, 39.0)],
    [(8.6, 41.6), (9.5, 41.6), (9.5, 42.9), (8.6, 42.6), (8.6, 41.6)],
    [(-6.0, 35.8), (-2.0, 35.1), (0.5, 35.9), (3.1, 36.8), (8.0, 36.9), (10.2, 36.8),
     (11.0, 37.0), (10.7, 34.7), (11.5, 33.2)],
]


def draw_fallback_europe_map(ax):
    for poly in EUROPE_COASTLINES:
        lons = [pt[0] for pt in poly]
        lats = [pt[1] for pt in poly]
        ax.plot(lons, lats, color="#1d3557", lw=0.95, alpha=0.8, zorder=2)


def standardize_coords(ds: xr.Dataset) -> xr.Dataset:
    rename_map = {
        k: v for k, v in [
            ("valid_time", "time"),
            ("pressure_level", "level"),
            ("isobaricInhPa", "level"),
            ("lat", "latitude"),
            ("lon", "longitude"),
        ] if k in ds.coords or k in ds.dims
    }
    if rename_map:
        ds = ds.rename(rename_map)

    if "expver" in ds.dims:
        ds = ds.mean("expver", skipna=True)
    for junk in ("number", "expver", "surface"):
        if junk in ds.coords and junk not in ds.dims:
            ds = ds.drop_vars(junk)

    if "longitude" in ds.coords and float(ds["longitude"].max()) > 180:
        ds = ds.assign_coords(longitude=((ds["longitude"] + 180) % 360) - 180)

    if "latitude" in ds.coords:
        ds = ds.sortby("latitude")
    if "longitude" in ds.coords:
        ds = ds.sortby("longitude")
    if "time" in ds.coords:
        ds = ds.sortby("time")
        _, unique_idx = np.unique(ds["time"].values, return_index=True)
        if len(unique_idx) < ds.sizes["time"]:
            ds = ds.isel(time=unique_idx)

    return ds


def e_sat_ice_hpa(t_c):
    t_k = t_c + 273.15
    ln_e_pa = (
        9.550426
        - 5723.265 / t_k
        + 3.53068 * np.log(t_k)
        - 0.00728332 * t_k
    )
    return np.exp(ln_e_pa) / 100.0


def e_sat_water_hpa(t_c):
    t_k = t_c + 273.15
    ln_e_pa = (
        54.842763
        - 6763.22 / t_k
        - 4.210 * np.log(t_k)
        + 0.000367 * t_k
        + np.tanh(0.0415 * (t_k - 218.8))
        * (53.878 - 1331.22 / t_k - 9.44523 * np.log(t_k) + 0.014025 * t_k)
    )
    return np.exp(ln_e_pa) / 100.0


def rhw_to_rhi(t_c, rhw):
    return rhw * (e_sat_water_hpa(t_c) / e_sat_ice_hpa(t_c))


def rhi_from_q(t_c, q, p_hpa):
    e = q * p_hpa / (0.622 + 0.378 * q)
    return 100.0 * e / e_sat_ice_hpa(t_c)


def schmidt_appleman_mask(t_c, rhi, p_hpa, eta=ETA_PROP, rhi_thresh=RHI_THRESHOLD, ei_h2o=EI_H2O, q_fuel=Q_FUEL):
    cp = 1004.0
    eps = 0.622
    p_pa = p_hpa * 100.0

    G = (ei_h2o * cp * p_pa) / (eps * q_fuel * (1.0 - eta))
    log_g = np.log(G - 0.053)
    t_lm_c = -46.46 + 9.43 * log_g + 0.720 * (log_g ** 2)

    e_sat_lm_pa = e_sat_water_hpa(t_lm_c) * 100.0
    e_ambient_pa = (rhi / 100.0) * (e_sat_ice_hpa(t_c) * 100.0)
    t_max_c = t_lm_c - (e_sat_lm_pa - e_ambient_pa) / G

    sac_met = (t_c <= t_lm_c) & (t_c <= t_max_c) & (rhi >= rhi_thresh)
    return sac_met, t_lm_c, G


def sens_slope_and_mk(years: np.ndarray, values: np.ndarray) -> dict:
    n = len(values)
    tau, p_val = stats.kendalltau(years, values)
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            if years[j] != years[i]:
                slopes.append((values[j] - values[i]) / (years[j] - years[i]))
    sen_yr = float(np.median(slopes)) if slopes else np.nan
    return {
        "kendall_tau": round(float(tau), 4),
        "p_value": round(float(p_val), 4),
        "sens_slope_per_yr": round(sen_yr, 4),
        "sens_slope_per_decade": round(sen_yr * 10.0, 4),
    }


def load_era5_dataset() -> xr.Dataset:
    ext_file = ERA5_DIR / "weather_europe_daynight_multilevel.nc"
    if ext_file.exists() and ext_file.stat().st_size > 10_000:
        nc_files = [ext_file]
    else:
        nc_files = [
            p for p in sorted(ERA5_DIR.glob("*.nc"))
            if not p.name.startswith("tmp_") and p.stat().st_size > 10_000
        ]
    if not nc_files:
        print(f"[ERROR] No NetCDF files found in {ERA5_DIR}.")
        sys.exit(1)

    print(f"[INFO] Loading {len(nc_files)} NetCDF file(s): {[p.name for p in nc_files]} ...")
    loaded = []
    for p in nc_files:
        ds_i = standardize_coords(xr.open_dataset(p))
        ds_i = ds_i.sel(
            time=slice("2008-01-01", "2023-12-31"),
            latitude=slice(34.0, 70.0),
            longitude=slice(-15.0, 35.0),
        )
        if ds_i.sizes.get("time", 0) > 0:
            loaded.append(ds_i)

    if len(loaded) == 1:
        return loaded[0]
    try:
        ds = xr.combine_by_coords(loaded, compat="override", combine_attrs="override", join="outer")
    except Exception:
        ds = xr.merge(loaded, compat="override", join="outer")
    return standardize_coords(ds)


def load_backup_cloud_cover() -> xr.DataArray | None:
    for p in sorted(ERA5_DIR.glob("*.nc")):
        if p.name.startswith("tmp_") or p.name == "weather_europe_daynight_multilevel.nc":
            continue
        try:
            ds_aux = standardize_coords(xr.open_dataset(p))
            for cname in ("cc", "tcc"):
                if cname in ds_aux.data_vars:
                    cc_aux = ds_aux[cname].sel(latitude=slice(34.0, 70.0), longitude=slice(-15.0, 35.0)).load()
                    if float(cc_aux.max()) <= 1.05:
                        cc_aux = cc_aux * 100.0
                    return cc_aux
        except Exception:
            continue
    return None


def main():
    ds = load_era5_dataset()

    is_ncep = ("air" in ds.data_vars) or ("rhum" in ds.data_vars)
    t_var = "t" if "t" in ds.data_vars else "air"
    if "r" in ds.data_vars:
        h_var = "r"
    elif "q" in ds.data_vars:
        h_var = "q"
    else:
        h_var = "rhum"

    yr_min = int(pd.DatetimeIndex(ds["time"].values).year.min())
    yr_max = int(pd.DatetimeIndex(ds["time"].values).year.max())
    span_label = f"{yr_min}–{yr_max}"
    source_label = ds.attrs.get(
        "source_dataset",
        f"Copernicus ERA5 ({span_label})",
    )

    avail_levels = []
    for lvl in ALL_TARGET_LEVELS:
        if lvl in ds["level"].values:
            sample = ds[h_var].sel(level=lvl).isel(time=slice(0, min(10, ds.sizes["time"]))).values
            if not bool(np.isnan(sample).all()):
                avail_levels.append(lvl)

    if not avail_levels:
        print("[ERROR] No valid humidity data found at target pressure levels.")
        sys.exit(1)

    ds = ds.sel(level=avail_levels)
    t_raw = ds[t_var].load()
    t_c = t_raw - 273.15 if float(t_raw.mean()) > 150 else t_raw

    h_raw = ds[h_var].load()
    p_3d = ds["level"].astype(float)

    if h_var == "q":
        rhi = rhi_from_q(t_c, h_raw, p_3d)
    elif is_ncep and h_var == "rhum":
        rhi = rhw_to_rhi(t_c, h_raw)
    else:
        rhi = h_raw

    cc_pct = None
    cc_is_3d = False
    if "cc" in ds.data_vars:
        cc = ds["cc"].load()
        cc_pct = cc * 100.0 if float(cc.max()) <= 1.05 else cc
    elif "tcc" in ds.data_vars:
        cc = ds["tcc"].load()
        cc_pct = cc * 100.0 if float(cc.max()) <= 1.05 else cc
        cc_is_3d = True

    cc_aux = load_backup_cloud_cover() if cc_pct is None else None

    def get_cc_for_level(lvl):
        if cc_pct is None:
            return None
        return cc_pct if cc_is_3d else cc_pct.sel(level=lvl)

    if USE_EXACT_SAC:
        sc_mask, t_lm_c, _ = schmidt_appleman_mask(t_c, rhi, p_3d, eta=ETA_PROP)
        t_crit_check = t_c <= t_lm_c
    else:
        sc_mask = (t_c <= T_THRESHOLD_C) & (rhi >= RHI_THRESHOLD)
        t_crit_check = t_c <= T_THRESHOLD_C

    lat_weights = np.cos(np.deg2rad(sc_mask["latitude"]))
    lat_weights.name = "weights"

    def area_time_mean_pct(da: xr.DataArray) -> float:
        spatial_mean = da.weighted(lat_weights).mean(dim=("latitude", "longitude"))
        return float(spatial_mean.mean(dim="time").values) * 100.0

    def area_time_mean_val(da: xr.DataArray) -> float:
        spatial_mean = da.weighted(lat_weights).mean(dim=("latitude", "longitude"))
        return float(spatial_mean.mean(dim="time").values)

    season_of = t_c["time"].dt.season
    summary_rows = []
    for lvl in avail_levels:
        m_lvl = sc_mask.sel(level=lvl)
        tc_lvl = t_crit_check.sel(level=lvl)
        r_lvl = rhi.sel(level=lvl) >= RHI_THRESHOLD
        cc_lvl = get_cc_for_level(lvl)

        if cc_lvl is not None:
            cc_mean = area_time_mean_val(cc_lvl)
        elif cc_aux is not None and "level" in cc_aux.dims and lvl in cc_aux["level"].values:
            w_aux = np.cos(np.deg2rad(cc_aux["latitude"]))
            cc_mean = float(cc_aux.sel(level=lvl).weighted(w_aux).mean().values)
        else:
            cc_mean = np.nan

        summary_rows.append({
            "source": source_label,
            "pressure_hPa": lvl,
            "season": f"ALL_{yr_min}_{yr_max}",
            "contrail_permissive_pct": round(area_time_mean_pct(m_lvl), 2),
            "sac_temp_met_pct": round(area_time_mean_pct(tc_lvl), 2),
            "rhi_ge_100_pct": round(area_time_mean_pct(r_lvl), 2),
            "mean_cloud_cover_Z2_pct": round(cc_mean, 2) if not np.isnan(cc_mean) else np.nan,
        })
        for season in ["DJF", "MAM", "JJA", "SON"]:
            sel = season_of == season
            if int(sel.sum()) == 0:
                continue
            cc_seas = area_time_mean_val(cc_lvl.sel(time=sel)) if cc_lvl is not None else np.nan
            summary_rows.append({
                "source": source_label,
                "pressure_hPa": lvl,
                "season": season,
                "contrail_permissive_pct": round(area_time_mean_pct(m_lvl.sel(time=sel)), 2),
                "sac_temp_met_pct": round(area_time_mean_pct(tc_lvl.sel(time=sel)), 2),
                "rhi_ge_100_pct": round(area_time_mean_pct(r_lvl.sel(time=sel)), 2),
                "mean_cloud_cover_Z2_pct": round(cc_seas, 2) if not np.isnan(cc_seas) else np.nan,
            })

    t3 = pd.DataFrame(summary_rows)
    t3.to_csv(TAB_DIR / "table3_h2_contrail_stats.csv", index=False)
    print(f"[OK] Saved {TAB_DIR / 'table3_h2_contrail_stats.csv'}")

    bias_rows = []
    for rhi_cut, note in [
        (100.0, "Baseline raw ERA5 (RHi >= 100%)"),
        (97.0, "Mild dry-bias correction (RHi >= 97%, ~1.03x scaling)"),
        (95.0, "Moderate dry-bias correction (RHi >= 95%, ~1.05x scaling)"),
        (90.0, "Upper-bound dry-bias correction (RHi >= 90%, ~1.11x scaling)"),
    ]:
        m_bias, _, _ = schmidt_appleman_mask(t_c, rhi, p_3d, eta=ETA_PROP, rhi_thresh=rhi_cut)
        for lvl in [l for l in [250, 300] if l in avail_levels]:
            bias_rows.append({
                "pressure_hPa": lvl,
                "rhi_threshold_pct": rhi_cut,
                "scenario": note,
                "contrail_permissive_pct": round(area_time_mean_pct(m_bias.sel(level=lvl)), 2),
            })
    bias_df = pd.DataFrame(bias_rows)
    bias_df.to_csv(TAB_DIR / "table3g_h2_rhi_bias_sensitivity.csv", index=False)
    print(f"[OK] Saved {TAB_DIR / 'table3g_h2_rhi_bias_sensitivity.csv'}")
    print(bias_df.to_string(index=False))

    hours = t_c["time"].dt.hour
    has_day_night = (int((hours == 0).sum()) > 0) and (int((hours == 12).sum()) > 0)
    dn_rows = []
    if has_day_night:
        for lvl in avail_levels:
            m_lvl = sc_mask.sel(level=lvl)
            day_pct = area_time_mean_pct(m_lvl.sel(time=(hours == 12)))
            night_pct = area_time_mean_pct(m_lvl.sel(time=(hours == 0)))
            night_share = (night_pct / (day_pct + night_pct)) * 100.0 if (day_pct + night_pct) > 0 else np.nan
            dn_rows.append({
                "pressure_hPa": lvl,
                "season": "ALL",
                "day_12z_contrail_pct": round(day_pct, 2),
                "night_00z_contrail_pct": round(night_pct, 2),
                "night_minus_day_pp": round(night_pct - day_pct, 2),
                "night_share_of_events_pct": round(night_share, 2),
            })
            for season in ["DJF", "MAM", "JJA", "SON"]:
                s_day = (hours == 12) & (season_of == season)
                s_night = (hours == 0) & (season_of == season)
                if int(s_day.sum()) > 0 and int(s_night.sum()) > 0:
                    dp = area_time_mean_pct(m_lvl.sel(time=s_day))
                    npct = area_time_mean_pct(m_lvl.sel(time=s_night))
                    nsh = (npct / (dp + npct)) * 100.0 if (dp + npct) > 0 else np.nan
                    dn_rows.append({
                        "pressure_hPa": lvl,
                        "season": season,
                        "day_12z_contrail_pct": round(dp, 2),
                        "night_00z_contrail_pct": round(npct, 2),
                        "night_minus_day_pp": round(npct - dp, 2),
                        "night_share_of_events_pct": round(nsh, 2),
                    })
    dn_df = pd.DataFrame(dn_rows)
    dn_df.to_csv(TAB_DIR / "table3c_h2_day_night_asymmetry.csv", index=False)
    print(f"[OK] Saved {TAB_DIR / 'table3c_h2_day_night_asymmetry.csv'}")

    sens_rows = []
    for eta_val in [0.25, 0.30, 0.35, 0.40]:
        m_eta, tlm_eta, g_eta = schmidt_appleman_mask(t_c, rhi, p_3d, eta=eta_val)
        for lvl in [l for l in [250, 300] if l in avail_levels]:
            tot_pct = area_time_mean_pct(m_eta.sel(level=lvl))
            tlm_scalar = float(tlm_eta.sel(level=lvl).values)
            g_scalar = float(g_eta.sel(level=lvl).values)
            cc_lvl = get_cc_for_level(lvl)
            if cc_lvl is not None:
                clear_pct = area_time_mean_pct(m_eta.sel(level=lvl) & (cc_lvl < 20.0))
                cloudy_pct = area_time_mean_pct(m_eta.sel(level=lvl) & (cc_lvl >= 20.0))
                clear_share = (clear_pct / tot_pct) * 100.0 if tot_pct > 0 else np.nan
            else:
                clear_pct, cloudy_pct, clear_share = np.nan, np.nan, np.nan

            sens_rows.append({
                "pressure_hPa": lvl,
                "engine_eta": eta_val,
                "mixing_slope_G_Pa_K": round(g_scalar, 4),
                "sac_threshold_T_LM_C": round(tlm_scalar, 2),
                "total_contrail_pct": round(tot_pct, 2),
                "clear_sky_Z2_lt_20_pct": round(clear_pct, 2) if not np.isnan(clear_pct) else np.nan,
                "cloud_overlap_Z2_ge_20_pct": round(cloudy_pct, 2) if not np.isnan(cloudy_pct) else np.nan,
                "clear_sky_share_of_contrails_pct": round(clear_share, 2) if not np.isnan(clear_share) else np.nan,
            })
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(TAB_DIR / "table3d_h2_eta_cloud_sensitivity.csv", index=False)
    print(f"[OK] Saved {TAB_DIR / 'table3d_h2_eta_cloud_sensitivity.csv'}")

    primary_lvl = 250 if 250 in avail_levels else avail_levels[0]
    t_ann = t_c.sel(level=primary_lvl).weighted(lat_weights).mean(dim=("latitude", "longitude")).groupby("time.year").mean()
    r_ann = rhi.sel(level=primary_lvl).weighted(lat_weights).mean(dim=("latitude", "longitude")).groupby("time.year").mean()
    c_ann = (sc_mask.sel(level=primary_lvl).weighted(lat_weights).mean(dim=("latitude", "longitude")) * 100.0).groupby("time.year").mean()

    trend_ts = pd.DataFrame({
        "year": t_ann["year"].values.astype(int),
        "mean_temp_C": np.round(t_ann.values, 3),
        "mean_rhi_pct": np.round(r_ann.values, 3),
        "contrail_permissive_pct": np.round(c_ann.values, 3),
    })

    yr_series = pd.Series(sc_mask["time"].dt.year.values)
    yr_counts = yr_series.value_counts()
    full_yrs = sorted(yr_counts[yr_counts >= 20].index.astype(int).tolist())
    trend_full = trend_ts[trend_ts["year"].isin(full_yrs)].copy()
    t_start, t_end = int(trend_full["year"].min()), int(trend_full["year"].max())

    trend_summary = []
    for col, label in [
        ("mean_temp_C", "Temperature at 250 hPa (°C)"),
        ("mean_rhi_pct", "Ice Relative Humidity at 250 hPa (%)"),
        ("contrail_permissive_pct", "Persistent Contrail Frequency at 250 hPa (%)"),
    ]:
        stats_res = sens_slope_and_mk(trend_full["year"].values, trend_full[col].values)
        trend_summary.append({
            "variable": label,
            "start_year": t_start,
            "end_year": t_end,
            "mean_value": round(float(trend_full[col].mean()), 3),
            **stats_res,
        })
    trend_df = pd.DataFrame(trend_summary)
    trend_df.to_csv(TAB_DIR / "table3e_h2_era5_trends.csv", index=False)
    print(f"[OK] Saved {TAB_DIR / 'table3e_h2_era5_trends.csv'} ({t_start}–{t_end})")

    prob_250 = (sc_mask.sel(level=250).mean(dim="time") * 100.0).transpose("latitude", "longitude")
    prob_300 = (sc_mask.sel(level=300).mean(dim="time") * 100.0).transpose("latitude", "longitude") if 300 in avail_levels else prob_250
    prob_day = (sc_mask.sel(level=250, time=(hours == 12)).mean(dim="time") * 100.0).transpose("latitude", "longitude") if has_day_night else prob_250
    prob_night = (sc_mask.sel(level=250, time=(hours == 0)).mean(dim="time") * 100.0).transpose("latitude", "longitude") if has_day_night else prob_250

    corr_rows = []
    corridor_freq_map = {}
    for orig, dest in CORRIDORS:
        lon1, lat1 = AIRPORTS[orig]
        lon2, lat2 = AIRPORTS[dest]
        lons = xr.DataArray(np.linspace(lon1, lon2, 50), dims="pt")
        lats = xr.DataArray(np.linspace(lat1, lat2, 50), dims="pt")

        f250 = round(float(prob_250.interp(longitude=lons, latitude=lats, method="linear").mean().values), 2)
        f300 = round(float(prob_300.interp(longitude=lons, latitude=lats, method="linear").mean().values), 2)
        fday = round(float(prob_day.interp(longitude=lons, latitude=lats, method="linear").mean().values), 2)
        fnight = round(float(prob_night.interp(longitude=lons, latitude=lats, method="linear").mean().values), 2)

        corridor_freq_map[(orig, dest)] = f250
        corr_rows.append({
            "corridor": f"{orig} - {dest}",
            "freq_250hPa_pct": f250,
            "freq_300hPa_pct": f300,
            "day_12z_250hPa_pct": fday,
            "night_00z_250hPa_pct": fnight,
        })
    corr_df = pd.DataFrame(corr_rows).sort_values("freq_250hPa_pct", ascending=False)
    corr_df.to_csv(TAB_DIR / "table3b_h2_corridor_frequencies.csv", index=False)
    print(f"[OK] Saved {TAB_DIR / 'table3b_h2_corridor_frequencies.csv'}")

    w_df = pd.DataFrame()
    if ROUTE_FILE.exists():
        rt = pd.read_csv(ROUTE_FILE)
        rt_2019 = rt[rt["year"] == 2019][[
            "corridor", "distance_km", "cruise_km", "typical_cruise_hpa",
            "pax_route", "flights_route", "pax_km_millions", "cruise_flight_km_millions"
        ]].rename(columns={
            "pax_route": "pax_2019",
            "flights_route": "flights_2019",
            "pax_km_millions": "pax_km_millions_2019",
            "cruise_flight_km_millions": "cruise_flight_km_millions_2019",
        })
        w_df = corr_df.merge(rt_2019, on="corridor", how="left")

        w_df["matched_cruise_freq_pct"] = np.where(
            w_df["typical_cruise_hpa"] == 300,
            w_df["freq_300hPa_pct"],
            w_df["freq_250hPa_pct"],
        )
        w_df["exposed_cruise_flight_km_M_2019"] = (
            w_df["cruise_flight_km_millions_2019"] * (w_df["matched_cruise_freq_pct"] / 100.0)
        ).round(3)
        w_df["exposed_pax_km_M_2019_250hPa"] = (
            w_df["pax_km_millions_2019"] * (w_df["freq_250hPa_pct"] / 100.0)
        ).round(2)

        w_df = w_df.sort_values("exposed_cruise_flight_km_M_2019", ascending=False)
        tot_flt_exp = w_df["exposed_cruise_flight_km_M_2019"].sum()
        w_df["share_flight_exposure_pct"] = ((w_df["exposed_cruise_flight_km_M_2019"] / tot_flt_exp) * 100.0).round(2) if tot_flt_exp > 0 else np.nan
        w_df["cumul_flight_exposure_pct"] = w_df["share_flight_exposure_pct"].cumsum().round(2)

        w_df.to_csv(TAB_DIR / "table3f_h2_weighted_corridor_exposure.csv", index=False)
        print(f"[OK] Saved {TAB_DIR / 'table3f_h2_weighted_corridor_exposure.csv'}")
        print(w_df[["corridor", "distance_km", "typical_cruise_hpa", "matched_cruise_freq_pct",
                    "exposed_cruise_flight_km_M_2019", "share_flight_exposure_pct"]].to_string(index=False))

    fig = plt.figure(figsize=(16, 6.8))
    vmax = max(1.0, float(np.nanpercentile(prob_250.values, 99.5)))

    if USE_CARTOPY:
        ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
        ax1.set_extent([-15, 35, 34, 70], crs=ccrs.PlateCarree())
        im = ax1.pcolormesh(
            prob_250["longitude"], prob_250["latitude"], prob_250.values,
            cmap="YlOrRd", vmin=0, vmax=vmax, transform=ccrs.PlateCarree(), shading="auto", zorder=1
        )
        ax1.add_feature(cfeature.COASTLINE, lw=0.9, edgecolor="#1d3557", zorder=2)
        ax1.add_feature(cfeature.BORDERS, lw=0.5, ls=":", edgecolor="#1d3557", zorder=2)
        gl = ax1.gridlines(draw_labels=True, lw=0.3, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        for o, d in CORRIDORS:
            p1, p2 = AIRPORTS[o], AIRPORTS[d]
            ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#0077b6", lw=2.0, alpha=0.95, transform=ccrs.Geodetic(), zorder=4)
            mid_lon, mid_lat = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
            ax1.text(
                mid_lon, mid_lat, f"{corridor_freq_map[(o, d)]:.1f}%",
                fontsize=6.5, fontweight="bold", color="#003049", ha="center", va="center",
                transform=ccrs.PlateCarree(), zorder=5,
                bbox=dict(boxstyle="round,pad=0.12", fc="#e0fbfc", ec="#0077b6", lw=0.6, alpha=0.9),
            )
        for name, (lon, lat) in AIRPORTS.items():
            code = name.split("(")[-1].rstrip(")")
            ax1.plot(lon, lat, marker="o", color="black", markersize=4.5, transform=ccrs.PlateCarree(), zorder=6)
            ax1.text(
                lon + 0.5, lat + 0.35, code,
                fontsize=8, fontweight="bold", color="#111111",
                transform=ccrs.PlateCarree(), zorder=7,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="black", lw=0.4, alpha=0.85),
            )
    else:
        ax1 = fig.add_subplot(1, 2, 1)
        im = ax1.pcolormesh(
            prob_250["longitude"], prob_250["latitude"], prob_250.values,
            cmap="YlOrRd", vmin=0, vmax=vmax, shading="auto", zorder=1
        )
        draw_fallback_europe_map(ax1)
        ax1.grid(True, ls="--", lw=0.3, color="#333333", alpha=0.35, zorder=2)
        for o, d in CORRIDORS:
            p1, p2 = AIRPORTS[o], AIRPORTS[d]
            ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#0077b6", lw=2.0, alpha=0.95, zorder=4)
            mid_lon, mid_lat = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
            ax1.text(
                mid_lon, mid_lat, f"{corridor_freq_map[(o, d)]:.1f}%",
                fontsize=6.5, fontweight="bold", color="#003049", ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.12", fc="#e0fbfc", ec="#0077b6", lw=0.6, alpha=0.9),
            )
        for name, (lon, lat) in AIRPORTS.items():
            code = name.split("(")[-1].rstrip(")")
            ax1.scatter(lon, lat, color="black", s=24, zorder=6)
            ax1.text(
                lon + 0.5, lat + 0.35, code,
                fontsize=8, fontweight="bold", color="#111111", zorder=7,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="black", lw=0.4, alpha=0.85),
            )
        ax1.set_xlim(-15, 35)
        ax1.set_ylim(34, 70)
        ax1.set_xlabel("Longitude (°E)", fontsize=9.5)
        ax1.set_ylabel("Latitude (°N)", fontsize=9.5)

    cbar = plt.colorbar(im, ax=ax1, orientation="vertical", pad=0.025)
    cbar.set_label("Persistent Contrail-Permissive Frequency (% of time)", fontsize=9.5)
    ax1.set_title(
        f"1. Spatial Contrail-Permissive Climatology ({span_label})\n"
        f"250 hPa (~34,000 ft / 10.4 km) | Exact SAC + RHi ≥ 100%",
        fontsize=10.5, fontweight="bold"
    )

    ax2 = fig.add_subplot(1, 2, 2)
    core_avail = [l for l in CORE_LEVELS if l in avail_levels]
    seas = t3[(t3["season"].isin(["DJF", "MAM", "JJA", "SON"])) & (t3["pressure_hPa"].isin(core_avail))].pivot(
        index="season", columns="pressure_hPa", values="contrail_permissive_pct"
    ).reindex(["DJF", "MAM", "JJA", "SON"])

    seas.index = [SEASON_LABELS[s] for s in seas.index]
    seas.columns = [LEVEL_LABELS.get(c, f"{c} hPa") for c in seas.columns]

    seas.plot(kind="bar", ax=ax2, color=["#1f77b4", "#ff7f0e"], edgecolor="black", rot=0, width=0.62)
    ax2.set_title(
        f"2. Seasonal & Vertical Cruise Profile ({span_label})\n"
        "(Winter Peaks at 300 hPa; Summer Drops Sharply at 300 hPa)",
        fontsize=10.5, fontweight="bold"
    )
    ax2.set_xlabel("Season", fontsize=10)
    ax2.set_ylabel("Contrail-Permissive Frequency (% of time)", fontsize=10)
    ax2.grid(axis="y", ls="--", alpha=0.5)
    ax2.legend(title="Cruising Altitude", loc="upper right", fontsize=8.5, title_fontsize=9)

    for container in ax2.containers:
        ax2.bar_label(container, fmt="%.1f%%", padding=3, fontsize=8.5, fontweight="bold")
    ax2.set_ylim(0, max(1.0, float(seas.max().max()) * 1.28))

    plt.tight_layout()
    out_fig = FIG_DIR / "fig4_h2_contrail_map.png"
    plt.savefig(out_fig, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {out_fig}")

    fig2, axes2 = plt.subplots(1, 3, figsize=(19.0, 5.5))

    if has_day_night and not dn_df.empty:
        dn_all = dn_df[dn_df["season"] == "ALL"].sort_values("pressure_hPa")
        x = np.arange(len(dn_all))
        w = 0.36
        b1 = axes2[0].bar(x - w / 2, dn_all["day_12z_contrail_pct"], width=w, color="#f4a261", edgecolor="black", label="Daytime (12:00 UTC)")
        b2 = axes2[0].bar(x + w / 2, dn_all["night_00z_contrail_pct"], width=w, color="#1d3557", edgecolor="black", label="Nighttime (00:00 UTC)")
        axes2[0].bar_label(b1, fmt="%.1f%%", padding=2, fontsize=7.5, fontweight="bold")
        axes2[0].bar_label(b2, fmt="%.1f%%", padding=2, fontsize=7.5, fontweight="bold")
        axes2[0].set_ylim(0, float(dn_all[["day_12z_contrail_pct", "night_00z_contrail_pct"]].max().max()) * 1.24)
        axes2[0].set_xticks(x)
        axes2[0].set_xticklabels([f"{int(p)} hPa" for p in dn_all["pressure_hPa"]])
        axes2[0].set_ylabel("Contrail-Permissive Frequency (% of time)")
        axes2[0].set_title("1. Day vs. Night & Vertical Cruise Profile\n(Nighttime Contrails Lack Solar Cooling)", fontweight="bold", fontsize=10)
        axes2[0].grid(axis="y", ls=":", alpha=0.5)
        axes2[0].legend(fontsize=8)

    for lvl, col, mk in [(250, "#1d3557", "o"), (300, "#e63946", "s")]:
        sub_e = sens_df[sens_df["pressure_hPa"] == lvl].sort_values("engine_eta")
        if not sub_e.empty:
            xs_e = sub_e["engine_eta"] * 100.0
            ys_e = sub_e["total_contrail_pct"]
            axes2[1].plot(xs_e, ys_e, marker=mk, lw=2.2, color=col, label=f"{lvl} hPa Cruise")
            for xv, yv in zip(xs_e, ys_e):
                axes2[1].annotate(f"{yv:.2f}%", (xv, yv + 0.08), ha="center", fontsize=7.5, fontweight="bold", color=col)
    axes2[1].set_ylim(sens_df["total_contrail_pct"].min() * 0.9, sens_df["total_contrail_pct"].max() * 1.1)
    axes2[1].set_xlabel("Overall Engine Propulsion Efficiency eta (%)")
    axes2[1].set_ylabel("Contrail-Permissive Frequency (% of time)")
    axes2[1].set_title("2. The Engine Efficiency–Contrail Paradox\n(Higher eta Raises T_LM & Expands Contrail Zones)", fontweight="bold", fontsize=10)
    axes2[1].grid(True, ls=":", alpha=0.5)
    axes2[1].legend(fontsize=8.5)

    s250 = sens_df[(sens_df["pressure_hPa"] == primary_lvl) & (sens_df["engine_eta"] == 0.30)]
    if not s250.empty and (not cc_is_3d) and pd.notna(s250["clear_sky_Z2_lt_20_pct"].values[0]):
        vals_z2 = [
            float(s250["clear_sky_Z2_lt_20_pct"].values[0]),
            float(s250["cloud_overlap_Z2_ge_20_pct"].values[0]),
        ]
        bars_z2 = axes2[2].bar(
            ["Clear-Sky Airspace\n(Z2 Cloud Cover < 20%)", "Pre-Existing Cloud\n(Z2 Cloud Cover ≥ 20%)"],
            vals_z2, color=["#e63946", "#457b9d"], edgecolor="black", width=0.52
        )
        axes2[2].bar_label(bars_z2, fmt="%.2f%%", padding=3, fontweight="bold", fontsize=9)
        axes2[2].set_ylim(0, max(vals_z2) * 1.25)
        axes2[2].set_ylabel("Share of All Observations at 250 hPa (%)")
        axes2[2].set_title("3. Clear-Sky vs. Cloud Overlap Control (Z2)\n(Isolating Clear-Sky Contrail Events)", fontweight="bold", fontsize=10)
        axes2[2].grid(axis="y", ls=":", alpha=0.5)
    else:
        b250 = bias_df[bias_df["pressure_hPa"] == 250].sort_values("rhi_threshold_pct", ascending=False)
        b300 = bias_df[bias_df["pressure_hPa"] == 300].sort_values("rhi_threshold_pct", ascending=False)
        xb = np.arange(len(b250))
        wb = 0.36
        bar1 = axes2[2].bar(xb - wb / 2, b250["contrail_permissive_pct"], width=wb, color="#1d3557", edgecolor="black", label="250 hPa")
        bar2 = axes2[2].bar(xb + wb / 2, b300["contrail_permissive_pct"], width=wb, color="#2a9d8f", edgecolor="black", label="300 hPa")
        axes2[2].bar_label(bar1, fmt="%.1f%%", padding=2, fontsize=7.5, fontweight="bold")
        axes2[2].bar_label(bar2, fmt="%.1f%%", padding=2, fontsize=7.5, fontweight="bold")
        axes2[2].set_xticks(xb)
        axes2[2].set_xticklabels([f"RHi≥{int(r)}%" for r in b250["rhi_threshold_pct"]])
        axes2[2].set_ylim(0, max(b250["contrail_permissive_pct"].max(), b300["contrail_permissive_pct"].max()) * 1.25)
        axes2[2].set_ylabel("Contrail-Permissive Frequency (% of time)")
        axes2[2].set_title("3. ERA5 Ice-Humidity Dry-Bias Sensitivity\n(Raw RHi ≥ 100% vs. Bias-Adjusted Thresholds)", fontweight="bold", fontsize=10)
        axes2[2].grid(axis="y", ls=":", alpha=0.5)
        axes2[2].legend(fontsize=8)

    plt.tight_layout()
    out_fig2 = FIG_DIR / "fig4b_h2_daynight_eta_cloud.png"
    plt.savefig(out_fig2, dpi=300)
    plt.close(fig2)
    print(f"[OK] Saved {out_fig2}")

    fig3, axes3 = plt.subplots(1, 2, figsize=(16.5, 5.5))
    mean_c = float(trend_full["contrail_permissive_pct"].mean())
    std_c = float(trend_full["contrail_permissive_pct"].std())
    axes3[0].plot(trend_full["year"], trend_full["contrail_permissive_pct"], marker="o", lw=2.2, color="#1d3557", label="Annual Contrail Frequency (%)")
    axes3[0].axhline(mean_c, color="#e63946", ls="--", lw=1.4, label=f"{t_start}–{t_end} Mean = {mean_c:.2f}%")
    axes3[0].axhspan(mean_c - std_c, mean_c + std_c, color="#457b9d", alpha=0.14, label=f"±1 SD Band ({std_c:.2f} pp)")
    mk_row = trend_df[trend_df["variable"].str.contains("Contrail")].iloc[0]
    axes3[0].set_title(
        f"1. Upper-Tropospheric Contrail Trend (250 hPa, {t_start}–{t_end})\n"
        f"(Sen's Slope = {mk_row['sens_slope_per_decade']:+.2f} pp/decade, Kendall p = {mk_row['p_value']:.3f})",
        fontweight="bold", fontsize=10
    )
    axes3[0].set_xlabel("Year")
    axes3[0].set_ylabel("European Domain Contrail Frequency (%)")
    axes3[0].grid(True, ls=":", alpha=0.5)
    axes3[0].legend(fontsize=8.5, loc="upper left")

    if not w_df.empty and w_df["exposed_cruise_flight_km_M_2019"].notna().sum() >= 5:
        w_plot = w_df.dropna(subset=["exposed_cruise_flight_km_M_2019"]).sort_values("exposed_cruise_flight_km_M_2019", ascending=True)
        colors_bar = ["#457b9d" if int(h) == 300 else "#e76f51" for h in w_plot["typical_cruise_hpa"]]
        bars_w = axes3[1].barh(w_plot["corridor"], w_plot["exposed_cruise_flight_km_M_2019"], color=colors_bar, edgecolor="black", height=0.68)
        for bar, (_, r) in zip(bars_w, w_plot.iterrows()):
            val = float(r["exposed_cruise_flight_km_M_2019"])
            shr = float(r["share_flight_exposure_pct"])
            lvl_tag = f"@{int(r['typical_cruise_hpa'])}"
            axes3[1].text(
                val + w_plot["exposed_cruise_flight_km_M_2019"].max() * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}M ({shr:.1f}%, {lvl_tag})", va="center", fontsize=7.5, fontweight="bold"
            )
        axes3[1].set_xlim(0, w_plot["exposed_cruise_flight_km_M_2019"].max() * 1.32)
        axes3[1].set_xlabel("Contrail-Exposed Cruise Flight-Kilometres (Million aircraft-km, 2019)")
        axes3[1].set_title(
            "2. Aircraft-km Corridor Contrail Exposure (Altitude & Cruise Adjusted)\n"
            "(Orange: 250 hPa ≥600 km | Blue: 300 hPa <600 km)",
            fontweight="bold", fontsize=10
        )
        axes3[1].tick_params(axis="y", labelsize=8)
        axes3[1].grid(axis="x", ls=":", alpha=0.5)

    plt.tight_layout()
    out_fig3 = FIG_DIR / "fig4c_h2_weighted_corridors_trends.png"
    plt.savefig(out_fig3, dpi=300)
    plt.close(fig3)
    print(f"[OK] Saved {out_fig3}")


if __name__ == "__main__":
    main()