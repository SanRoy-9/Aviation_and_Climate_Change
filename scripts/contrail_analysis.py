import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    USE_CARTOPY = True
except ImportError:
    USE_CARTOPY = False

BASE_DIR = Path(__file__).resolve().parent.parent
ERA5_DIR = BASE_DIR / "data" / "raw" / "era5"
FIG_DIR = BASE_DIR / "reports" / "figures"
TAB_DIR = BASE_DIR / "reports" / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# ---- Settings ---------------------------------------------------------------
USE_EXACT_SAC = True
T_THRESHOLD_C = -40.0
RHI_THRESHOLD = 100.0

# Propulsion/fuel constants for standard kerosene jet (Schumann, 1996)
ETA_PROP = 0.30          # Overall propulsion efficiency
EI_H2O = 1.223           # Water emission index (kg H2O / kg fuel)
Q_FUEL = 43.2e6          # Lower heating value of jet fuel (J / kg)

LEVELS = [250, 300]

LEVEL_LABELS = {
    250: "250 hPa (~34,000 ft / 10.4 km — High Cruise)",
    300: "300 hPa (~30,000 ft / 9.1 km — Low Cruise)",
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

# Built-in simplified European coastline polygons (lon, lat) so the map looks
# clear and recognizable even when Cartopy is not installed.
EUROPE_COASTLINES = [
    # Great Britain
    [(-5.7, 50.1), (-5.0, 50.2), (-3.5, 50.4), (-1.9, 50.7), (0.2, 50.8), (1.4, 51.3),
     (0.8, 51.5), (1.7, 52.5), (0.5, 52.9), (0.1, 53.5), (-0.2, 54.1), (-1.5, 55.0),
     (-2.0, 55.8), (-3.1, 56.0), (-2.1, 57.0), (-1.8, 57.7), (-4.0, 57.8), (-3.1, 58.6),
     (-5.0, 58.6), (-5.8, 57.3), (-5.6, 56.0), (-4.8, 55.0), (-3.5, 54.5), (-3.1, 53.4),
     (-4.5, 53.3), (-4.1, 52.5), (-5.3, 51.9), (-4.1, 51.5), (-5.7, 50.1)],
    # Ireland
    [(-10.0, 51.6), (-8.5, 51.9), (-6.4, 52.2), (-6.0, 53.3), (-6.2, 54.0), (-5.5, 54.6),
     (-6.1, 55.2), (-7.5, 55.3), (-8.5, 54.9), (-8.5, 54.3), (-10.0, 54.2), (-9.9, 53.5),
     (-9.0, 53.1), (-9.9, 52.6), (-10.0, 51.6)],
    # Continental Europe (Iberia -> France -> Low Countries -> Denmark -> Baltic -> Italy -> Balkans)
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
    # Mediterranean Coastline (Spain -> France -> Italy -> Adriatic -> Greece)
    [(-5.6, 36.0), (-4.4, 36.7), (-2.2, 36.8), (-0.7, 37.6), (-0.1, 38.8), (-0.3, 39.5),
     (0.9, 41.0), (3.2, 42.4), (3.1, 43.3), (4.8, 43.4), (6.6, 43.2), (7.5, 43.8),
     (8.9, 44.4), (10.3, 43.7), (11.7, 42.1), (12.9, 41.2), (14.2, 40.8), (15.2, 40.0),
     (15.8, 38.0), (16.6, 38.8), (17.1, 40.4), (18.4, 40.0), (18.0, 40.7), (16.2, 41.9),
     (14.0, 42.5), (13.5, 43.6), (12.3, 44.5), (12.3, 45.4), (13.8, 45.6), (13.6, 45.1),
     (15.2, 44.1), (16.4, 43.5), (18.1, 42.6), (19.5, 41.8), (19.3, 40.5), (20.8, 38.9),
     (21.4, 37.7), (22.5, 36.4), (23.0, 36.5), (23.2, 37.8), (24.0, 37.7), (23.0, 39.3),
     (22.6, 40.5), (24.5, 40.8), (26.5, 40.9), (29.0, 41.0)],
    # Sicily, Sardinia, Corsica
    [(12.5, 37.7), (15.2, 36.7), (15.6, 38.2), (13.3, 38.2), (12.5, 37.7)],
    [(8.2, 39.0), (9.6, 39.2), (9.7, 41.2), (8.2, 41.0), (8.2, 39.0)],
    [(8.6, 41.6), (9.5, 41.6), (9.5, 42.9), (8.6, 42.6), (8.6, 41.6)],
    # North Africa (Morocco / Algeria / Tunisia)
    [(-6.0, 35.8), (-2.0, 35.1), (0.5, 35.9), (3.1, 36.8), (8.0, 36.9), (10.2, 36.8),
     (11.0, 37.0), (10.7, 34.7), (11.5, 33.2)],
]


def draw_fallback_europe_map(ax):
    """Draw built-in European coastlines when Cartopy is not installed."""
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
    """Saturation vapour pressure over hexagonal ice (hPa), Murphy & Koop (2005)."""
    t_k = t_c + 273.15
    ln_e_pa = (
        9.550426
        - 5723.265 / t_k
        + 3.53068 * np.log(t_k)
        - 0.00728332 * t_k
    )
    return np.exp(ln_e_pa) / 100.0


def e_sat_water_hpa(t_c):
    """Saturation vapour pressure over supercooled liquid water (hPa), Murphy & Koop (2005)."""
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


def schmidt_appleman_mask(t_c, rhi, p_hpa, eta=ETA_PROP, ei_h2o=EI_H2O, q_fuel=Q_FUEL):
    cp = 1004.0
    eps = 0.622
    p_pa = p_hpa * 100.0

    G = (ei_h2o * cp * p_pa) / (eps * q_fuel * (1.0 - eta))
    log_g = np.log(G - 0.053)
    t_lm_c = -46.46 + 9.43 * log_g + 0.720 * (log_g ** 2)

    e_sat_lm_pa = e_sat_water_hpa(t_lm_c) * 100.0
    e_ambient_pa = (rhi / 100.0) * (e_sat_ice_hpa(t_c) * 100.0)
    t_max_c = t_lm_c - (e_sat_lm_pa - e_ambient_pa) / G

    sac_met = (t_c <= t_lm_c) & (t_c <= t_max_c) & (rhi >= RHI_THRESHOLD)
    return sac_met, t_lm_c


def median_step_days(time: xr.DataArray) -> float:
    if time.size < 2:
        return float("nan")
    dt = np.diff(time.values).astype("timedelta64[s]").astype(float) / 86400.0
    return float(np.median(dt))


def load_and_combine_era5(nc_files: list[Path]) -> xr.Dataset:
    loaded = []
    for p in nc_files:
        ds_i = standardize_coords(xr.open_dataset(p))
        ds_i = ds_i.sel(
            time=slice("2008-01-01", "2023-12-31"),
            latitude=slice(34.0, 70.0),
            longitude=slice(-15.0, 35.0),
        )
        if ds_i.sizes.get("time", 0) == 0:
            continue
        step_i = median_step_days(ds_i["time"])
        loaded.append((p, ds_i, step_i))

    if not loaded:
        print("[ERROR] No valid NetCDF files with 2008-2023 data found.")
        sys.exit(1)

    datasets = [item[1] for item in loaded]
    if len(datasets) == 1:
        return datasets[0]

    try:
        ds = xr.combine_by_coords(datasets, compat="override", combine_attrs="override", join="outer")
    except Exception:
        ds = xr.merge(datasets, compat="override", join="outer")
    return standardize_coords(ds)


def main():
    nc_files = [
        p for p in sorted(ERA5_DIR.glob("*.nc"))
        if not p.name.startswith("tmp_") and p.stat().st_size > 10_000
    ]
    if not nc_files:
        print(f"[ERROR] No NetCDF files found in {ERA5_DIR}.")
        sys.exit(1)

    print(f"[INFO] Loading {len(nc_files)} NetCDF file(s) from {ERA5_DIR} ...")
    ds = load_and_combine_era5(nc_files)

    is_ncep = ("air" in ds.data_vars) or ("rhum" in ds.data_vars)
    t_var = "t" if "t" in ds.data_vars else "air"
    if "q" in ds.data_vars:
        h_var = "q"
    elif "r" in ds.data_vars:
        h_var = "r"
    else:
        h_var = "rhum"

    source_label = ds.attrs.get(
        "source_dataset",
        "NOAA NCEP-R2 (2008–2023)" if is_ncep else "Copernicus ERA5 (2008–2023)",
    )

    avail_levels = []
    for lvl in LEVELS:
        if lvl in ds["level"].values:
            sample = ds[h_var].sel(level=lvl).isel(time=slice(0, min(10, ds.sizes["time"]))).values
            if not bool(np.isnan(sample).all()):
                avail_levels.append(lvl)

    if not avail_levels:
        print("[ERROR] No valid humidity data found at 250 or 300 hPa.")
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

    cc = ds["cc"].load() if "cc" in ds.data_vars else None

    if USE_EXACT_SAC:
        sc_mask, t_lm_c = schmidt_appleman_mask(t_c, rhi, p_3d)
        t_crit_check = t_c <= t_lm_c
    else:
        sc_mask = (t_c <= T_THRESHOLD_C) & (rhi >= RHI_THRESHOLD)
        t_crit_check = t_c <= T_THRESHOLD_C

    lat_weights = np.cos(np.deg2rad(sc_mask["latitude"]))
    lat_weights.name = "weights"

    def area_time_mean_pct(da: xr.DataArray) -> float:
        spatial_mean = da.weighted(lat_weights).mean(dim=("latitude", "longitude"))
        return float(spatial_mean.mean(dim="time").values) * 100.0

    season_of = t_c["time"].dt.season
    summary_rows = []

    for lvl in avail_levels:
        m_lvl = sc_mask.sel(level=lvl)
        tc_lvl = t_crit_check.sel(level=lvl)
        r_lvl = rhi.sel(level=lvl) >= RHI_THRESHOLD
        cc_mean = area_time_mean_pct(cc.sel(level=lvl)) if cc is not None else np.nan

        summary_rows.append({
            "source": source_label,
            "pressure_hPa": lvl,
            "season": "ALL_2008_2023",
            "contrail_permissive_pct": round(area_time_mean_pct(m_lvl), 2),
            "temp_le_minus40_pct": round(area_time_mean_pct(tc_lvl), 2),
            "rhi_ge_100_pct": round(area_time_mean_pct(r_lvl), 2),
            "mean_cloud_cover_Z2_pct": round(cc_mean, 2) if not np.isnan(cc_mean) else np.nan,
        })
        for season in ["DJF", "MAM", "JJA", "SON"]:
            sel = season_of == season
            if int(sel.sum()) == 0:
                continue
            summary_rows.append({
                "source": source_label,
                "pressure_hPa": lvl,
                "season": season,
                "contrail_permissive_pct": round(area_time_mean_pct(m_lvl.sel(time=sel)), 2),
                "temp_le_minus40_pct": round(area_time_mean_pct(tc_lvl.sel(time=sel)), 2),
                "rhi_ge_100_pct": round(area_time_mean_pct(r_lvl.sel(time=sel)), 2),
                "mean_cloud_cover_Z2_pct": np.nan,
            })

    t3 = pd.DataFrame(summary_rows)
    t3.to_csv(TAB_DIR / "table3_h2_contrail_stats.csv", index=False)
    print(f"[OK] Saved {TAB_DIR / 'table3_h2_contrail_stats.csv'}")
    print(t3[t3["season"] == "ALL_2008_2023"].to_string(index=False))

    primary_lvl = 250 if 250 in avail_levels else avail_levels[0]
    prob_map = sc_mask.sel(level=primary_lvl).mean(dim="time") * 100.0

    # ---- Corridor Sampling --------------------------------------------------
    corr_rows = []
    corridor_freq_map = {}
    for orig, dest in CORRIDORS:
        lon1, lat1 = AIRPORTS[orig]
        lon2, lat2 = AIRPORTS[dest]
        lons = xr.DataArray(np.linspace(lon1, lon2, 50), dims="pt")
        lats = xr.DataArray(np.linspace(lat1, lat2, 50), dims="pt")
        sampled = prob_map.interp(longitude=lons, latitude=lats, method="linear")
        freq_val = round(float(sampled.mean().values), 1)
        corridor_freq_map[(orig, dest)] = freq_val
        corr_rows.append({
            "corridor": f"{orig} - {dest}",
            "pressure_hPa": primary_lvl,
            "contrail_freq_pct": freq_val,
        })
    corr_df = pd.DataFrame(corr_rows).sort_values("contrail_freq_pct", ascending=False)
    corr_df.to_csv(TAB_DIR / "table3b_h2_corridor_frequencies.csv", index=False)
    print(f"[OK] Saved {TAB_DIR / 'table3b_h2_corridor_frequencies.csv'}")

    # ---- Figure: Clear, Self-Explanatory Layout -----------------------------
    fig = plt.figure(figsize=(16, 6.8))
    vmax = max(1.0, float(np.nanpercentile(prob_map.values, 99.5)))

    if USE_CARTOPY:
        ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
        ax1.set_extent([-15, 35, 34, 70], crs=ccrs.PlateCarree())
        im = ax1.pcolormesh(
            prob_map["longitude"], prob_map["latitude"], prob_map.values,
            cmap="YlOrRd", vmin=0, vmax=vmax, transform=ccrs.PlateCarree(), shading="auto", zorder=1
        )
        ax1.add_feature(cfeature.COASTLINE, lw=0.9, edgecolor="#1d3557", zorder=2)
        ax1.add_feature(cfeature.BORDERS, lw=0.5, ls=":", edgecolor="#1d3557", zorder=2)
        gl = ax1.gridlines(draw_labels=True, lw=0.3, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        for o, d in CORRIDORS:
            p1, p2 = AIRPORTS[o], AIRPORTS[d]
            ax1.plot(
                [p1[0], p2[0]], [p1[1], p2[1]],
                color="#0077b6", lw=2.0, alpha=0.95, transform=ccrs.Geodetic(), zorder=4
            )
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
            prob_map["longitude"], prob_map["latitude"], prob_map.values,
            cmap="YlOrRd", vmin=0, vmax=vmax, shading="auto", zorder=1
        )
        # Draw built-in European coastlines so the map is immediately recognizable
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

        ax1.text(
            -14.0, 66.2,
            "Northern Europe / North Atlantic Track\n(High Risk: 11–14% of flights form persistent contrails)",
            fontsize=8, fontweight="bold", color="white", zorder=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="#6a040f", ec="white", lw=0.6, alpha=0.85),
        )
        ax1.text(
            -14.0, 35.2,
            "Southern Europe / Mediterranean\n(Low Risk: 2–5% of flights form persistent contrails)",
            fontsize=8, fontweight="bold", color="#222222", zorder=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="#fff3b0", ec="#999999", lw=0.6, alpha=0.9),
        )
        ax1.set_xlim(-15, 35)
        ax1.set_ylim(34, 70)
        ax1.set_xlabel("Longitude (°E — West to East across Europe)", fontsize=9.5)
        ax1.set_ylabel("Latitude (°N — South to North across Europe)", fontsize=9.5)

    cbar = plt.colorbar(im, ax=ax1, orientation="vertical", pad=0.025)
    cbar.set_label("Chance of Persistent Contrail Formation (% of time)", fontsize=9.5)
    ax1.set_title(
        "1. Where Do Persistent Contrails Form over Europe?\n"
        f"Cruising Altitude: 250 hPa (~34,000 ft / 10.4 km) | {source_label}",
        fontsize=10.5, fontweight="bold"
    )

    # ---- Panel 2: Seasonal Bar Chart ----------------------------------------
    ax2 = fig.add_subplot(1, 2, 2)
    seas = t3[t3["season"].isin(["DJF", "MAM", "JJA", "SON"])].pivot(
        index="season", columns="pressure_hPa", values="contrail_permissive_pct"
    ).reindex(["DJF", "MAM", "JJA", "SON"])

    seas.index = [SEASON_LABELS[s] for s in seas.index]
    seas.columns = [LEVEL_LABELS.get(c, f"{c} hPa") for c in seas.columns]

    seas.plot(kind="bar", ax=ax2, color=["#1f77b4", "#ff7f0e"], edgecolor="black", rot=0, width=0.62)
    ax2.set_title(
        "2. How Often Do Contrails Form by Season & Altitude?\n"
        "(Cold & Ice-Supersaturated Air: Schmidt–Appleman Criterion + RH_ice ≥ 100%)",
        fontsize=10.5, fontweight="bold"
    )
    ax2.set_xlabel("Season of the Year", fontsize=10)
    ax2.set_ylabel("Chance of Persistent Contrail Formation (% of time)", fontsize=10)
    ax2.grid(axis="y", ls="--", alpha=0.5)
    ax2.legend(title="Cruising Altitude", loc="upper right", fontsize=8.5, title_fontsize=9)

    for container in ax2.containers:
        ax2.bar_label(container, fmt="%.1f%%", padding=3, fontsize=8.5, fontweight="bold")
    ax2.set_ylim(0, max(1.0, float(seas.max().max()) * 1.28))

    if len(avail_levels) >= 2 and 300 in avail_levels:
        jja_300 = float(t3[(t3["season"] == "JJA") & (t3["pressure_hPa"] == 300)]["contrail_permissive_pct"].values[0])
        ax2.annotate(
            f"Summer Takeaway:\nAt 30,000 ft (orange), summer air\nis too warm for contrails ({jja_300:.1f}%),\n"
            "whereas at 34,000 ft (blue) air\nis colder and still forms contrails!",
            xy=(2.15, jja_300 + 0.3), xytext=(1.35, float(seas.max().max()) * 0.58),
            arrowprops=dict(arrowstyle="->", color="#d62828", lw=1.6),
            fontsize=8, fontweight="bold", color="#9d0208",
            bbox=dict(boxstyle="round,pad=0.35", fc="#fff8f6", ec="#d62828", lw=1.0),
        )

    plt.tight_layout()
    out_fig = FIG_DIR / "fig4_h2_contrail_map.png"
    plt.savefig(out_fig, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {out_fig}")


if __name__ == "__main__":
    main()