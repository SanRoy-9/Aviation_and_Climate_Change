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
RAW_DIR = BASE_DIR / "data" / "raw"
FIG_DIR = BASE_DIR / "reports" / "figures"
TAB_DIR = BASE_DIR / "reports" / "tables"

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


def standardize_coords(ds: xr.Dataset) -> xr.Dataset:
    rename_map = {}
    if "valid_time" in ds.coords:
        rename_map["valid_time"] = "time"
    if "pressure_level" in ds.coords:
        rename_map["pressure_level"] = "level"
    if "lat" in ds.coords:
        rename_map["lat"] = "latitude"
    if "lon" in ds.coords:
        rename_map["lon"] = "longitude"
    if rename_map:
        ds = ds.rename(rename_map)
    return ds


def corridor_contrail_stats(mask_250: xr.DataArray) -> pd.DataFrame:
    rows = []
    for orig, dest in CORRIDORS:
        lon1, lat1 = AIRPORTS[orig]
        lon2, lat2 = AIRPORTS[dest]
        lons = np.linspace(lon1, lon2, 30)
        lats = np.linspace(lat1, lat2, 30)
        sampled = mask_250.sel(
            longitude=xr.DataArray(lons, dims="pt"),
            latitude=xr.DataArray(lats, dims="pt"),
            method="nearest",
        )
        freq_pct = float(sampled.mean().values) * 100.0
        rows.append({"corridor": f"{orig} - {dest}", "contrail_freq_pct": round(freq_pct, 2)})
    return pd.DataFrame(rows).sort_values("contrail_freq_pct", ascending=False)


def main():
    nc_files = sorted(ERA5_DIR.glob("era5_europe_250_300hPa_*.nc"))
    if not nc_files:
        print(f"[ERROR] No ERA5 NetCDF files found in {ERA5_DIR}. Run 02_get_era5_data.py first.")
        sys.exit(1)

    print(f"[INFO] Loading {len(nc_files)} ERA5 NetCDF files ...")
    ds = xr.open_mfdataset(nc_files, combine="by_coords")
    ds = standardize_coords(ds)

    # Convert Temperature K -> C
    # Note: In ECMWF IFS / ERA5, relative_humidity (r) below -23C is already saturation over ice (RHi).
    t_c = ds["t"] - 273.15
    rhi = ds["r"]
    cc = ds["cc"] if "cc" in ds.data_vars else None

    # Schmidt-Appleman contrail-permissive mask: T <= -40C and RHi >= 100%
    sc_mask = (t_c <= -40.0) & (rhi >= 100.0)

    summary_rows = []
    for lvl in [250, 300]:
        if lvl not in sc_mask["level"].values:
            continue
        m_lvl = sc_mask.sel(level=lvl).compute()
        t_lvl = t_c.sel(level=lvl).compute()
        r_lvl = rhi.sel(level=lvl).compute()

        overall_pct = float(m_lvl.mean().values) * 100.0
        t_ok_pct = float((t_lvl <= -40.0).mean().values) * 100.0
        r_ok_pct = float((r_lvl >= 100.0).mean().values) * 100.0
        cc_mean = float(cc.sel(level=lvl).mean().values) * 100.0 if cc is not None else np.nan

        summary_rows.append({
            "pressure_hPa": lvl,
            "season": "ALL_YEARS_2008_2023",
            "contrail_permissive_pct": round(overall_pct, 2),
            "temp_le_minus40_pct": round(t_ok_pct, 2),
            "rhi_ge_100_pct": round(r_ok_pct, 2),
            "mean_cloud_cover_Z2_pct": round(cc_mean, 2) if not np.isnan(cc_mean) else np.nan,
        })

        # Seasonal breakdown
        for season, grp in m_lvl.groupby("time.season"):
            summary_rows.append({
                "pressure_hPa": lvl,
                "season": season,
                "contrail_permissive_pct": round(float(grp.mean().values) * 100.0, 2),
                "temp_le_minus40_pct": round(float((t_lvl.sel(time=grp["time"]) <= -40.0).mean().values) * 100.0, 2),
                "rhi_ge_100_pct": round(float((r_lvl.sel(time=grp["time"]) >= 100.0).mean().values) * 100.0, 2),
                "mean_cloud_cover_Z2_pct": np.nan,
            })

    t3 = pd.DataFrame(summary_rows)
    t3.to_csv(TAB_DIR / "table3_h2_contrail_stats.csv", index=False)
    print(f"[OK] Saved {TAB_DIR / 'table3_h2_contrail_stats.csv'}")

    # Corridor-specific frequency at 250 hPa
    m_250 = sc_mask.sel(level=250).compute()
    prob_map = m_250.mean(dim="time") * 100.0
    corr_df = corridor_contrail_stats(prob_map)
    corr_df.to_csv(TAB_DIR / "table3b_h2_corridor_frequencies.csv", index=False)
    print(f"[OK] Saved {TAB_DIR / 'table3b_h2_corridor_frequencies.csv'}")

    # Optional Lee et al. (2021) matrix check
    lee_candidates = list(RAW_DIR.glob("*Lee*.*")) + list(RAW_DIR.glob("*RF*.*"))
    if lee_candidates:
        print(f"[INFO] Found Lee et al. benchmark file: {lee_candidates[0].name}")
    else:
        print("[NOTE] Place Lee et al. (2021) Zenodo data table in data/raw/ to auto-parse component ERF rows.")

    # Plot Figure 4: 250 hPa Schmidt-Appleman probability map + Seasonal Bar Chart
    fig = plt.figure(figsize=(15, 6))

    if USE_CARTOPY:
        ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
        ax1.set_extent([-15, 35, 34, 70], crs=ccrs.PlateCarree())
        im = ax1.pcolormesh(
            prob_map["longitude"], prob_map["latitude"], prob_map.values,
            cmap="YlOrRd", vmin=0, vmax=max(20.0, float(prob_map.max())),
            transform=ccrs.PlateCarree(), shading="auto"
        )
        ax1.add_feature(cfeature.COASTLINE, lw=0.8, edgecolor="#1d3557")
        ax1.add_feature(cfeature.BORDERS, lw=0.4, linestyle=":", edgecolor="#1d3557")
        for o, d in CORRIDORS:
            p1, p2 = AIRPORTS[o], AIRPORTS[d]
            ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#0077b6", lw=1.6, alpha=0.85, transform=ccrs.PlateCarree())
        for name, (lon, lat) in AIRPORTS.items():
            ax1.plot(lon, lat, marker="o", color="black", markersize=4, transform=ccrs.PlateCarree())
    else:
        ax1 = fig.add_subplot(1, 2, 1)
        im = ax1.pcolormesh(
            prob_map["longitude"], prob_map["latitude"], prob_map.values,
            cmap="YlOrRd", vmin=0, vmax=max(20.0, float(prob_map.max())), shading="auto"
        )
        for o, d in CORRIDORS:
            p1, p2 = AIRPORTS[o], AIRPORTS[d]
            ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#0077b6", lw=1.6, alpha=0.85)
        for name, (lon, lat) in AIRPORTS.items():
            ax1.scatter(lon, lat, color="black", s=20)
        ax1.set_xlabel("Longitude (°E)")
        ax1.set_ylabel("Latitude (°N)")

    cbar = plt.colorbar(im, ax=ax1, orientation="vertical", pad=0.03)
    cbar.set_label("Contrail-Permissive Months (%)", fontsize=9)
    ax1.set_title("1. 250 hPa Schmidt-Appleman Frequency (2008–2023)\n(T <= -40°C and RHi >= 100%)", fontsize=10, fontweight="bold")

    # Panel 2: Seasonal comparison (250 hPa vs 300 hPa)
    ax2 = fig.add_subplot(1, 2, 2)
    seas = t3[t3["season"].isin(["DJF", "MAM", "JJA", "SON"])].pivot(
        index="season", columns="pressure_hPa", values="contrail_permissive_pct"
    ).reindex(["DJF", "MAM", "JJA", "SON"])
    seas.plot(kind="bar", ax=ax2, color=["#e63946", "#457b9d"], edgecolor="black")
    ax2.set_title("2. Seasonal Contrail-Permissive Frequency\n(250 hPa vs. 300 hPa Sensitivity)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Season")
    ax2.set_ylabel("Grid-Months Meeting Criterion (%)")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(title="Pressure Level (hPa)")

    plt.tight_layout()
    out_fig = FIG_DIR / "fig4_h2_contrail_map.png"
    plt.savefig(out_fig, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {out_fig}")


if __name__ == "__main__":
    main()