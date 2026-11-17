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
    rename_map = {
        k: v for k, v in [
            ("valid_time", "time"),
            ("pressure_level", "level"),
            ("lat", "latitude"),
            ("lon", "longitude"),
        ] if k in ds.coords
    }
    return ds.rename(rename_map) if rename_map else ds


def main():
    nc_files = sorted(ERA5_DIR.glob("*.nc"))
    if not nc_files:
        print(f"[ERROR] No NetCDF files found in {ERA5_DIR}. Run 02_get_era5_data.py first.")
        sys.exit(1)

    ds = xr.open_mfdataset(nc_files, combine="by_coords")
    source_label = ds.attrs.get("source_dataset", "Upper-Air Reanalysis")
    ds = standardize_coords(ds)

    t_var = "t" if "t" in ds.data_vars else "air"
    r_var = "r" if "r" in ds.data_vars else "rhum"

    t_raw = ds[t_var]
    t_c = t_raw - 273.15 if float(t_raw.mean()) > 150 else t_raw
    rhi = ds[r_var]
    cc = ds["cc"] if "cc" in ds.data_vars else None

    sc_mask = (t_c <= -40.0) & (rhi >= 100.0)
    levels = [int(l) for l in sc_mask["level"].values]

    summary_rows = []
    for lvl in levels:
        m_lvl = sc_mask.sel(level=lvl).compute()
        t_lvl = t_c.sel(level=lvl).compute()
        r_lvl = rhi.sel(level=lvl).compute()
        cc_mean = float(cc.sel(level=lvl).mean().values) * 100.0 if cc is not None else np.nan

        summary_rows.append({
            "source": source_label,
            "pressure_hPa": lvl,
            "season": "ALL_2008_2023",
            "contrail_permissive_pct": round(float(m_lvl.mean().values) * 100.0, 2),
            "temp_le_minus40_pct": round(float((t_lvl <= -40.0).mean().values) * 100.0, 2),
            "rhi_ge_100_pct": round(float((r_lvl >= 100.0).mean().values) * 100.0, 2),
            "mean_cloud_cover_Z2_pct": round(cc_mean, 2) if not np.isnan(cc_mean) else np.nan,
        })
        for season, grp in m_lvl.groupby("time.season"):
            summary_rows.append({
                "source": source_label,
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
    print(t3[t3["season"] == "ALL_2008_2023"].to_string(index=False))

    primary_lvl = 250 if 250 in levels else levels[0]
    prob_map = sc_mask.sel(level=primary_lvl).mean(dim="time").compute() * 100.0

    corr_rows = []
    for orig, dest in CORRIDORS:
        lon1, lat1 = AIRPORTS[orig]
        lon2, lat2 = AIRPORTS[dest]
        lons = np.linspace(lon1, lon2, 30)
        lats = np.linspace(lat1, lat2, 30)
        sampled = prob_map.sel(
            longitude=xr.DataArray(lons, dims="pt"),
            latitude=xr.DataArray(lats, dims="pt"),
            method="nearest",
        )
        corr_rows.append({
            "corridor": f"{orig} - {dest}",
            "pressure_hPa": primary_lvl,
            "contrail_freq_pct": round(float(sampled.mean().values), 2),
        })
    corr_df = pd.DataFrame(corr_rows).sort_values("contrail_freq_pct", ascending=False)
    corr_df.to_csv(TAB_DIR / "table3b_h2_corridor_frequencies.csv", index=False)
    print(f"[OK] Saved {TAB_DIR / 'table3b_h2_corridor_frequencies.csv'}")

    fig = plt.figure(figsize=(15, 6))
    vmax = max(15.0, float(prob_map.max()))

    if USE_CARTOPY:
        ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
        ax1.set_extent([-15, 35, 34, 70], crs=ccrs.PlateCarree())
        im = ax1.pcolormesh(
            prob_map["longitude"], prob_map["latitude"], prob_map.values,
            cmap="YlOrRd", vmin=0, vmax=vmax, transform=ccrs.PlateCarree(), shading="auto",
        )
        ax1.add_feature(cfeature.COASTLINE, lw=0.8, edgecolor="#1d3557")
        ax1.add_feature(cfeature.BORDERS, lw=0.4, ls=":", edgecolor="#1d3557")
        for o, d in CORRIDORS:
            p1, p2 = AIRPORTS[o], AIRPORTS[d]
            ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#0077b6", lw=1.6, alpha=0.85, transform=ccrs.PlateCarree())
        for _, (lon, lat) in AIRPORTS.items():
            ax1.plot(lon, lat, marker="o", color="black", markersize=4, transform=ccrs.PlateCarree())
    else:
        ax1 = fig.add_subplot(1, 2, 1)
        im = ax1.pcolormesh(prob_map["longitude"], prob_map["latitude"], prob_map.values, cmap="YlOrRd", vmin=0, vmax=vmax, shading="auto")
        for o, d in CORRIDORS:
            p1, p2 = AIRPORTS[o], AIRPORTS[d]
            ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#0077b6", lw=1.6, alpha=0.85)
        for _, (lon, lat) in AIRPORTS.items():
            ax1.scatter(lon, lat, color="black", s=20)
        ax1.set_xlabel("Longitude (°E)")
        ax1.set_ylabel("Latitude (°N)")

    cbar = plt.colorbar(im, ax=ax1, orientation="vertical", pad=0.03)
    cbar.set_label("Contrail-Permissive Months (%)", fontsize=9)
    ax1.set_title(f"1. {primary_lvl} hPa Schmidt-Appleman Map (2008–2023)\n({source_label})", fontsize=10, fontweight="bold")

    ax2 = fig.add_subplot(1, 2, 2)
    seas = t3[t3["season"].isin(["DJF", "MAM", "JJA", "SON"])].pivot(
        index="season", columns="pressure_hPa", values="contrail_permissive_pct"
    ).reindex(["DJF", "MAM", "JJA", "SON"])
    seas.plot(kind="bar", ax=ax2, edgecolor="black")
    ax2.set_title("2. Seasonal Contrail-Permissive Frequency\n(T <= -40°C and RH_ice >= 100%)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Season")
    ax2.set_ylabel("Grid-Months Meeting Criterion (%)")
    ax2.grid(axis="y", ls="--", alpha=0.5)
    ax2.legend(title="Level (hPa)")

    plt.tight_layout()
    out_fig = FIG_DIR / "fig4_h2_contrail_map.png"
    plt.savefig(out_fig, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {out_fig}")


if __name__ == "__main__":
    main()