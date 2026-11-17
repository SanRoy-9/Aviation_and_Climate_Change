import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

BASE_DIR = Path(__file__).resolve().parent.parent
ERA5_DIR = BASE_DIR / "data" / "raw" / "era5"
ERA5_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = ERA5_DIR / "weather_europe_2008_2023.nc"
CHECKSUM_FILE = ERA5_DIR / "era5_checksums.sha256"


def sha256_sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_rhi_from_q(temp_k: xr.DataArray, spec_hum: xr.DataArray, pressure_hpa: float) -> xr.DataArray:
    """
    Alduchov & Eskridge (1996) saturation vapor pressure over ice:
      e_i(t) = 6.1121 * exp(22.587 * t_C / (273.86 + t_C))  [hPa]
    """
    t_c = temp_k - 273.15
    e_sat_ice = 6.1121 * np.exp((22.587 * t_c) / (273.86 + t_c))
    e_vap = (spec_hum * pressure_hpa) / (0.62198 + 0.37802 * spec_hum)
    return 100.0 * (e_vap / e_sat_ice)


def fetch_arco_era5() -> xr.Dataset:
    print("[INFO] Trying public Google Cloud ARCO-ERA5 (0.25 deg, NO API key needed) ...")
    ds = xr.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        chunks=None,
        storage_options={"token": "anon"},
    )
    if float(ds["longitude"].max()) > 180:
        ds = ds.assign_coords(longitude=((ds["longitude"] + 180) % 360) - 180).sortby("longitude")
    lat_slice = slice(70, 34) if float(ds["latitude"][0]) > float(ds["latitude"][-1]) else slice(34, 70)

    # 15th of each month at 12:00 UTC across 2008-2023 (192 monthly snapshots)
    sample_times = pd.date_range("2008-01-01T12:00", "2023-12-01T12:00", freq="MS") + pd.Timedelta(days=14)
    sub = ds[["temperature", "specific_humidity", "fraction_of_cloud_cover"]].sel(
        time=sample_times,
        level=[250, 300],
        latitude=lat_slice,
        longitude=slice(-15, 35),
    )

    t_k = sub["temperature"]
    rhi_250 = compute_rhi_from_q(t_k.sel(level=250), sub["specific_humidity"].sel(level=250), 250.0)
    rhi_300 = compute_rhi_from_q(t_k.sel(level=300), sub["specific_humidity"].sel(level=300), 300.0)
    rhi = xr.concat([rhi_250, rhi_300], dim="level")

    out = xr.Dataset({
        "t": t_k,
        "r": rhi,
        "cc": sub["fraction_of_cloud_cover"],
    })
    out.attrs["source_dataset"] = "ARCO-ERA5 0.25 deg (No-Key Cloud Zarr)"
    return out


def fetch_ncep_r2() -> xr.Dataset:
    print("[INFO] Using public NOAA PSL NCEP-DOE Reanalysis 2 OPeNDAP (NO API key needed) ...")
    air_url = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis2/Monthlies/pressure/air.mon.mean.nc"
    rhum_url = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis2/Monthlies/pressure/rhum.mon.mean.nc"

    ds_air = xr.open_dataset(air_url)
    ds_rh = xr.open_dataset(rhum_url)

    def prep(d):
        if float(d["lon"].max()) > 180:
            d = d.assign_coords(lon=((d["lon"] + 180) % 360) - 180).sortby("lon")
        d = d.rename({"lat": "latitude", "lon": "longitude"})
        lat_sl = slice(70, 34) if float(d["latitude"][0]) > float(d["latitude"][-1]) else slice(34, 70)
        return d.sel(time=slice("2008-01-01", "2023-12-31"), latitude=lat_sl, longitude=slice(-15, 35))

    ds_air = prep(ds_air)
    ds_rh = prep(ds_rh)
    common_levels = [int(l) for l in [250, 300] if l in ds_air["level"].values and l in ds_rh["level"].values]

    out = xr.Dataset({
        "t": ds_air["air"].sel(level=common_levels).load(),
        "r": ds_rh["rhum"].sel(level=common_levels).load(),
    })
    out.attrs["source_dataset"] = "NOAA/NCEP Reanalysis 2 (No-Key OPeNDAP)"
    return out


def main():
    print("=" * 65)
    print("STEP 02: DOWNLOADING UPPER-AIR WEATHER DATA (2008-2023)")
    print("=" * 65)

    existing = [p for p in ERA5_DIR.glob("*.nc") if p.stat().st_size > 10_000]
    if existing:
        lines = [f"{sha256_sum(p)}  {p.name}" for p in sorted(existing)]
        CHECKSUM_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[EXISTS] Found {len(existing)} NetCDF file(s) in {ERA5_DIR}. Skipping download.")
        return

    try:
        ds_out = fetch_arco_era5()
    except Exception as e_arco:
        print(f"[NOTE] ARCO-ERA5 unavailable ({e_arco}). Falling back to NOAA/NCEP-R2 ...")
        try:
            ds_out = fetch_ncep_r2()
        except Exception as e_ncep:
            print(f"[ERROR] Both ARCO-ERA5 and NOAA/NCEP-R2 failed: {e_ncep}", file=sys.stderr)
            sys.exit(1)

    ds_out.to_netcdf(OUT_FILE)
    digest = sha256_sum(OUT_FILE)
    CHECKSUM_FILE.write_text(f"{digest}  {OUT_FILE.name}\n", encoding="utf-8")
    print(f"[OK] Saved {OUT_FILE} ({OUT_FILE.stat().st_size:,} bytes)")
    print(f"[SHA-256] {digest}")


if __name__ == "__main__":
    main()