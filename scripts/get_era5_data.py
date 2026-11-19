import hashlib
import itertools
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

BASE_DIR = Path(__file__).resolve().parent.parent
ERA5_DIR = BASE_DIR / "data" / "raw" / "era5"
ERA5_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = ERA5_DIR / "weather_europe_2008_2023.nc"
CHECKSUM_FILE = ERA5_DIR / "era5_checksums.sha256"
TMP_ZARR = ERA5_DIR / "tmp_wb2_zarr"

WB2_BASE = (
    "https://storage.googleapis.com/weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
)


def sha256_sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_rhi_from_q(temp_k: xr.DataArray, spec_hum: xr.DataArray, pressure_hpa: float) -> xr.DataArray:
    """Alduchov & Eskridge (1996) saturation vapor pressure over ice."""
    t_c = temp_k - 273.15
    e_sat_ice = 6.1121 * np.exp((22.587 * t_c) / (273.86 + t_c))
    e_vap = (spec_hum * pressure_hpa) / (0.62198 + 0.37802 * spec_hum)
    return 100.0 * (e_vap / e_sat_ice)


def download_chunk(session: requests.Session, rel_key: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{WB2_BASE}/{rel_key}"
    r = session.get(url, timeout=30)
    r.raise_for_status()
    dest.write_bytes(r.content)


def fetch_wb2_via_requests() -> xr.Dataset:
    session = requests.Session()

    if TMP_ZARR.exists():
        shutil.rmtree(TMP_ZARR, ignore_errors=True)
    TMP_ZARR.mkdir(parents=True, exist_ok=True)

    print("[INFO] Fetching WeatherBench2 ERA5 metadata over HTTPS ...", flush=True)
    r_meta = session.get(f"{WB2_BASE}/.zmetadata", timeout=30)
    r_meta.raise_for_status()
    zmeta = r_meta.json()["metadata"]

    # Write local Zarr v2 metadata files
    for key, val in zmeta.items():
        p = TMP_ZARR / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(val), encoding="utf-8")

    # Download 1D coordinate chunks (time, level, latitude, longitude)
    print("[INFO] Downloading coordinate arrays ...", flush=True)
    for coord in ["time", "level", "latitude", "longitude"]:
        arr_meta = zmeta[f"{coord}/.zarray"]
        shape = arr_meta["shape"][0]
        chunk = arr_meta["chunks"][0]
        n_chunks = (shape + chunk - 1) // chunk
        for c in range(n_chunks):
            download_chunk(session, f"{coord}/{c}", TMP_ZARR / coord / str(c))

    ds = xr.open_zarr(TMP_ZARR, consolidated=False)

    # Pick 15th of each month at 12:00 UTC (2008-2022 = 180 months)
    target_times = pd.date_range("2008-01-01T12:00", "2022-12-01T12:00", freq="MS") + pd.Timedelta(days=14)
    time_vals = pd.DatetimeIndex(ds["time"].values)
    t_indices = sorted({int(np.abs(time_vals - t).argmin()) for t in target_times})

    lvl_vals = ds["level"].values
    l_indices = [int(i) for i, v in enumerate(lvl_vals) if int(v) in (250, 300)]

    lat_vals = ds["latitude"].values
    lat_indices = [int(i) for i, v in enumerate(lat_vals) if 34.0 <= float(v) <= 70.0]

    lon_vals = ds["longitude"].values
    if float(lon_vals.max()) > 180:
        lon_indices = [int(i) for i, v in enumerate(lon_vals) if float(v) >= 345.0 or float(v) <= 35.0]
    else:
        lon_indices = [int(i) for i, v in enumerate(lon_vals) if -15.0 <= float(v) <= 35.0]

    idx_map = {
        "time": t_indices,
        "level": l_indices,
        "latitude": lat_indices,
        "longitude": lon_indices,
    }

    want_vars = [v for v in ["temperature", "specific_humidity", "fraction_of_cloud_cover"] if f"{v}/.zarray" in zmeta]

    # Collect unique chunks needed across all selected variables
    chunk_jobs = []
    for var in want_vars:
        arr_meta = zmeta[f"{var}/.zarray"]
        dims = zmeta[f"{var}/.zattrs"]["_ARRAY_DIMENSIONS"]
        chunks = arr_meta["chunks"]
        sep = arr_meta.get("dimension_separator", ".")
        per_dim_chunks = [sorted({idx // c for idx in idx_map[d]}) for d, c in zip(dims, chunks)]
        for combo in itertools.product(*per_dim_chunks):
            chunk_jobs.append(f"{var}/{sep.join(map(str, combo))}")

    total_jobs = len(chunk_jobs)
    print(f"[INFO] Downloading {total_jobs} European ERA5 chunks (2008-2022, 250/300 hPa) ...", flush=True)
    for k, rel_key in enumerate(chunk_jobs, start=1):
        download_chunk(session, rel_key, TMP_ZARR / rel_key)
        if k == 1 or k % 20 == 0 or k == total_jobs:
            print(f"\r  [PROGRESS] {k}/{total_jobs} chunks ({100.0 * k / total_jobs:.0f}%)", end="", flush=True)
    print()

    sub = ds[want_vars].isel(
        time=t_indices,
        level=l_indices,
        latitude=lat_indices,
        longitude=lon_indices,
    ).load()
    ds.close()

    if float(sub["longitude"].max()) > 180:
        sub = sub.assign_coords(longitude=((sub["longitude"] + 180) % 360) - 180).sortby("longitude")

    t_k = sub["temperature"]
    rhi_250 = compute_rhi_from_q(t_k.sel(level=250), sub["specific_humidity"].sel(level=250), 250.0)
    rhi_300 = compute_rhi_from_q(t_k.sel(level=300), sub["specific_humidity"].sel(level=300), 300.0)
    rhi = xr.concat([rhi_250, rhi_300], dim="level")

    data_dict = {"t": t_k, "r": rhi}
    if "fraction_of_cloud_cover" in sub.data_vars:
        data_dict["cc"] = sub["fraction_of_cloud_cover"]

    out = xr.Dataset(data_dict)
    out.attrs["source_dataset"] = "WeatherBench2 ERA5 1.5 deg (2008-2022, No-Key HTTPS)"
    shutil.rmtree(TMP_ZARR, ignore_errors=True)
    return out


def main():
    print("=" * 65)
    print("STEP 02: DOWNLOADING ERA5 WEATHER DATA (NO API KEY)")
    print("=" * 65)

    existing = [p for p in ERA5_DIR.glob("*.nc") if p.stat().st_size > 10_000 and not p.name.startswith("tmp_")]
    if existing:
        lines = [f"{sha256_sum(p)}  {p.name}" for p in sorted(existing)]
        CHECKSUM_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[EXISTS] Found {len(existing)} NetCDF file(s) in {ERA5_DIR}. Skipping download.")
        return

    try:
        ds_out = fetch_wb2_via_requests()
    except Exception as e:
        print(f"\n[ERROR] ERA5 HTTPS fetch failed: {e}", file=sys.stderr)
        sys.exit(1)

    ds_out.to_netcdf(OUT_FILE)
    digest = sha256_sum(OUT_FILE)
    CHECKSUM_FILE.write_text(f"{digest}  {OUT_FILE.name}\n", encoding="utf-8")
    print(f"[OK] Saved {OUT_FILE} ({OUT_FILE.stat().st_size:,} bytes)")
    print(f"[SHA-256] {digest}")


if __name__ == "__main__":
    main()