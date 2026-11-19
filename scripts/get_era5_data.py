import hashlib
import itertools
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import xarray as xr

BASE_DIR = Path(__file__).resolve().parent.parent
ERA5_DIR = BASE_DIR / "data" / "raw" / "era5"
ERA5_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = ERA5_DIR / "weather_europe_daynight_multilevel.nc"
CHECKSUM_FILE = ERA5_DIR / "era5_checksums.sha256"
TMP_ZARR = ERA5_DIR / "tmp_wb2_zarr"

WB2_BASE = (
    "https://storage.googleapis.com/weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
)

TARGET_LEVELS = (200, 250, 300, 400)


def sha256_sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_resilient_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def e_sat_ice_hpa(t_c: xr.DataArray) -> xr.DataArray:
    """Saturation vapour pressure over hexagonal ice (hPa), Murphy & Koop (2005)."""
    t_k = t_c + 273.15
    ln_e_pa = (
        9.550426
        - 5723.265 / t_k
        + 3.53068 * np.log(t_k)
        - 0.00728332 * t_k
    )
    return np.exp(ln_e_pa) / 100.0


def compute_rhi_from_q(temp_k: xr.DataArray, spec_hum: xr.DataArray, pressure_hpa: xr.DataArray) -> xr.DataArray:
    t_c = temp_k - 273.15
    e_vap = (spec_hum * pressure_hpa) / (0.622 + 0.378 * spec_hum)
    return 100.0 * (e_vap / e_sat_ice_hpa(t_c))


def download_chunk(session: requests.Session, rel_key: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{WB2_BASE}/{rel_key}"
    for attempt in range(1, 5):
        try:
            r = session.get(url, timeout=(15, 90))
            r.raise_for_status()
            tmp_part = dest.with_suffix(".part")
            tmp_part.write_bytes(r.content)
            tmp_part.replace(dest)
            return
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2.0 * attempt)


def fetch_wb2_daynight_multilevel() -> xr.Dataset:
    session = build_resilient_session()

    # Preserve TMP_ZARR across runs so interrupted downloads resume automatically
    TMP_ZARR.mkdir(parents=True, exist_ok=True)

    print("[INFO] Fetching WeatherBench2 ERA5 metadata over HTTPS ...", flush=True)
    r_meta = session.get(f"{WB2_BASE}/.zmetadata", timeout=(15, 60))
    r_meta.raise_for_status()
    zmeta = r_meta.json()["metadata"]

    for key, val in zmeta.items():
        p = TMP_ZARR / key
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(val), encoding="utf-8")

    print("[INFO] Downloading/verifying coordinate arrays ...", flush=True)
    for coord in ["time", "level", "latitude", "longitude"]:
        arr_meta = zmeta[f"{coord}/.zarray"]
        shape = arr_meta["shape"][0]
        chunk = arr_meta["chunks"][0]
        n_chunks = (shape + chunk - 1) // chunk
        for c in range(n_chunks):
            download_chunk(session, f"{coord}/{c}", TMP_ZARR / coord / str(c))

    ds = xr.open_zarr(TMP_ZARR, consolidated=False)

    # Sample 1st & 15th of each month at BOTH 00:00 UTC (Nighttime) and 12:00 UTC (Daytime)
    months = pd.date_range("2008-01-01", "2023-01-01", freq="MS")
    target_list = []
    for m in months:
        for day_offset in (0, 14):
            dt_base = m + pd.Timedelta(days=day_offset)
            for hr in (0, 12):
                cand = dt_base + pd.Timedelta(hours=hr)
                if cand <= pd.Timestamp("2023-01-10T18:00"):
                    target_list.append(cand)

    time_vals = pd.DatetimeIndex(ds["time"].values)
    t_indices = sorted({int(np.abs(time_vals - t).argmin()) for t in target_list})

    lvl_vals = ds["level"].values
    l_indices = [int(i) for i, v in enumerate(lvl_vals) if int(v) in TARGET_LEVELS]

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

    want_vars = [
        v for v in ["temperature", "specific_humidity", "fraction_of_cloud_cover"]
        if f"{v}/.zarray" in zmeta
    ]

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
    already_cached = sum(1 for k in chunk_jobs if (TMP_ZARR / k).exists() and (TMP_ZARR / k).stat().st_size > 0)
    print(
        f"[INFO] Downloading {total_jobs} European ERA5 chunks "
        f"({already_cached} already cached, {total_jobs - already_cached} remaining) ...",
        flush=True,
    )
    for k, rel_key in enumerate(chunk_jobs, start=1):
        download_chunk(session, rel_key, TMP_ZARR / rel_key)
        if k == 1 or k % 25 == 0 or k == total_jobs:
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
    p_3d = sub["level"].astype(float)
    rhi = compute_rhi_from_q(t_k, sub["specific_humidity"], p_3d)

    data_dict = {"t": t_k, "r": rhi, "q": sub["specific_humidity"]}
    if "fraction_of_cloud_cover" in sub.data_vars:
        data_dict["cc"] = sub["fraction_of_cloud_cover"]

    out = xr.Dataset(data_dict)
    out.attrs["source_dataset"] = "Copernicus ERA5 / WeatherBench2 (Day 12Z & Night 00Z, 200–400 hPa)"
    return out


def main():
    print("=" * 65)
    print("STEP 02: DOWNLOADING EXTENDED DAY/NIGHT MULTI-LEVEL ERA5 DATA")
    print("=" * 65)

    if OUT_FILE.exists() and OUT_FILE.stat().st_size > 10_000:
        digest = sha256_sum(OUT_FILE)
        CHECKSUM_FILE.write_text(f"{digest}  {OUT_FILE.name}\n", encoding="utf-8")
        print(f"[EXISTS] Found {OUT_FILE.name} ({OUT_FILE.stat().st_size:,} bytes). Skipping download.")
        return

    try:
        ds_out = fetch_wb2_daynight_multilevel()
    except Exception as e:
        print(f"\n[ERROR] ERA5 HTTPS fetch failed: {e}", file=sys.stderr)
        sys.exit(1)

    ds_out.to_netcdf(OUT_FILE)
    shutil.rmtree(TMP_ZARR, ignore_errors=True)
    digest = sha256_sum(OUT_FILE)
    CHECKSUM_FILE.write_text(f"{digest}  {OUT_FILE.name}\n", encoding="utf-8")
    print(f"[OK] Saved {OUT_FILE} ({OUT_FILE.stat().st_size:,} bytes)")
    print(f"[SHA-256] {digest}")


if __name__ == "__main__":
    main()