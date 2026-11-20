from concurrent.futures import ThreadPoolExecutor, as_completed
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

START_DATE = "2008-01-01"
END_DATE = "2022-12-31"
SAMPLE_STEP_DAYS = 5
TARGET_LEVELS = (200, 250, 300, 400)
MAX_WORKERS = 8


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
        backoff_factor=1.2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def e_sat_ice_hpa(t_c: xr.DataArray) -> xr.DataArray:
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
            time.sleep(1.5 * attempt)


def fetch_wb2_daynight_multilevel() -> xr.Dataset:
    session = build_resilient_session()
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

    sample_days = pd.date_range(START_DATE, END_DATE, freq=f"{SAMPLE_STEP_DAYS}D")
    target_list = []
    for d in sample_days:
        target_list.append(d + pd.Timedelta(hours=0))
        target_list.append(d + pd.Timedelta(hours=12))

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

    candidate_vars = ["temperature", "specific_humidity", "fraction_of_cloud_cover", "total_cloud_cover"]
    want_vars = [v for v in candidate_vars if f"{v}/.zarray" in zmeta]
    print(f"[INFO] Available ERA5 variables in WB2 store: {want_vars}", flush=True)

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
        f"[INFO] Fetching {total_jobs} chunks across {len(t_indices)} timestamps (2008–2022, every {SAMPLE_STEP_DAYS}d, 00Z & 12Z) "
        f"[{already_cached} cached, {total_jobs - already_cached} to download] ...",
        flush=True,
    )

    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_chunk, session, key, TMP_ZARR / key): key for key in chunk_jobs}
        for fut in as_completed(futures):
            fut.result()
            completed += 1
            if completed == 1 or completed % 50 == 0 or completed == total_jobs:
                print(f"\r  [PROGRESS] {completed}/{total_jobs} chunks ({100.0 * completed / total_jobs:.0f}%)", end="", flush=True)
    print()

    sub_vars = {}
    for var in want_vars:
        sel_kwargs = {d: idx_map[d] for d in ds[var].dims if d in idx_map}
        sub_vars[var] = ds[var].isel(**sel_kwargs).load()
    ds.close()

    sub = xr.Dataset(sub_vars)
    if float(sub["longitude"].max()) > 180:
        sub = sub.assign_coords(longitude=((sub["longitude"] + 180) % 360) - 180).sortby("longitude")

    t_k = sub["temperature"]
    p_3d = sub["level"].astype(float)
    rhi = compute_rhi_from_q(t_k, sub["specific_humidity"], p_3d)

    data_dict = {"t": t_k, "r": rhi, "q": sub["specific_humidity"]}
    if "fraction_of_cloud_cover" in sub.data_vars:
        data_dict["cc"] = sub["fraction_of_cloud_cover"]
    if "total_cloud_cover" in sub.data_vars:
        data_dict["tcc"] = sub["total_cloud_cover"]

    out = xr.Dataset(data_dict)
    out.attrs["source_dataset"] = f"Copernicus ERA5 / WeatherBench2 (2008–2022, {SAMPLE_STEP_DAYS}-day 00Z/12Z, 200–400 hPa)"
    out.attrs["sampling_step_days"] = SAMPLE_STEP_DAYS
    return out


def main():
    print("=" * 68)
    print("STEP 02: DOWNLOADING MULTI-LEVEL DAY/NIGHT ERA5 DATA (2008–2022)")
    print("=" * 68)

    if OUT_FILE.exists() and OUT_FILE.stat().st_size > 10_000:
        try:
            with xr.open_dataset(OUT_FILE) as check_ds:
                n_times = int(check_ds.sizes.get("time", 0))
            if n_times >= 1500:
                digest = sha256_sum(OUT_FILE)
                CHECKSUM_FILE.write_text(f"{digest}  {OUT_FILE.name}\n", encoding="utf-8")
                print(f"[EXISTS] {OUT_FILE.name} ({n_times} timestamps, {OUT_FILE.stat().st_size:,} bytes) — verified.")
                return
            print(f"[REFRESH] Existing file has only {n_times} timestamps; upgrading to {SAMPLE_STEP_DAYS}-day sampling ...")
        except Exception:
            print("[REFRESH] Re-downloading ERA5 NetCDF ...")

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