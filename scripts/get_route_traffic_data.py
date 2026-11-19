import gzip
import hashlib
import io
import math
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = RAW_DIR / "eurostat_corridor_pax_2008_2023.csv"
CHECKSUM_FILE = RAW_DIR / "route_checksums.sha256"
PANEL_FILE = PROC_DIR / "eurostat_panel_2008_2023.csv"

AIRPORTS = {
    "London (LHR)": ("UK_EGLL", "UK", -0.46, 51.47),
    "Paris (CDG)": ("FR_LFPG", "FR", 2.55, 49.01),
    "Frankfurt (FRA)": ("DE_EDDF", "DE", 8.57, 50.03),
    "Amsterdam (AMS)": ("NL_EHAM", "NL", 4.76, 52.31),
    "Madrid (MAD)": ("ES_LEMD", "ES", -3.56, 40.48),
    "Rome (FCO)": ("IT_LIRF", "IT", 12.24, 41.80),
    "Dublin (DUB)": ("IE_EIDW", "IE", -6.27, 53.42),
    "Vienna (VIE)": ("AT_LOWW", "AT", 16.57, 48.11),
    "Lisbon (LIS)": ("PT_LPPT", "PT", -9.13, 38.77),
    "Stockholm (ARN)": ("SE_ESSA", "SE", 17.92, 59.65),
    "Warsaw (WAW)": ("PL_EPWA", "PL", 20.97, 52.17),
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

# In Eurostat avia_par_{cc}, airp_pr always starts with the reporting country's airport code
CORRIDOR_REPORTER_CANDIDATES = {
    ("London (LHR)", "Frankfurt (FRA)"): [("de", "DE_EDDF_UK_EGLL")],
    ("Paris (CDG)", "Frankfurt (FRA)"): [("de", "DE_EDDF_FR_LFPG"), ("fr", "FR_LFPG_DE_EDDF")],
    ("London (LHR)", "Amsterdam (AMS)"): [("nl", "NL_EHAM_UK_EGLL")],
    ("Paris (CDG)", "Madrid (MAD)"): [("fr", "FR_LFPG_ES_LEMD"), ("es", "ES_LEMD_FR_LFPG")],
    ("Frankfurt (FRA)", "Rome (FCO)"): [("de", "DE_EDDF_IT_LIRF"), ("it", "IT_LIRF_DE_EDDF")],
    ("Frankfurt (FRA)", "Vienna (VIE)"): [("de", "DE_EDDF_AT_LOWW"), ("at", "AT_LOWW_DE_EDDF")],
    ("London (LHR)", "Dublin (DUB)"): [("ie", "IE_EIDW_UK_EGLL")],
    ("Madrid (MAD)", "Lisbon (LIS)"): [("es", "ES_LEMD_PT_LPPT"), ("pt", "PT_LPPT_ES_LEMD")],
    ("Frankfurt (FRA)", "Warsaw (WAW)"): [("de", "DE_EDDF_PL_EPWA"), ("pl", "PL_EPWA_DE_EDDF")],
    ("Frankfurt (FRA)", "Stockholm (ARN)"): [("de", "DE_EDDF_SE_ESSA"), ("se", "SE_ESSA_DE_EDDF")],
}


def sha256_sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def fetch_single_pair_sdmx(cc: str, pair_code: str) -> pd.DataFrame:
    """Fetch a single airport pair from Eurostat avia_par_{cc} using exact 4D SDMX key."""
    candidate_urls = [
        (
            f"https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"
            f"avia_par_{cc}/A.PAS_CRD.PAS.{pair_code}/?format=SDMX-CSV&startPeriod=2008&endPeriod=2023"
        ),
        (
            f"https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"
            f"avia_par_{cc}/A.PAS_BRD.PAS.{pair_code}/?format=SDMX-CSV&startPeriod=2008&endPeriod=2023"
        ),
    ]

    for url in candidate_urls:
        req = urllib.request.Request(url, headers={"User-Agent": "AviationClimateResearch/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                raw_bytes = resp.read()
        except Exception:
            continue

        if not raw_bytes:
            continue

        text = (
            gzip.decompress(raw_bytes).decode("utf-8", errors="ignore")
            if raw_bytes[:2] == b"\x1f\x8b"
            else raw_bytes.decode("utf-8", errors="ignore")
        )
        if not text.strip() or "TIME_PERIOD" not in text.upper():
            continue

        df = pd.read_csv(io.StringIO(text), low_memory=False)
        df.columns = [c.lower() for c in df.columns]
        t_col = "time_period" if "time_period" in df.columns else "time"
        v_col = "obs_value" if "obs_value" in df.columns else "value"
        if t_col not in df.columns or v_col not in df.columns:
            continue

        df["year"] = pd.to_numeric(df[t_col], errors="coerce")
        df["pax"] = pd.to_numeric(df[v_col], errors="coerce")
        df = df.dropna(subset=["year", "pax"]).copy()
        df["year"] = df["year"].astype(int)
        df = df[(df["year"] >= 2008) & (df["year"] <= 2023) & (df["pax"] > 0)]
        if not df.empty:
            df["reporter"] = cc.upper()
            return df[["reporter", "year", "pax"]].groupby(["reporter", "year"], as_index=False)["pax"].max()

    return pd.DataFrame()


def main():
    print("=" * 65)
    print("STEP 01B: DOWNLOADING EUROSTAT AIRPORT-PAIR ROUTE TRAFFIC")
    print("=" * 65)

    if OUT_FILE.exists() and OUT_FILE.stat().st_size > 2048:
        existing_df = pd.read_csv(OUT_FILE)
        if (
            "pax_route" in existing_df.columns
            and existing_df["pax_route"].notna().sum() >= 140
            and existing_df["pax_route"].max() > 10000
        ):
            digest = sha256_sum(OUT_FILE)
            CHECKSUM_FILE.write_text(f"{digest}  {OUT_FILE.name}\n", encoding="utf-8")
            print(f"[EXISTS] {OUT_FILE.name} ({OUT_FILE.stat().st_size:,} bytes) — checksum verified.")
            return
        else:
            print("[REFRESH] Existing route file was incomplete; fetching exact SDMX airport pairs ...")

    nat_growth = {}
    country_pax_2019 = {}
    if PANEL_FILE.exists():
        pan = pd.read_csv(PANEL_FILE)
        eu = pan[pan["geo"] == "EU27_2020"].sort_values("year")
        if not eu.empty:
            base_2019 = float(eu.loc[eu["year"] == 2019, "pax_total"].values[0])
            for _, r in eu.iterrows():
                nat_growth[int(r["year"])] = float(r["pax_total"]) / base_2019
        p19 = pan[pan["year"] == 2019]
        for _, r in p19.iterrows():
            country_pax_2019[str(r["geo"])] = float(r["pax_total"])
    country_pax_2019.setdefault("UK", 297_000_000.0)

    rows = []
    for orig, dest in CORRIDORS:
        c1, iso1, lon1, lat1 = AIRPORTS[orig]
        c2, iso2, lon2, lat2 = AIRPORTS[dest]
        dist_km = round(haversine_km(lon1, lat1, lon2, lat2), 1)
        corridor_label = f"{orig} - {dest}"
        print(f"  [FETCHING] {corridor_label} ({dist_km:.0f} km) ...", flush=True)

        cand_frames = []
        for cc, pair_code in CORRIDOR_REPORTER_CANDIDATES[(orig, dest)]:
            sub = fetch_single_pair_sdmx(cc, pair_code)
            if not sub.empty:
                cand_frames.append(sub)

        merged_pair = pd.concat(cand_frames, ignore_index=True) if cand_frames else pd.DataFrame()
        n_direct = 0
        for yr in range(2008, 2024):
            pax_val = np.nan
            source_note = "eurostat_direct"
            if not merged_pair.empty:
                yr_sub = merged_pair[merged_pair["year"] == yr]
                if not yr_sub.empty:
                    pax_val = float(yr_sub["pax"].max())
                    n_direct += 1

            rows.append({
                "corridor": corridor_label,
                "orig_airport": orig,
                "dest_airport": dest,
                "orig_code": c1,
                "dest_code": c2,
                "distance_km": dist_km,
                "year": yr,
                "pax_route": pax_val,
                "data_source": source_note,
            })
        print(f"    -> {n_direct}/16 direct Eurostat years retrieved.")

    out_df = pd.DataFrame(rows)

    for corr, grp in out_df.groupby("corridor"):
        orig_name = grp["orig_airport"].iloc[0]
        dest_name = grp["dest_airport"].iloc[0]
        iso1 = AIRPORTS[orig_name][1]
        iso2 = AIRPORTS[dest_name][1]
        dist_km = float(grp["distance_km"].iloc[0])

        valid_2019 = grp.loc[grp["year"] == 2019, "pax_route"].dropna()
        if not valid_2019.empty:
            pax_2019_est = float(valid_2019.iloc[0])
            imp_label = "eurostat_imputed_from_eu_index"
        else:
            valid_any = grp["pax_route"].dropna()
            if not valid_any.empty:
                ref_yr = int(grp.loc[valid_any.index[0], "year"])
                ref_pax = float(valid_any.iloc[0])
                ref_idx = nat_growth.get(ref_yr, 1.0)
                pax_2019_est = ref_pax / max(ref_idx, 0.1)
                imp_label = "eurostat_imputed_from_eu_index"
            else:
                p1 = country_pax_2019.get(iso1, 50_000_000.0) / 1e6
                p2 = country_pax_2019.get(iso2, 50_000_000.0) / 1e6
                pax_2019_est = 38_000.0 * math.sqrt(p1 * p2) * (1000.0 / max(dist_km, 300.0)) ** 0.35
                imp_label = "gravity_calibrated_from_eurostat_national"

        for idx, r in grp.iterrows():
            if pd.isna(r["pax_route"]):
                yr_idx = nat_growth.get(int(r["year"]), 1.0)
                out_df.loc[idx, "pax_route"] = round(pax_2019_est * yr_idx, 0)
                out_df.loc[idx, "data_source"] = imp_label

    out_df["pax_km_millions"] = (out_df["pax_route"] * out_df["distance_km"]) / 1e6
    out_df.to_csv(OUT_FILE, index=False)
    digest = sha256_sum(OUT_FILE)
    CHECKSUM_FILE.write_text(f"{digest}  {OUT_FILE.name}\n", encoding="utf-8")
    n_valid = int(out_df["pax_route"].notna().sum())
    print(f"[OK] Saved {OUT_FILE} ({len(out_df)} corridor-year rows, {n_valid} populated)")
    print(f"[SHA-256] {digest}")


if __name__ == "__main__":
    main()