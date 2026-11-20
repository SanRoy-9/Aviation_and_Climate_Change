import gzip
import hashlib
import io
import math
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

CLIMB_DESCENT_KM = 220.0

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


def fetch_single_pair_metric(cc: str, pair_code: str, candidate_specs: list[tuple[str, str, float]]) -> pd.DataFrame:
    for tra_meas, unit, mult in candidate_specs:
        # Use exact positional SDMX key first (freq.unit.tra_meas.airp_pr) + strict pandas filtering
        urls = [
            (
                f"https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"
                f"avia_par_{cc}/A.{unit}.{tra_meas}.{pair_code}/?format=SDMX-CSV&compressed=false&startPeriod=2008&endPeriod=2023"
            ),
            (
                f"https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"
                f"avia_par_{cc}/A.{tra_meas}.{unit}.{pair_code}/?format=SDMX-CSV&compressed=false&startPeriod=2008&endPeriod=2023"
            ),
        ]
        for url in urls:
            req = urllib.request.Request(url, headers={"User-Agent": "EuropeanAviationClimateResearch/2.0"})
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

            # Strict in-memory filter so wrong units/measures can NEVER leak in
            if "unit" in df.columns:
                df = df[df["unit"] == unit]
            if "tra_meas" in df.columns:
                df = df[df["tra_meas"] == tra_meas]
            if "airp_pr" in df.columns:
                df = df[df["airp_pr"] == pair_code]

            t_col = "time_period" if "time_period" in df.columns else "time"
            v_col = "obs_value" if "obs_value" in df.columns else "value"
            if df.empty or t_col not in df.columns or v_col not in df.columns:
                continue

            df["year"] = pd.to_numeric(df[t_col], errors="coerce")
            df["val"] = pd.to_numeric(df[v_col], errors="coerce") * mult
            df = df.dropna(subset=["year", "val"]).copy()
            df["year"] = df["year"].astype(int)
            df = df[(df["year"] >= 2008) & (df["year"] <= 2023) & (df["val"] > 0)]
            if not df.empty:
                df["reporter"] = cc.upper()
                return df[["reporter", "year", "val"]].groupby(["reporter", "year"], as_index=False)["val"].max()

    return pd.DataFrame()


def main():
    print("=" * 68)
    print("STEP 01B: DOWNLOADING EUROSTAT AIRPORT-PAIR PASSENGERS & FLIGHTS")
    print("=" * 68)

    # Always refresh if flights look corrupted (e.g., pax_per_flight < 20)
    if OUT_FILE.exists() and OUT_FILE.stat().st_size > 2048:
        existing_df = pd.read_csv(OUT_FILE)
        required_cols = {"pax_route", "flights_route", "pax_per_flight", "cruise_flight_km_millions"}
        if (
            required_cols.issubset(existing_df.columns)
            and existing_df["pax_route"].notna().sum() >= 140
            and existing_df["flights_route"].notna().sum() >= 140
            and float(existing_df["pax_per_flight"].min()) >= 30.0
        ):
            digest = sha256_sum(OUT_FILE)
            CHECKSUM_FILE.write_text(f"{digest}  {OUT_FILE.name}\n", encoding="utf-8")
            print(f"[EXISTS] {OUT_FILE.name} — pax & flights sanity-checked (min pax/flight = {existing_df['pax_per_flight'].min():.1f}).")
            return
        print("[REFRESH] Detected corrupted flight units in cached file; re-fetching strictly filtered Eurostat series ...")

    nat_growth = {}
    if PANEL_FILE.exists():
        pan = pd.read_csv(PANEL_FILE)
        eu = pan[pan["geo"] == "EU27_2020"].sort_values("year")
        if not eu.empty and (eu["year"] == 2019).any():
            base_2019 = float(eu.loc[eu["year"] == 2019, "pax_total"].values[0])
            for _, r in eu.iterrows():
                nat_growth[int(r["year"])] = float(r["pax_total"]) / base_2019

    pax_specs = [("PAS_CRD", "PAS", 1.0), ("PAS_BRD", "PAS", 1.0)]
    flt_specs = [("CAF_PAS", "FLIGHT", 1.0), ("CAF_PAS_DEP", "FLIGHT", 2.0)]

    rows = []
    for orig, dest in CORRIDORS:
        c1, _, lon1, lat1 = AIRPORTS[orig]
        c2, _, lon2, lat2 = AIRPORTS[dest]
        dist_km = round(haversine_km(lon1, lat1, lon2, lat2), 1)
        cruise_km = round(max(dist_km - CLIMB_DESCENT_KM, 0.25 * dist_km), 1)
        cruise_level_hpa = 300 if dist_km < 600.0 else 250
        corridor_label = f"{orig} - {dest}"
        print(f"  [FETCHING] {corridor_label} ({dist_km:.0f} km | cruise {cruise_km:.0f} km @ {cruise_level_hpa} hPa) ...", flush=True)

        pax_frames, flt_frames = [], []
        for cc, pair_code in CORRIDOR_REPORTER_CANDIDATES[(orig, dest)]:
            p_sub = fetch_single_pair_metric(cc, pair_code, pax_specs)
            if not p_sub.empty:
                pax_frames.append(p_sub)
            f_sub = fetch_single_pair_metric(cc, pair_code, flt_specs)
            if not f_sub.empty:
                flt_frames.append(f_sub)

        merged_pax = pd.concat(pax_frames, ignore_index=True) if pax_frames else pd.DataFrame()
        merged_flt = pd.concat(flt_frames, ignore_index=True) if flt_frames else pd.DataFrame()

        n_pax_dir, n_flt_dir = 0, 0
        for yr in range(2008, 2024):
            pax_val, flt_val = np.nan, np.nan
            if not merged_pax.empty:
                yr_p = merged_pax[merged_pax["year"] == yr]
                if not yr_p.empty:
                    pax_val = float(yr_p["val"].max())
                    n_pax_dir += 1
            if not merged_flt.empty:
                yr_f = merged_flt[merged_flt["year"] == yr]
                if not yr_f.empty:
                    cand_flt = float(yr_f["val"].max())
                    # Sanity check: commercial flights must imply >= 25 passengers/flight
                    if np.isnan(pax_val) or (pax_val / cand_flt >= 25.0):
                        flt_val = cand_flt
                        n_flt_dir += 1

            rows.append({
                "corridor": corridor_label,
                "orig_airport": orig,
                "dest_airport": dest,
                "orig_code": c1,
                "dest_code": c2,
                "distance_km": dist_km,
                "cruise_km": cruise_km,
                "typical_cruise_hpa": cruise_level_hpa,
                "year": yr,
                "pax_route": pax_val,
                "flights_route": flt_val,
                "pax_source": "eurostat_direct" if not np.isnan(pax_val) else "imputed",
                "flights_source": "eurostat_direct" if not np.isnan(flt_val) else "imputed",
            })
        print(f"    -> Direct Eurostat years: {n_pax_dir}/16 pax | {n_flt_dir}/16 flights.")

    out_df = pd.DataFrame(rows)

    for corr, grp in out_df.groupby("corridor"):
        both = grp.dropna(subset=["pax_route", "flights_route"])
        pax_per_flt = float((both["pax_route"] / both["flights_route"]).median()) if not both.empty else 130.0

        valid_p19 = grp.loc[grp["year"] == 2019, "pax_route"].dropna()
        if not valid_p19.empty:
            p19_est = float(valid_p19.iloc[0])
        else:
            valid_any = grp["pax_route"].dropna()
            if not valid_any.empty:
                ref_yr = int(grp.loc[valid_any.index[0], "year"])
                p19_est = float(valid_any.iloc[0]) / max(nat_growth.get(ref_yr, 1.0), 0.1)
            else:
                p19_est = np.nan

        for idx, r in grp.iterrows():
            yr_idx = nat_growth.get(int(r["year"]), 1.0)
            if pd.isna(r["pax_route"]) and pd.notna(p19_est):
                out_df.loc[idx, "pax_route"] = round(p19_est * yr_idx, 0)
            if pd.isna(r["flights_route"]) and pd.notna(out_df.loc[idx, "pax_route"]):
                out_df.loc[idx, "flights_route"] = round(out_df.loc[idx, "pax_route"] / pax_per_flt, 0)

    out_df["pax_per_flight"] = (out_df["pax_route"] / out_df["flights_route"]).round(1)
    out_df["pax_km_millions"] = ((out_df["pax_route"] * out_df["distance_km"]) / 1e6).round(3)
    out_df["flight_km_millions"] = ((out_df["flights_route"] * out_df["distance_km"]) / 1e6).round(4)
    out_df["cruise_flight_km_millions"] = ((out_df["flights_route"] * out_df["cruise_km"]) / 1e6).round(4)

    out_df.to_csv(OUT_FILE, index=False)
    digest = sha256_sum(OUT_FILE)
    CHECKSUM_FILE.write_text(f"{digest}  {OUT_FILE.name}\n", encoding="utf-8")
    print(f"[OK] Saved {OUT_FILE} ({len(out_df)} corridor-year rows)")
    print("[2019 CORRIDOR SANITY CHECK]:")
    print(out_df[out_df["year"] == 2019][["corridor", "pax_route", "flights_route", "pax_per_flight", "cruise_flight_km_millions"]].to_string(index=False))


if __name__ == "__main__":
    main()