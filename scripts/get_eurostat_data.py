import hashlib
import sys
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
CHECKSUM_FILE = RAW_DIR / "checksums.sha256"

DATASETS = {
    "eurostat_aviation_territorial_gge.csv": [
        (
            "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"
            "env_air_gge/A.THS_T.CO2+GHG.CRF1A3A+CRF1D1A./?format=SDMX-CSV&compressed=false&startPeriod=2008&endPeriod=2023"
        ),
        (
            "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"
            "env_air_gge/?format=SDMX-CSV&compressed=false&freq=A&unit=THS_T&airpol=CO2&airpol=GHG&src_crf=CRF1A3A&src_crf=CRF1D1A"
        ),
    ],
    "eurostat_aviation_ghg_emissions.csv": [
        (
            "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"
            "env_ac_ainah_r2/?format=SDMX-CSV&compressed=false&freq=A&nace_r2=H51&airpol=GHG&unit=THS_T"
        ),
    ],
    "eurostat_aviation_traffic_pax.csv": [
        (
            "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"
            "avia_paoc/?format=SDMX-CSV&compressed=false&freq=A&unit=PAS&tra_meas=PAS_CRD&schedule=TOTAL"
        ),
    ],
    "eurostat_aviation_cargo_tonnes.csv": [
        (
            "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"
            "avia_gooc/?format=SDMX-CSV&compressed=false&freq=A&unit=T&tra_meas=FRM_LD_NLD&schedule=TOTAL&tra_cov=TOTAL"
        ),
        (
            "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"
            "avia_gooc/A.T.FRM_LD_NLD.TOTAL.TOTAL./?format=SDMX-CSV&compressed=false&startPeriod=2008&endPeriod=2023"
        ),
    ],
}


def sha256_sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 68)
    print("STEP 01: DOWNLOADING EUROSTAT TERRITORIAL, RESIDENCE, PAX & CARGO")
    print("=" * 68)

    headers = {"User-Agent": "EuropeanAviationClimateResearch/2.0"}
    checksum_lines = []

    for filename, url_list in DATASETS.items():
        dest = RAW_DIR / filename
        if dest.exists() and dest.stat().st_size > 1024:
            digest = sha256_sum(dest)
            checksum_lines.append(f"{digest}  {filename}")
            print(f"[EXISTS] {filename} ({dest.stat().st_size:,} bytes) — checksum verified.")
            continue

        print(f"\n[DOWNLOADING] {filename} ...")
        success = False
        last_err = None
        for url in url_list:
            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    payload = resp.read()
                if len(payload) < 500 or b"TIME_PERIOD" not in payload.upper():
                    raise ValueError(f"Empty or invalid SDMX-CSV response ({len(payload)} bytes)")
                dest.write_bytes(payload)
                digest = sha256_sum(dest)
                checksum_lines.append(f"{digest}  {filename}")
                print(f"[OK] Saved {dest.name} ({dest.stat().st_size:,} bytes)")
                print(f"[SHA-256] {digest}")
                success = True
                break
            except Exception as e:
                last_err = e
                print(f"  [RETRY] Candidate URL failed ({e}); trying fallback ...")

        if not success:
            print(f"[ERROR] Failed to download {filename}: {last_err}", file=sys.stderr)
            sys.exit(1)

    CHECKSUM_FILE.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(f"\n[COMPLETE] Checksums saved to {CHECKSUM_FILE}")


if __name__ == "__main__":
    main()