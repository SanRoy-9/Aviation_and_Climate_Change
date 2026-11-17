import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw_data"
RAW_DIR.mkdir(parents=True, exist_ok=True)
CHECKSUM_FILE = RAW_DIR / "checksums.sha256"

DATASETS = {
    # Response Y1: Air transport GHG emissions (Thousand tonnes CO2e)
    "eurostat_aviation_ghg_emissions.csv": (
        "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"
        "env_ac_ainah_r2/?format=SDMX-CSV&compressed=false&freq=A&nace_r2=H51&airpol=GHG&unit=THS_T"
    ),
    # Predictor X1 & Control Z1: Air passenger transport (annual, passengers carried, all schedules)
    # Kept all tra_cov levels (TOTAL, NAT, INTL_EU, INTL_NEU) so Z1 share can be computed cleanly
    "eurostat_aviation_traffic_pax.csv": (
        "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"
        "avia_paoc/?format=SDMX-CSV&compressed=false&freq=A&unit=PAS&tra_meas=PAS_CRD&schedule=TOTAL"
    ),
}


def sha256_sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 65)
    print("STEP 01: DOWNLOADING EUROSTAT DATA INTO data/raw/")
    print("=" * 65)

    headers = {"User-Agent": "AviationClimateResearch/1.0 (Matriculation_6385648)"}
    checksum_lines = []

    for filename, url in DATASETS.items():
        dest = RAW_DIR / filename
        print(f"\n[INFO] Downloading {filename} ...")
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f_out:
                f_out.write(resp.read())
            digest = sha256_sum(dest)
            checksum_lines.append(f"{digest}  {filename}")
            print(f"[OK] Saved {dest} ({dest.stat().st_size:,} bytes)")
            print(f"[SHA-256] {digest}")
        except urllib.error.URLError as e:
            print(f"[ERROR] Failed to download {filename}: {e}", file=sys.stderr)
            sys.exit(1)

    CHECKSUM_FILE.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(f"\n[COMPLETE] Checksums saved to {CHECKSUM_FILE}")


if __name__ == "__main__":
    main()