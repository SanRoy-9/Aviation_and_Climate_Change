import json
import sys
import urllib.request
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    USE_CARTOPY = True
except ImportError:
    USE_CARTOPY = False

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"
FIG_DIR = BASE_DIR / "reports" / "figures"
TAB_DIR = BASE_DIR / "reports" / "tables"

for d in (PROC_DIR, FIG_DIR, TAB_DIR):
    d.mkdir(parents=True, exist_ok=True)

TERR_FILE = RAW_DIR / "eurostat_aviation_territorial_gge.csv"
RES_GHG_FILE = RAW_DIR / "eurostat_aviation_ghg_emissions.csv"
PAX_FILE = RAW_DIR / "eurostat_aviation_traffic_pax.csv"
CARGO_FILE = RAW_DIR / "eurostat_aviation_cargo_tonnes.csv"

CORE_STATES = ["AT", "BE", "DE", "ES", "FR", "IE", "IT", "NL", "PL", "PT", "SE"]

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


def load_territorial_emissions() -> pd.DataFrame:
    df = pd.read_csv(TERR_FILE, low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    for col, val in [("unit", "THS_T"), ("freq", "A")]:
        if col in df.columns:
            df = df[df[col] == val]

    t_col = "time_period" if "time_period" in df.columns else "time"
    v_col = "obs_value" if "obs_value" in df.columns else "value"
    df["year"] = pd.to_numeric(df[t_col], errors="coerce")
    df["val"] = pd.to_numeric(df[v_col], errors="coerce")
    df = df.dropna(subset=["geo", "year", "val", "src_crf", "airpol"]).copy()
    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= 2008) & (df["year"] <= 2023)]
    df = df[df["src_crf"].isin(["CRF1A3A", "CRF1D1A"]) & df["airpol"].isin(["CO2", "GHG"])]

    piv = df.pivot_table(index=["geo", "year"], columns=["airpol", "src_crf"], values="val", aggfunc="sum")
    out = pd.DataFrame(index=piv.index)
    for pol, prefix in [("CO2", "co2"), ("GHG", "ghg")]:
        dom = piv[(pol, "CRF1A3A")] if (pol, "CRF1A3A") in piv.columns else pd.Series(0.0, index=piv.index)
        intl = piv[(pol, "CRF1D1A")] if (pol, "CRF1D1A") in piv.columns else pd.Series(0.0, index=piv.index)
        both_missing = dom.isna() & intl.isna()
        out[f"dom_{prefix}_ths_t"] = dom.fillna(0.0).where(~both_missing, np.nan)
        out[f"intl_{prefix}_ths_t"] = intl.fillna(0.0).where(~both_missing, np.nan)
        out[f"{prefix}_terr_ths_t"] = (dom.fillna(0.0) + intl.fillna(0.0)).where(~both_missing, np.nan)

    return out.reset_index()


def load_residence_ghg() -> pd.DataFrame:
    if not RES_GHG_FILE.exists():
        return pd.DataFrame(columns=["geo", "year", "ghg_res_ths_t"])
    df = pd.read_csv(RES_GHG_FILE, low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    for col, val in [("unit", "THS_T"), ("freq", "A"), ("airpol", "GHG"), ("nace_r2", "H51")]:
        if col in df.columns:
            df = df[df[col] == val]

    t_col = "time_period" if "time_period" in df.columns else "time"
    v_col = "obs_value" if "obs_value" in df.columns else "value"
    out = df[["geo", t_col, v_col]].rename(columns={t_col: "year", v_col: "ghg_res_ths_t"}).copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["ghg_res_ths_t"] = pd.to_numeric(out["ghg_res_ths_t"], errors="coerce")
    out = out.dropna()
    out["year"] = out["year"].astype(int)
    return out[(out["year"] >= 2008) & (out["year"] <= 2023)]


def load_eurostat_pax() -> pd.DataFrame:
    df = pd.read_csv(PAX_FILE, low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    for col, val in [("freq", "A"), ("unit", "PAS"), ("tra_meas", "PAS_CRD"), ("schedule", "TOTAL")]:
        if col in df.columns:
            df = df[df[col] == val]

    t_col = "time_period" if "time_period" in df.columns else "time"
    v_col = "obs_value" if "obs_value" in df.columns else "value"
    df["year"] = pd.to_numeric(df[t_col], errors="coerce")
    df["pax"] = pd.to_numeric(df[v_col], errors="coerce")
    df = df.dropna(subset=["geo", "year", "pax"])
    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= 2008) & (df["year"] <= 2023)]

    tot = df[df["tra_cov"] == "TOTAL"][["geo", "year", "pax"]].rename(columns={"pax": "pax_total"})
    cov_piv = df.pivot_table(index=["geo", "year"], columns="tra_cov", values="pax", aggfunc="sum").reset_index()
    for c in ["NAT", "INTL", "INTL_EU", "INTL_NEU", "TOTAL"]:
        if c not in cov_piv.columns:
            cov_piv[c] = np.nan

    cov_piv["share_intl_pct"] = (cov_piv["INTL"] / cov_piv["TOTAL"]) * 100.0
    cov_piv["share_extra_eu_pct"] = (cov_piv["INTL_NEU"] / cov_piv["TOTAL"]) * 100.0
    return tot.merge(cov_piv[["geo", "year", "share_intl_pct", "share_extra_eu_pct"]], on=["geo", "year"], how="left")


def load_eurostat_cargo() -> pd.DataFrame:
    if not CARGO_FILE.exists():
        return pd.DataFrame(columns=["geo", "year", "cargo_tonnes"])
    df = pd.read_csv(CARGO_FILE, low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    for col, val in [("freq", "A"), ("unit", "T"), ("tra_meas", "FRM_LD_NLD"), ("schedule", "TOTAL"), ("tra_cov", "TOTAL")]:
        if col in df.columns:
            df = df[df[col] == val]

    t_col = "time_period" if "time_period" in df.columns else "time"
    v_col = "obs_value" if "obs_value" in df.columns else "value"
    out = df[["geo", t_col, v_col]].rename(columns={t_col: "year", v_col: "cargo_tonnes"}).copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["cargo_tonnes"] = pd.to_numeric(out["cargo_tonnes"], errors="coerce")
    out = out.dropna()
    out["year"] = out["year"].astype(int)
    return out[(out["year"] >= 2008) & (out["year"] <= 2023)].groupby(["geo", "year"], as_index=False)["cargo_tonnes"].max()


def plot_fig1_coverage(terr: pd.DataFrame):
    pivot = terr[terr["geo"].isin(CORE_STATES)].pivot_table(
        index="geo", columns="year", values="co2_terr_ths_t", aggfunc="sum"
    )
    fig = plt.figure(figsize=(20, 6))

    if USE_CARTOPY:
        ax1 = fig.add_subplot(1, 3, 1, projection=ccrs.PlateCarree())
        ax1.set_extent([-15, 35, 34, 70], crs=ccrs.PlateCarree())
        ax1.add_feature(cfeature.OCEAN, facecolor="#0d1b2a")
        ax1.add_feature(cfeature.LAND, facecolor="#1e3048", edgecolor="#2c4260")
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="#5c7080", zorder=2)
        ax1.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5, edgecolor="#3a506b", zorder=2)
        ax1.plot(
            [-15, 35, 35, -15, -15], [34, 34, 70, 70, 34],
            color="#e63946", lw=2.2, ls="--", transform=ccrs.PlateCarree(),
            label="Study Area Bounding Box (34°N–70°N, 15°W–35°E)",
        )
        for o, d in CORRIDORS:
            p1, p2 = AIRPORTS[o], AIRPORTS[d]
            ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#00b4d8", lw=1.6, alpha=0.75, transform=ccrs.PlateCarree())
        for name, (lon, lat) in AIRPORTS.items():
            c = "#ffb703" if "FRA" in name else "#f77f00"
            s = 7 if "FRA" in name else 5
            ax1.plot(lon, lat, marker="o", color=c, markersize=s, transform=ccrs.PlateCarree(), zorder=5)
            ax1.text(lon + 0.7, lat - 0.2, name.split()[0], color="white", fontsize=7.5, fontweight="bold", transform=ccrs.PlateCarree(), zorder=6)
    else:
        ax1 = fig.add_subplot(1, 3, 1, facecolor="#0d1b2a")
        ax1.set_xlim(-16, 36)
        ax1.set_ylim(33, 71)
        geojson_url = "https://raw.githubusercontent.com/leakyMirror/map-of-europe/master/GeoJSON/europe.geojson"
        try:
            req = urllib.request.Request(geojson_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                geo_data = json.loads(resp.read().decode())
            for feat in geo_data["features"]:
                coords = feat["geometry"]["coordinates"]
                if feat["geometry"]["type"] == "Polygon":
                    coords = [coords]
                for poly in coords:
                    xs, ys = zip(*poly[0])
                    ax1.fill(xs, ys, facecolor="#1e3048", edgecolor="#2c4260", lw=0.5)
        except Exception:
            ax1.fill([-15, 35, 35, -15], [34, 34, 70, 70], facecolor="#1e3048", alpha=0.4)
        ax1.add_patch(mpatches.Rectangle(
            (-15, 34), 50, 36, lw=2.2, edgecolor="#e63946", facecolor="none", ls="--",
            label="Study Area Bounding Box (34°N–70°N, 15°W–35°E)"
        ))
        for o, d in CORRIDORS:
            p1, p2 = AIRPORTS[o], AIRPORTS[d]
            ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#00b4d8", lw=1.6, alpha=0.75)
        for name, (lon, lat) in AIRPORTS.items():
            c = "#ffb703" if "FRA" in name else "#f77f00"
            s = 55 if "FRA" in name else 35
            ax1.scatter(lon, lat, color=c, s=s, zorder=5)
            ax1.annotate(name.split()[0], (lon + 0.7, lat - 0.3), color="white", fontsize=7.5, fontweight="bold")

    ax1.set_xlabel("Longitude (°E)", fontsize=8.5, color="#c0d6df")
    ax1.set_ylabel("Latitude (°N)", fontsize=8.5, color="#c0d6df")
    ax1.tick_params(colors="#c0d6df")
    ax1.grid(True, linestyle=":", alpha=0.25, color="white")
    ax1.legend(loc="lower left", fontsize=7.5)
    ax1.set_title("1. European Study Domain & Core Corridors\n(34°N–70°N, 15°W–35°E)", fontsize=10, fontweight="bold")

    ax2 = fig.add_subplot(1, 3, 2)
    cols = [y for y in [2019, 2020, 2023] if y in pivot.columns]
    pivot[cols].plot(kind="bar", ax=ax2, colormap="cividis", edgecolor="black", width=0.8)
    ax2.set_title("2. Territorial Aviation CO2 in Core States\n(UNFCCC Bunkers + Domestic: 2019 vs. 2020 vs. 2023)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Thousand Tonnes CO2", fontsize=9)
    ax2.set_xlabel("Member State", fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(title="Year", fontsize=8)

    ax3 = fig.add_subplot(1, 3, 3)
    ts = terr[terr["geo"].isin(["EU27_2020", "DE", "ES", "FR"])].pivot_table(
        index="year", columns="geo", values="co2_terr_ths_t", aggfunc="sum"
    )
    if "EU27_2020" in ts.columns:
        ax3.plot(ts.index, ts["EU27_2020"], marker="o", color="#b33939", lw=2.2, ms=4, label="EU27 (2020)")
    if "DE" in ts.columns:
        ax3.plot(ts.index, ts["DE"], marker="s", color="#1d3557", lw=1.6, ms=3.5, label="Germany (DE)")
    if "FR" in ts.columns:
        ax3.plot(ts.index, ts["FR"], marker="^", color="#457b9d", lw=1.6, ms=3.5, label="France (FR)")
    if "ES" in ts.columns:
        ax3.plot(ts.index, ts["ES"], marker="d", color="#2a9d8f", lw=1.6, ms=3.5, label="Spain (ES)")
    ax3.axvspan(2019.8, 2021.2, color="#f7b731", alpha=0.25, label="COVID-19 Shock (2020–2021)")
    ax3.set_title("3. Territorial Aviation CO2 Trajectory (2008–2023)\n(Domestic + International Bunkers)", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Year", fontsize=9)
    ax3.set_ylabel("Thousand Tonnes CO2", fontsize=9)
    ax3.grid(True, linestyle=":", alpha=0.5)
    ax3.legend(fontsize=8)

    plt.tight_layout()
    out = FIG_DIR / "fig1_coverage_map.png"
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {out}")


def build_tables(panel: pd.DataFrame):
    sub = panel[panel["geo"].isin(CORE_STATES + ["EU27_2020"])].copy()
    piv_co2 = sub.pivot_table(index="geo", columns="year", values="co2_terr_ths_t", aggfunc="sum")
    piv_pax = sub.pivot_table(index="geo", columns="year", values="pax_total", aggfunc="sum")

    t1 = pd.DataFrame(index=piv_co2.index)
    for y in [2008, 2019, 2020, 2022, 2023]:
        t1[f"co2_terr_{y}_ths_t"] = piv_co2.get(y)
        t1[f"pax_{y}_millions"] = piv_pax.get(y) / 1e6

    t1["co2_drop_2020_vs_2019_pct"] = ((t1["co2_terr_2020_ths_t"] - t1["co2_terr_2019_ths_t"]) / t1["co2_terr_2019_ths_t"]) * 100
    t1["pax_drop_2020_vs_2019_pct"] = ((t1["pax_2020_millions"] - t1["pax_2019_millions"]) / t1["pax_2019_millions"]) * 100
    t1["co2_recovery_2023_vs_2019_pct"] = (t1["co2_terr_2023_ths_t"] / t1["co2_terr_2019_ths_t"]) * 100
    t1["pax_recovery_2023_vs_2019_pct"] = (t1["pax_2023_millions"] / t1["pax_2019_millions"]) * 100

    out1 = TAB_DIR / "table1_descriptive_stats.csv"
    t1.round(2).to_csv(out1)
    print(f"[OK] Saved {out1}")

    p19 = sub[sub["year"] == 2019].copy()
    audit = p19[["geo", "co2_terr_ths_t", "ghg_terr_ths_t", "ghg_res_ths_t", "pax_total", "cargo_tonnes"]].copy()
    audit["pax_millions"] = audit["pax_total"] / 1e6
    audit["kg_co2_per_pax_territorial"] = (audit["co2_terr_ths_t"] * 1e6) / audit["pax_total"]
    audit["kg_ghg_per_pax_residence"] = (audit["ghg_res_ths_t"] * 1e6) / audit["pax_total"]
    audit["residence_vs_territorial_ratio"] = audit["ghg_res_ths_t"] / audit["ghg_terr_ths_t"]
    audit = audit.sort_values("co2_terr_ths_t", ascending=False)

    out1b = TAB_DIR / "table1b_territorial_vs_residence_audit.csv"
    audit.round(2).to_csv(out1b, index=False)
    print(f"[OK] Saved {out1b}")
    print("\n[SANITY CHECK — 2019 TERRITORIAL vs. RESIDENCE ACCOUNTING]:")
    print(audit[audit["geo"].isin(["EU27_2020", "DE", "ES", "FR", "IE", "NL"])][
        ["geo", "co2_terr_ths_t", "ghg_res_ths_t", "kg_co2_per_pax_territorial", "kg_ghg_per_pax_residence", "residence_vs_territorial_ratio"]
    ].round(2).to_string(index=False))


def main():
    if not TERR_FILE.exists() or not PAX_FILE.exists():
        print("[ERROR] Eurostat CSVs missing. Run get_eurostat_data.py first.")
        sys.exit(1)

    terr = load_territorial_emissions()
    res_ghg = load_residence_ghg()
    pax = load_eurostat_pax()
    cargo = load_eurostat_cargo()

    panel = terr.merge(pax, on=["geo", "year"], how="inner")
    panel = panel.merge(res_ghg, on=["geo", "year"], how="left")
    panel = panel.merge(cargo, on=["geo", "year"], how="left")

    panel.to_csv(PROC_DIR / "eurostat_panel_2008_2023.csv", index=False)
    n_cargo = int(panel["cargo_tonnes"].notna().sum())
    print(f"[OK] Saved merged panel ({len(panel)} rows, {panel['geo'].nunique()} geos, {n_cargo} rows with cargo data)")

    plot_fig1_coverage(terr)
    build_tables(panel)


if __name__ == "__main__":
    main()