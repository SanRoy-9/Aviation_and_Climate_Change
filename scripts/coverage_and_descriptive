import json
import sys
import urllib.request
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
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

GHG_FILE = RAW_DIR / "eurostat_aviation_ghg_emissions.csv"
PAX_FILE = RAW_DIR / "eurostat_aviation_traffic_pax.csv"

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
    ("Amsterdam (AMS)", "Frankfurt (FRA)"),
]


def load_eurostat_ghg() -> pd.DataFrame:
    df = pd.read_csv(GHG_FILE)
    df.columns = [c.lower() for c in df.columns]
    for col, val in [("unit", "THS_T"), ("freq", "A"), ("airpol", "GHG"), ("nace_r2", "H51")]:
        if col in df.columns:
            df = df[df[col] == val]

    t_col = "time_period" if "time_period" in df.columns else "time"
    v_col = "obs_value" if "obs_value" in df.columns else "value"
    out = df[["geo", t_col, v_col]].rename(columns={t_col: "year", v_col: "ghg_ths_t"}).copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["ghg_ths_t"] = pd.to_numeric(out["ghg_ths_t"], errors="coerce")
    out = out.dropna()
    out["year"] = out["year"].astype(int)
    return out[(out["year"] >= 2008) & (out["year"] <= 2023)]


def load_eurostat_pax() -> pd.DataFrame:
    df = pd.read_csv(PAX_FILE)
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

    # Total passengers (tra_cov == TOTAL)
    tot = df[df["tra_cov"] == "TOTAL"][["geo", "year", "pax"]].rename(columns={"pax": "pax_total"})

    # Route network distribution Z1: international / extra-EU shares
    cov_piv = df.pivot_table(index=["geo", "year"], columns="tra_cov", values="pax", aggfunc="sum").reset_index()
    for c in ["NAT", "INTL", "INTL_EU", "INTL_NEU", "TOTAL"]:
        if c not in cov_piv.columns:
            cov_piv[c] = pd.NA

    cov_piv["share_intl_pct"] = (cov_piv["INTL"] / cov_piv["TOTAL"]) * 100.0
    cov_piv["share_extra_eu_pct"] = (cov_piv["INTL_NEU"] / cov_piv["TOTAL"]) * 100.0
    z1 = cov_piv[["geo", "year", "share_intl_pct", "share_extra_eu_pct"]]

    return tot.merge(z1, on=["geo", "year"], how="left")


def plot_fig1_coverage(ghg: pd.DataFrame):
    pivot = ghg[ghg["geo"].isin(CORE_STATES)].pivot_table(
        index="geo", columns="year", values="ghg_ths_t", aggfunc="sum"
    )
    fig = plt.figure(figsize=(20, 6))

    # Panel 1: Map
    if USE_CARTOPY:
        ax1 = fig.add_subplot(1, 3, 1, projection=ccrs.PlateCarree())
        ax1.set_extent([-15, 35, 34, 70], crs=ccrs.PlateCarree())
        ax1.add_feature(cfeature.OCEAN, facecolor="#0d1b2a")
        ax1.add_feature(cfeature.LAND, facecolor="#1e3048", edgecolor="#2c4260")
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="#5c7080", zorder=2)
        ax1.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5, edgecolor="#3a506b", zorder=2)
        ax1.plot(
            [-15, 35, 35, -15, -15], [34, 34, 70, 70, 34],
            color="#e63946", linewidth=2.2, linestyle="--", transform=ccrs.PlateCarree(),
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
            (-15, 34), 50, 36, lw=2.2, edgecolor="#e63946", facecolor="none", linestyle="--",
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
    ax1.set_title("1. Geographic Study Area & Corridors\n(European Airspace: 34°N–70°N, 15°W–35°E)", fontsize=10, fontweight="bold")

    # Panel 2: Spatial Emissions
    ax2 = fig.add_subplot(1, 3, 2)
    pivot[[2019, 2020, 2022]].plot(kind="bar", ax=ax2, colormap="cividis", edgecolor="black", width=0.8)
    ax2.set_title("2. Aviation GHG Emissions in Core States\n(2019 vs. 2020 vs. 2022)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Thousand Tonnes CO2e", fontsize=9)
    ax2.set_xlabel("Member State", fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(title="Year", fontsize=8)

    # Panel 3: Temporal Coverage (distinct lines, no double counting)
    ax3 = fig.add_subplot(1, 3, 3)
    ts = ghg[ghg["geo"].isin(["EU27_2020", "DE", "FR"])].pivot_table(
        index="year", columns="geo", values="ghg_ths_t", aggfunc="sum"
    )
    if "EU27_2020" in ts.columns:
        ax3.plot(ts.index, ts["EU27_2020"], marker="o", color="#b33939", lw=2, ms=4, label="EU27 (2020)")
    if "DE" in ts.columns:
        ax3.plot(ts.index, ts["DE"], marker="s", color="#1d3557", lw=1.6, ms=3.5, label="Germany (DE)")
    if "FR" in ts.columns:
        ax3.plot(ts.index, ts["FR"], marker="^", color="#457b9d", lw=1.6, ms=3.5, label="France (FR)")
    ax3.axvspan(2019.8, 2021.2, color="#f7b731", alpha=0.25, label="COVID-19 Shock (2020–2021)")
    ax3.set_title("3. Temporal Coverage (2008–2023)\n(Continuous Annual Reporting)", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Year", fontsize=9)
    ax3.set_ylabel("Thousand Tonnes CO2e", fontsize=9)
    ax3.grid(True, linestyle=":", alpha=0.5)
    ax3.legend(fontsize=8)

    plt.tight_layout()
    out = FIG_DIR / "fig1_coverage_map.png"
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {out}")


def build_table1(panel: pd.DataFrame):
    sub = panel[panel["geo"].isin(CORE_STATES + ["EU27_2020"])].copy()
    piv_ghg = sub.pivot_table(index="geo", columns="year", values="ghg_ths_t", aggfunc="sum")
    piv_pax = sub.pivot_table(index="geo", columns="year", values="pax_total", aggfunc="sum")

    t1 = pd.DataFrame(index=piv_ghg.index)
    for y in [2008, 2019, 2020, 2022, 2023]:
        t1[f"ghg_{y}_ths_t"] = piv_ghg.get(y)
        t1[f"pax_{y}_millions"] = piv_pax.get(y) / 1e6

    t1["ghg_drop_2020_vs_2019_pct"] = ((t1["ghg_2020_ths_t"] - t1["ghg_2019_ths_t"]) / t1["ghg_2019_ths_t"]) * 100
    t1["pax_drop_2020_vs_2019_pct"] = ((t1["pax_2020_millions"] - t1["pax_2019_millions"]) / t1["pax_2019_millions"]) * 100
    t1["ghg_recovery_2023_vs_2019_pct"] = (t1["ghg_2023_ths_t"] / t1["ghg_2019_ths_t"]) * 100
    t1["pax_recovery_2023_vs_2019_pct"] = (t1["pax_2023_millions"] / t1["pax_2019_millions"]) * 100

    out = TAB_DIR / "table1_descriptive_stats.csv"
    t1.round(2).to_csv(out)
    print(f"[OK] Saved {out}")


def main():
    if not GHG_FILE.exists() or not PAX_FILE.exists():
        print("[ERROR] Eurostat CSVs missing. Run 01_get_eurostat_data.py first.")
        sys.exit(1)

    ghg = load_eurostat_ghg()
    pax = load_eurostat_pax()
    panel = ghg.merge(pax, on=["geo", "year"], how="inner")

    panel_path = PROC_DIR / "eurostat_panel_2008_2023.csv"
    panel.to_csv(panel_path, index=False)
    print(f"[OK] Saved merged panel ({len(panel)} rows) to {panel_path}")

    plot_fig1_coverage(ghg)
    build_table1(panel)


if __name__ == "__main__":
    main()