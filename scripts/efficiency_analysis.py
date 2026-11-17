import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"
FIG_DIR = BASE_DIR / "reports" / "figures"
TAB_DIR = BASE_DIR / "reports" / "tables"

PANEL_FILE = PROC_DIR / "eurostat_panel_2008_2023.csv"


def load_optional_eaer() -> pd.DataFrame | None:
    candidates = list(RAW_DIR.glob("*EAER*.csv")) + list(RAW_DIR.glob("*eaer*.csv"))
    if not candidates:
        return None
    df = pd.read_csv(candidates[0])
    df.columns = [c.lower() for c in df.columns]
    if "year" in df.columns and "g_co2_pkm" in df.columns:
        return df[["year", "g_co2_pkm"]]
    return None


def main():
    if not PANEL_FILE.exists():
        print("[ERROR] Merged panel missing. Run 03_coverage_and_descriptive.py first.")
        sys.exit(1)

    df = pd.read_csv(PANEL_FILE)

    # Prefer EU27_2020 aggregate if available for all years; otherwise sum reporting states cleanly
    eu = df[df["geo"] == "EU27_2020"].sort_values("year").copy()
    if len(eu) < 12:
        agg = df[~df["geo"].isin(["EU27_2020", "EU28", "EA19", "EA20"])].groupby("year", as_index=False)[
            ["ghg_ths_t", "pax_total"]
        ].sum()
        source_label = "Sum of Reporting European States"
    else:
        agg = eu[["year", "ghg_ths_t", "pax_total"]].copy()
        source_label = "EU27_2020 Official Aggregate"

    # Derived intensity: kg CO2e per passenger carried
    # 1 thousand tonne = 1,000,000 kg
    agg["kg_co2_per_pax"] = (agg["ghg_ths_t"] * 1e6) / agg["pax_total"]

    # Optional EASA EAER g CO2/pkm merge
    eaer = load_optional_eaer()
    if eaer is not None:
        agg = agg.merge(eaer, on="year", how="left")
        print("[INFO] Merged EASA EAER g_co2_pkm series.")
    else:
        print("[NOTE] No EASA EAER CSV found in data/raw/ (using Eurostat kg_co2_per_pax; add EAER CSV with columns year,g_co2_pkm if desired).")

    # Base 2008 = 100 indices
    base = agg.loc[agg["year"] == agg["year"].min()].iloc[0]
    agg["index_co2_2008_100"] = (agg["ghg_ths_t"] / base["ghg_ths_t"]) * 100.0
    agg["index_pax_2008_100"] = (agg["pax_total"] / base["pax_total"]) * 100.0
    agg["index_intensity_2008_100"] = (agg["kg_co2_per_pax"] / base["kg_co2_per_pax"]) * 100.0
    if "g_co2_pkm" in agg.columns and pd.notna(base.get("g_co2_pkm")):
        agg["index_g_co2_pkm_2008_100"] = (agg["g_co2_pkm"] / base["g_co2_pkm"]) * 100.0

    agg["yoy_co2_pct"] = agg["ghg_ths_t"].pct_change() * 100.0
    agg["yoy_pax_pct"] = agg["pax_total"].pct_change() * 100.0
    agg["yoy_intensity_pct"] = agg["kg_co2_per_pax"].pct_change() * 100.0

    out_tab = TAB_DIR / "table4_h3_efficiency_decomposition.csv"
    agg.round(3).to_csv(out_tab, index=False)
    print(f"[OK] Saved {out_tab}")

    # Plot Figure 5: Indexed Traffic vs. Intensity vs. Total CO2
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    axes[0].plot(agg["year"], agg["index_pax_2008_100"], marker="o", lw=2, color="#1d3557", label="Air Traffic (Passengers, 2008=100)")
    axes[0].plot(agg["year"], agg["index_co2_2008_100"], marker="s", lw=2, color="#e63946", label="Total Aviation GHG (2008=100)")
    axes[0].plot(agg["year"], agg["index_intensity_2008_100"], marker="^", lw=2, color="#2a9d8f", label="Emissions per Passenger (2008=100)")
    if "index_g_co2_pkm_2008_100" in agg.columns:
        axes[0].plot(agg["year"], agg["index_g_co2_pkm_2008_100"], marker="d", lw=1.8, linestyle="--", color="#f4a261", label="EASA g CO2/pkm (2008=100)")
    axes[0].axhline(100, color="gray", lw=1, linestyle=":")
    axes[0].axvspan(2019.8, 2021.2, color="#f7b731", alpha=0.22, label="COVID-19 Shock")
    axes[0].set_title(f"1. Efficiency Gains vs. Demand Growth (H3)\n({source_label})", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Index (2008 = 100)")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(fontsize=8)

    # Panel 2: Pre-COVID CAGR comparison bar chart (2008-2019)
    pre = agg[agg["year"] <= 2019]
    n_yrs = pre["year"].max() - pre["year"].min()
    cagr_pax = ((pre["pax_total"].iloc[-1] / pre["pax_total"].iloc[0]) ** (1 / n_yrs) - 1) * 100
    cagr_co2 = ((pre["ghg_ths_t"].iloc[-1] / pre["ghg_ths_t"].iloc[0]) ** (1 / n_yrs) - 1) * 100
    cagr_int = ((pre["kg_co2_per_pax"].iloc[-1] / pre["kg_co2_per_pax"].iloc[0]) ** (1 / n_yrs) - 1) * 100

    labels = ["Air Traffic\nGrowth (%/yr)", "Intensity\nChange (%/yr)", "Net Total CO2\nGrowth (%/yr)"]
    vals = [cagr_pax, cagr_int, cagr_co2]
    colors = ["#1d3557", "#2a9d8f", "#e63946"]
    bars = axes[1].bar(labels, vals, color=colors, edgecolor="black", width=0.55)
    axes[1].axhline(0, color="black", lw=1)
    for bar, v in zip(bars, vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + (0.15 if v >= 0 else -0.35), f"{v:+.2f}%", ha="center", fontweight="bold", fontsize=9)
    axes[1].set_title("2. Pre-COVID Annual Growth Rates (2008–2019)\n(Why Demand Outpaces Efficiency)", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Compound Annual Change (% / year)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_fig = FIG_DIR / "fig5_h3_efficiency_vs_demand.png"
    plt.savefig(out_fig, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {out_fig}")


if __name__ == "__main__":
    main()