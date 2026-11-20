import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"
FIG_DIR = BASE_DIR / "reports" / "figures"
TAB_DIR = BASE_DIR / "reports" / "tables"

PANEL_FILE = PROC_DIR / "eurostat_panel_2008_2023.csv"
AGGREGATE_CODES = {"EU27_2020", "EU28", "EA19", "EA20"}
CORE_STATES = ["AT", "BE", "DE", "ES", "FR", "IE", "IT", "NL", "PL", "PT", "SE"]
EU27_CODES = [
    "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES", "FI", "FR", "HR",
    "HU", "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK"
]


def classify_tapio(pct_co2: float, pct_pax: float) -> tuple[float, str]:
    if abs(pct_pax) < 1e-9:
        return np.nan, "Undefined"
    e = pct_co2 / pct_pax
    if pct_pax > 0 and pct_co2 > 0:
        if e > 1.2:
            regime = "Expansive Negative Decoupling"
        elif e >= 0.8:
            regime = "Expansive Coupling"
        else:
            regime = "Weak Decoupling"
    elif pct_pax > 0 and pct_co2 <= 0:
        regime = "Strong Decoupling"
    elif pct_pax < 0 and pct_co2 < 0:
        if e > 1.2:
            regime = "Recessive Decoupling"
        elif e >= 0.8:
            regime = "Recessive Coupling"
        else:
            regime = "Weak Negative Decoupling"
    else:
        regime = "Strong Negative Decoupling"
    return round(float(e), 3), regime


def compute_lmdi_period(df_panel: pd.DataFrame, countries: list[str], y0: int, y1: int, group_label: str) -> dict:
    sub0 = df_panel[(df_panel["geo"].isin(countries)) & (df_panel["year"] == y0)].set_index("geo")
    sub1 = df_panel[(df_panel["geo"].isin(countries)) & (df_panel["year"] == y1)].set_index("geo")
    common = sorted(set(sub0.index).intersection(sub1.index))
    sub0 = sub0.loc[common]
    sub1 = sub1.loc[common]

    valid = (sub0["co2_terr_ths_t"] > 0) & (sub1["co2_terr_ths_t"] > 0) & (sub0["pax_total"] > 0) & (sub1["pax_total"] > 0)
    sub0 = sub0[valid]
    sub1 = sub1[valid]

    C0 = sub0["co2_terr_ths_t"].values
    C1 = sub1["co2_terr_ths_t"].values
    P0 = sub0["pax_total"].values
    P1 = sub1["pax_total"].values

    Ptot0, Ptot1 = P0.sum(), P1.sum()
    S0, S1 = P0 / Ptot0, P1 / Ptot1
    I0, I1 = C0 / P0, C1 / P1

    L = np.where(np.isclose(C1, C0), C0, (C1 - C0) / (np.log(C1) - np.log(C0)))

    dC_act = float(np.sum(L * np.log(Ptot1 / Ptot0)))
    dC_str = float(np.sum(L * np.log(S1 / S0)))
    dC_int = float(np.sum(L * np.log(I1 / I0)))
    dC_tot = float(C1.sum() - C0.sum())

    base_c = float(C0.sum())
    return {
        "sample_group": group_label,
        "n_countries": len(sub0),
        "period": f"{y0}–{y1}",
        "start_co2_ths_t": round(base_c, 2),
        "end_co2_ths_t": round(float(C1.sum()), 2),
        "total_change_ths_t": round(dC_tot, 2),
        "activity_effect_ths_t": round(dC_act, 2),
        "country_mix_effect_ths_t": round(dC_str, 2),
        "intensity_effect_ths_t": round(dC_int, 2),
        "total_change_pct": round((dC_tot / base_c) * 100.0, 2),
        "activity_effect_pct": round((dC_act / base_c) * 100.0, 2),
        "country_mix_effect_pct": round((dC_str / base_c) * 100.0, 2),
        "intensity_effect_pct": round((dC_int / base_c) * 100.0, 2),
    }


def main():
    if not PANEL_FILE.exists():
        print("[ERROR] Merged panel missing. Run coverage_and_descriptive.py first.")
        sys.exit(1)

    df = pd.read_csv(PANEL_FILE)
    eu = df[df["geo"] == "EU27_2020"].sort_values("year").copy()
    if len(eu) < 12:
        agg = df[df["geo"].isin(EU27_CODES)].groupby("year", as_index=False)[
            ["co2_terr_ths_t", "pax_total"]
        ].sum().sort_values("year")
        source_label = "Sum of EU27 Member States (Territorial CO2)"
    else:
        agg = eu[["year", "co2_terr_ths_t", "pax_total"]].copy()
        source_label = "EU27_2020 Official Aggregate (Territorial CO2)"

    agg["kg_co2_per_pax"] = (agg["co2_terr_ths_t"] * 1e6) / agg["pax_total"]

    base = agg.iloc[0]
    agg["index_co2_2008_100"] = (agg["co2_terr_ths_t"] / base["co2_terr_ths_t"]) * 100.0
    agg["index_pax_2008_100"] = (agg["pax_total"] / base["pax_total"]) * 100.0
    agg["index_intensity_2008_100"] = (agg["kg_co2_per_pax"] / base["kg_co2_per_pax"]) * 100.0

    agg["yoy_co2_pct"] = agg["co2_terr_ths_t"].pct_change() * 100.0
    agg["yoy_pax_pct"] = agg["pax_total"].pct_change() * 100.0
    agg["yoy_intensity_pct"] = agg["kg_co2_per_pax"].pct_change() * 100.0

    out_tab = TAB_DIR / "table4_h3_efficiency_decomposition.csv"
    agg.round(3).to_csv(out_tab, index=False)
    print(f"[OK] Saved {out_tab}")

    tapio_rows = []
    for _, r in agg.iloc[1:].iterrows():
        el, reg = classify_tapio(float(r["yoy_co2_pct"]), float(r["yoy_pax_pct"]))
        tapio_rows.append({
            "year": int(r["year"]),
            "yoy_pax_pct": round(float(r["yoy_pax_pct"]), 2),
            "yoy_co2_pct": round(float(r["yoy_co2_pct"]), 2),
            "yoy_intensity_pct": round(float(r["yoy_intensity_pct"]), 2),
            "tapio_elasticity": el,
            "decoupling_regime": reg,
        })
    tapio_df = pd.DataFrame(tapio_rows)
    tapio_df.to_csv(TAB_DIR / "table4c_h3_tapio_decoupling.csv", index=False)
    print(f"[OK] Saved {TAB_DIR / 'table4c_h3_tapio_decoupling.csv'}")

    all_states = sorted(set(df["geo"].unique()) - AGGREGATE_CODES)
    lmdi_rows = []
    groups = [
        ("Core_11_EU_States", CORE_STATES),
        ("EU27_Member_States", EU27_CODES),
        ("All_Reporting_European_States", all_states),
    ]
    for grp_name, c_list in groups:
        for y0, y1 in [(2008, 2019), (2019, 2020), (2020, 2023), (2008, 2023)]:
            lmdi_rows.append(compute_lmdi_period(df, c_list, y0, y1, grp_name))
    lmdi_df = pd.DataFrame(lmdi_rows)
    lmdi_df.to_csv(TAB_DIR / "table4b_h3_lmdi_decomposition.csv", index=False)
    print(f"[OK] Saved {TAB_DIR / 'table4b_h3_lmdi_decomposition.csv'}")
    print(lmdi_df[lmdi_df["sample_group"] == "EU27_Member_States"][
        ["period", "n_countries", "total_change_ths_t", "activity_effect_ths_t", "country_mix_effect_ths_t", "intensity_effect_ths_t"]
    ].to_string(index=False))

    pre = agg[agg["year"] <= 2019]
    n_yrs = pre["year"].iloc[-1] - pre["year"].iloc[0]
    cagr_pax = ((pre["pax_total"].iloc[-1] / pre["pax_total"].iloc[0]) ** (1 / n_yrs) - 1) * 100
    cagr_co2 = ((pre["co2_terr_ths_t"].iloc[-1] / pre["co2_terr_ths_t"].iloc[0]) ** (1 / n_yrs) - 1) * 100
    cagr_int = ((pre["kg_co2_per_pax"].iloc[-1] / pre["kg_co2_per_pax"].iloc[0]) ** (1 / n_yrs) - 1) * 100
    print(f"[H3 2008-2019 CAGR] Traffic: {cagr_pax:+.2f}%/yr | CO2/pax Intensity: {cagr_int:+.2f}%/yr | Territorial CO2: {cagr_co2:+.2f}%/yr")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    axes[0].plot(agg["year"], agg["index_pax_2008_100"], marker="o", lw=2, color="#1d3557", label="Air Traffic (Passengers, 2008=100)")
    axes[0].plot(agg["year"], agg["index_co2_2008_100"], marker="s", lw=2, color="#e63946", label="Territorial Aviation CO2 (2008=100)")
    axes[0].plot(agg["year"], agg["index_intensity_2008_100"], marker="^", lw=2, color="#2a9d8f", label="Per-Passenger Emissions Intensity (2008=100)")
    axes[0].axhline(100, color="gray", lw=1, ls=":")
    axes[0].axvspan(2019.8, 2021.2, color="#f7b731", alpha=0.22, label="COVID-19 Shock")
    axes[0].set_title(f"1. Emissions Intensity Gains vs. Demand Growth (H3)\n({source_label})", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Index (2008 = 100)")
    axes[0].grid(True, ls=":", alpha=0.5)
    axes[0].legend(fontsize=8)

    labels = ["Air Traffic\nGrowth (%/yr)", "CO2/Pax Intensity\nChange (%/yr)", "Net Territorial CO2\nGrowth (%/yr)"]
    vals = [cagr_pax, cagr_int, cagr_co2]
    bars = axes[1].bar(labels, vals, color=["#1d3557", "#2a9d8f", "#e63946"], edgecolor="black", width=0.55)
    axes[1].axhline(0, color="black", lw=1)
    for bar, v in zip(bars, vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + (0.15 if v >= 0 else -0.4), f"{v:+.2f}%", ha="center", fontweight="bold", fontsize=9)
    axes[1].set_title("2. Pre-COVID Compound Annual Rates (2008–2019)\n(Traffic Growth Outpaces Intensity Reductions)", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Annual Change (% / year)")
    axes[1].grid(axis="y", ls="--", alpha=0.5)

    plt.tight_layout()
    out_fig = FIG_DIR / "fig5_h3_efficiency_vs_demand.png"
    plt.savefig(out_fig, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {out_fig}")

    lmdi_eu = lmdi_df[lmdi_df["sample_group"] == "EU27_Member_States"].copy()
    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5.4))

    x = np.arange(len(lmdi_eu))
    w = 0.25
    axes2[0].bar(x - w, lmdi_eu["activity_effect_ths_t"] / 1000.0, width=w, color="#1d3557", edgecolor="black", label="Activity Effect (Traffic Volume)")
    axes2[0].bar(x, lmdi_eu["country_mix_effect_ths_t"] / 1000.0, width=w, color="#f4a261", edgecolor="black", label="Country-Mix Structure Effect")
    axes2[0].bar(x + w, lmdi_eu["intensity_effect_ths_t"] / 1000.0, width=w, color="#2a9d8f", edgecolor="black", label="Per-Passenger Intensity Effect")
    axes2[0].plot(x, lmdi_eu["total_change_ths_t"] / 1000.0, "o--", color="#e63946", lw=2, ms=7, label="Net Total CO2 Change")
    axes2[0].axhline(0, color="black", lw=1)
    axes2[0].set_xticks(x)
    axes2[0].set_xticklabels(lmdi_eu["period"], fontsize=9.5)
    axes2[0].set_ylabel("Territorial Aviation CO2 Change (Million Tonnes CO2)")
    axes2[0].set_title("1. EU27 Multi-Country LMDI Decomposition\n(Activity vs. Country Mix vs. Per-Passenger Intensity)", fontsize=10, fontweight="bold")
    axes2[0].grid(axis="y", ls=":", alpha=0.5)
    axes2[0].legend(fontsize=8)

    pre_t = tapio_df[tapio_df["year"] <= 2019]
    cov_t = tapio_df[tapio_df["year"].isin([2020, 2021])]
    rec_t = tapio_df[tapio_df["year"] >= 2022]
    axes2[1].scatter(pre_t["yoy_pax_pct"], pre_t["yoy_co2_pct"], color="#1d3557", s=45, label="2009–2019 (Pre-COVID)")
    axes2[1].scatter(cov_t["yoy_pax_pct"], cov_t["yoy_co2_pct"], color="#e63946", marker="X", s=65, label="2020–2021 (COVID Shock)")
    axes2[1].scatter(rec_t["yoy_pax_pct"], rec_t["yoy_co2_pct"], color="#2a9d8f", marker="^", s=55, label="2022–2023 (Recovery)")
    for _, r in tapio_df.iterrows():
        axes2[1].annotate(str(int(r["year"])), (r["yoy_pax_pct"] + 1.2, r["yoy_co2_pct"] + 0.8), fontsize=7.5)

    lim = max(abs(tapio_df["yoy_pax_pct"]).max(), abs(tapio_df["yoy_co2_pct"]).max()) * 1.15
    gx = np.linspace(-lim, lim, 200)
    axes2[1].plot(gx, gx, "k--", lw=1, alpha=0.6, label="e = 1.0 (Coupling Line)")
    axes2[1].plot(gx, 0.8 * gx, color="#2a9d8f", ls=":", lw=1.2, label="e = 0.8 (Weak Decoupling Cutoff)")
    axes2[1].axhline(0, color="gray", lw=0.9)
    axes2[1].axvline(0, color="gray", lw=0.9)
    axes2[1].set_xlabel("Annual Passenger Growth (% YoY)")
    axes2[1].set_ylabel("Annual Territorial Aviation CO2 Growth (% YoY)")
    axes2[1].set_title("2. Tapio Decoupling Elasticity Trajectory (EU27)\n(Pre-COVID Years Dominated by Weak Decoupling)", fontsize=10, fontweight="bold")
    axes2[1].grid(True, ls=":", alpha=0.5)
    axes2[1].legend(fontsize=7.5, loc="upper left")

    plt.tight_layout()
    out_fig2 = FIG_DIR / "fig5b_h3_lmdi_and_tapio.png"
    plt.savefig(out_fig2, dpi=300)
    plt.close(fig2)
    print(f"[OK] Saved {out_fig2}")


if __name__ == "__main__":
    main()