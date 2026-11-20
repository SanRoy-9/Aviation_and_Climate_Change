import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"
FIG_DIR = BASE_DIR / "reports" / "figures"
TAB_DIR = BASE_DIR / "reports" / "tables"

PANEL_FILE = PROC_DIR / "eurostat_panel_2008_2023.csv"
AGGREGATE_CODES = {"EU27_2020", "EU28", "EA19", "EA20"}


def durbin_watson_stat(resid: np.ndarray) -> float:
    diff = np.diff(resid)
    return float(np.sum(diff ** 2) / np.sum(resid ** 2))


def fit_and_collect(name: str, formula: str, df: pd.DataFrame, pax_term: str = "ln_pax", cargo_term: str = "ln_cargo"):
    if len(df) < 10 or df["geo"].nunique() < 2:
        print(f"[SKIP] {name}: insufficient rows ({len(df)}) or countries ({df['geo'].nunique()}).")
        return None, None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["geo"]})

    resid = np.asarray(res.resid)
    _, jb_p = stats.jarque_bera(resid)
    _, sw_p = stats.shapiro(resid) if len(resid) <= 5000 else (np.nan, np.nan)
    dw = durbin_watson_stat(resid)

    ci = res.conf_int()
    row = {
        "model": name,
        "n_obs": int(res.nobs),
        "n_countries": int(df["geo"].nunique()),
        "r2": res.rsquared,
        "adj_r2": res.rsquared_adj,
        "beta_ln_pax": res.params.get(pax_term, np.nan),
        "se_ln_pax": res.bse.get(pax_term, np.nan),
        "p_ln_pax": res.pvalues.get(pax_term, np.nan),
        "ci_low_pax": ci.loc[pax_term, 0] if pax_term in res.params else np.nan,
        "ci_high_pax": ci.loc[pax_term, 1] if pax_term in res.params else np.nan,
        "beta_ln_cargo": res.params.get(cargo_term, np.nan),
        "se_ln_cargo": res.bse.get(cargo_term, np.nan),
        "p_ln_cargo": res.pvalues.get(cargo_term, np.nan),
        "beta_z3_covid": res.params.get("z3_covid", res.params.get("d_z3_covid", np.nan)),
        "se_z3_covid": res.bse.get("z3_covid", res.bse.get("d_z3_covid", np.nan)),
        "p_z3_covid": res.pvalues.get("z3_covid", res.pvalues.get("d_z3_covid", np.nan)),
        "beta_z1_intl_share": res.params.get("share_intl_pct", np.nan),
        "se_z1_intl_share": res.bse.get("share_intl_pct", np.nan),
        "p_z1_intl_share": res.pvalues.get("share_intl_pct", np.nan),
        "durbin_watson": dw,
        "jarque_bera_p": jb_p,
        "shapiro_wilk_p": sw_p,
    }
    return res, row


def main():
    if not PANEL_FILE.exists():
        print("[ERROR] Merged panel missing. Run coverage_and_descriptive.py first.")
        sys.exit(1)

    df = pd.read_csv(PANEL_FILE)
    df = df[~df["geo"].isin(AGGREGATE_CODES)].copy()
    df = df[(df["co2_terr_ths_t"] > 0) & (df["pax_total"] > 0)].sort_values(["geo", "year"]).copy()

    df["ln_co2"] = np.log(df["co2_terr_ths_t"])
    df["ln_pax"] = np.log(df["pax_total"])
    df["ln_cargo"] = np.where(df["cargo_tonnes"] > 0, np.log(df["cargo_tonnes"]), np.nan)
    df["ln_ghg_res"] = np.where(df["ghg_res_ths_t"] > 0, np.log(df["ghg_res_ths_t"]), np.nan)
    df["z3_covid"] = df["year"].isin([2020, 2021]).astype(int)
    df["year_c"] = df["year"] - 2008
    df["regime"] = np.where(
        df["year"] <= 2019,
        "1_PreCOVID_2008_2019",
        np.where(df["year"].isin([2020, 2021]), "2_COVID_2020_2021", "3_Recovery_2022_2023"),
    )

    df_pre = df[df["year"] <= 2019].copy()
    df_cargo = df.dropna(subset=["ln_cargo"]).copy()
    df_z1_cargo = df.dropna(subset=["share_intl_pct", "ln_cargo"]).copy()
    df_res = df.dropna(subset=["ln_ghg_res", "ln_cargo"]).copy()

    all_geos = set(df["geo"].unique())
    missing_cargo_geos = sorted(all_geos - set(df_cargo["geo"].unique()))
    if missing_cargo_geos:
        print(f"[INFO] Countries without positive cargo series: {missing_cargo_geos}")

    df_fd = df_cargo.sort_values(["geo", "year"]).copy()
    df_fd["year_gap"] = df_fd.groupby("geo")["year"].diff()
    df_fd["d_ln_co2"] = df_fd.groupby("geo")["ln_co2"].diff()
    df_fd["d_ln_pax"] = df_fd.groupby("geo")["ln_pax"].diff()
    df_fd["d_ln_cargo"] = df_fd.groupby("geo")["ln_cargo"].diff()
    df_fd["d_z3_covid"] = df_fd.groupby("geo")["z3_covid"].diff()
    df_fd = df_fd[
        (df_fd["year_gap"] == 1)
        & df_fd["d_ln_co2"].notna()
        & df_fd["d_ln_pax"].notna()
        & df_fd["d_ln_cargo"].notna()
    ].copy()

    dropped_fd = sorted(set(df_cargo["geo"].unique()) - set(df_fd["geo"].unique()))
    if dropped_fd:
        print(f"[INFO] Countries dropped in First Differences (no consecutive years): {dropped_fd}")

    core_models = [
        ("M1_Pooled_Full_Benchmark", "ln_co2 ~ ln_pax + z3_covid", df, "ln_pax", "ln_cargo"),
        ("M2_CountryFE_PreCOVID", "ln_co2 ~ ln_pax + C(geo)", df_pre, "ln_pax", "ln_cargo"),
        ("M3_CountryFE_Full_Z3", "ln_co2 ~ ln_pax + z3_covid + C(geo)", df, "ln_pax", "ln_cargo"),
        ("M4_CountryFE_Full_Z3_Z1_Cargo", "ln_co2 ~ ln_pax + ln_cargo + z3_covid + share_intl_pct + C(geo)", df_z1_cargo, "ln_pax", "ln_cargo"),
        ("M5_TwoWay_FE_Primary_Cargo", "ln_co2 ~ ln_pax + ln_cargo + C(geo) + C(year)", df_cargo, "ln_pax", "ln_cargo"),
    ]

    core_rows = []
    fitted = {}
    with open(TAB_DIR / "h1_full_model_summaries.txt", "w", encoding="utf-8") as f_txt:
        for name, formula, data, pterm, cterm in core_models:
            res, row = fit_and_collect(name, formula, data, pterm, cterm)
            if res is not None:
                fitted[name] = (res, data)
                core_rows.append(row)
                f_txt.write(f"\n{'='*72}\n{name}\n{'='*72}\n{res.summary().as_text()}\n")

        res_m5_raw = smf.ols("ln_co2 ~ ln_pax + ln_cargo + C(geo) + C(year)", data=df_cargo).fit()
        infl = res_m5_raw.get_influence()
        df_cargo["m5_resid"] = res_m5_raw.resid
        df_cargo["m5_stud_resid"] = infl.resid_studentized_internal
        df_cargo["m5_cooks_d"] = infl.cooks_distance[0]

        outliers = df_cargo[df_cargo["m5_stud_resid"].abs() > 3.5].sort_values("m5_cooks_d", ascending=False)
        print("\n[OUTLIER AUDIT] Observations with |studentized residual| > 3.5 in Primary M5 (Territorial CO2):")
        if not outliers.empty:
            print(outliers[["geo", "year", "co2_terr_ths_t", "pax_total", "cargo_tonnes", "m5_stud_resid", "m5_cooks_d"]].to_string(index=False))
        else:
            print("  None — territorial alignment eliminated extreme carrier-liquidation outliers!")
        df_no_out = df_cargo[df_cargo["m5_stud_resid"].abs() <= 3.5].copy()

        robust_models = [
            ("M6_First_Differences_Primary_Cargo", "d_ln_co2 ~ d_ln_pax + d_ln_cargo + d_z3_covid", df_fd, "d_ln_pax", "d_ln_cargo"),
            ("M7_CountryFE_CountryTrends_Cargo", "ln_co2 ~ ln_pax + ln_cargo + z3_covid + C(geo) + C(geo):year_c", df_cargo, "ln_pax", "ln_cargo"),
            ("M8_Regime_Slope_Interaction", "ln_co2 ~ ln_pax * C(regime) + ln_cargo + C(geo)", df_cargo, "ln_pax", "ln_cargo"),
            ("M9_TwoWay_FE_Outlier_Trimmed", "ln_co2 ~ ln_pax + ln_cargo + C(geo) + C(year)", df_no_out, "ln_pax", "ln_cargo"),
            ("M10_Residence_Accounting_Comparison", "ln_ghg_res ~ ln_pax + ln_cargo + C(geo) + C(year)", df_res, "ln_pax", "ln_cargo"),
        ]

        robust_rows = []
        for name, formula, data, pterm, cterm in robust_models:
            res, row = fit_and_collect(name, formula, data, pterm, cterm)
            if res is not None:
                fitted[name] = (res, data)
                robust_rows.append(row)
                f_txt.write(f"\n{'='*72}\n{name}\n{'='*72}\n{res.summary().as_text()}\n")

    t2 = pd.DataFrame(core_rows)
    t2.round(4).to_csv(TAB_DIR / "table2_h1_regression.csv", index=False)
    print(f"\n[OK] Saved {TAB_DIR / 'table2_h1_regression.csv'}")
    print(t2[["model", "n_obs", "n_countries", "r2", "beta_ln_pax", "beta_ln_cargo", "durbin_watson"]].round(4).to_string(index=False))

    t2b = pd.DataFrame(robust_rows)
    t2b.round(4).to_csv(TAB_DIR / "table2b_h1_robustness_models.csv", index=False)
    print(f"\n[OK] Saved {TAB_DIR / 'table2b_h1_robustness_models.csv'}")
    print(t2b[["model", "n_obs", "n_countries", "r2", "beta_ln_pax", "beta_ln_cargo", "durbin_watson"]].round(4).to_string(index=False))

    jk_rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for c in sorted(df_cargo["geo"].unique()):
            sub_c = df_cargo[df_cargo["geo"] != c]
            r_c = smf.ols("ln_co2 ~ ln_pax + ln_cargo + C(geo) + C(year)", data=sub_c).fit(
                cov_type="cluster", cov_kwds={"groups": sub_c["geo"]}
            )
            jk_rows.append({
                "excluded_country": c,
                "n_obs": int(r_c.nobs),
                "beta_ln_pax": round(float(r_c.params["ln_pax"]), 4),
                "se_ln_pax": round(float(r_c.bse["ln_pax"]), 4),
                "p_ln_pax": round(float(r_c.pvalues["ln_pax"]), 6),
                "beta_ln_cargo": round(float(r_c.params["ln_cargo"]), 4),
                "r2": round(float(r_c.rsquared), 4),
            })
    jk_df = pd.DataFrame(jk_rows)
    jk_df.to_csv(TAB_DIR / "table2c_h1_outlier_jackknife.csv", index=False)
    print(f"\n[OK] Saved {TAB_DIR / 'table2c_h1_outlier_jackknife.csv'} "
          f"(M5 Jackknife beta_pax range: {jk_df['beta_ln_pax'].min():.4f} to {jk_df['beta_ln_pax'].max():.4f})")

    res_m5, df_m5 = fitted["M5_TwoWay_FE_Primary_Cargo"]
    m1_res, _ = fitted["M1_Pooled_Full_Benchmark"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    normal = df[df["year"] <= 2019]
    covid = df[df["z3_covid"] == 1]
    recov = df[df["year"] >= 2022]

    axes[0].scatter(normal["ln_pax"], normal["ln_co2"], alpha=0.55, s=22, color="#1d3557", label="2008–2019 (Pre-COVID)")
    axes[0].scatter(covid["ln_pax"], covid["ln_co2"], alpha=0.8, s=28, color="#e63946", marker="x", label="2020–2021 (COVID Shock)")
    axes[0].scatter(recov["ln_pax"], recov["ln_co2"], alpha=0.75, s=24, color="#2a9d8f", marker="^", label="2022–2023 (Recovery)")
    xs = np.linspace(df["ln_pax"].min(), df["ln_pax"].max(), 100)
    axes[0].plot(xs, m1_res.params["Intercept"] + m1_res.params["ln_pax"] * xs, "k--", lw=1.8, label=f"Cross-Sectional Slope = {m1_res.params['ln_pax']:.2f}")
    axes[0].set_xlabel("ln(Passengers Carried)")
    axes[0].set_ylabel("ln(Territorial Aviation CO2, Thousand Tonnes)")
    axes[0].set_title("1. Territorial Aviation CO2 vs. Passenger Traffic (H1)", fontweight="bold")
    axes[0].grid(True, ls=":", alpha=0.5)
    axes[0].legend(fontsize=8)

    axes[1].scatter(df_m5["year"], res_m5.resid, alpha=0.55, s=18, color="#457b9d")
    axes[1].axhline(0, color="black", lw=1.2, ls="--")
    axes[1].axvspan(2019.6, 2021.4, color="#f7b731", alpha=0.2, label="COVID-19 Shock Window")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Model Residual (Two-Way FE + Cargo)")
    axes[1].set_title("2. M5 Residuals Over Time (Time FE & Cargo Controlled)", fontweight="bold")
    axes[1].grid(True, ls=":", alpha=0.5)
    axes[1].legend(fontsize=8)

    stats.probplot(res_m5.resid, dist="norm", plot=axes[2])
    axes[2].set_title("3. Residual Normality Q-Q Plot (Primary M5)", fontweight="bold")
    axes[2].grid(True, ls=":", alpha=0.5)

    plt.tight_layout()
    out_fig = FIG_DIR / "fig3_h1_elasticity_and_residuals.png"
    plt.savefig(out_fig, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {out_fig}")

    all_specs = pd.concat([t2, t2b], ignore_index=True)
    label_map = {
        "M1_Pooled_Full_Benchmark": "M1 Pooled (Naive)",
        "M2_CountryFE_PreCOVID": "M2 Country FE Pre",
        "M3_CountryFE_Full_Z3": "M3 Country FE Full",
        "M4_CountryFE_Full_Z3_Z1_Cargo": "M4 FE + Z1 + Cargo",
        "M5_TwoWay_FE_Primary_Cargo": "M5 Two-Way FE + Cargo*",
        "M6_First_Differences_Primary_Cargo": "M6 First-Diff + Cargo*",
        "M7_CountryFE_CountryTrends_Cargo": "M7 Country Trends",
        "M8_Regime_Slope_Interaction": "M8 Regime Slope",
        "M9_TwoWay_FE_Outlier_Trimmed": "M9 M5 Trimmed",
        "M10_Residence_Accounting_Comparison": "M10 Residence GHG",
    }
    short_names = [label_map.get(m, m) for m in all_specs["model"]]
    fig2, axes2 = plt.subplots(1, 3, figsize=(19.0, 5.5))

    y_pos = np.arange(len(all_specs))
    betas = all_specs["beta_ln_pax"].values
    err_low = betas - all_specs["ci_low_pax"].values
    err_high = all_specs["ci_high_pax"].values - betas
    axes2[0].errorbar(betas, y_pos, xerr=[err_low, err_high], fmt="o", color="#1d3557", ecolor="#457b9d", elinewidth=1.8, capsize=3)
    axes2[0].axvline(1.0, color="#e63946", ls="--", lw=1.2, label="Unit Elasticity (1.0)")
    axes2[0].set_yticks(y_pos)
    axes2[0].set_yticklabels(short_names, fontsize=8.5)
    axes2[0].invert_yaxis()
    axes2[0].set_xlabel("Passenger Elasticity (beta_ln_pax ± 95% CI)")
    axes2[0].set_title("1. Elasticity Across 10 Specifications (*Primary)", fontweight="bold", fontsize=10)
    axes2[0].grid(True, ls=":", alpha=0.5)
    axes2[0].legend(fontsize=8)

    jk_sorted = jk_df.sort_values("beta_ln_pax")
    axes2[1].barh(jk_sorted["excluded_country"], jk_sorted["beta_ln_pax"], color="#2a9d8f", edgecolor="black", height=0.72)
    axes2[1].axvline(float(t2.loc[t2["model"] == "M5_TwoWay_FE_Primary_Cargo", "beta_ln_pax"].values[0]), color="#e63946", ls="--", lw=1.5, label="Full M5 Baseline")
    axes2[1].set_xlabel("M5 Elasticity When Country Excluded")
    axes2[1].set_title("2. Leave-One-Country-Out Jackknife Audit (M5)", fontweight="bold", fontsize=10)
    axes2[1].tick_params(axis="y", labelsize=7)
    axes2[1].grid(axis="x", ls=":", alpha=0.5)
    axes2[1].legend(fontsize=8)

    m6_res, _ = fitted["M6_First_Differences_Primary_Cargo"]
    axes2[2].scatter(df_fd["d_ln_pax"], df_fd["d_ln_co2"], alpha=0.5, s=20, color="#e76f51")
    x_fd = np.linspace(df_fd["d_ln_pax"].min(), df_fd["d_ln_pax"].max(), 100)
    axes2[2].plot(
        x_fd,
        m6_res.params["Intercept"] + m6_res.params["d_ln_pax"] * x_fd,
        "k--", lw=1.8,
        label=f"FD Slope = {m6_res.params['d_ln_pax']:.2f} (DW={durbin_watson_stat(np.asarray(m6_res.resid)):.2f})",
    )
    axes2[2].axhline(0, color="gray", lw=0.8, ls=":")
    axes2[2].axvline(0, color="gray", lw=0.8, ls=":")
    axes2[2].set_xlabel("Annual Change: d_ln(Passengers)")
    axes2[2].set_ylabel("Annual Change: d_ln(Territorial CO2)")
    axes2[2].set_title("3. First-Differences Growth Elasticity (M6 + Cargo)", fontweight="bold", fontsize=10)
    axes2[2].grid(True, ls=":", alpha=0.5)
    axes2[2].legend(fontsize=8)

    plt.tight_layout()
    out_fig2 = FIG_DIR / "fig3b_h1_robustness_diagnostics.png"
    plt.savefig(out_fig2, dpi=300)
    plt.close(fig2)
    print(f"[OK] Saved {out_fig2}")


if __name__ == "__main__":
    main()