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


def fit_and_collect(name: str, formula: str, df: pd.DataFrame, pax_term: str = "ln_pax"):
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
    row = {
        "model": name,
        "n_obs": int(res.nobs),
        "n_countries": int(df["geo"].nunique()),
        "r2": res.rsquared,
        "adj_r2": res.rsquared_adj,
        "beta_ln_pax": res.params.get(pax_term, np.nan),
        "se_ln_pax": res.bse.get(pax_term, np.nan),
        "p_ln_pax": res.pvalues.get(pax_term, np.nan),
        "ci_low_pax": res.conf_int().loc[pax_term, 0] if pax_term in res.params else np.nan,
        "ci_high_pax": res.conf_int().loc[pax_term, 1] if pax_term in res.params else np.nan,
        "beta_z3_covid": res.params.get("z3_covid", res.params.get("d_z3_covid", np.nan)),
        "se_z3_covid": res.bse.get("z3_covid", res.bse.get("d_z3_covid", np.nan)),
        "p_z3_covid": res.pvalues.get("z3_covid", res.pvalues.get("d_z3_covid", np.nan)),
        "beta_z1_intl_share": res.params.get("share_intl_pct", np.nan),
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
    df = df[(df["ghg_ths_t"] > 0) & (df["pax_total"] > 0)].sort_values(["geo", "year"]).copy()

    df["ln_co2"] = np.log(df["ghg_ths_t"])
    df["ln_pax"] = np.log(df["pax_total"])
    df["z3_covid"] = df["year"].isin([2020, 2021]).astype(int)
    df["year_c"] = df["year"] - 2008
    df["regime"] = np.where(
        df["year"] <= 2019,
        "1_PreCOVID_2008_2019",
        np.where(df["year"].isin([2020, 2021]), "2_COVID_2020_2021", "3_Recovery_2022_2023"),
    )

    df_pre = df[df["year"] <= 2019].copy()
    df_z1 = df.dropna(subset=["share_intl_pct"]).copy()

    # First-differences dataset within each country
    df_fd = df.copy()
    df_fd["d_ln_co2"] = df_fd.groupby("geo")["ln_co2"].diff()
    df_fd["d_ln_pax"] = df_fd.groupby("geo")["ln_pax"].diff()
    df_fd["d_z3_covid"] = df_fd.groupby("geo")["z3_covid"].diff()
    df_fd = df_fd.dropna(subset=["d_ln_co2", "d_ln_pax"]).copy()

    core_models = [
        ("M1_Pooled_PreCOVID_2008_2019", "ln_co2 ~ ln_pax", df_pre, "ln_pax"),
        ("M2_FE_PreCOVID_2008_2019", "ln_co2 ~ ln_pax + C(geo)", df_pre, "ln_pax"),
        ("M3_Pooled_Full_with_Z3", "ln_co2 ~ ln_pax + z3_covid", df, "ln_pax"),
        ("M4_FE_Full_with_Z3", "ln_co2 ~ ln_pax + z3_covid + C(geo)", df, "ln_pax"),
        ("M5_FE_Full_with_Z3_and_Z1", "ln_co2 ~ ln_pax + z3_covid + share_intl_pct + C(geo)", df_z1, "ln_pax"),
    ]

    core_rows = []
    fitted = {}
    with open(TAB_DIR / "h1_full_model_summaries.txt", "w", encoding="utf-8") as f_txt:
        for name, formula, data, pterm in core_models:
            res, row = fit_and_collect(name, formula, data, pterm)
            if res is not None:
                fitted[name] = (res, data)
                core_rows.append(row)
                f_txt.write(f"\n{'='*70}\n{name}\n{'='*70}\n{res.summary().as_text()}\n")

        # Outlier audit on M4
        res_m4_raw = smf.ols("ln_co2 ~ ln_pax + z3_covid + C(geo)", data=df).fit()
        infl = res_m4_raw.get_influence()
        df["m4_resid"] = res_m4_raw.resid
        df["m4_stud_resid"] = infl.resid_studentized_internal
        df["m4_cooks_d"] = infl.cooks_distance[0]

        outliers = df[df["m4_stud_resid"].abs() > 3.5].sort_values("m4_cooks_d", ascending=False)
        print("\n[OUTLIER AUDIT] Observations with |studentized residual| > 3.5 in M4:")
        if not outliers.empty:
            print(outliers[["geo", "year", "ghg_ths_t", "pax_total", "m4_resid", "m4_stud_resid", "m4_cooks_d"]].to_string(index=False))
        df_no_out = df[df["m4_stud_resid"].abs() <= 3.5].copy()

        # Extended Robustness Models M6–M10
        robust_models = [
            ("M6_FE_Country_Specific_Trends", "ln_co2 ~ ln_pax + z3_covid + C(geo) + C(geo):year_c", df, "ln_pax"),
            ("M7_TwoWay_Country_Year_FE", "ln_co2 ~ ln_pax + C(geo) + C(year)", df, "ln_pax"),
            ("M8_First_Differences_Panel", "d_ln_co2 ~ d_ln_pax + d_z3_covid", df_fd, "d_ln_pax"),
            ("M9_Regime_Slope_Interaction", "ln_co2 ~ ln_pax * C(regime) + C(geo)", df, "ln_pax"),
            ("M10_M4_FE_Outlier_Trimmed", "ln_co2 ~ ln_pax + z3_covid + C(geo)", df_no_out, "ln_pax"),
        ]

        robust_rows = []
        for name, formula, data, pterm in robust_models:
            res, row = fit_and_collect(name, formula, data, pterm)
            if res is not None:
                fitted[name] = (res, data)
                robust_rows.append(row)
                f_txt.write(f"\n{'='*70}\n{name}\n{'='*70}\n{res.summary().as_text()}\n")

    t2 = pd.DataFrame(core_rows)
    t2.round(4).to_csv(TAB_DIR / "table2_h1_regression.csv", index=False)
    print(f"[OK] Saved {TAB_DIR / 'table2_h1_regression.csv'}")

    t2b = pd.DataFrame(robust_rows)
    t2b.round(4).to_csv(TAB_DIR / "table2b_h1_robustness_models.csv", index=False)
    print(f"[OK] Saved {TAB_DIR / 'table2b_h1_robustness_models.csv'}")
    print(t2b[["model", "n_obs", "r2", "beta_ln_pax", "p_ln_pax", "durbin_watson"]].round(4).to_string(index=False))

    # Leave-One-Country-Out Jackknife on M4
    jk_rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for c in sorted(df["geo"].unique()):
            sub_c = df[df["geo"] != c]
            r_c = smf.ols("ln_co2 ~ ln_pax + z3_covid + C(geo)", data=sub_c).fit(
                cov_type="cluster", cov_kwds={"groups": sub_c["geo"]}
            )
            jk_rows.append({
                "excluded_country": c,
                "n_obs": int(r_c.nobs),
                "beta_ln_pax": round(float(r_c.params["ln_pax"]), 4),
                "se_ln_pax": round(float(r_c.bse["ln_pax"]), 4),
                "p_ln_pax": round(float(r_c.pvalues["ln_pax"]), 6),
                "r2": round(float(r_c.rsquared), 4),
            })
    jk_df = pd.DataFrame(jk_rows)
    jk_df.to_csv(TAB_DIR / "table2c_h1_outlier_jackknife.csv", index=False)
    print(f"[OK] Saved {TAB_DIR / 'table2c_h1_outlier_jackknife.csv'} "
          f"(Jackknife beta range: {jk_df['beta_ln_pax'].min():.4f} to {jk_df['beta_ln_pax'].max():.4f})")

    # ---- Original Figure 3 --------------------------------------------------
    res_m4, df_m4 = fitted["M4_FE_Full_with_Z3"]
    m3_res, _ = fitted["M3_Pooled_Full_with_Z3"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    normal = df[df["year"] <= 2019]
    covid = df[df["z3_covid"] == 1]
    recov = df[df["year"] >= 2022]

    axes[0].scatter(normal["ln_pax"], normal["ln_co2"], alpha=0.55, s=22, color="#1d3557", label="2008–2019 (Pre-COVID)")
    axes[0].scatter(covid["ln_pax"], covid["ln_co2"], alpha=0.8, s=28, color="#e63946", marker="x", label="2020–2021 (COVID Shock)")
    axes[0].scatter(recov["ln_pax"], recov["ln_co2"], alpha=0.75, s=24, color="#2a9d8f", marker="^", label="2022–2023 (Recovery)")
    xs = np.linspace(df["ln_pax"].min(), df["ln_pax"].max(), 100)
    axes[0].plot(xs, m3_res.params["Intercept"] + m3_res.params["ln_pax"] * xs, "k--", lw=1.8, label=f"Slope = {m3_res.params['ln_pax']:.2f}")
    axes[0].set_xlabel("ln(Passengers Carried)")
    axes[0].set_ylabel("ln(Aviation GHG, Thousand Tonnes)")
    axes[0].set_title("1. Log-Log Traffic vs. CO2 Elasticity (H1)")
    axes[0].grid(True, ls=":", alpha=0.5)
    axes[0].legend(fontsize=8)

    axes[1].scatter(df_m4["year"], res_m4.resid, alpha=0.55, s=18, color="#457b9d")
    axes[1].axhline(0, color="black", lw=1.2, ls="--")
    axes[1].axvspan(2019.6, 2021.4, color="#f7b731", alpha=0.2, label="Z3 = 1 Window")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Model Residual (FE + Z3)")
    axes[1].set_title("2. Residuals Over Time (COVID Control Check)")
    axes[1].grid(True, ls=":", alpha=0.5)
    axes[1].legend(fontsize=8)

    stats.probplot(res_m4.resid, dist="norm", plot=axes[2])
    axes[2].set_title("3. Residual Normality Q-Q Plot (M4)")
    axes[2].grid(True, ls=":", alpha=0.5)

    plt.tight_layout()
    out_fig = FIG_DIR / "fig3_h1_elasticity_and_residuals.png"
    plt.savefig(out_fig, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {out_fig}")

    # ---- Figure 3b: Robustness & Jackknife Diagnostics ----------------------
    all_specs = pd.concat([t2, t2b], ignore_index=True)
    label_map = {
        "M1_Pooled_PreCOVID_2008_2019": "M1 Pooled Pre",
        "M2_FE_PreCOVID_2008_2019": "M2 FE Pre",
        "M3_Pooled_Full_with_Z3": "M3 Pooled Full",
        "M4_FE_Full_with_Z3": "M4 FE Full",
        "M5_FE_Full_with_Z3_and_Z1": "M5 FE + Z1",
        "M6_FE_Country_Specific_Trends": "M6 Country Trends",
        "M7_TwoWay_Country_Year_FE": "M7 Two-Way FE",
        "M8_First_Differences_Panel": "M8 First-Diff",
        "M9_Regime_Slope_Interaction": "M9 Regime Slope",
        "M10_M4_FE_Outlier_Trimmed": "M10 Trimmed",
    }
    short_names = [label_map.get(m, m) for m in all_specs["model"]]
    fig2, axes2 = plt.subplots(1, 3, figsize=(18.5, 5.4))

    y_pos = np.arange(len(all_specs))
    betas = all_specs["beta_ln_pax"].values
    err_low = betas - all_specs["ci_low_pax"].values
    err_high = all_specs["ci_high_pax"].values - betas
    axes2[0].errorbar(betas, y_pos, xerr=[err_low, err_high], fmt="o", color="#1d3557", ecolor="#457b9d", elinewidth=1.8, capsize=3)
    axes2[0].axvline(1.0, color="#e63946", ls="--", lw=1.2, label="Unit Elasticity (1.0)")
    axes2[0].set_yticks(y_pos)
    axes2[0].set_yticklabels(short_names, fontsize=8.5)
    axes2[0].invert_yaxis()
    axes2[0].set_xlabel("Traffic Elasticity (beta_ln_pax ± 95% CI)")
    axes2[0].set_title("1. Elasticity Stability Across 10 Specifications", fontweight="bold", fontsize=10)
    axes2[0].grid(True, ls=":", alpha=0.5)
    axes2[0].legend(fontsize=8)

    jk_sorted = jk_df.sort_values("beta_ln_pax")
    axes2[1].barh(jk_sorted["excluded_country"], jk_sorted["beta_ln_pax"], color="#2a9d8f", edgecolor="black", height=0.72)
    axes2[1].axvline(float(t2.loc[t2["model"] == "M4_FE_Full_with_Z3", "beta_ln_pax"].values[0]), color="#e63946", ls="--", lw=1.5, label="Full M4 Baseline")
    axes2[1].set_xlabel("M4 Elasticity When Country Excluded")
    axes2[1].set_title("2. Leave-One-Country-Out Jackknife Audit", fontweight="bold", fontsize=10)
    axes2[1].tick_params(axis="y", labelsize=7)
    axes2[1].grid(axis="x", ls=":", alpha=0.5)
    axes2[1].legend(fontsize=8)

    m8_res, _ = fitted["M8_First_Differences_Panel"]
    axes2[2].scatter(df_fd["d_ln_pax"], df_fd["d_ln_co2"], alpha=0.5, s=20, color="#e76f51")
    x_fd = np.linspace(df_fd["d_ln_pax"].min(), df_fd["d_ln_pax"].max(), 100)
    axes2[2].plot(x_fd, m8_res.params["Intercept"] + m8_res.params["d_ln_pax"] * x_fd, "k--", lw=1.8,
                  label=f"FD Slope = {m8_res.params['d_ln_pax']:.2f} (DW={durbin_watson_stat(np.asarray(m8_res.resid)):.2f})")
    axes2[2].axhline(0, color="gray", lw=0.8, ls=":")
    axes2[2].axvline(0, color="gray", lw=0.8, ls=":")
    axes2[2].set_xlabel("Annual Change: d_ln(Passengers)")
    axes2[2].set_ylabel("Annual Change: d_ln(Aviation GHG)")
    axes2[2].set_title("3. First-Differences Growth Elasticity (M8)", fontweight="bold", fontsize=10)
    axes2[2].grid(True, ls=":", alpha=0.5)
    axes2[2].legend(fontsize=8)

    plt.tight_layout()
    out_fig2 = FIG_DIR / "fig3b_h1_robustness_diagnostics.png"
    plt.savefig(out_fig2, dpi=300)
    plt.close(fig2)
    print(f"[OK] Saved {out_fig2}")


if __name__ == "__main__":
    main()