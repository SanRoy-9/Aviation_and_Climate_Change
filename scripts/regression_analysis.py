import sys
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


def fit_and_collect(name: str, formula: str, df: pd.DataFrame):
    res = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["geo"]})
    resid = res.resid
    _, jb_p = stats.jarque_bera(resid)
    _, sw_p = stats.shapiro(resid) if len(resid) <= 5000 else (np.nan, np.nan)
    row = {
        "model": name,
        "n_obs": int(res.nobs),
        "n_countries": int(df["geo"].nunique()),
        "r2": res.rsquared,
        "adj_r2": res.rsquared_adj,
        "beta_ln_pax": res.params.get("ln_pax", np.nan),
        "se_ln_pax": res.bse.get("ln_pax", np.nan),
        "p_ln_pax": res.pvalues.get("ln_pax", np.nan),
        "beta_z3_covid": res.params.get("z3_covid", np.nan),
        "se_z3_covid": res.bse.get("z3_covid", np.nan),
        "p_z3_covid": res.pvalues.get("z3_covid", np.nan),
        "beta_z1_intl_share": res.params.get("share_intl_pct", np.nan),
        "p_z1_intl_share": res.pvalues.get("share_intl_pct", np.nan),
        "jarque_bera_p": jb_p,
        "shapiro_wilk_p": sw_p,
    }
    return res, row


def main():
    if not PANEL_FILE.exists():
        print("[ERROR] Merged panel missing. Run 03_coverage_and_descriptive.py first.")
        sys.exit(1)

    df = pd.read_csv(PANEL_FILE)
    df = df[~df["geo"].isin(AGGREGATE_CODES)].copy()
    df = df[(df["ghg_ths_t"] > 0) & (df["pax_total"] > 0)].copy()

    df["ln_co2"] = np.log(df["ghg_ths_t"])
    df["ln_pax"] = np.log(df["pax_total"])
    df["z3_covid"] = df["year"].isin([2020, 2021]).astype(int)

    df_pre = df[df["year"] <= 2019].copy()
    df_z1 = df.dropna(subset=["share_intl_pct"]).copy()

    models = [
        ("M1_Pooled_PreCOVID_2008_2019", "ln_co2 ~ ln_pax", df_pre),
        ("M2_FE_PreCOVID_2008_2019", "ln_co2 ~ ln_pax + C(geo)", df_pre),
        ("M3_Pooled_Full_with_Z3", "ln_co2 ~ ln_pax + z3_covid", df),
        ("M4_FE_Full_with_Z3", "ln_co2 ~ ln_pax + z3_covid + C(geo)", df),
        ("M5_FE_Full_with_Z3_and_Z1", "ln_co2 ~ ln_pax + z3_covid + share_intl_pct + C(geo)", df_z1),
    ]

    rows = []
    fitted = {}
    with open(TAB_DIR / "h1_full_model_summaries.txt", "w", encoding="utf-8") as f_txt:
        for name, formula, data in models:
            res, row = fit_and_collect(name, formula, data)
            fitted[name] = (res, data)
            rows.append(row)
            f_txt.write(f"\n{'='*70}\n{name}\n{'='*70}\n{res.summary().as_text()}\n")

    t2 = pd.DataFrame(rows)
    t2.round(4).to_csv(TAB_DIR / "table2_h1_regression.csv", index=False)
    print(f"[OK] Saved {TAB_DIR / 'table2_h1_regression.csv'}")
    print(t2[["model", "n_obs", "r2", "beta_ln_pax", "p_ln_pax", "beta_z3_covid"]].round(4).to_string(index=False))

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


if __name__ == "__main__":
    main()