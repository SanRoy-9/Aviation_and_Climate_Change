# Aviation and Climate Change: Greenhouse Gas Emissions, Contrails, and Aircraft Fuel

---

## 1. Project Overview

Commercial aviation drives global mobility and trade, but it exerts complex radiative forcing pressures on the global climate system. Beyond direct carbon dioxide (CO₂) emitted from jet fuel combustion, aircraft operations in the cold, ice-supersaturated upper troposphere trigger persistent condensation trails (contrails) and contrail-induced cirrus clouds. These artificial clouds trap outgoing longwave radiation while partially reflecting incoming solar radiation, yielding a net positive warming effect. 

Although modern airframe and engine designs continuously improve fuel efficiency per passenger-kilometre, global traffic demand growth risks offsetting these technological gains. This research project investigates the empirical relationship between air traffic volume, direct combustion emissions, atmospheric conditions governing contrail persistence, and fleet fuel efficiency across European airspace.

---

## 2. Research Question & Hypotheses

### Primary Research Question
> *How does commercial aviation activity contribute to climate change through direct CO₂ emissions and persistent contrails, and to what extent can improvements in aircraft fuel efficiency reduce emissions per passenger-kilometre despite increasing traffic demand?*

### Refined Hypotheses
- **H1 (Traffic Activity vs. CO₂ Emissions):** Aviation-related CO₂ emissions increase monotonically with commercial air traffic activity.
- **H2 (Contrail Formation & Radiative Forcing):** Persistent aircraft contrails contribute to atmospheric warming by trapping outgoing longwave radiation while partially reflecting incoming solar radiation, resulting in net positive radiative forcing.
- **H3 (Fuel Efficiency vs. Demand Growth):** Technological and operational improvements reduce CO₂ emissions per passenger-kilometre, but absolute sector emissions continue to rise if passenger demand growth outpaces efficiency improvements.

---

## 3. Variable Operationalization

| Role | Code | Variable Name | Operational Definition | Metric / Unit | Expected Direction | Hypothesis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Response** | **Y1** | Aviation CO₂ Emissions | Annual mass of carbon dioxide emitted from jet fuel combustion | Mt CO₂ / year | Positive (+) | H1 |
| **Predictor** | **X1** | Air Traffic Activity | Commercial IFR flight movements & Revenue Passenger-Kilometres | Flights / year & RPK | Positive (+) | H1 |
| **Response** | **Y2** | Contrail Radiative Forcing | Net perturbation of Earth's radiation balance by contrail-cirrus (LW minus SW) | mW / m² | Positive (+) | H2 |
| **Predictor** | **X2** | Contrail Permissibility Index | Upper tropospheric conditions meeting Schmidt-Appleman criterion (T ≤ -40°C & RHi ≥ 100% at 200–300 hPa) | % cruise time / grid frequency | Positive (+) | H2 |
| **Response** | **Y3** | Specific Emission Intensity | Average greenhouse gas emissions generated per unit of passenger transport work | g CO₂ / pkm | Negative (–) | H3 |
| **Predictor** | **X3** | Fleet Fuel Efficiency | Fleet-averaged Specific Fuel Consumption (SFC) and passenger load factors | kg fuel / seat-km & % load | Negative (–) | H3 |
| **Control** | **Z1** | Route Network Distribution | Proportion of short-haul (<500 km), medium-haul (500–1500 km), and long-haul (>1500 km) flights | % of total departures | Confounder | H1, H3 |
| **Control** | **Z2** | Background Cloudiness | Pre-existing natural cirrus optical depth and Top-of-Atmosphere (TOA) background albedo | Dimensionless optical depth (τ) | Confounder | H2 |
| **Control** | **Z3** | Exogenous Macro Shocks | COVID-19 pandemic restrictions (2020) and global jet fuel price shocks | Binary dummy / Brent USD per barrel | Confounder | H1, H3 |

---

## 4. Repository Structure

```text
├── data/
│   ├── README.md               # Data inventory, sources, versioning, licences, and citations
│   └── raw/                    # Raw, unedited data downloaded by automated scripts
│       ├── eurostat_aviation_ghg_emissions.csv
│       └── eurostat_aviation_traffic_pax.csv
│
├── scripts/
│   ├── 01_get_data.py          # Script downloading primary candidate datasets into data/raw/
│   └── plot_quick_map.py       # Script generating coverage verification plots
│
├── reports/
│   └── figures/                # Output visualizations and verification plots
│       └── coverage_verification_map.png
│
├── requirements.txt            # Python dependencies (pandas, matplotlib)
└── README.md                   # Root repository overview (this file)
