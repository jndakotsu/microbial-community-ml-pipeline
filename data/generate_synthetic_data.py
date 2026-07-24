"""
generate_synthetic_data.py

Creates a SYNTHETIC soil-microbiome dataset shaped like a typical
screenhouse amendment trial (10 treatments x 3 replicates x 7 sampling
weeks spanning 0-12 weeks), including a genus-level abundance table,
soil physicochemical variables, soil enzyme activities, a shoot dry
matter growth trajectory, and a harvest-time grain yield response.

This is fabricated data for portfolio-demonstration purposes only.
To turn this into a real project, swap this script for your own
lab export and keep the column names the same, or edit the column
names in analysis/diversity_and_ml_analysis.py to match yours.
"""

import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

TREATMENTS = {
    "A": "Unamended control",
    "B": "NPK inorganic fertilizer",
    "C": "Farmyard manure compost",
    "D": "Compost + leaf-litter blend",
    "E": "Biochar amendment",
    "F": "Vermicompost",
    "G": "Humic acid extract",
    "H": "Microbial inoculant blend 1",
    "I": "Microbial inoculant blend 2",
    "J": "Combined organic-mineral blend",
}

# Relative "quality" score per treatment used to bias simulated outcomes.
# H is deliberately the strongest, A (control) the weakest, mirroring a
# typical amendment-trial result.
QUALITY = {
    "A": 0.20, "B": 0.50, "C": 0.45, "D": 0.55, "E": 0.60,
    "F": 0.65, "G": 0.58, "H": 0.85, "I": 0.70, "J": 0.72,
}

REPLICATES = [1, 2, 3, 4, 5, 6]
WEEKS = [0, 2, 4, 6, 8, 10, 12]
HARVEST_WEEK = max(WEEKS)

GENERA = [
    "Bacillus", "Pseudomonas", "Azotobacter", "Rhizobium", "Streptomyces",
    "Arthrobacter", "Enterobacter", "Nitrosomonas", "Flavobacterium", "Serratia",
    "Micrococcus", "Paenibacillus",
]

# Putative functional guild per genus — used to tie genus-level abundance
# to environmental drivers, so community composition is not just noise
# around an overall diversity level, but shifts systematically with
# treatment/environment (as in a real amendment trial).
GUILD_OF = {
    "Bacillus": "decomposer",
    "Pseudomonas": "phosphate_solubilizer",
    "Azotobacter": "nitrogen_fixer",
    "Rhizobium": "nitrogen_fixer",
    "Streptomyces": "antifungal_biocontrol",
    "Arthrobacter": "decomposer",
    "Enterobacter": "nitrogen_fixer",
    "Nitrosomonas": "nitrifier",
    "Flavobacterium": "decomposer",
    "Serratia": "phosphate_solubilizer",
    "Micrococcus": "decomposer",
    "Paenibacillus": "nitrogen_fixer",
}

def clip(x, lo, hi):
    return max(lo, min(hi, x))

rows = []
for trt, label in TREATMENTS.items():
    q = QUALITY[trt]
    for rep in REPLICATES:
        for wk in WEEKS:
            week_factor = 1 + 0.15 * wk / HARVEST_WEEK  # microbial buildup over time
            row = {
                "treatment": trt,
                "treatment_label": label,
                "replicate": rep,
                "week": wk,
            }

            # Soil physicochemical variables
            row["soil_pH"] = round(rng.normal(6.0 + 0.3 * q, 0.15), 2)
            row["organic_carbon_pct"] = round(rng.normal(1.2 + 1.5 * q * week_factor, 0.12), 2)
            row["available_N_mgkg"] = round(rng.normal(40 + 60 * q * week_factor, 5), 1)
            row["available_P_mgkg"] = round(rng.normal(8 + 12 * q * week_factor, 1.5), 1)
            row["available_K_mgkg"] = round(rng.normal(45 + 70 * q * week_factor, 6), 1)

            # Microbial biomass nutrients
            row["MBC_ugg"] = round(rng.normal(80 + 220 * q * week_factor, 15), 1)
            row["MBN_ugg"] = round(rng.normal(12 + 28 * q * week_factor, 3), 1)
            row["MBP_ugg"] = round(rng.normal(6 + 14 * q * week_factor, 1.8), 1)
            row["MBK_ugg"] = round(rng.normal(9 + 20 * q * week_factor, 2.5), 1)

            # Soil enzyme activities
            row["urease_activity"] = round(rng.normal(10 + 25 * q * week_factor, 3), 2)
            row["amylase_activity"] = round(rng.normal(4 + 12 * q * week_factor, 1.6), 2)
            row["dehydrogenase_activity"] = round(rng.normal(3 + 9 * q * week_factor, 1.2), 2)

            # Guild-level environmental weights: each functional guild's
            # abundance is pushed up or down by the environmental variable
            # it plausibly tracks, so genus composition carries a real,
            # learnable signal rather than pure noise around a diversity level.
            n_norm = (row["available_N_mgkg"] - 70) / 40
            p_norm = (row["available_P_mgkg"] - 15) / 8
            c_norm = (row["organic_carbon_pct"] - 2.0) / 1.0
            urease_norm = (row["urease_activity"] - 22) / 12
            dehydro_norm = (row["dehydrogenase_activity"] - 7) / 4

            guild_weight = {
                "nitrogen_fixer": clip(1 + 0.8 * n_norm, 0.25, 2.5),
                "phosphate_solubilizer": clip(1 + 0.8 * p_norm, 0.25, 2.5),
                "decomposer": clip(1 + 0.8 * c_norm, 0.25, 2.5),
                # nitrifiers tend to be suppressed under high-urease,
                # high-amendment conditions relative to the unamended control
                "nitrifier": clip(1.6 - 0.7 * urease_norm, 0.25, 2.5),
                "antifungal_biocontrol": clip(1 + 0.8 * dehydro_norm, 0.25, 2.5),
            }

            # Genus-level counts (used to compute diversity indices)
            base_conc = 1 + 4 * q  # overall evenness/concentration, as before
            alpha_vec = np.array([
                base_conc * guild_weight[GUILD_OF[g]] for g in GENERA
            ])
            base_counts = rng.dirichlet(alpha=alpha_vec)
            total_reads = rng.integers(8000, 12000)
            counts = (base_counts * total_reads).round().astype(int)
            for genus, c in zip(GENERA, counts):
                row[f"genus_{genus}"] = int(c)

            # Plant growth response (what we'll try to predict later).
            # shoot_dry_matter_g is tracked at every sampling week across
            # the full 12-week trial; grain_yield_kg_ha is a one-off
            # harvest measurement taken only at the final (12th) week,
            # the way yield actually gets recorded in the field.
            row["leaf_number_6wap"] = round(rng.normal(8 + 6 * q, 0.8), 1) if wk == 6 else np.nan
            row["plant_height_cm"] = round(rng.normal(25 + 20 * q * week_factor, 3), 1)
            row["shoot_dry_matter_g"] = round(rng.normal(3 + 5 * q * week_factor, 0.6), 2)
            row["grain_yield_kg_ha"] = (
                round(rng.normal(1800 + 3200 * q, 250), 1) if wk == HARVEST_WEEK else np.nan
            )

            rows.append(row)

df = pd.DataFrame(rows)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
df.to_csv(os.path.join(OUT_DIR, "soil_microbiome_data.csv"), index=False)
print(f"Wrote {len(df)} rows, {df.shape[1]} columns")
print(df.head(3).to_string())
