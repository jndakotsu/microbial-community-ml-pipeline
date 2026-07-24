"""
diversity_and_ml_analysis.py

Portfolio demo: takes a soil-amendment trial dataset (genus-level
abundance table + soil physicochemical variables + microbial biomass
nutrients + plant growth) and runs an expanded analysis:

  1. Alpha diversity (Shannon, Simpson, inverse Simpson, Chao1,
     Pielou's evenness, richness) from raw genus counts
  2. Log10-transformation of count-like variables before parametric tests
  3. One-way ANOVA + Tukey HSD on diversity across treatments
  4. Two-way ANOVA (treatment x week) on microbial biomass carbon
  5. Pearson correlation matrix across physicochemical / microbial
     biomass / diversity / growth variables, as a heatmap
  6. Beta diversity (Bray-Curtis) + PCoA ordination by treatment
  7. Mean relative abundance stacked bar chart by treatment
  8. Model comparison: Ridge, Random Forest, Gradient Boosting
     (predicting plant growth from environment + community variables)
  9. SHAP-based feature importance for the best model
 10. Residual diagnostics for the chosen model
 11. Putative functional guild inference from genus identity
     (nitrogen-fixers, phosphate-solubilizers, decomposers, nitrifiers,
     biocontrol/antifungal taxa) — a lightweight stand-in for
     PICRUSt/Tax4Fun-style functional prediction from taxonomy
 12. Metagenomics-style classification: predict treatment identity
     from the microbial community fingerprint (genus relative
     abundances) using a Random Forest classifier
 13. Metagenomics-style regression: predict individual genus relative
     abundances directly from environmental variables (multi-output
     regression) — i.e. predicting the microbes, not just growth
 14. Treatment summary table (mean +/- SE) written to RESULTS.md

Run: python3 analysis/diversity_and_ml_analysis.py
Expects: data/soil_microbiome_data.csv (see data/generate_synthetic_data.py)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    r2_score, mean_absolute_error, accuracy_score,
    confusion_matrix, classification_report,
)
from sklearn.manifold import MDS
import shap

sns.set_style("whitegrid")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "soil_microbiome_data.csv")
FIG_DIR = os.path.join(BASE_DIR, "figures")
RESULTS_PATH = os.path.join(BASE_DIR, "RESULTS.md")

df = pd.read_csv(DATA_PATH)
genus_cols = [c for c in df.columns if c.startswith("genus_")]

# ---------------------------------------------------------------------------
# 1. Alpha diversity: Shannon and Simpson indices per sample
# ---------------------------------------------------------------------------
def shannon_index(counts):
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    p = counts / counts.sum()
    return -np.sum(p * np.log(p))

def simpson_index(counts):
    counts = np.asarray(counts, dtype=float)
    p = counts / counts.sum()
    return 1 - np.sum(p**2)

def inverse_simpson_index(counts):
    counts = np.asarray(counts, dtype=float)
    p = counts / counts.sum()
    return 1 / np.sum(p**2)

def chao1_index(counts):
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    s_obs = len(counts)
    f1 = np.sum(counts == 1)
    f2 = np.sum(counts == 2)
    if f2 == 0:
        return s_obs + (f1 * (f1 - 1)) / 2
    return s_obs + (f1 ** 2) / (2 * f2)

def pielou_evenness(counts):
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    s = len(counts)
    if s <= 1:
        return np.nan
    h = shannon_index(counts)
    return h / np.log(s)

df["shannon"] = df[genus_cols].apply(shannon_index, axis=1)
df["simpson"] = df[genus_cols].apply(simpson_index, axis=1)
df["inverse_simpson"] = df[genus_cols].apply(inverse_simpson_index, axis=1)
df["chao1"] = df[genus_cols].apply(chao1_index, axis=1)
df["pielou_evenness"] = df[genus_cols].apply(pielou_evenness, axis=1)
df["genus_richness"] = (df[genus_cols] > 0).sum(axis=1)

print("=== Alpha diversity summary by treatment (mean Shannon) ===")
summary = df.groupby("treatment")["shannon"].agg(["mean", "std"]).sort_values("mean", ascending=False)
print(summary.round(3).to_string())

# ---------------------------------------------------------------------------
# 2. Log10-transform count-like variables before parametric testing
#    (mirrors log10-transforming microbial counts before DMRT/ANOVA)
# ---------------------------------------------------------------------------
df["total_genus_reads"] = df[genus_cols].sum(axis=1)
df["log_total_reads"] = np.log10(df["total_genus_reads"])

# ---------------------------------------------------------------------------
# 3. One-way ANOVA + Tukey HSD on Shannon diversity across treatments
# ---------------------------------------------------------------------------
groups = [g["shannon"].values for _, g in df.groupby("treatment")]
f_stat, p_val = stats.f_oneway(*groups)
print(f"\n=== One-way ANOVA on Shannon diversity ===\nF = {f_stat:.3f}, p = {p_val:.4f}")

tukey = pairwise_tukeyhsd(df["shannon"], df["treatment"], alpha=0.05)
print("\n=== Tukey HSD (post-hoc mean separation) ===")
print(tukey.summary())

order = summary.index.tolist()
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="treatment", y="shannon", order=order, hue="treatment", palette="viridis", legend=False)
plt.title("Shannon diversity by treatment (synthetic demo data)")
plt.xlabel("Treatment")
plt.ylabel("Shannon diversity index")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/shannon_by_treatment.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 4. Two-way ANOVA: treatment x week on microbial biomass carbon
# ---------------------------------------------------------------------------
model = ols("MBC_ugg ~ C(treatment) + C(week) + C(treatment):C(week)", data=df).fit()
anova_table = anova_lm(model, typ=2)
print("\n=== Two-way ANOVA: MBC_ugg ~ treatment * week ===")
print(anova_table.round(4).to_string())

plt.figure(figsize=(9, 5))
sns.lineplot(data=df, x="week", y="MBC_ugg", hue="treatment", palette="tab10", marker="o", errorbar="se")
plt.title("Microbial biomass carbon over time by treatment")
plt.xlabel("Week")
plt.ylabel("MBC (µg/g)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/mbc_by_treatment_week.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 5. Correlation matrix (Pearson) across key variables
# ---------------------------------------------------------------------------
corr_vars = [
    "soil_pH", "organic_carbon_pct", "available_N_mgkg", "available_P_mgkg",
    "MBC_ugg", "MBN_ugg", "MBP_ugg", "MBK_ugg",
    "urease_activity", "amylase_activity", "dehydrogenase_activity",
    "shannon", "simpson", "inverse_simpson", "chao1", "pielou_evenness", "genus_richness",
    "plant_height_cm", "shoot_dry_matter_g", "grain_yield_kg_ha",
]
corr = df[corr_vars].corr(method="pearson")

plt.figure(figsize=(11, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
            square=True, cbar_kws={"label": "Pearson r"}, annot_kws={"size": 7})
plt.title("Correlation matrix: soil, microbial biomass, diversity, growth")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/correlation_heatmap.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 6. Beta diversity (Bray-Curtis) + PCoA ordination
# ---------------------------------------------------------------------------
rel_abund = df[genus_cols].div(df[genus_cols].sum(axis=1), axis=0)
bray_curtis = squareform(pdist(rel_abund.values, metric="braycurtis"))

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress="auto", n_init=4)
coords = mds.fit_transform(bray_curtis)
df["pcoa1"], df["pcoa2"] = coords[:, 0], coords[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="pcoa1", y="pcoa2", hue="treatment", palette="tab10", s=70, edgecolor="k")
plt.title("PCoA ordination of microbial community (Bray-Curtis dissimilarity)")
plt.xlabel("PCoA 1")
plt.ylabel("PCoA 2")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/beta_diversity_pcoa.png", dpi=150)
plt.close()
print(f"\n=== Beta diversity ===\nMean Bray-Curtis dissimilarity: {bray_curtis[np.triu_indices_from(bray_curtis, k=1)].mean():.3f}")

# ---------------------------------------------------------------------------
# 7. Mean relative abundance stacked bar chart by treatment
# ---------------------------------------------------------------------------
rel_abund_named = rel_abund.copy()
rel_abund_named.columns = [c.replace("genus_", "") for c in genus_cols]
rel_abund_named["treatment"] = df["treatment"]
mean_rel_abund = rel_abund_named.groupby("treatment").mean().loc[
    [t for t in df["treatment"].unique() if t in rel_abund_named["treatment"].unique()]
]
mean_rel_abund = mean_rel_abund.loc[sorted(mean_rel_abund.index)]

plt.figure(figsize=(10, 6))
mean_rel_abund.plot(kind="bar", stacked=True, colormap="tab20", ax=plt.gca(), width=0.8)
plt.title("Mean relative genus abundance by treatment")
plt.xlabel("Treatment")
plt.ylabel("Mean relative abundance")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, title="Genus")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/relative_abundance_stacked.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 8. Model comparison: Ridge, Random Forest, Gradient Boosting
# ---------------------------------------------------------------------------
feature_cols = [
    "soil_pH", "organic_carbon_pct", "available_N_mgkg", "available_P_mgkg",
    "MBC_ugg", "MBN_ugg", "MBP_ugg", "MBK_ugg",
    "urease_activity", "amylase_activity", "dehydrogenase_activity",
    "shannon", "simpson", "inverse_simpson", "chao1", "pielou_evenness",
    "genus_richness", "week",
]
target_col = "shoot_dry_matter_g"

X = df[feature_cols]
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

models = {
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42),
}

print(f"\n=== Model comparison: predicting {target_col} ===")
results = []
fitted = {}
for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    pred = mdl.predict(X_test)
    cv = cross_val_score(mdl, X, y, cv=5, scoring="r2")
    results.append({
        "model": name,
        "test_r2": r2_score(y_test, pred),
        "test_mae": mean_absolute_error(y_test, pred),
        "cv_r2_mean": cv.mean(),
        "cv_r2_std": cv.std(),
    })
    fitted[name] = mdl

results_df = pd.DataFrame(results).set_index("model")
print(results_df.round(3).to_string())

plt.figure(figsize=(7, 5))
results_df["test_r2"].plot(kind="bar", color=["#4C72B0", "#55A868", "#C44E52"])
plt.title(f"Model comparison: test R² predicting {target_col}")
plt.ylabel("Test R²")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/model_comparison.png", dpi=150)
plt.close()

# Best model by test R^2
best_name = results_df["test_r2"].idxmax()
best_model = fitted[best_name]
print(f"\nBest model: {best_name}")

# Keep RF-specific plots for continuity with earlier version
rf = fitted["RandomForest"]
y_pred_rf = rf.predict(X_test)
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n=== Random Forest feature importance ===")
print(importances.round(3).to_string())

plt.figure(figsize=(8, 6))
importances.sort_values().plot(kind="barh", color="teal")
plt.title(f"Random forest feature importance — predicting {target_col}")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/feature_importance.png", dpi=150)
plt.close()

plt.figure(figsize=(5.5, 5.5))
plt.scatter(y_test, y_pred_rf, alpha=0.7, color="darkorange", edgecolor="k")
lims = [min(y_test.min(), y_pred_rf.min()), max(y_test.max(), y_pred_rf.max())]
plt.plot(lims, lims, "k--", linewidth=1)
plt.xlabel("Actual shoot dry matter (g)")
plt.ylabel("Predicted shoot dry matter (g)")
plt.title("Random forest: predicted vs. actual")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/predicted_vs_actual.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 9. SHAP feature importance for the best model
# ---------------------------------------------------------------------------
explainer = shap.TreeExplainer(best_model) if best_name != "Ridge" else shap.LinearExplainer(best_model, X_train)
shap_values = explainer.shap_values(X_test)

plt.figure()
shap.summary_plot(shap_values, X_test, show=False, plot_size=(8, 6))
plt.title(f"SHAP feature importance ({best_name})")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 10. Residual diagnostics for the best model
# ---------------------------------------------------------------------------
y_pred_best = best_model.predict(X_test)
residuals = y_test.values - y_pred_best

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].scatter(y_pred_best, residuals, alpha=0.7, color="steelblue", edgecolor="k")
axes[0].axhline(0, color="k", linestyle="--", linewidth=1)
axes[0].set_xlabel("Predicted shoot dry matter (g)")
axes[0].set_ylabel("Residual")
axes[0].set_title(f"Residuals vs. predicted ({best_name})")

axes[1].hist(residuals, bins=12, color="steelblue", edgecolor="k")
axes[1].set_xlabel("Residual")
axes[1].set_ylabel("Count")
axes[1].set_title("Residual distribution")

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/residual_diagnostics.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 10b. Yield prediction at harvest (week 12)
#      grain_yield_kg_ha is only recorded once per plot, at the final
#      (12th) sampling week, so this model is fit on that subset only
#      (n = 10 treatments x 6 replicates = 60 plots).
#
#      Feature set trimmed to 8 variables to reduce overfitting on this
#      small n: the standard soil-fertility trio N, P, K (available_K_mgkg)
#      plus organic carbon, and the four variables that ranked highest
#      for the shoot-dry-matter model above (MBC_ugg, urease/amylase/
#      dehydrogenase activity).
# ---------------------------------------------------------------------------
yield_target_col = "grain_yield_kg_ha"
yield_feature_cols = [
    "available_N_mgkg", "available_P_mgkg", "available_K_mgkg", "organic_carbon_pct",
    "MBC_ugg", "urease_activity", "amylase_activity", "dehydrogenase_activity",
]
HARVEST_WEEK = df["week"].max()

df_yield = df.loc[df["week"] == HARVEST_WEEK].dropna(subset=[yield_target_col])
Xy = df_yield[yield_feature_cols]
yy = df_yield[yield_target_col]
yield_treatment = df_yield["treatment"]
Xy_train, Xy_test, yy_train, yy_test = train_test_split(
    Xy, yy, test_size=0.25, random_state=42, stratify=yield_treatment
)

# Stratify CV folds by treatment (not by the continuous yield target) so
# every fold trains and tests on all 10 treatments rather than risking a
# fold that never sees a given treatment, which otherwise inflates CV
# variance for reasons unrelated to the model or feature set.
yield_cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
yield_cv_folds = list(yield_cv_splitter.split(Xy, yield_treatment))

yield_models = {
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=4, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=200, max_depth=2, random_state=42),
}

print(f"\n=== Model comparison: predicting {yield_target_col} (harvest, week {HARVEST_WEEK}) ===")
yield_results = []
yield_fitted = {}
for name, mdl in yield_models.items():
    mdl.fit(Xy_train, yy_train)
    pred = mdl.predict(Xy_test)
    cv = cross_val_score(mdl, Xy, yy, cv=yield_cv_folds, scoring="r2")
    yield_results.append({
        "model": name,
        "test_r2": r2_score(yy_test, pred),
        "test_mae": mean_absolute_error(yy_test, pred),
        "cv_r2_mean": cv.mean(),
        "cv_r2_std": cv.std(),
    })
    yield_fitted[name] = mdl

yield_results_df = pd.DataFrame(yield_results).set_index("model")
print(yield_results_df.round(3).to_string())

yield_best_name = yield_results_df["test_r2"].idxmax()
yield_best_model = yield_fitted[yield_best_name]
yy_pred_best = yield_best_model.predict(Xy_test)

plt.figure(figsize=(7, 5))
yield_results_df["test_r2"].plot(kind="bar", color=["#4C72B0", "#55A868", "#C44E52"])
plt.title(f"Model comparison: test R² predicting {yield_target_col}")
plt.ylabel("Test R²")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/yield_model_comparison.png", dpi=150)
plt.close()

plt.figure(figsize=(5.5, 5.5))
plt.scatter(yy_test, yy_pred_best, alpha=0.7, color="darkorange", edgecolor="k")
lims = [min(yy_test.min(), yy_pred_best.min()), max(yy_test.max(), yy_pred_best.max())]
plt.plot(lims, lims, "k--", linewidth=1)
plt.xlabel("Actual grain yield (kg/ha)")
plt.ylabel("Predicted grain yield (kg/ha)")
plt.title(f"{yield_best_name}: predicted vs. actual yield (n={len(yy)} plots)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/yield_predicted_vs_actual.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 11. Putative functional guild inference from genus identity
#     (lightweight stand-in for PICRUSt/Tax4Fun-style functional
#     prediction: map each genus to its best-known ecological role and
#     aggregate relative abundance by guild)
# ---------------------------------------------------------------------------
GUILD_MAP = {
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

guild_abund = pd.DataFrame(index=rel_abund.index)
for guild in sorted(set(GUILD_MAP.values())):
    guild_genera = [f"genus_{g}" for g, gd in GUILD_MAP.items() if gd == guild]
    guild_abund[guild] = rel_abund[guild_genera].sum(axis=1)

guild_abund["treatment"] = df["treatment"]
mean_guild_abund = guild_abund.groupby("treatment").mean()
mean_guild_abund = mean_guild_abund.loc[sorted(mean_guild_abund.index)]

plt.figure(figsize=(9, 6))
mean_guild_abund.plot(kind="bar", stacked=True, colormap="Set2", ax=plt.gca(), width=0.8)
plt.title("Putative functional guild composition by treatment\n(inferred from genus identity — PICRUSt-style stand-in)")
plt.xlabel("Treatment")
plt.ylabel("Mean relative abundance")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, title="Functional guild")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/functional_guilds_by_treatment.png", dpi=150)
plt.close()

print("\n=== Mean functional guild abundance by treatment ===")
print(mean_guild_abund.round(3).to_string())

# ---------------------------------------------------------------------------
# 12. Metagenomics-style classification: predict treatment identity from
#     the microbial community fingerprint (genus relative abundances)
# ---------------------------------------------------------------------------
X_micro = rel_abund.copy()
X_micro.columns = [c.replace("genus_", "") for c in genus_cols]
y_treatment = df["treatment"]

Xm_train, Xm_test, ym_train, ym_test = train_test_split(
    X_micro, y_treatment, test_size=0.25, random_state=42, stratify=y_treatment
)

clf = RandomForestClassifier(n_estimators=400, max_depth=6, random_state=42)
clf.fit(Xm_train, ym_train)
ym_pred = clf.predict(Xm_test)

acc = accuracy_score(ym_test, ym_pred)
print(f"\n=== Community-fingerprint classifier: predicting treatment from genus abundances ===")
print(f"Test accuracy: {acc:.3f}  (chance level ~ {1/len(y_treatment.unique()):.3f})")
print(classification_report(ym_test, ym_pred, zero_division=0))

labels = sorted(y_treatment.unique())
cm = confusion_matrix(ym_test, ym_pred, labels=labels)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title(f"Confusion matrix: treatment prediction from microbiome\n(accuracy = {acc:.2f})")
plt.xlabel("Predicted treatment")
plt.ylabel("True treatment")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/treatment_classification_confusion.png", dpi=150)
plt.close()

genus_importance = pd.Series(clf.feature_importances_, index=X_micro.columns).sort_values(ascending=False)
plt.figure(figsize=(7, 5))
genus_importance.sort_values().plot(kind="barh", color="indianred")
plt.title("Most discriminative genera for treatment classification")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/discriminative_genera.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 13. Metagenomics-style regression: predict individual genus relative
#     abundances directly from environmental variables (multi-output
#     regression) — predicting the microbes, not just plant growth
# ---------------------------------------------------------------------------
env_features = [
    "soil_pH", "organic_carbon_pct", "available_N_mgkg", "available_P_mgkg",
    "urease_activity", "amylase_activity", "dehydrogenase_activity", "week",
]
Xe = df[env_features]
Ye = X_micro  # genus relative abundances, columns already renamed

Xe_train, Xe_test, Ye_train, Ye_test = train_test_split(Xe, Ye, test_size=0.25, random_state=42)

multi_rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42))
multi_rf.fit(Xe_train, Ye_train)
Ye_pred = multi_rf.predict(Xe_test)

per_genus_r2 = pd.Series(
    {genus: r2_score(Ye_test[genus], Ye_pred[:, i]) for i, genus in enumerate(Ye.columns)}
).sort_values(ascending=False)

print("\n=== Predicting individual genus relative abundance from environment ===")
print(f"Mean R^2 across genera: {per_genus_r2.mean():.3f}")
print(per_genus_r2.round(3).to_string())

plt.figure(figsize=(8, 5))
per_genus_r2.sort_values().plot(kind="barh", color="mediumpurple")
plt.axvline(0, color="k", linewidth=0.8)
plt.title("Per-genus prediction accuracy from environmental variables\n(multi-output Random Forest)")
plt.xlabel("Test R²")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/genus_prediction_r2.png", dpi=150)
plt.close()

# Functional-guild-level prediction from the same environmental variables.
# Individual genera within a guild are randomly partitioned by the
# community assembly process, so genus-level prediction is noisy; guild
# (functional) level abundance is usually far more predictable — the same
# pattern seen in real amplicon/metagenomic studies, where functional
# profiling (e.g. PICRUSt) is often more robust than taxon-level modelling.
Ye_guild = guild_abund.drop(columns="treatment")
Ye_guild_train, Ye_guild_test = Ye_guild.loc[Xe_train.index], Ye_guild.loc[Xe_test.index]

multi_rf_guild = MultiOutputRegressor(RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42))
multi_rf_guild.fit(Xe_train, Ye_guild_train)
Ye_guild_pred = multi_rf_guild.predict(Xe_test)

per_guild_r2 = pd.Series(
    {g: r2_score(Ye_guild_test[g], Ye_guild_pred[:, i]) for i, g in enumerate(Ye_guild.columns)}
).sort_values(ascending=False)

print("\n=== Predicting functional guild abundance from environment ===")
print(f"Mean R^2 across guilds: {per_guild_r2.mean():.3f}")
print(per_guild_r2.round(3).to_string())

plt.figure(figsize=(7, 4))
per_guild_r2.sort_values().plot(kind="barh", color="darkseagreen")
plt.axvline(0, color="k", linewidth=0.8)
plt.title("Functional guild prediction accuracy from environment\n(genus-level R² vs. guild-level R²)")
plt.xlabel("Test R²")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/guild_prediction_r2.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 14. Treatment summary table (mean +/- SE) -> RESULTS.md
# ---------------------------------------------------------------------------
summary_vars = [
    "soil_pH", "organic_carbon_pct", "available_N_mgkg", "available_P_mgkg",
    "MBC_ugg", "MBN_ugg", "MBP_ugg", "MBK_ugg",
    "urease_activity", "amylase_activity", "dehydrogenase_activity",
    "shannon", "plant_height_cm", "shoot_dry_matter_g", "grain_yield_kg_ha",
]

def mean_se(x):
    return f"{x.mean():.2f} ± {x.sem():.2f}"

summary_table = df.groupby("treatment")[summary_vars].agg(mean_se)
summary_table = summary_table.loc[sorted(summary_table.index)]
label_map = df.drop_duplicates("treatment").set_index("treatment")["treatment_label"]
summary_table.insert(0, "treatment_label", label_map)

with open(RESULTS_PATH, "w") as f:
    f.write("# Results summary (synthetic demo data)\n\n")
    f.write("Mean ± SE by treatment.\n\n")
    f.write(summary_table.to_markdown())
    f.write("\n\n## Model comparison (predicting shoot dry matter, all sampling weeks)\n\n")
    f.write(results_df.round(3).to_markdown())
    f.write(f"\n\nBest model on held-out test set: **{best_name}**.\n")
    f.write(f"\n\n## Model comparison (predicting grain yield, harvest week {HARVEST_WEEK} only)\n\n")
    f.write(yield_results_df.round(3).to_markdown())
    f.write(f"\n\nBest model on held-out test set: **{yield_best_name}** "
            f"(n={len(yy)} plots — one yield value per plot, so this is a small-sample fit).\n")
    f.write("\n\n## Functional guild composition (mean relative abundance)\n\n")
    f.write(mean_guild_abund.round(3).to_markdown())
    f.write("\n\n## Community-fingerprint classification\n\n")
    f.write(f"Predicting treatment identity from genus relative abundances: "
            f"test accuracy = {acc:.3f} (chance level ~{1/len(y_treatment.unique()):.3f}).\n")
    f.write("\n\n## Genus-level abundance prediction from environment\n\n")
    f.write(f"Multi-output Random Forest, mean test R² across {len(per_genus_r2)} genera: "
            f"{per_genus_r2.mean():.3f}\n\n")
    f.write(per_genus_r2.round(3).to_frame("test_r2").to_markdown())
    f.write("\n\n## Functional guild-level abundance prediction from environment\n\n")
    f.write(f"Multi-output Random Forest, mean test R² across {len(per_guild_r2)} guilds: "
            f"{per_guild_r2.mean():.3f}\n\n")
    f.write(per_guild_r2.round(3).to_frame("test_r2").to_markdown())
    f.write("\n")

print(f"\nTreatment summary table + model comparison written to {RESULTS_PATH}")
print(f"Figures written to {FIG_DIR}/")
