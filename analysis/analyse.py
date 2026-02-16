

from __future__ import annotations

"""
analyse.py

Ce module contient les fonctions d'analyse statistique simples et
interprétables utilisées dans le mémoire.

Idée générale :
- on part du tableau "propre" construit par data.build_tidy_dataframe
- on ajoute des informations de type de dent (avant / arrière) et de "zone"
- on construit, pour chaque méthode non manuelle, des paires
  (manuel, méthode) dent par dent
- on résume, par zone, trois aspects :
  1) biais (différence méthode - manuel)
  2) proximité dent par dent (erreur absolue)
  3) corrélation entre méthode et manuel

Les fonctions sont écrites de manière transparente et commentée,
pour être facilement compréhensibles dans un contexte médical.
"""

import numpy as np
import pandas as pd

from data import build_tidy_dataframe

# ---------------------------------------------------------------------------
# 1. Paramètres globaux (modifiables facilement ou via Streamlit plus tard)
# ---------------------------------------------------------------------------

# Seuil par défaut pour une "bonne" dent en erreur absolue (Gy)
DEFAULT_ABS_THRESHOLD_GY = 0.5

# Seuil par défaut pour une "bonne" dent en erreur relative (ex: 0.10 = 10 %)
DEFAULT_REL_THRESHOLD = 0.10

# Nombre de rééchantillonnages pour les intervalles de confiance par bootstrap
DEFAULT_N_BOOTSTRAP = 2000

# Niveau de confiance (0.95 = 95 %)
DEFAULT_ALPHA = 0.05


# ---------------------------------------------------------------------------
# 2. Classification des dents : avant / arrière
# ---------------------------------------------------------------------------

def dent_type_from_fdi(tooth: int) -> str:
    """Détermine si une dent (numérotation FDI) est "avant" ou "arrière".

    Hypothèse utilisée :
    - unités 1, 2, 3 -> incisives + canines -> "avant"
    - unités 4, 5, 6, 7, 8 -> prémolaires + molaires -> "arrière"
    """
    tooth = int(tooth)
    unit = tooth % 10
    if unit in (1, 2, 3):
        return "avant"
    return "arriere"


def add_dent_type(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute une colonne 'dent_type' = "avant" / "arriere" à partir de la colonne 'dent'."""
    df = df.copy()
    df["dent_type"] = df["dent"].apply(dent_type_from_fdi)
    return df


# ---------------------------------------------------------------------------
# 3. Définition des zones : 2 zones (avant/arrière) ou 4 zones (sup/inf x avant/arrière)
# ---------------------------------------------------------------------------

def add_zone(df: pd.DataFrame, mode: str = "2zones") -> pd.DataFrame:
    """Ajoute une colonne 'zone' en fonction du mode choisi.

    - mode = "2zones" :
        zone = "avant"  si dent_type = avant
        zone = "arriere" sinon

    - mode = "4zones" :
        zone = "sup-avant", "sup-arriere", "inf-avant", "inf-arriere"
        en combinant 'arcade' (sup/inf) et 'dent_type' (avant/arriere)

    Cette fonction suppose que df contient déjà les colonnes :
    - 'arcade' (sup / inf), ajoutée dans data.py
    - 'dent_type', que l'on ajoutera juste avant avec add_dent_type()
    """
    df = df.copy()

    if "dent_type" not in df.columns:
        df = add_dent_type(df)

    if mode == "2zones":
        df["zone"] = df["dent_type"]
        return df

    if mode == "4zones":
        def _zone(row):
            return f"{row['arcade']}-{row['dent_type']}"

        df["zone"] = df.apply(_zone, axis=1)
        return df

    raise ValueError(f"Mode de zone non reconnu : {mode!r}. Utiliser '2zones' ou '4zones'.")


# ---------------------------------------------------------------------------
# 4. Construction des paires (manuel / méthode) dent par dent
# ---------------------------------------------------------------------------

def prepare_pairs(
    df_tidy: pd.DataFrame,
    zone_mode: str = "2zones",
    methods: list[str] | None = None,
) -> pd.DataFrame:
    """Construit un tableau de paires (manuel, méthode) dent par dent, par patient.

    Entrée :
    - df_tidy : DataFrame issu de build_tidy_dataframe (data.py)
    - zone_mode : "2zones" ou "4zones"
    - methods : liste des méthodes non manuelles à inclure
                (par défaut : toutes sauf 'manuel')

    Sortie :
    - DataFrame avec une ligne par paire (patient, dent, méthode), contenant
      notamment :
      - zone
      - dose_manuel
      - dose_method
      - diff = method - manuel
      - abs_error
      - rel_error
    """
    df = df_tidy.copy()
    df = add_dent_type(df)
    df = add_zone(df, mode=zone_mode)

    # Séparation des mesures manuelles et non manuelles
    df_manuel = df[df["method"] == "manuel"].copy()
    if df_manuel.empty:
        raise ValueError("Aucune mesure 'manuel' trouvée dans les données.")

    # Colonnes d'intérêt pour le manuel
    df_manuel = df_manuel[["patient_id", "dent", "zone", "dose_gy"]].rename(
        columns={"dose_gy": "dose_manuel"}
    )

    # Liste des méthodes non manuelles
    all_methods = sorted(df["method"].unique().tolist())
    non_manual_methods = [m for m in all_methods if m != "manuel"]

    if methods is None:
        methods = non_manual_methods
    else:
        # On ne garde que les méthodes qui existent effectivement
        methods = [m for m in methods if m in non_manual_methods]

    pairs_list: list[pd.DataFrame] = []

    for method in methods:
        df_m = df[df["method"] == method].copy()
        df_m = df_m[["patient_id", "dent", "zone", "dose_gy"]].rename(
            columns={"dose_gy": "dose_method"}
        )

        # Jointure interne : on ne garde que les dents où on a manuel ET méthode
        merged = pd.merge(
            df_manuel,
            df_m,
            on=["patient_id", "dent", "zone"],
            how="inner",
        )

        if merged.empty:
            continue

        merged["method"] = method
        merged["diff"] = merged["dose_method"] - merged["dose_manuel"]
        merged["abs_error"] = merged["diff"].abs()

        # Erreur relative : on évite de diviser par 0
        merged["rel_error"] = np.where(
            merged["dose_manuel"] != 0,
            merged["abs_error"] / merged["dose_manuel"].abs(),
            np.nan,
        )

        pairs_list.append(merged)

    if not pairs_list:
        raise ValueError("Aucune paire manuel/méthode trouvée. Vérifier les données.")

    df_pairs = pd.concat(pairs_list, ignore_index=True)
    return df_pairs


# ---------------------------------------------------------------------------
# 5. Outils pour intervalles de confiance (bootstrap simple + proportion)
# ---------------------------------------------------------------------------

def bootstrap_ci(
    data: np.ndarray,
    stat_func,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
    random_state: int | None = 0,
) -> tuple[float, float]:
    """Calcule un intervalle de confiance bootstrap pour une statistique.

    - data : tableau 1D de valeurs (par ex. diff, abs_error, etc.)
    - stat_func : fonction qui prend un tableau et retourne un nombre
                  (par ex. np.median, np.mean)
    - n_bootstrap : nombre de rééchantillonnages
    - alpha : 0.05 -> IC à 95 %
    - random_state : graine pour reproductibilité

    Retour :
    - (borne_inf, borne_sup)
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return (np.nan, np.nan)

    rng = np.random.default_rng(random_state)
    stats = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        stats.append(stat_func(sample))

    lower = np.percentile(stats, 100 * alpha / 2)
    upper = np.percentile(stats, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def proportion_ci_normal(
    successes: int,
    n: int,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[float, float]:
    """Intervalle de confiance approximatif (méthode normale) pour une proportion.

    - successes : nombre de cas "réussis" (par ex. dents avec abs_error <= seuil)
    - n : nombre total de cas
    - alpha : 0.05 -> IC à 95 %

    Cette approximation est simple et suffisante ici, tout en restant
    compréhensible pour un lecteur non statisticien.
    """
    if n == 0:
        return (np.nan, np.nan)

    p = successes / n
    z = 1.96  # environ pour 95 %
    se = np.sqrt(p * (1 - p) / n)
    lower = p - z * se
    upper = p + z * se
    # On tronque entre 0 et 1
    lower = max(0.0, lower)
    upper = min(1.0, upper)
    return float(lower), float(upper)


# ---------------------------------------------------------------------------
# 6. Résumés par zone : biais, erreurs, corrélation
# ---------------------------------------------------------------------------

def compute_zone_summary(
    df_pairs: pd.DataFrame,
    abs_threshold_gy: float = DEFAULT_ABS_THRESHOLD_GY,
    rel_threshold: float | None = DEFAULT_REL_THRESHOLD,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """Résume, par méthode et par zone, les principaux indicateurs :

    - n_pairs : nombre de dents utilisées
    - biais (médiane de diff = méthode - manuel) + IC bootstrap
    - biais moyen (pour info)
    - erreur absolue médiane + IC bootstrap
    - proportion de dents avec erreur absolue <= seuil + IC approximatif
    - (facultatif) proportion avec erreur relative <= seuil relatif
    - corrélation de Spearman entre méthode et manuel + IC bootstrap

    Retourne un DataFrame avec une ligne par (method, zone).
    """
    rows = []

    grouped = df_pairs.groupby(["method", "zone"], dropna=False)

    for (method, zone), group in grouped:
        n_pairs = len(group)
        if n_pairs == 0:
            continue

        diff = group["diff"].to_numpy()
        abs_error = group["abs_error"].to_numpy()
        rel_error = group["rel_error"].to_numpy()
        dose_manuel = group["dose_manuel"].to_numpy()
        dose_method = group["dose_method"].to_numpy()

        # Biais (différence méthode - manuel)
        median_diff = np.median(diff)
        mean_diff = float(np.mean(diff))
        median_diff_ci_low, median_diff_ci_high = bootstrap_ci(
            diff, np.median, n_bootstrap=n_bootstrap, alpha=alpha
        )

        # Erreur absolue
        median_abs_error = np.median(abs_error)
        median_abs_error_ci_low, median_abs_error_ci_high = bootstrap_ci(
            abs_error, np.median, n_bootstrap=n_bootstrap, alpha=alpha
        )

        # Proportion de dents "correctes" en absolu
        if abs_threshold_gy is not None:
            good_abs = np.sum(abs_error <= abs_threshold_gy)
            prop_good_abs = good_abs / n_pairs
            prop_good_abs_ci_low, prop_good_abs_ci_high = proportion_ci_normal(
                good_abs, n_pairs, alpha=alpha
            )
        else:
            prop_good_abs = np.nan
            prop_good_abs_ci_low = np.nan
            prop_good_abs_ci_high = np.nan

        # Proportion de dents "correctes" en relatif (si seuil fourni)
        if rel_threshold is not None:
            valid_rel = ~np.isnan(rel_error)
            rel_vals = rel_error[valid_rel]
            if rel_vals.size > 0:
                good_rel = np.sum(rel_vals <= rel_threshold)
                n_rel = len(rel_vals)
                prop_good_rel = good_rel / n_rel
                prop_good_rel_ci_low, prop_good_rel_ci_high = proportion_ci_normal(
                    good_rel, n_rel, alpha=alpha
                )
            else:
                prop_good_rel = np.nan
                prop_good_rel_ci_low = np.nan
                prop_good_rel_ci_high = np.nan
        else:
            prop_good_rel = np.nan
            prop_good_rel_ci_low = np.nan
            prop_good_rel_ci_high = np.nan

        # Corrélation de Spearman (via pandas, sans dépendance externe)
        if n_pairs >= 3:
            s_manuel = pd.Series(dose_manuel)
            s_method = pd.Series(dose_method)
            spearman_r = float(s_manuel.corr(s_method, method="spearman"))

            # IC bootstrap pour la corrélation
            rng = np.random.default_rng(0)
            rs = []
            for _ in range(n_bootstrap):
                idx = rng.integers(0, n_pairs, size=n_pairs)
                r = pd.Series(dose_manuel[idx]).corr(
                    pd.Series(dose_method[idx]),
                    method="spearman",
                )
                rs.append(r)
            spearman_ci_low = float(np.nanpercentile(rs, 100 * alpha / 2))
            spearman_ci_high = float(np.nanpercentile(rs, 100 * (1 - alpha / 2)))
        else:
            spearman_r = np.nan
            spearman_ci_low = np.nan
            spearman_ci_high = np.nan

        rows.append(
            {
                "method": method,
                "zone": zone,
                "n_pairs": n_pairs,
                # Biais
                "median_diff": float(median_diff),
                "median_diff_ci_low": median_diff_ci_low,
                "median_diff_ci_high": median_diff_ci_high,
                "mean_diff": mean_diff,
                # Erreur absolue
                "median_abs_error": float(median_abs_error),
                "median_abs_error_ci_low": median_abs_error_ci_low,
                "median_abs_error_ci_high": median_abs_error_ci_high,
                # Proportions "bonnes" en absolu
                "prop_good_abs": float(prop_good_abs),
                "prop_good_abs_ci_low": prop_good_abs_ci_low,
                "prop_good_abs_ci_high": prop_good_abs_ci_high,
                # Proportions "bonnes" en relatif
                "prop_good_rel": float(prop_good_rel),
                "prop_good_rel_ci_low": prop_good_rel_ci_low,
                "prop_good_rel_ci_high": prop_good_rel_ci_high,
                # Corrélation
                "spearman_r": spearman_r,
                "spearman_ci_low": spearman_ci_low,
                "spearman_ci_high": spearman_ci_high,
            }
        )

    summary = pd.DataFrame(rows)
    # On ordonne un peu les colonnes pour la lisibilité
    col_order = [
        "method",
        "zone",
        "n_pairs",
        "median_diff",
        "median_diff_ci_low",
        "median_diff_ci_high",
        "mean_diff",
        "median_abs_error",
        "median_abs_error_ci_low",
        "median_abs_error_ci_high",
        "prop_good_abs",
        "prop_good_abs_ci_low",
        "prop_good_abs_ci_high",
        "prop_good_rel",
        "prop_good_rel_ci_low",
        "prop_good_rel_ci_high",
        "spearman_r",
        "spearman_ci_low",
        "spearman_ci_high",
    ]
    summary = summary[col_order]
    return summary


# ---------------------------------------------------------------------------
# 7. Fonctions "haut niveau" pour l'analyse complète
# ---------------------------------------------------------------------------

def run_analysis(
    excel_file: str | None = None,
    sheet_name: int | str = 0,
    zone_mode: str = "2zones",
    abs_threshold_gy: float = DEFAULT_ABS_THRESHOLD_GY,
    rel_threshold: float | None = DEFAULT_REL_THRESHOLD,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lance l'analyse complète à partir du fichier Excel brut.

    Retourne :
    - df_pairs : tableau détaillé des paires (manuel, méthode) dent par dent
    - df_summary : résumé par zone et par méthode

    Cette fonction sera typiquement appelée depuis le futur fichier streamlit.py.
    """
    if excel_file is None:
        # On utilise le chemin par défaut défini dans data.py
        df_tidy = build_tidy_dataframe()
    else:
        df_tidy = build_tidy_dataframe(excel_file=excel_file, sheet_name=sheet_name)

    df_pairs = prepare_pairs(df_tidy, zone_mode=zone_mode)
    df_summary = compute_zone_summary(
        df_pairs,
        abs_threshold_gy=abs_threshold_gy,
        rel_threshold=rel_threshold,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
    )
    return df_pairs, df_summary


# ---------------------------------------------------------------------------
# 8. Exemple d'utilisation en ligne de commande (sans Streamlit)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== ANALYSE AVEC 2 ZONES (avant / arriere) ===")
    pairs_2, summary_2 = run_analysis(zone_mode="2zones")
    print("\nRésumé par méthode et par zone (2 zones) :")
    print(summary_2)
    summary_2.to_csv("resume_2zones.csv", index=False)
    pairs_2.to_csv("paires_2zones.csv", index=False)
    print("\nFichiers 'resume_2zones.csv' et 'paires_2zones.csv' créés.")