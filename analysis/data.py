

"""data.py

Script simple et transparent pour préparer les données de la thèse :
- lecture du fichier Excel brut
- réorganisation des données au format « long » (une ligne = 1 dent, 1 patient, 1 méthode)
- ajout de variables utiles (arcade, côté, paire symétrique)

Ce fichier ne fait pour l'instant que de la MISE EN FORME des données,
aucune statistique avancée. L'idée est que tout soit lisible pour un
lecteur de médecine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Paramètres généraux
# ---------------------------------------------------------------------------

# Nom du fichier Excel contenant les données brutes.
# Par défaut, on s'attend à un fichier local privé (non versionné).
EXCEL_FILE = Path(__file__).resolve().parent / "private_data" / "thesis_source_data.xlsx"
SHEET_NAME = 0  # première feuille ; adapter si besoin


# ---------------------------------------------------------------------------
# 2. Fonctions utilitaires pour repérer les secteurs et leurs limites
# ---------------------------------------------------------------------------

def find_secteur_rows(df: pd.DataFrame) -> list[int]:
    """Retourne la liste des indices de lignes où apparaît "Secteur".

    Dans le fichier Excel brut, chaque bloc commence par une ligne
    de type "Secteur 1", "Secteur 2", etc.
    """
    mask = df[0].astype(str).str.startswith("Secteur")
    return df.index[mask].tolist()


def get_sector_bounds(df: pd.DataFrame, secteur_rows: list[int]) -> list[tuple[int, int, int, int]]:
    """Pour chaque secteur, retourne un tuple :
    (row_secteur, header_row, patient_start, end_row)

    - row_secteur : ligne où se trouve le texte "Secteur X"
    - header_row : ligne avec les noms de colonnes (Nom, Prénom, etc.)
    - patient_start : première ligne de patients
    - end_row : dernière ligne de patients pour ce secteur
    """
    bounds: list[tuple[int, int, int, int]] = []

    for i, row in enumerate(secteur_rows):
        header_row = row + 1

        if i + 1 < len(secteur_rows):
            end_row = secteur_rows[i + 1] - 1
        else:
            end_row = len(df) - 1

        patient_start = header_row + 1
        bounds.append((row, header_row, patient_start, end_row))

    return bounds


# ---------------------------------------------------------------------------
# 3. Détection des méthodes (manuel, DERO, AMSA, ajusté) et des dents
# ---------------------------------------------------------------------------

def detect_method_from_header(header_value: object) -> str | None:
    """Traduit l'intitulé de colonne en nom de méthode standardisé.

    On simplifie volontairement :
    - "dose moyenne" ou "manuel" -> "manuel"
    - "DERO" -> "dero"
    - "AMSA" -> "amsa"
    - colonnes de différence (ex: "Manuel-DERO", "M- DERO ajusté") -> ignorées (None)
    - colonnes "ajustées" (ex: "DERO-ajusté", "AMSA-ajusté") -> "ajuste"
    - le reste -> ignoré (None)
    """
    if header_value is None or (isinstance(header_value, float) and np.isnan(header_value)):
        return None

    h = str(header_value).strip().lower()

    # Colonnes de différences manuel - méthode -> on ne les garde pas ici
    # Exemples : "Manuel-DERO", "Manuel-AMSA", "M- DERO ajusté", etc.
    if h.startswith("manuel-") or h.startswith("manuel -"):
        return None
    if h.startswith("m- ") or h.startswith("m -"):
        return None
    if "manuel-dero" in h or "manuel-amsa" in h or "manuel - dero" in h or "manuel - amsa" in h:
        return None
    if "m- dero" in h or "m- amsa" in h:
        return None

    # Mesure manuelle :
    if "dose moyenne" in h or h == "manuel":
        return "manuel"

    # Méthodes automatiques :
    if h == "dero":
        return "dero"
    if h == "amsa":
        return "amsa"

    # Colonnes "ajustées" (peut être DERO-ajusté, AMSA-ajusté, etc.)
    if "ajust" in h:
        return "ajuste"

    # Par défaut : on ignore
    return None


def get_tooth_number(df: pd.DataFrame, secteur_row: int, col: int) -> int | None:
    """Retourne le numéro de dent pour une colonne donnée.

    Les numéros de dents (11, 12, 13, ...) se trouvent sur la ligne
    du secteur (row_secteur). On propage ce numéro vers la droite
    tant qu'on ne rencontre pas un autre numéro de dent.
    """
    value = df.loc[secteur_row, col]

    # Si la cellule de la ligne "Secteur" contient un numéro, on le prend
    if not pd.isna(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    # Sinon, on cherche le numéro de dent en se déplaçant vers la gauche
    c = col - 1
    while c >= 0 and pd.isna(df.loc[secteur_row, c]):
        c -= 1

    if c >= 0:
        value = df.loc[secteur_row, c]
        if not pd.isna(value):
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

    return None


# ---------------------------------------------------------------------------
# 4. Extraction d'un secteur en format « long »
# ---------------------------------------------------------------------------

def extract_sector_long(
    df: pd.DataFrame,
    row_secteur: int,
    header_row: int,
    patient_start: int,
    end_row: int,
) -> pd.DataFrame:
    """Extrait les données d'un secteur et les convertit au format "long".

    Format "long" = une ligne par (patient, dent, méthode) avec :
    - dose en Gy
    - informations patient (nom, prénom, id, âge, sexe)
    - numéro de secteur
    """
    secteur_label = str(df.loc[row_secteur, 0])
    # Exemple : "Secteur 1" -> 1
    try:
        secteur_num = int(secteur_label.split()[-1])
    except (IndexError, ValueError):
        secteur_num = np.nan

    header = df.loc[header_row]

    # Lignes de patients : on filtre sur la colonne ID (colonne 2) non vide
    patients = df.loc[patient_start:end_row].copy()
    patients = patients[patients[2].notna()]

    records: list[dict[str, object]] = []

    for col in df.columns:
        # Les 5 premières colonnes correspondent aux infos patient (Nom, Prénom, ID, Age, Sexe)
        if col <= 4:
            continue

        method = detect_method_from_header(header[col])
        if method is None:
            # On ignore les colonnes qui ne correspondent pas à une méthode
            continue

        tooth = get_tooth_number(df, row_secteur, col)
        if tooth is None:
            # On ignore les colonnes sans numéro de dent associé
            continue

        # Pour chaque patient, on récupère la valeur de dose pour cette dent et cette méthode
        for _, row in patients.iterrows():
            dose = row[col]
            if pd.isna(dose):
                continue

            record = {
                "secteur": secteur_num,
                "dent": int(tooth),
                "method": method,
                "dose_gy": float(dose),
                "nom": row[0],
                "prenom": row[1],
                "patient_id": row[2],
                "age": row[3],
                "sexe": row[4],
            }
            records.append(record)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 5. Variables supplémentaires : arcade, côté, paire symétrique
# ---------------------------------------------------------------------------

def add_arcade_and_side(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les colonnes "arcade" (sup/inf) et "cote" (droit/gauche).

    On utilise ici la codification FDI :
    - 1 = quadrant supérieur droit
    - 2 = quadrant supérieur gauche
    - 3 = quadrant inférieur gauche
    - 4 = quadrant inférieur droit
    """
    df = df.copy()
    df["dent"] = df["dent"].astype(int)

    first_digit = df["dent"] // 10

    df["arcade"] = np.where(first_digit.isin([1, 2]), "sup", "inf")
    df["cote"] = np.where(first_digit.isin([1, 4]), "droit", "gauche")

    return df


def symmetric_pair_code(tooth: int) -> str | None:
    """Retourne un identifiant de paire symétrique pour une dent.

    Exemple :
    - 11 et 21 -> "11-21"
    - 36 et 46 -> "36-46"
    """
    tooth = int(tooth)
    q = tooth // 10  # dizaine = quadrant
    u = tooth % 10   # unité = position dans le quadrant

    if q in (1, 2):
        sym_q = 3 - q  # 1 <-> 2
    elif q in (3, 4):
        sym_q = 7 - q  # 3 <-> 4
    else:
        return None

    sym = sym_q * 10 + u
    # On crée un code commun pour la paire (ex: "11-21")
    return f"{min(tooth, sym)}-{max(tooth, sym)}"


def add_symmetric_pair(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute une colonne "pair_sym" indiquant la paire de dents symétriques.

    Exemple : 11 et 21 auront tous les deux pair_sym = "11-21".
    """
    df = df.copy()
    df["pair_sym"] = df["dent"].apply(symmetric_pair_code)
    return df


# ---------------------------------------------------------------------------
# 6. Fonction principale : construire le tableau propre ("tidy")
# ---------------------------------------------------------------------------

def build_tidy_dataframe(
    excel_file: str | Path = EXCEL_FILE,
    sheet_name: int | str = SHEET_NAME,
) -> pd.DataFrame:
    """Lit le fichier Excel brut et renvoie un DataFrame au format "long".

    Étapes :
    1. Lecture brute du fichier (sans ligne d'en-tête automatique).
    2. Identification des lignes "Secteur".
    3. Pour chaque secteur : extraction des lignes patients et des mesures.
    4. Fusion de tous les secteurs.
    5. Ajout des colonnes arcade / côté / paire symétrique.
    """
    excel_path = Path(excel_file)
    if not excel_path.exists():
        raise FileNotFoundError(
            f"Fichier Excel introuvable: {excel_path}. "
            "Placez le fichier privé dans analysis/private_data/thesis_source_data.xlsx."
        )

    # 1. Lecture brute (header=None car la première ligne du fichier n'est
    #    pas la vraie ligne d'en-tête de nos données).
    df_raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)

    # 2. Lignes des secteurs
    secteur_rows = find_secteur_rows(df_raw)
    sector_bounds = get_sector_bounds(df_raw, secteur_rows)

    # 3. Extraction de chaque secteur au format long
    all_sectors: list[pd.DataFrame] = []
    for (row_secteur, header_row, patient_start, end_row) in sector_bounds:
        df_sector = extract_sector_long(
            df_raw,
            row_secteur=row_secteur,
            header_row=header_row,
            patient_start=patient_start,
            end_row=end_row,
        )
        all_sectors.append(df_sector)

    # 4. Fusion
    df_tidy = pd.concat(all_sectors, ignore_index=True)

    # 5. Ajout des variables supplémentaires
    df_tidy = add_arcade_and_side(df_tidy)
    df_tidy = add_symmetric_pair(df_tidy)

    return df_tidy


# ---------------------------------------------------------------------------
# 7. Point d'entrée du script : exemple d'utilisation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Construction du tableau "propre"
    print("Lecture du fichier Excel et préparation des données...")
    df_tidy = build_tidy_dataframe()

    print("\nAperçu des données transformées :")
    print(df_tidy.head())

    print("\nQuelques informations de base :")
    print("Nombre total de lignes (patient, dent, méthode) :", len(df_tidy))
    print("Nombre de patients uniques :", df_tidy["patient_id"].nunique())
    print("Nombre de dents différentes :", sorted(df_tidy["dent"].unique()))
    print("Méthodes présentes :", df_tidy["method"].unique())

    # On peut enregistrer ce tableau intermédiaire dans un fichier CSV
    # pour l'utiliser dans d'autres scripts (statistiques, graphiques, etc.).
    output_file = "donnees_tidy.csv"
    df_tidy.to_csv(output_file, index=False)
    print(f"\nFichier CSV créé : {output_file}")
