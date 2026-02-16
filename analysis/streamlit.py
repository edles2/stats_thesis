

from __future__ import annotations

"""
streamlit.py

Application Streamlit pour explorer les résultats de la thèse :
- compare les méthodes non manuelles à la méthode manuelle
- permet de choisir le regroupement des zones (2 ou 4)
- permet d'ajuster les paramètres de fiabilité (seuils, bootstrap…)
- affiche tableaux et graphiques interactifs

L'idée est qu'on puisse envoyer ce fichier + les scripts data.py / analyse.py
+ le fichier Excel à quelqu'un, et qu'il puisse tout explorer sans toucher au code.
"""

import textwrap

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from analyse import (
    run_analysis,
    DEFAULT_ABS_THRESHOLD_GY,
    DEFAULT_REL_THRESHOLD,
    DEFAULT_N_BOOTSTRAP,
    DEFAULT_ALPHA,
)


# ---------------------------------------------------------------------------
# Configuration de base de la page
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Analyse des méthodes de dose dentaire",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Fonctions utilitaires pour l'affichage
# ---------------------------------------------------------------------------

def format_percent(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return "NA"
    return f"{x * 100:.1f} %"


def format_float(x: float | int | None, digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "NA"
    return f"{x:.{digits}f}"


def apply_plot_theme(fig, ax, theme_mode: str, primary_color: str):
    """
    Applique un thème simple clair/sombre aux graphiques matplotlib.

    theme_mode : "auto", "clair", "sombre"
    """
    if theme_mode == "sombre":
        bg = "#111111"
        fg = "#f0f0f0"
    elif theme_mode == "clair":
        bg = "#ffffff"
        fg = "#000000"
    else:
        # auto : on reste neutre
        bg = fig.get_facecolor()
        fg = "#000000"

    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    # Couleurs des axes / textes
    ax.tick_params(colors=fg)
    for spine in ax.spines.values():
        spine.set_color(fg)
    ax.xaxis.label.set_color(fg)
    ax.yaxis.label.set_color(fg)
    ax.title.set_color(fg)

    # Petites lignes de grille discrètes
    ax.grid(True, alpha=0.2, color=fg)

    # Retourner les couleurs si besoin ailleurs
    return bg, fg


# ---------------------------------------------------------------------------
# Cache de l'analyse pour éviter de recalculer à chaque interaction
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=True)
def cached_run_analysis(
    excel_bytes,
    zone_mode: str,
    abs_threshold_gy: float,
    rel_threshold: float | None,
    n_bootstrap: int,
    alpha: float,
):
    """
    Wrapper autour de run_analysis pour qu'il soit mis en cache par Streamlit.

    - excel_bytes : None -> utiliser le fichier Excel par défaut (chemin de data.py)
                    sinon : bytes d'un fichier uploadé (streamlit file_uploader)
    """
    if excel_bytes is not None:
        # Streamlit fournit un UploadedFile, que pandas/Excel sait lire directement
        df_pairs, df_summary = run_analysis(
            excel_file=excel_bytes,
            sheet_name=0,
            zone_mode=zone_mode,
            abs_threshold_gy=abs_threshold_gy,
            rel_threshold=rel_threshold,
            n_bootstrap=n_bootstrap,
            alpha=alpha,
        )
    else:
        # On laisse run_analysis utiliser le chemin par défaut
        df_pairs, df_summary = run_analysis(
            excel_file=None,
            sheet_name=0,
            zone_mode=zone_mode,
            abs_threshold_gy=abs_threshold_gy,
            rel_threshold=rel_threshold,
            n_bootstrap=n_bootstrap,
            alpha=alpha,
        )
    return df_pairs, df_summary


# ---------------------------------------------------------------------------
# Application principale
# ---------------------------------------------------------------------------

def main():
    st.title("Évaluation des méthodes de mesure de dose dentaire")
    st.write(
        textwrap.dedent(
            """
            Cette interface permet de comparer différentes méthodes de mesure de dose
            (DERO, AMSA, ajustée, etc.) à la référence **manuelle**, dent par dent.

            Vous pouvez :
            - choisir le regroupement des zones (2 zones : avant / arrière, ou 4 zones : sup/inf × avant/arrière),
            - ajuster les seuils de fiabilité (erreur absolue, erreur relative),
            - explorer les résultats par méthode, par zone et dent par dent,
            - visualiser des graphiques (nuage de points, Bland–Altman).
            """
        )
    )

    # -----------------------------
    # SIDEBAR : paramètres globaux
    # -----------------------------

    st.sidebar.header("Paramètres d'analyse")

    # Fichier Excel (optionnel : si non fourni, on utilise le fichier par défaut)
    st.sidebar.subheader("Données")
    uploaded_file = st.sidebar.file_uploader(
        "Fichier Excel des données brutes",
        type=["xlsx"],
        help=(
            "Si aucun fichier n'est fourni, l'application utilise le chemin par défaut "
            "défini dans data.py."
        ),
    )

    # Choix regroupement zones
    st.sidebar.subheader("Zones")
    zone_label = st.sidebar.radio(
        "Regroupement de zones",
        ["2 zones (avant / arrière)", "4 zones (sup/inf × avant/arrière)"],
        index=0,
    )
    zone_mode = "2zones" if "2 zones" in zone_label else "4zones"

    # Paramètres de fiabilité
    st.sidebar.subheader("Seuils de fiabilité")
    abs_threshold = st.sidebar.number_input(
        "Seuil d'erreur absolue (Gy)",
        min_value=0.0,
        max_value=5.0,
        value=float(DEFAULT_ABS_THRESHOLD_GY),
        step=0.1,
    )
    rel_threshold_percent = st.sidebar.number_input(
        "Seuil d'erreur relative (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(DEFAULT_REL_THRESHOLD * 100),
        step=1.0,
    )
    rel_threshold = rel_threshold_percent / 100.0 if rel_threshold_percent > 0 else None

    # Options avancées : bootstrap, alpha
    with st.sidebar.expander("Options statistiques avancées"):
        n_bootstrap = st.number_input(
            "Nombre de rééchantillonnages bootstrap",
            min_value=500,
            max_value=10000,
            value=int(DEFAULT_N_BOOTSTRAP),
            step=500,
            help="Augmenter ce nombre rend les intervalles de confiance plus stables mais plus longs à calculer.",
        )
        alpha = st.selectbox(
            "Niveau alpha (1 - IC)",
            options=[0.10, 0.05, 0.01],
            index=1,  # 0.05
            format_func=lambda x: f"{int((1-x)*100)} % d'intervalle de confiance",
        )

    # Paramètres d'affichage (thème & couleurs)
    st.sidebar.subheader("Affichage des graphiques")
    theme_choice = st.sidebar.radio(
        "Thème des graphiques",
        ["Auto", "Clair", "Sombre"],
        index=0,
        help="Contrôle seulement l'apparence des graphiques matplotlib (pas le thème global Streamlit).",
    )
    theme_mode = theme_choice.lower()  # "auto", "clair", "sombre"

    primary_color = st.sidebar.color_picker(
        "Couleur principale des graphiques",
        value="#1f77b4",
    )

    outlier_color = st.sidebar.color_picker(
        "Couleur des points hors tolérance",
        value="#ff7f0e",  # orange standard
    )

    point_size = st.sidebar.slider(
        "Taille des points",
        min_value=10,
        max_value=80,
        value=30,
    )

    # -----------------------------------
    # Lancer l'analyse (avec cache)
    # -----------------------------------

    with st.spinner("Analyse en cours..."):
        df_pairs, df_summary = cached_run_analysis(
            uploaded_file,
            zone_mode=zone_mode,
            abs_threshold_gy=abs_threshold,
            rel_threshold=rel_threshold,
            n_bootstrap=int(n_bootstrap),
            alpha=float(alpha),
        )

    if df_summary.empty:
        st.error("Aucune donnée d'analyse disponible. Vérifier le fichier de données.")
        return

    # Méthodes disponibles
    methods_available = sorted(df_summary["method"].unique().tolist())

    # ----------------------
    # TABS / ONGLET PRINCIPAUX
    # ----------------------

    tab_resume, tab_zones, tab_dent, tab_graphs = st.tabs(
        ["Résumé global", "Par zones", "Détail dent par dent", "Graphiques"]
    )

    # --------------------------------------------------------------------
    # Onglet 1 : Résumé global
    # --------------------------------------------------------------------
    with tab_resume:
        st.subheader("Résumé global par méthode et par zone")

        # Formatage pour l'affichage
        df_display = df_summary.copy()
        df_display["median_diff"] = df_display["median_diff"].apply(lambda x: format_float(x, 2))
        df_display["median_abs_error"] = df_display["median_abs_error"].apply(
            lambda x: format_float(x, 2)
        )
        df_display["prop_good_abs"] = df_display["prop_good_abs"].apply(format_percent)
        df_display["prop_good_rel"] = df_display["prop_good_rel"].apply(format_percent)
        df_display["spearman_r"] = df_display["spearman_r"].apply(lambda x: format_float(x, 2))

        st.dataframe(df_display, use_container_width=True)

        st.markdown(
            """
            **Rappel des indicateurs :**
            - `median_diff` : biais médian (méthode - manuel) en Gy.
            - `median_abs_error` : erreur absolue médiane |méthode - manuel| en Gy.
            - `prop_good_abs` : % de dents avec erreur absolue ≤ seuil choisi.
            - `prop_good_rel` : % de dents avec erreur relative ≤ seuil choisi.
            - `spearman_r` : corrélation de Spearman entre méthode et manuel.
            """
        )

    # --------------------------------------------------------------------
    # Onglet 2 : Analyse par zones
    # --------------------------------------------------------------------
    with tab_zones:
        st.subheader("Analyse par zones")

        col1, col2 = st.columns(2)
        with col1:
            method_selected = st.selectbox(
                "Méthode à afficher",
                methods_available,
                index=0,
                key="method_zones",
            )

        df_m = df_summary[df_summary["method"] == method_selected].copy()
        if df_m.empty:
            st.warning("Aucune donnée pour cette méthode.")
        else:
            st.markdown(f"### Méthode : `{method_selected}`")

            # Tableau de synthèse par zone pour cette méthode
            df_m_display = df_m.copy()
            df_m_display["median_diff"] = df_m_display["median_diff"].apply(
                lambda x: format_float(x, 2)
            )
            df_m_display["median_abs_error"] = df_m_display["median_abs_error"].apply(
                lambda x: format_float(x, 2)
            )
            df_m_display["prop_good_abs"] = df_m_display["prop_good_abs"].apply(
                format_percent
            )
            df_m_display["prop_good_rel"] = df_m_display["prop_good_rel"].apply(
                format_percent
            )
            df_m_display["spearman_r"] = df_m_display["spearman_r"].apply(
                lambda x: format_float(x, 2)
            )

            st.dataframe(df_m_display, use_container_width=True)

            # Graphique simple : proportion de dents "bonnes" par zone
            st.markdown("#### Proportion de dents dans la tolérance (erreur absolue)")
            fig, ax = plt.subplots()
            zones = df_m["zone"].tolist()
            props = df_m["prop_good_abs"].to_numpy()
            ax.bar(zones, props, color=primary_color)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Proportion de dents correctes")
            ax.set_xlabel("Zone")
            ax.set_title(f"% de dents avec erreur ≤ {abs_threshold} Gy")
            apply_plot_theme(fig, ax, theme_mode=theme_mode, primary_color=primary_color)
            st.pyplot(fig, use_container_width=True)

    # --------------------------------------------------------------------
    # Onglet 3 : Détail dent par dent
    # --------------------------------------------------------------------
    with tab_dent:
        st.subheader("Détail dent par dent")

        col1, col2, col3 = st.columns(3)
        with col1:
            method_dent = st.selectbox(
                "Méthode",
                methods_available,
                index=0,
                key="method_dent",
            )
        with col2:
            zones_available = sorted(df_pairs["zone"].dropna().unique().tolist())
            zone_dent = st.selectbox(
                "Zone",
                zones_available,
                index=0,
            )
        with col3:
            max_error_display = st.number_input(
                "Filtrer sur erreur absolue ≤ (Gy) (0 = pas de filtre)",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.1,
            )

        df_d = df_pairs[
            (df_pairs["method"] == method_dent) & (df_pairs["zone"] == zone_dent)
        ].copy()

        if max_error_display > 0:
            df_d = df_d[df_d["abs_error"] <= max_error_display]

        if df_d.empty:
            st.warning("Aucune dent correspondant aux filtres sélectionnés.")
        else:
            st.markdown(
                f"### Dents pour la méthode `{method_dent}` en zone `{zone_dent}`  "
                f"(n = {len(df_d)})"
            )
            # Quelques colonnes utiles seulement
            cols_show = [
                "patient_id",
                "dent",
                "dose_manuel",
                "dose_method",
                "diff",
                "abs_error",
                "rel_error",
            ]
            for c in cols_show:
                if c not in df_d.columns:
                    df_d[c] = np.nan
            df_show = df_d[cols_show].copy()
            st.dataframe(df_show, use_container_width=True)

            csv = df_show.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Télécharger ce tableau (CSV)",
                data=csv,
                file_name=f"dents_{method_dent}_{zone_dent}.csv",
                mime="text/csv",
            )

    # --------------------------------------------------------------------
    # Onglet 4 : Graphiques comparatifs
    # --------------------------------------------------------------------
    with tab_graphs:
        st.subheader("Graphiques comparatifs (méthode vs manuel)")

        col1, col2 = st.columns(2)
        with col1:
            method_graph = st.selectbox(
                "Méthode",
                methods_available,
                index=0,
                key="method_graph",
            )
        with col2:
            zones_graph_available = sorted(df_pairs["zone"].dropna().unique().tolist())
            zone_graph = st.selectbox(
                "Zone",
                zones_graph_available,
                index=0,
                key="zone_graph",
            )

        df_g = df_pairs[
            (df_pairs["method"] == method_graph) & (df_pairs["zone"] == zone_graph)
        ].copy()

        if len(df_g) < 3:
            st.warning("Trop peu de dents dans cette combinaison méthode/zone pour tracer des graphiques.")
        else:
            st.markdown(
                f"### Méthode `{method_graph}` – Zone `{zone_graph}`  "
                f"(n = {len(df_g)} dents)"
            )
            
            # 1) Nuage de points Manuel vs Méthode
            st.markdown("#### Nuage de points : Manuel vs Méthode")
            fig1, ax1 = plt.subplots()
            
            x = df_g["dose_manuel"].to_numpy()
            y = df_g["dose_method"].to_numpy()
            
            diff_vals = y - x
            T = float(abs_threshold)
            ok = np.abs(diff_vals) <= T
            
            min_val = float(min(np.min(x), np.min(y)))
            max_val = float(max(np.max(x), np.max(y)))
            
            # Debug visuel : si tu ne vois pas cette ligne, ce bloc ne s'exécute pas
            st.caption(f"[Debug scatter] T={T:.2f} Gy — hors tolérance: {(~ok).sum()} / {len(ok)}")
            
            # Bande de tolérance (derrière)
            if T > 0:
                xs = np.linspace(min_val, max_val, 400)
                ax1.fill_between(xs, xs - T, xs + T, color=outlier_color, alpha=0.22, linewidth=0, zorder=0)
                # Limites de tolérance (au-dessus, épaisses -> visibles)
                #ax1.plot([min_val, max_val], [min_val + T, max_val + T], ":", color=outlier_color, linewidth=3.0, alpha=0.95, zorder=10)
                #ax1.plot([min_val, max_val], [min_val - T, max_val - T], ":", color=outlier_color, linewidth=3.0, alpha=0.95, zorder=10)
            
            # Diagonale y=x
            ax1.plot([min_val, max_val], [min_val, max_val], "--", color="gray", alpha=0.7, zorder=1)
            
            # Points OK / hors tolérance
            ax1.scatter(x[ok], y[ok], s=point_size, alpha=0.85, color=primary_color, label=f"|diff| ≤ {T:.2f} Gy", zorder=3)
            ax1.scatter(x[~ok], y[~ok], s=point_size, alpha=0.85, color=outlier_color, label=f"|diff| > {T:.2f} Gy", zorder=4)
            
            ax1.set_xlabel("Dose manuelle (Gy)")
            ax1.set_ylabel("Dose méthode (Gy)")
            ax1.set_title("Comparaison Manuel vs Méthode")
            ax1.legend()
            
            apply_plot_theme(fig1, ax1, theme_mode=theme_mode, primary_color=primary_color)
            st.pyplot(fig1, use_container_width=True)

            # 2) Bland–Altman
            st.markdown("#### Bland–Altman : différence vs moyenne")
            fig2, ax2 = plt.subplots()
            mean_vals = 0.5 * (x + y)
            diff_vals = y - x
            bias = np.mean(diff_vals)
            sd = np.std(diff_vals, ddof=1)
            # Coloration selon la tolérance (seuil d'erreur absolue ±T Gy)
            T = abs_threshold
            ok = np.abs(diff_vals) <= T
            # Points dans la tolérance
            ax2.scatter(
                mean_vals[ok],
                diff_vals[ok],
                s=point_size,
                alpha=0.85,
                color=primary_color,
                label=f"|diff| ≤ {T:.2f} Gy",
            )
            # Points hors tolérance
            ax2.scatter(
                mean_vals[~ok],
                diff_vals[~ok],
                s=point_size,
                alpha=0.85,
                color="orange",   # tu peux mettre une color_picker si tu veux
                label=f"|diff| > {T:.2f} Gy",
            )
            ax2.axhline(bias, color="gray", linestyle="-", label=f"Mean bias = {bias:.2f} Gy")
            ax2.axhline(bias + 1.96 * sd, color="gray", linestyle="--", alpha=0.7)
            ax2.axhline(bias - 1.96 * sd, color="gray", linestyle="--", alpha=0.7)
            ax2.set_ylim(-10, 25)
            ax2.set_xlabel("Moyenne (Manuel, Méthode) [Gy]")
            ax2.set_ylabel("Méthode - Manuel (Gy)")
            ax2.set_title("Diagramme de Bland–Altman")
            ax2.legend()
            apply_plot_theme(fig2, ax2, theme_mode=theme_mode, primary_color=primary_color)
            st.pyplot(fig2, use_container_width=True)


if __name__ == "__main__":
    main()
