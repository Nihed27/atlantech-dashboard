import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import time

# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="ATLANTECH — Détection de chute",
    page_icon="🚨",
    layout="wide"
)

# ============================================================
# CHARGEMENT DU MODÈLE ET DES DONNÉES (mis en cache pour la performance)
# ============================================================
@st.cache_resource
def load_model():
    with open('model_rf_FINAL_v7.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('feature_names_FINAL.json', 'r') as f:
        feature_names = json.load(f)
    return model, feature_names

@st.cache_data
def load_sample_data():
    df = pd.read_csv('sample_stream_data_FINAL.csv')
    return df

model, feature_names = load_model()
sample_data = load_sample_data()

LABELS = {0: "Normal", 1: "Marche", 2: "Chute"}
SEUIL_CHUTE = 0.4  # seuil retenu dans le rapport final

# ============================================================
# EN-TÊTE
# ============================================================
st.title("🚨 ATLANTECH — Dashboard de détection de chute")
st.caption("PoC bracelet connecté pour agriculteurs isolés · Modèle Random Forest (21 features, seuil 0.4)")

# ============================================================
# BARRE LATÉRALE — CONTRÔLES
# ============================================================
st.sidebar.header("Contrôles de simulation")

mode = st.sidebar.radio(
    "Mode de simulation",
    ["Lecture automatique (flux temps réel)", "Navigation manuelle"]
)

vitesse = st.sidebar.slider("Vitesse de lecture (secondes entre fenêtres)", 0.1, 2.0, 0.5, 0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("À propos du modèle")
st.sidebar.write(f"**Rappel Chute (validation) :** 79,68 %")
st.sidebar.write(f"**Faux positifs/heure :** 815,82")
st.sidebar.write(f"**Seuil de décision :** {SEUIL_CHUTE}")
st.sidebar.write(f"**Nombre de features :** {len(feature_names)}")

# ============================================================
# ZONE PRINCIPALE — ÉTAT DE SESSION POUR LA NAVIGATION
# ============================================================
if 'index' not in st.session_state:
    st.session_state.index = 0
if 'historique' not in st.session_state:
    st.session_state.historique = []
if 'running' not in st.session_state:
    st.session_state.running = False

def predict_window(row):
    X = row[feature_names].values.reshape(1, -1).astype(float)
    proba = model.predict_proba(X)[0]
    chute_idx = list(model.classes_).index(2)
    if proba[chute_idx] >= SEUIL_CHUTE:
        pred = 2
    else:
        # sinon on garde argmax parmi les autres classes
        autres = [c for c in model.classes_ if c != 2]
        autres_proba = {c: proba[list(model.classes_).index(c)] for c in autres}
        pred = max(autres_proba, key=autres_proba.get)
    return pred, proba

# ============================================================
# NAVIGATION MANUELLE
# ============================================================
if mode == "Navigation manuelle":
    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
    with col_nav1:
        if st.button("⬅ Fenêtre précédente"):
            st.session_state.index = max(0, st.session_state.index - 1)
    with col_nav3:
        if st.button("Fenêtre suivante ➡"):
            st.session_state.index = min(len(sample_data) - 1, st.session_state.index + 1)
    with col_nav2:
        st.session_state.index = st.slider(
            "Position dans le flux", 0, len(sample_data) - 1, st.session_state.index
        )

# ============================================================
# LECTURE AUTOMATIQUE
# ============================================================
else:
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        if st.button("▶ Démarrer le flux"):
            st.session_state.running = True
    with col_ctrl2:
        if st.button("⏸ Arrêter"):
            st.session_state.running = False

# ============================================================
# AFFICHAGE DE LA FENÊTRE COURANTE
# ============================================================
placeholder = st.empty()

def render_frame(idx):
    row = sample_data.iloc[idx]
    pred, proba = predict_window(row)
    label_reel = LABELS.get(int(row['ActivityLabel']), "?")
    label_pred = LABELS.get(pred, "?")
    confiance = proba[list(model.classes_).index(pred)]

    with placeholder.container():
        # --- Alerte visuelle si chute détectée ---
        if pred == 2:
            st.error(f"🚨 ALERTE CHUTE DÉTECTÉE — Fenêtre {idx+1}/{len(sample_data)}")
        else:
            st.success(f"✅ Activité normale détectée — Fenêtre {idx+1}/{len(sample_data)}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Prédiction", label_pred)
        col2.metric("Label réel (référence)", label_reel)
        col3.metric("Confiance", f"{confiance*100:.1f} %")
        col4.metric("Source des données", row.get('source', '—'))

        # --- Probabilités par classe ---
        st.subheader("Probabilités par classe")
        proba_df = pd.DataFrame({
            'Classe': [LABELS[c] for c in model.classes_],
            'Probabilité': proba
        })
        st.bar_chart(proba_df.set_index('Classe'))

        # --- Détail des features de la fenêtre courante ---
        with st.expander("Voir les features détaillées de cette fenêtre"):
            st.dataframe(row[feature_names].to_frame(name="Valeur"))

        # --- Historique ---
        st.session_state.historique.append({
            'Fenêtre': idx,
            'Prédiction': label_pred,
            'Réel': label_reel,
            'Confiance': f"{confiance*100:.1f}%"
        })
        if len(st.session_state.historique) > 20:
            st.session_state.historique = st.session_state.historique[-20:]

        st.subheader("Historique des 20 dernières prédictions")
        st.dataframe(pd.DataFrame(st.session_state.historique), use_container_width=True)

# ============================================================
# BOUCLE D'EXÉCUTION
# ============================================================
if mode == "Navigation manuelle":
    render_frame(st.session_state.index)
else:
    if st.session_state.running:
        render_frame(st.session_state.index)
        st.session_state.index += 1
        if st.session_state.index >= len(sample_data):
            st.session_state.index = 0
            st.session_state.running = False
            st.warning("Fin du flux de données atteinte — relancez pour recommencer.")
        else:
            time.sleep(vitesse)
            st.rerun()
    else:
        render_frame(st.session_state.index)