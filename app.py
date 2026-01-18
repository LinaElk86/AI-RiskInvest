import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="AI-RiskInvest", layout="wide")

# ===================== HEADER =====================
try:
    st.image("header.png", use_container_width=True)
except:
    pass

st.title(" AI-RiskInvest")
st.write("Application de prédiction boursière et gestion du risque")

# ===================== LOAD MODEL =====================
model = joblib.load("riskinvest_model.pkl")
scaler = joblib.load("scaler.pkl")

# ===================== SESSION STATE INIT =====================
if "predicted_price" not in st.session_state:
    st.session_state.predicted_price = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ===================== INPUT PRICES =====================
st.subheader("📥 Entrer les 60 derniers prix de clôture")

texte_prix = st.text_area(
    "Entrez les 60 prix (séparés par des virgules ou retour à la ligne)",
    height=200,
    placeholder="Exemple :\n1.25\n1.30\n1.28\n...\n(60 valeurs)"
)

prices = [0.0] * 60

if texte_prix:
    try:
        texte_prix = texte_prix.replace("\n", ",")
        valeurs = [float(v.strip()) for v in texte_prix.split(",") if v.strip()]

        for i in range(min(len(valeurs), 60)):
            prices[i] = valeurs[i]

        if len(valeurs) == 60:
            st.success("✅ 60 prix chargés avec succès")
        else:
            st.warning(f"⚠️ {len(valeurs)} prix saisis — 60 requis")

    except ValueError:
        st.error("❌ Veuillez entrer uniquement des nombres.")

# ===================== DISPLAY PRICES =====================
st.markdown("###  Détail des 60 prix")

idx = 0
for _ in range(6):
    cols = st.columns(10)
    for col in cols:
        col.number_input(
            f"{idx + 1}",
            value=prices[idx],
            disabled=True
        )
        idx += 1

# ===================== PREDICTION =====================
st.markdown("##   Résultat de la prédiction")

if st.button("🔮 Prédire"):
    prices_array = np.array(prices).reshape(-1, 1)
    prices_scaled = scaler.transform(prices_array)
    X_input = prices_scaled.reshape(1, -1)

    prediction = model.predict(X_input)

    st.session_state.predicted_price = scaler.inverse_transform(
        prediction.reshape(-1, 1)
    )[0][0]

    st.success("✅ Prédiction effectuée avec succès")

# ===================== RESULT DISPLAY (PERSISTENT) =====================
if st.session_state.predicted_price is not None:
    st.metric(
        "📈 Prix prédit",
        f"{st.session_state.predicted_price:.4f}"
    )

    st.subheader("📉 Évolution des prix")

    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.plot(range(1, 61), prices, label="Prix historiques", linewidth=2)
    ax.scatter(
        61,
        st.session_state.predicted_price,
        color="red",
        label="Prix prédit",
        zorder=5
    )
    ax.plot(
        [60, 61],
        [prices[-1], st.session_state.predicted_price],
        linestyle="--",
        color="red"
    )

    ax.set_xlabel("Temps")
    ax.set_ylabel("Prix")
    ax.set_title("Prédiction du prochain prix")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig, use_container_width=False)

# ===================== CHATBOT =====================
st.divider()
st.subheader("💬 Chatbot AI-RiskInvest")

st.markdown("### 💡 Questions suggérées")

c1, c2, c3 = st.columns(3)
if c1.button("👋 Hello / Who are you"):
    st.session_state.messages.append(
        {"role": "user", "content": "Hello, who are you?"}
    )
if c2.button("📊 Explique le résultat"):
    st.session_state.messages.append(
        {"role": "user", "content": "Explique le résultat"}
    )
if c3.button("⚠️ Quel est le risque ?"):
    st.session_state.messages.append(
        {"role": "user", "content": "Quel est le risque ?"}
    )

# ===================== CHAT HISTORY =====================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ===================== CHAT INPUT =====================
user_input = st.chat_input("Posez votre question (FR / EN / AR)")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    q = user_input.lower()

    if "hello" in q or "bonjour" in q or "who" in q:
        reply = (
            "👋 Je suis **AI-RiskInvest**.\n\n"
            "Je vous aide à comprendre les prédictions, "
            "les risques et l’utilisation de l’application."
        )

    elif "résultat" in q or "prediction" in q:
        if st.session_state.predicted_price:
            reply = (
                f"📊 Le prix prédit est **{st.session_state.predicted_price:.4f}**.\n\n"
                "Basé sur les 60 derniers prix.\n"
                "⚠️ Ce n’est pas une garantie."
            )
        else:
            reply = "ℹ️ Veuillez d’abord cliquer sur **Prédire**."

    elif "risque" in q:
        reply = (
            "⚠️ Le marché est imprévisible.\n"
            "Cette prédiction n’est PAS un conseil financier.\n"
            "Utilisez toujours une gestion du risque."
        )

    else:
        reply = (
            "🤖 Je n’ai pas bien compris.\n\n"
            "Essayez :\n"
            "• Explique le résultat\n"
            "• Quel est le risque ?"
        )

    st.session_state.messages.append(
        {"role": "assistant", "content": reply}
    )

    with st.chat_message("assistant"):
        st.markdown(reply)
