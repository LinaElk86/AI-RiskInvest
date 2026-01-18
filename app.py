import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="AI-RiskInvest",
    layout="wide"
)

# ===================== LOAD MODEL =====================
model = joblib.load("riskinvest_model.pkl")
scaler = joblib.load("scaler.pkl")

# ===================== HEADER =====================
st.image("header.png", use_column_width=True)
st.markdown("<br>", unsafe_allow_html=True)

st.title("📈 AI-RiskInvest")
st.write("Application de prédiction boursière et gestion du risque")

# ===================== INPUT PRICES =====================
st.subheader("📥 Entrer les 60 derniers prix de clôture")

texte_prix = st.text_area(
    "Entrez les 60 prix (séparés par des virgules ou retour à la ligne)",
    height=180,
    placeholder="Exemple :\n1.25\n1.30\n1.28\n...\n(60 valeurs)"
)

prices = [0.0] * 60

if texte_prix:
    try:
        texte_prix = texte_prix.replace("\n", ",")
        valeurs = [float(v.strip()) for v in texte_prix.split(",") if v.strip()]

        for i in range(min(len(valeurs), 60)):
            prices[i] = valeurs[i]

        if len(valeurs) != 60:
            st.warning(f"⚠️ Vous avez entré {len(valeurs)} prix. Il faut exactement 60.")
        else:
            st.success("✅ 60 prix chargés avec succès")

    except ValueError:
        st.error("❌ Veuillez entrer uniquement des nombres.")

# ===================== DISPLAY 60 PRICES =====================
st.markdown("### 📋 Détail des 60 prix")

index = 0
for _ in range(6):
    cols = st.columns(10)
    for col in cols:
        col.number_input(
            f"{index + 1}",
            value=prices[index],
            disabled=True
        )
        index += 1

# ===================== PREDICTION =====================
st.markdown("## 📊 Résultat de la prédiction")

predicted_price = None

if st.button("🔮 Prédire"):
    prices_array = np.array(prices).reshape(-1, 1)
    prices_scaled = scaler.transform(prices_array)
    X_input = prices_scaled.reshape(1, -1)

    prediction = model.predict(X_input)
    predicted_price = scaler.inverse_transform(
        prediction.reshape(-1, 1)
    )[0][0]

    st.success("✅ Prédiction effectuée avec succès")
    st.metric("📈 Prix prédit", f"{predicted_price:.4f}")

    # ===================== GRAPH =====================
    st.subheader("📉 Évolution des prix")

    fig, ax = plt.subplots(figsize=(4, 2.2))
    ax.plot(range(1, 61), prices, label="Prix historiques", linewidth=2)
    ax.scatter(61, predicted_price, color="red", label="Prix prédit")
    ax.plot([60, 61], [prices[-1], predicted_price], linestyle="--", color="red")

    ax.set_xlabel("Temps")
    ax.set_ylabel("Prix")
    ax.set_title("Prédiction du prochain prix")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig, use_container_width=False)

# ===================== CHATBOT =====================
st.divider()
st.subheader("💬 Chatbot AI-RiskInvest")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Buttons ----------
st.markdown("### 💡 Questions suggérées")

c1, c2, c3 = st.columns(3)
if c1.button("👋 Hello / Who are you"):
    st.session_state.messages.append({"role": "user", "content": "hello"})
if c2.button("📊 Explique le résultat"):
    st.session_state.messages.append({"role": "user", "content": "explique le résultat"})
if c3.button("⚠️ Quel est le risque ?"):
    st.session_state.messages.append({"role": "user", "content": "quel est le risque"})

# ---------- Display Chat ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- Input ----------
user_input = st.chat_input("Posez votre question (FR / EN / AR)")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    q = user_input.lower()

    if any(w in q for w in ["hello", "bonjour", "salam", "who", "من انت", "شكون"]):
        reply = (
            "👋 Je suis **AI-RiskInvest** 🤖.\n\n"
            "Je vous aide à comprendre les prédictions boursières "
            "et la gestion du risque.\n\n"
            "أستطيع المساعدة بالعربية، الفرنسية والإنجليزية."
        )

    elif any(w in q for w in ["résultat", "prediction", "prix", "نتيجة", "توقع"]):
        if predicted_price:
            reply = f"📊 Le prix prédit est **{predicted_price:.4f}**.\nBasé sur 60 prix historiques."
        else:
            reply = "ℹ️ Veuillez d’abord cliquer sur **Prédire**."

    elif any(w in q for w in ["risque", "risk", "خطر"]):
        reply = (
            "⚠️ Ceci n’est PAS un conseil financier.\n"
            "Le marché peut être imprévisible.\n"
            "Utilisez toujours une bonne gestion du risque."
        )

    elif any(w in q for w in ["comment", "utiliser", "how", "كيف"]):
        reply = (
            "1️⃣ Entrer 60 prix\n"
            "2️⃣ Cliquer sur Prédire\n"
            "3️⃣ Analyser le graphique"
        )

    else:
        reply = "🤖 Je n’ai pas compris. Essayez : Résultat, Risque, Comment utiliser."

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)

# ===================== STYLE =====================
st.markdown("""
<style>
.stApp {
    background-color: #0f172a;
    color: #e5e7eb;
    font-family: Segoe UI, sans-serif;
}
.stButton>button {
    background-color: #1e293b;
    color: white;
    border-radius: 6px;
}
[data-testid="chat-message-assistant"] {
    background-color: #020617;
    border-left: 4px solid #e11d48;
    padding: 10px;
    border-radius: 8px;
}
[data-testid="chat-message-user"] {
    background-color: #1e293b;
    padding: 10px;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)
