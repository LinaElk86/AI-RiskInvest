import streamlit as st
import numpy as np
import joblib


# ===================== LOAD MODEL =====================
model = joblib.load("riskinvest_model.pkl")
scaler = joblib.load("scaler.pkl")

# ===================== TITLE =====================
st.title("📈 AI-RiskInvest")
st.write("Application de prédiction boursière et gestion du risque")

# ===================== INPUT PRICES =====================
st.subheader("📥 Entrer les 60 derniers prix de clôture")

texte_prix = st.text_area(
    "Entrez les 60 prix (séparés par des virgules ou retour à la ligne)",
    height=200,
    placeholder="Exemple :\n1.25\n1.30\n1.28\n...\n(60 valeurs)"
)

# Liste fixe de 60 prix (visible toujours)
prices = [0.0] * 60

if texte_prix:
    try:
        texte_prix = texte_prix.replace("\n", ",")
        valeurs = [float(p.strip()) for p in texte_prix.split(",") if p.strip() != ""]

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
for ligne in range(6):  # 6 lignes
    cols = st.columns(10)  # 10 colonnes
    for col in cols:
        col.number_input(
            f"{index + 1}",
            value=prices[index],
            disabled=True
        )
        index += 1

# ===================== PREDICTION =====================
st.markdown("## Résultat de la prédiction")

if st.button("Prédire"):
    if len(prices) != 60:
        st.error("❌ Il faut exactement 60 prix pour prédire.")
    else:
        prices_array = np.array(prices).reshape(-1, 1)
        prices_scaled = scaler.transform(prices_array)
        X_input = prices_scaled.reshape(1, -1)

        prediction = model.predict(X_input)
        predicted_price = scaler.inverse_transform(
            prediction.reshape(-1, 1)
        )[0][0]

        st.success("✅ Prédiction effectuée avec succès")
        st.metric(
            label="📊 Prix prédit",
            value=f"{predicted_price:.4f}"
        )




#___________________________________________________________



import matplotlib.pyplot as plt

# ===================== GRAPH =====================
st.subheader("📈 Évolution des prix")

# Axe X : 60 périodes + prediction
x_prices = list(range(1, 61))
x_pred = 61

# Création du graphique
fig, ax = plt.subplots(figsize=(10, 4))

# Prix historiques
ax.plot(x_prices, prices, label="Prix historiques", color="blue")

# Point de prédiction
ax.scatter(
    x_pred,
    predicted_price,
    color="red",
    label="Prix prédit",
    zorder=5
)

# Ligne pointillée vers la prédiction
ax.plot(
    [60, x_pred],
    [prices[-1], predicted_price],
    linestyle="--",
    color="red",
    alpha=0.7
)

ax.set_xlabel("Temps")
ax.set_ylabel("Prix")
ax.set_title("Prédiction du prochain prix")
ax.legend()
ax.grid(True)

st.pyplot(fig)
#______________________________________________________________________________2__________________________________________________________________

# ===================== CHATBOT =====================


st.divider()
st.subheader("💬 Chatbot AI-RiskInvest")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Posez votre question ici...")

if user_input:
    # Save user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    question = user_input.lower()

    # Chatbot logic
    if any(word in question for word in ["hello", "hi", "salut", "bonjour", "salam"]):
        reply = (
            "Bonjour 👋 Je suis le chatbot AI-RiskInvest 🤖.\n"
            "Je peux vous aider à comprendre l’application et les prédictions."
        )

    elif any(word in question for word in ["quoi", "what", "application"]):
        reply = (
            "AI-RiskInvest est une application de prédiction boursière "
            "basée sur le Machine Learning et la gestion du risque."
        )

    elif any(word in question for word in ["comment", "utiliser"]):
        reply = (
            "Entrez les 60 derniers prix de clôture "
            "puis cliquez sur le bouton « Prédire »."
        )

    elif any(word in question for word in ["prediction", "prédit", "résultat"]):
        reply = (
            "La prédiction est une estimation du prochain prix "
            "basée sur les données historiques."
        )

    elif any(word in question for word in ["risque", "risk"]):
        reply = (
            "Le risque représente l’incertitude du marché. "
            "Cette application aide à mieux l’anticiper."
        )

    elif any(word in question for word in ["merci", "thanks"]):
        reply = "Avec plaisir 😊 N’hésitez pas à poser d’autres questions."

    else:
        reply = (
            "Je n’ai pas bien compris 🤖.\n"
            "Essayez par exemple : hello, comment utiliser, prédiction, risque."
        )

    # Save assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": reply}
    )

    # Display assistant message
    with st.chat_message("assistant"):
        st.markdown(reply)



#_________________________________________________________________________________________3______________________________________________________________________________________
st.markdown("""
<style>

/* Global background */
.stApp {
    background-color: #0f172a;
    color: #e5e7eb;
    font-family: "Segoe UI", sans-serif;
}

/* Titles */
h1, h2, h3 {
    color: #f1f5f9;
    font-weight: 600;
}

/* Subtitles */
h4, h5, h6 {
    color: #cbd5f5;
}

/* Buttons */
.stButton > button {
    background-color: #1e293b;
    color: white;
    border-radius: 6px;
    border: 1px solid #475569;
    padding: 8px 16px;
    font-weight: 500;
}
.stButton > button:hover {
    background-color: #334155;
    border-color: #e11d48;
}

/* Inputs */
input {
    background-color: #020617 !important;
    color: white !important;
    border: 1px solid #334155 !important;
    border-radius: 6px !important;
}

/* Chat user */
[data-testid="chat-message-user"] {
    background-color: #1e293b;
    border-radius: 10px;
    padding: 8px;
}

/* Chat assistant */
[data-testid="chat-message-assistant"] {
    background-color: #020617;
    border-radius: 10px;
    padding: 8px;
    border-left: 3px solid #e11d48;
}

</style>
""", unsafe_allow_html=True)







