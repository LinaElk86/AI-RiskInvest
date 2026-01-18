import streamlit as st
import numpy as np
import joblib

# INSTALLER LE MODELE scaler
model = joblib.load("riskinvest_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("📈 AI-RiskInvest")
st.write("Application de prédiction boursière et gestion du risque")

st.subheader("Entrer les 60 derniers prix de clôture")

prices = []
for i in range(60):
    price = st.number_input(f"Prix {i+1}", value=0.0)
    prices.append(price)

if st.button("Prédire"):
    prices_array = np.array(prices).reshape(-1, 1)
    prices_scaled = scaler.transform(prices_array)
    X_input = prices_scaled.reshape(1, -1)

    prediction = model.predict(X_input)
    predicted_price = scaler.inverse_transform(prediction.reshape(-1,1))[0][0]

    st.success(f"📊 Prix prédit : {predicted_price:.2f}")



st.divider()
st.subheader("💬 Chatbot AI-RiskInvest")

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
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

 # Simple chatbot logic
question = user_input.lower() if user_input else ""

# Greetings
if any(word in question for word in ["hello", "hi", "bonjour", "salut", "salam", "slm"]):
    reply = (
        "Bonjour 👋 Je suis le chatbot AI-RiskInvest 🤖.\n"
        "Je peux vous aider à comprendre l’application, le modèle et la prédiction."
    )

elif any(word in question for word in ["quoi", "what", "application", "ai-riskinvest"]):
    reply = (
        "AI-RiskInvest est une application de prédiction boursière "
        "basée sur le Machine Learning et la gestion du risque."
    )

elif any(word in question for word in ["comment", "utiliser", "use"]):
    reply = (
        "Entrez les 60 derniers prix de clôture "
        "puis cliquez sur le bouton « Prédire »."
    )

elif any(word in question for word in ["prediction", "prédit", "résultat"]):
    reply = (
        "La prédiction représente une estimation du prochain prix "
        "basée sur les données historiques."
    )

elif any(word in question for word in ["risque", "risk"]):
    reply = (
        "Le risque correspond à l’incertitude des marchés financiers. "
        "Cette application aide à mieux l’anticiper."
    )

elif any(word in question for word in ["modèle", "modele", "model", "machine learning"]):
    reply = (
        "Le modèle utilise le Machine Learning pour analyser "
        "les prix passés et identifier des tendances."
    )

elif any(word in question for word in ["merci", "thanks"]):
    reply = "Avec plaisir 😊 N’hésitez pas si vous avez d’autres questions."

else:
    reply = (
        "Je n’ai pas compris votre question 🤖.\n"
        "Essayez par exemple : hello, comment utiliser, prédiction, risque."
    )

