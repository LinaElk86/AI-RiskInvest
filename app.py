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



#______________________________________________________________________________2__________________________________________________________________



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
.stApp {
    background-color: #0d1b2a;
    color: white;
}

h1, h2, h3 {
    color: #e63946;
}

.stButton>button {
    background-color: #1d3557;
    color: white;
    border-radius: 8px;
    border: 1px solid #e63946;
}
.stButton>button:hover {
    background-color: #e63946;
}

input {
    background-color: #1b263b !important;
    color: white !important;
}

[data-testid="chat-message-user"] {
    background-color: #1d3557;
    border-radius: 10px;
}

[data-testid="chat-message-assistant"] {
    background-color: #e63946;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


