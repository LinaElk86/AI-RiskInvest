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
    if "quoi" in user_input.lower() or "what" in user_input.lower():
        reply = (
            "Je suis le chatbot de AI-RiskInvest. "
            "Je vous aide à comprendre l'application et les prédictions."
        )
    elif "comment" in user_input.lower():
        reply = (
            "Entrez les 60 derniers prix de clôture "
            "puis cliquez sur le bouton « Prédire »."
        )
    else:
        reply = (
            "Bonne question 👍 "
            "Pour le moment, je réponds uniquement à des questions simples."
        )

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": reply})

    with st.chat_message("assistant"):
        st.markdown(reply)




st.divider()
st.subheader("💬 Chatbot AI-RiskInvest")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Posez votre question ici...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    if "what" in user_input.lower() or "quoi" in user_input.lower():
        reply = "Je suis le chatbot AI-RiskInvest. Je vous aide à comprendre l'application et les prédictions."
    elif "comment" in user_input.lower():
        reply = "Entrez les 60 derniers prix de clôture puis cliquez sur Prédire."
    else:
        reply = "Bonne question 👍 Pour le moment, je réponds uniquement à des questions simples."

    st.session_state.messages.append({"role": "assistant", "content": reply})

    with st.chat_message("assistant"):
        st.markdown(reply)
