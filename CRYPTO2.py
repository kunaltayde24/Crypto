import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

# Load trained model
model = load_model(r"E:\ML\Projects\CRYTO\crypto_predictor.h5")

# Set background image
def set_background_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background_image("https://academy.education.investing.com/wp-content/uploads/2022/03/bitcoin-what-is-crypto-scaled.jpg")

# Set up Streamlit
st.title("📈 Cryptocurrency Price Prediction")
st.markdown("**Welcome to the Cryptocurrency Price Predictor!** Explore the trends and forecasts for your favorite crypto assets.")

# Sidebar Navigation
st.sidebar.title("Navigation")
st.sidebar.button("Home")
st.sidebar.button("Predict Prices")
st.sidebar.button("Contact Us")

# Input for cryptocurrency symbol
crypto = st.text_input("Enter Cryptocurrency Symbol (e.g., BTC-USD):", "BTC-USD")
start_date = st.date_input("Start Date:", value=pd.to_datetime('2020-01-01'))
end_date = st.date_input("End Date:", value=pd.to_datetime('2025-02-01'))

# Fetch data
data = yf.download(crypto, start=start_date, end=end_date)

if not data.empty:
    st.subheader(f"📊 {crypto} Historical Prices")
    st.write(data.tail())

    # Plot historical prices
    st.subheader("📉 Price Chart")
    plt.figure(figsize=(10, 5))
    plt.plot(data['Close'], label="Actual Prices")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Price")
    st.pyplot(plt)

    # Prepare data for prediction
    prices = data['Close'].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(prices.reshape(-1, 1))

    # Prepare last 60 days for future prediction
    last_60_days = scaled_data[-60:]
    future_predictions = []
    input_seq = last_60_days.reshape(1, 60, 1)

    # Predict the next 30 days
    for _ in range(30):
        pred = model.predict(input_seq)
        future_predictions.append(pred[0, 0])
        input_seq = np.roll(input_seq, -1)
        input_seq[0, -1, 0] = pred[0, 0]

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Generate future dates
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')

    # Determine Market Trend
    trend = "📈 Bullish" if future_predictions[-1] > future_predictions[0] else "📉 Bearish"

    # Display Market Trend
    st.subheader("📢 Market Trend Prediction")
    st.success(f"The predicted market trend is: **{trend}**")

    # Define the cutoff date (April 2025)
    cutoff_date = pd.Timestamp('2025-04-01')

    # Check if the user-selected end date is beyond April 2025
    if pd.Timestamp(end_date) <= cutoff_date:
        # Plot actual vs predicted prices only if within the valid range
        st.subheader("📊 Actual vs Predicted Prices")
        plt.figure(figsize=(10, 5))
        plt.plot(data.index, prices, label="Actual Prices")
        plt.plot(future_dates, future_predictions, label="Predicted Prices", linestyle='dashed', color='red')
        plt.xticks(rotation=45)
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Price")
        st.pyplot(plt)
    else:
        st.warning("⚠️ You selected an end date beyond April 2025, so the Actual vs Predicted graph is disabled.")

    # Plot only future predictions
    st.subheader("🔮 Future Price Prediction")
    plt.figure(figsize=(10, 5))
    plt.plot(future_dates, future_predictions, label="Predicted Prices", linestyle='dashed', color='blue')
    plt.xticks(rotation=45)
    plt.legend()
    plt.xlabel("Date (Months)")
    plt.ylabel("Price")
    st.pyplot(plt)

else:
    st.error("❌ No data found for the selected cryptocurrency and date range.")


import google.generativeai as genai


# Set API Key
genai.configure(api_key="AIzaSyBQdeTR88yVO6axiBCc79b0H1efIWPi9pY")  # Replace with your actual API key

# Use the best suitable Gemini model
MODEL_NAME = "gemini-1.5-pro"  # Recommended latest model

# Function to get chatbot response
def get_chatbot_response(user_input):
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(user_input)
    return response.text

# Streamlit Chatbot UI
st.sidebar.subheader("Crypto Chatbot 🤖")
user_input = st.sidebar.text_input("Ask me about crypto!", key="chat_input")

if user_input:
    chatbot_response = get_chatbot_response(user_input)
    st.sidebar.write(f"**Chatbot:** {chatbot_response}")


# Footer
st.markdown(
    """
    <hr>
    <footer style='text-align: center;'>
        <p style='color: white;'>© 2025 Cryptocurrency Price Predictor | Designed by Kunal Tayde & Team</p>
    </footer>
    """,
    unsafe_allow_html=True
)
