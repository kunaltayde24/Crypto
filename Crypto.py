import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

# Load trained model
model = load_model(r"E:\ML\Projects\CRYTO\crypto_predictor.h5")


# Set up Streamlit
st.title("Cryptocurrency Price Prediction")

# Input for cryptocurrency symbol
crypto = st.text_input("Enter Cryptocurrency Symbol (e.g., BTC-USD):", "BTC-USD")
start_date = st.date_input("Start Date:", value=pd.to_datetime('2020-01-01'))
end_date = st.date_input("End Date:", value=pd.to_datetime('2022-12-31'))

# Fetch data
data = yf.download(crypto, start=start_date, end=end_date)

if not data.empty:
    st.subheader(f"{crypto} Historical Prices")
    st.write(data.tail())

    # Plot historical prices
    st.subheader("Price Chart")
    plt.figure(figsize=(10, 5))
    plt.plot(data['Close'], label="Closing Price")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Price")
    st.pyplot(plt)

    # Prepare data for prediction
    prices = data['Close'].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(prices.reshape(-1, 1))

    x_test = []
    for i in range(60, len(scaled_data)):
        x_test.append(scaled_data[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Make predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Plot predictions
    st.subheader("Predicted Prices vs. Actual Prices")
    plt.figure(figsize=(10, 5))
    plt.plot(prices[60:], label="Actual Prices")
    plt.plot(range(60, len(predictions) + 60), predictions, label="Predicted Prices")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Price")
    st.pyplot(plt)
else:
    st.error("No data found for the selected cryptocurrency and date range.")




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
import base64

# Usage
set_background_image("https://academy.education.investing.com/wp-content/uploads/2022/03/bitcoin-what-is-crypto-scaled.jpg")


# App Content
st.title("📈 Cryptocurrency Price Prediction")
st.markdown("**Welcome to the Cryptocurrency Price Predictor!** Explore the trends and forecasts for your favorite crypto assets.")

# Sidebar Navigation
st.sidebar.title("Navigation")
st.sidebar.button("Home")
st.sidebar.button("Predict Prices")
st.sidebar.button("Contact Us")

# Add a line chart
import numpy as np
data = np.random.randn(50, 3)
st.line_chart(data)

# Footer
st.markdown(
    """
    <hr>
    <footer style='text-align: center;'>
        <p style='color: white;'>© 2025 Cryptocurrency Price Predictor | Designed by Kunal Tayde</p>
    </footer>
    """,
    unsafe_allow_html=True
)
