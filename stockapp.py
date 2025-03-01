import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

def main():
    st.title("Stock Price Prediction App")
    st.write("Upload a CSV file containing historical stock prices.")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data:")
        st.write(df.head())
        
        # Ensure dataset has necessary columns
        if 'Date' in df.columns and 'Close' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            df['Days'] = (df['Date'] - df['Date'].min()).dt.days
            
            X = df[['Days']]
            y = df['Close']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Model training
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Display performance metrics
            st.write("### Model Performance Metrics:")
            st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
            
            # Plot results
            fig, ax = plt.subplots()
            ax.scatter(X_test, y_test, color='blue', label='Actual Prices')
            ax.scatter(X_test, y_pred, color='red', label='Predicted Prices')
            ax.set_xlabel("Days")
            ax.set_ylabel("Stock Price")
            ax.legend()
            st.pyplot(fig)
            
            # Future Prediction Input
            days_ahead = st.number_input("Predict stock price for how many days ahead?", min_value=1, value=10)
            future_day = np.array([[df['Days'].max() + days_ahead]])
            future_price = model.predict(future_day)[0]
            st.write(f"### Predicted Stock Price after {days_ahead} days: ${future_price:.2f}")
        else:
            st.error("The dataset must contain 'Date' and 'Close' columns.")

if __name__ == "__main__":
    main()
