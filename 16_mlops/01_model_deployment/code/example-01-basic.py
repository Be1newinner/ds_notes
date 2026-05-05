import streamlit as st
import pandas as pd

# Title of the application
st.title("Customer Spending Predictor (Mock)")

st.write("""
This is a simple interactive web application built with Streamlit.
It demonstrates how to take user inputs and display a simulated prediction.
""")

# Create an input form
st.header("Enter Customer Details")

# Various input widgets
age = st.slider("Customer Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income ($)", min_value=10000, max_value=200000, value=50000)
membership = st.selectbox("Membership Tier", options=["Basic", "Premium", "VIP"])

# Prediction Button
if st.button("Predict Expected Spending"):
    # Mock prediction logic (In reality, this would be model.predict())
    base_spend = income * 0.1
    
    if membership == "Premium":
        base_spend *= 1.2
    elif membership == "VIP":
        base_spend *= 1.5
        
    if age < 30:
        base_spend *= 0.9
        
    predicted_spend = round(base_spend, 2)
    
    # Display the result
    st.success(f"The predicted annual spending for this customer is: ${predicted_spend}")
    
    # Show the inputted data as a table
    st.write("Based on the following data:")
    data = pd.DataFrame({"Age": [age], "Income": [income], "Tier": [membership]})
    st.dataframe(data)

# To run this app, open your terminal and type:
# streamlit run example-01-basic.py
