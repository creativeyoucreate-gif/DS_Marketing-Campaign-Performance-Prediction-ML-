import streamlit as st
import pandas as pd
import joblib

# Step 1: Load the saved models and lists (run `streamlit run app.py` from the project folder)
revenue_model = joblib.load("models/revenue_model.pkl")
profit_model = joblib.load("models/profit_model.pkl")
model_columns = joblib.load("models/model_columns.pkl")
options = joblib.load("models/dropdown_options.pkl")

st.title("Marketing Campaign Performance Predictor")
st.write("Enter the campaign details on the left, then click Predict.")

# Step 2: Inputs
st.sidebar.header("Campaign details")

brand = st.sidebar.selectbox("Brand", options["Brand"])
campaign_type = st.sidebar.selectbox("Campaign type", options["Campaign_Type"])
target_audience = st.sidebar.selectbox("Target audience", options["Target_Audience"])
language = st.sidebar.selectbox("Language", options["Language"])
customer_segment = st.sidebar.selectbox("Customer segment", options["Customer_Segment"])

channels = st.sidebar.multiselect(
    "Channels used",
    ["Instagram", "Email", "Google", "Facebook", "YouTube", "WhatsApp"],
    default=["Email"],
)

duration = st.sidebar.number_input("Duration (days)", min_value=1, value=17)
impressions = st.sidebar.number_input("Impressions", min_value=0, value=55000)
clicks = st.sidebar.number_input("Clicks", min_value=0, value=4000)
leads = st.sidebar.number_input("Leads", min_value=0, value=1500)
conversions = st.sidebar.number_input("Conversions", min_value=1, value=775)
acquisition_cost = st.sidebar.number_input("Acquisition cost (cost per conversion)", min_value=0.0, value=190.0)
engagement_score = st.sidebar.number_input("Engagement score", min_value=0.0, value=15.0)
year = st.sidebar.selectbox("Year", [2024, 2025])
month = st.sidebar.selectbox("Month", list(range(1, 13)))

# Step 3: Predict when the button is clicked
if st.button("Predict"):

    # Start with every model column set to 0
    row = {}
    for col in model_columns:
        row[col] = 0

    # Fill the number columns
    row["Duration"] = duration
    row["Impressions"] = impressions
    row["Clicks"] = clicks
    row["Leads"] = leads
    row["Conversions"] = conversions
    row["Acquisition_Cost"] = acquisition_cost
    row["Engagement_Score"] = engagement_score
    row["Year"] = year
    row["Month"] = month

    # Fill the channel columns
    row["Channel_Instagram"] = 1 if "Instagram" in channels else 0
    row["Channel_Email"] = 1 if "Email" in channels else 0
    row["Channel_Google"] = 1 if "Google" in channels else 0
    row["Channel_Facebook"] = 1 if "Facebook" in channels else 0
    row["Channel_YouTube"] = 1 if "YouTube" in channels else 0
    row["Channel_WhatsApp"] = 1 if "WhatsApp" in channels else 0
    row["Channel_Count"] = len(channels)

    # Fill the text columns the same way get_dummies did.
    # The first value of each list has no column (drop_first=True), so it stays all 0.
    for name in ["Brand_" + brand, "Campaign_Type_" + campaign_type,
                 "Target_Audience_" + target_audience, "Language_" + language,
                 "Customer_Segment_" + customer_segment]:
        if name in row:
            row[name] = 1

    # One-row table in the same column order the models were trained on
    input_df = pd.DataFrame([row])[model_columns]

    # Predictions
    predicted_revenue = revenue_model.predict(input_df)[0]
    profit_class = profit_model.predict(input_df)[0]
    profit_chance = profit_model.predict_proba(input_df)[0][1]

    total_cost = acquisition_cost * conversions

    # Step 4: Show the results
    st.subheader("Prediction")
    st.metric("Predicted Revenue", format(int(predicted_revenue), ","))

    if profit_class == 1:
        st.success("Predicted result: PROFIT (chance of profit " + str(round(profit_chance * 100, 1)) + "%)")
    else:
        st.error("Predicted result: LOSS (chance of profit " + str(round(profit_chance * 100, 1)) + "%)")

    st.subheader("Predicted Revenue vs Total Cost")
    chart_df = pd.DataFrame({"Amount": [total_cost, predicted_revenue]}, index=["Total cost", "Predicted revenue"])
    st.bar_chart(chart_df)
    st.caption("Total cost = acquisition cost x conversions")
