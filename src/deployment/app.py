import os
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

MODEL_REPO_ID = "HfStan/tourism-wellness-model"  # TODO: update
MODEL_FILENAME = "best_model.pkl"

@st.cache_resource
def load_model():
    local_model_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        filename=MODEL_FILENAME,
    )
    model = joblib.load(local_model_path)
    return model

def main():
    st.title("Wellness Tourism Package Purchase Prediction")

    st.markdown(
        "Predict whether a customer is likely to purchase the Wellness Tourism Package based on their profile and interaction data."
    )

    age = st.number_input("Age", min_value=18, max_value=80, value=35)
    typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=60, value=10)
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Free Lancer", "Large Business"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    num_person = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    preferred_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    num_trips = st.number_input("Number of Trips per Year", min_value=0, max_value=20, value=2)
    passport = st.selectbox("Has Passport?", [0, 1])
    pitch_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
    own_car = st.selectbox("Owns Car?", [0, 1])
    num_children = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", min_value=0, max_value=200000, value=20000)

    if st.button("Predict"):
        model = load_model()
        input_df = pd.DataFrame([
            {
                "Age": age,
                "TypeofContact": typeof_contact,
                "CityTier": city_tier,
                "DurationOfPitch": duration_of_pitch,
                "Occupation": occupation,
                "Gender": gender,
                "NumberOfPersonVisiting": num_person,
                "NumberOfFollowups": num_followups,
                "ProductPitched": product_pitched,
                "PreferredPropertyStar": preferred_star,
                "MaritalStatus": marital_status,
                "NumberOfTrips": num_trips,
                "Passport": passport,
                "PitchSatisfactionScore": pitch_score,
                "OwnCar": own_car,
                "NumberOfChildrenVisiting": num_children,
                "Designation": designation,
                "MonthlyIncome": monthly_income,
            }
        ])

        prob = model.predict_proba(input_df)[0, 1]
        label = "LIKELY to purchase" if prob >= 0.5 else "Unlikely to purchase"
        st.metric("Prediction", label, f"{prob:.2%}")

if __name__ == "__main__":
    main()
