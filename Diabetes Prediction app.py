import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# I load the diabetes dataset directly with headers
def load_data():
    df = pd.read_csv("diabetes.csv")  # Assumes headers are present
    return df

# This function handles training: scaling the features and fitting the RandomForest model
def train_model(df):
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X.columns.tolist(), scaler

# Here I generate a horizontal bar chart to visualize what the model considers important
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)

    fig, ax = plt.subplots()
    ax.barh(range(len(indices)), importances[indices], align='center')
    ax.set_yticks(np.arange(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_title("Feature Importance")
    st.pyplot(fig)

# This is the core Streamlit app logic
def main():
    st.set_page_config(page_title="Diabetes Prediction App", layout="centered")
    st.title(" Diabetes Prediction App")
    st.markdown("Enter patient details below to predict the likelihood of diabetes.")

    # Load dataset and train model
    df = load_data()
    model, features, scaler = train_model(df)

    st.subheader(" Input Patient Data")

    # These default values help guide the user for realistic input
    example_values = {
        'Pregnancies': 2,
        'Glucose': 120,
        'BloodPressure': 70,
        'SkinThickness': 20,
        'Insulin': 80,
        'BMI': 28.5,
        'DiabetesPedigreeFunction': 0.5,
        'Age': 35
    }

    # Collect input from user
    user_input = []
    for feature in features:
        example = example_values.get(feature, 0)
        if feature in ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'Age']:
            val = st.number_input(f"{feature} (e.g., {example})", value=int(example), step=1)
        else:
            val = st.number_input(f"{feature} (e.g., {example})", value=float(example), format="%.2f")
        user_input.append(val)

    # When the user hits the Predict button
    if st.button("🔍 Predict"):
        input_scaled = scaler.transform([user_input])
        prediction = model.predict(input_scaled)[0]
        confidence = model.predict_proba(input_scaled)[0][prediction]

        result = "Positive (Diabetes)" if prediction == 1 else "Negative (No Diabetes)"
        st.success(f"**Prediction:** {result}")
        st.info(f"**Confidence:** {confidence:.2%}")

        st.subheader(" Feature Importance")
        plot_feature_importance(model, features)

        with st.expander("View Sample Dataset"):
            st.dataframe(df.head())
            
    st.markdown("**Designed by Atik Zafar**")

# Run the app
if __name__ == "__main__":
    main()
