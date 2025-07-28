import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Sample dataset
data = {
    'Weather': ['Sunny', 'Rainy', 'Overcast', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Rainy', 'Sunny', 'Rainy'],
    'TimeOfDay': ['Morning', 'Morning', 'Afternoon', 'Afternoon', 'Evening', 'Morning', 'Morning', 'Afternoon', 'Evening', 'Morning'],
    'SleepQuality': ['Poor', 'Good', 'Poor', 'Good', 'Poor', 'Good', 'Poor', 'Good', 'Good', 'Poor'],
    'Mood': ['Tired', 'Fresh', 'Tired', 'Energetic', 'Tired', 'Fresh', 'Tired', 'Tired', 'Energetic', 'Tired'],
    'BuyCoffee': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

# Encode categorical variables
df_encoded = pd.get_dummies(df.drop('BuyCoffee', axis=1))
target = df['BuyCoffee'].map({'Yes': 1, 'No': 0})

# Train the decision tree
model = DecisionTreeClassifier(criterion='entropy')
model.fit(df_encoded, target)

# Streamlit UI
st.title("☕ Coffee Prediction App")
st.write("Predict if a customer will buy coffee based on their conditions.")

# User input
weather = st.selectbox("Weather", ['Sunny', 'Rainy', 'Overcast'])
time_of_day = st.selectbox("Time of Day", ['Morning', 'Afternoon', 'Evening'])
sleep_quality = st.selectbox("Sleep Quality", ['Good', 'Poor'])
mood = st.selectbox("Mood", ['Fresh', 'Tired', 'Energetic'])

# Create input for prediction
user_input = pd.DataFrame({
    'Weather_' + weather: [1],
    'TimeOfDay_' + time_of_day: [1],
    'SleepQuality_' + sleep_quality: [1],
    'Mood_' + mood: [1]
})

# Ensure all columns match training data
for col in df_encoded.columns:
    if col not in user_input.columns:
        user_input[col] = 0
user_input = user_input[df_encoded.columns]  # Order columns

# Prediction
prediction = model.predict(user_input)[0]
st.subheader("Prediction:")
st.write("✅ **Will Buy Coffee**" if prediction == 1 else "❌ **Will Not Buy Coffee**")

# Plot tree
st.subheader("Decision Tree")
fig, ax = plt.subplots(figsize=(12, 6))
plot_tree(model, feature_names=df_encoded.columns, class_names=["No", "Yes"], filled=True)
st.pyplot(fig)
