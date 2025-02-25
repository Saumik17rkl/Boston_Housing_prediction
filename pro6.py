import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import seaborn as sns

# App Title
st.write("""
# Boston House Prediction App
This app predicts the **Boston House Price** using the uploaded dataset.
""")
st.write("--------------")

# Load the dataset from the uploaded file
uploaded_file = "HousingData.csv"
data = pd.read_csv(uploaded_file)

# Assuming the target column is "MEDV"
X = data.drop("MEDV", axis=1)
Y = data["MEDV"]

# Sidebar header
st.sidebar.header("Specify Input Parameters")

# Function to take user inputs
def user_input_features():
    inputs = {}
    for col in X.columns:
        if X[col].dtype in [int, float]:
            inputs[col] = st.sidebar.slider(
                f"{col} ({col} values)", float(X[col].min()), float(X[col].max()), float(X[col].mean())
            )
        else:
            inputs[col] = st.sidebar.selectbox(f"{col}", sorted(X[col].unique()))
    features = pd.DataFrame(inputs, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Display user input
st.write("### User Input Parameters")
st.write(input_df)

# Train the Random Forest Regressor
model = RandomForestRegressor()
model.fit(X, Y)

# Predict and display the result
prediction = model.predict(input_df)
st.write("### Predicted Median Value of Owner-Occupied Homes ($1000s):")
st.write(prediction[0])

# Display Histogram of Target Variable
st.write("### Distribution of Median Value (MEDV)")
plt.figure(figsize=(10, 6))
sns.histplot(Y, kde=True, color='blue', bins=30)
plt.title("Distribution of Median Value (MEDV)")
plt.xlabel("Median Value ($1000s)")
plt.ylabel("Frequency")
st.pyplot(plt.gcf())

# Scatter Plot: Feature vs Target
st.write("### Relationship between RM (Average Number of Rooms) and MEDV")
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X['RM'], y=Y, color='green')
plt.title("Relationship between RM and MEDV")
plt.xlabel("Average Number of Rooms (RM)")
plt.ylabel("Median Value ($1000s)")
st.pyplot(plt.gcf())

# SHAP for interpretability
st.write("### Feature Importance using SHAP")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X, plot_type="bar", show=False)
st.pyplot(plt.gcf())
