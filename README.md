# Boston House Price Prediction App  

This is a **Streamlit-based web application** that predicts Boston house prices using a **Random Forest Regressor**. The app allows users to interactively input parameters and visualize key insights from the dataset.  

## Features  
- **User Inputs**: Adjust housing parameters via sliders.  
- **Machine Learning Model**: Predicts median house prices using **Random Forest Regression**.  
- **Data Visualization**:  
  - Histogram of house prices.  
  - Scatter plot showing the relationship between house size and price.  
  - SHAP feature importance analysis.  

## Installation  
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```  
2. Install dependencies:  
   ```bash
   pip install pandas shap matplotlib seaborn streamlit scikit-learn
   ```

## Running the App  
Run the following command:  
```bash
streamlit run pro6.py
```

## Future Enhancements  
- Add multiple ML models for comparison.  
- Implement hyperparameter tuning.  
- Deploy as a web service.  
