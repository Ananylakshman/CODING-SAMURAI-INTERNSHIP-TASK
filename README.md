# House Price Prediction App

## Overview
This project predicts house prices based on features such as average area income, house age, number of rooms, number of bedrooms, and area population. The prediction model is built using **Linear Regression** and deployed as a **Streamlit web app**.

---

## Features
- Input house details: Average Area Income, House Age, Number of Rooms, Number of Bedrooms, Area Population
- Calculates **Rooms per Bedroom** as an additional feature
- Predicts house price in USD
- Interactive and user-friendly interface with Streamlit

---

## Dataset
- The dataset used is the **USA Housing Dataset** (available publicly):  
  [USA Housing Dataset](https://raw.githubusercontent.com/selva86/datasets/master/USA_Housing.csv)

---

## Preprocessing Steps
- Feature engineering: Added `Rooms_per_Bedroom`
- Removed outliers (top 1% of prices)
- Feature scaling using `StandardScaler`
- Train-test split (80%-20%)

---

## Model
- **Algorithm:** Linear Regression
- **Evaluation Metrics:**
  - Mean Squared Error (MSE): *example output*  
  - Root Mean Squared Error (RMSE): *example output*  
  - R² Score: *example output*

---

## How to Run
1. Clone the repository:
```bash
git clone https://github.com/your-username/house-price-prediction.git
