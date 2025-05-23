import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import tkinter as tk
from tkinter import scrolledtext

# Load dataset
df = pd.read_csv('clv_data.csv')

# Features and target variable
X = df[['tenure_days', 'frequency', 'avg_order_value', 'total_spent']]
y = df['CLV']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
coefficients = model.coef_
intercept = model.intercept_

# Predict CLV for a new customer example
new_customer = pd.DataFrame({
    'tenure_days': [300],
    'frequency': [10],
    'avg_order_value': [60],
    'total_spent': [600]
})
predicted_clv = model.predict(new_customer)[0]

# Prepare output text
output_text = (
    f"Mean Squared Error: {mse:.2f}\n"
    f"R² Score: {r2:.2f}\n"
    f"Model Coefficients: {coefficients}\n"
    f"Intercept: {intercept:.2f}\n\n"
    f"Predicted Customer Lifetime Value for new customer: ${predicted_clv:.2f}"
)

# Create GUI window
window = tk.Tk()
window.title("CLV Prediction Results")

# Create a scrolled text widget
text_area = scrolledtext.ScrolledText(window, width=60, height=10, font=("Arial", 12))
text_area.pack(padx=10, pady=10)

# Insert the output text
text_area.insert(tk.INSERT, output_text)
text_area.configure(state='disabled')  # Make read-only

# Start the GUI event loop
window.mainloop()
