import gradio as gr
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle

# Modell laden
model_filename = "week2/apartment_price_model.pkl"
try:
    with open(model_filename, "rb") as f:
        model = pickle.load(f)
    print(f"Loaded model from {model_filename}")
except FileNotFoundError:
    print(f"Model file {model_filename} not found. Using a placeholder model.")
    model = RandomForestRegressor(
        n_estimators=1000,
        max_depth=100,
        max_features=9,
        min_samples_leaf=4,
        min_samples_split=10,
        random_state=42
    )

# Standardwerte aus dem Trainingsdatensatz
default_pop = 24990.00
default_pop_dens = 1662.60
default_frg_pct = 28.87
default_emp = 19226.00
default_tax_income = 80977.00

def predict_price(rooms, area, desclength, cows):
    # Sicherstellen, dass keine None-Werte verarbeitet werden
    rooms = float(rooms) if rooms is not None else 1
    area = float(area) if area is not None else 50
    desclength = int(desclength) if desclength is not None else 0
    cows = int(cows) if cows is not None else 0

    # Input-Daten mit den ursprünglich verwendeten Features
    input_data = pd.DataFrame({
        'rooms': [rooms],
        'area': [area],
        'pop': [default_pop],
        'pop_dens': [default_pop_dens],
        'frg_pct': [default_frg_pct],
        'emp': [default_emp],
        'tax_income': [default_tax_income],
        'desclength': [desclength],
        'cows': [cows]
    })
    
    # Vorhersage mit dem Modell
    predicted_price = model.predict(input_data)[0]
    
    return f"Estimated monthly rent: CHF {predicted_price:.2f}"

# Gradio Interface
demo = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Number of Rooms", value=3, minimum=1, maximum=10),
        gr.Number(label="Area (m²)", value=75, minimum=10, maximum=500),
        gr.Slider(label="Description Length", minimum=0, maximum=500, value=5, step=10),
        gr.Slider(label="Number of Cows", minimum=0, maximum=5000, value=5, step=50)
    ],
    outputs="text",
    title="Swiss Apartment Price Prediction",
    description="Enter apartment details to predict the monthly rental price in Swiss Francs (CHF).",
    examples=[
        [2, 55, 100, 3],  # Small apartment
        [3, 80, 250, 7],   # Medium apartment
        [4, 120, 300, 10], # Large apartment
        [2, 60, 50, 5]     # Small apartment with short description
    ]
)

if __name__ == "__main__":
    demo.launch()
