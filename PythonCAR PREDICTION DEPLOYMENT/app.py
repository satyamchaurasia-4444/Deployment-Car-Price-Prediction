import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# Set up the Streamlit page.
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered",
)

# Find the files in the same folder as this Python file.
app_folder = Path(__file__).parent
model_file_path = app_folder / "LinearRegressionModel.pkl"
data_file_path = app_folder / "Cleaned car.csv"

# The model expects its input columns in this order.
input_columns = ["name", "company", "year", "kms_driven", "fuel_type"]

st.title("🚗 Car Price Predictor")
st.write("Enter the car details to estimate its selling price.")


@st.cache_data
# Read the CSV file and get the car names and company names for the menus.
def load_car_choices(csv_file_path):
    car_data = pd.read_csv(csv_file_path)

    # Make sure the CSV has the columns needed by the menus.
    if "name" not in car_data.columns or "company" not in car_data.columns:
        raise ValueError("The CSV must have 'name' and 'company' columns.")

    car_names = car_data["name"].dropna().astype(str).unique().tolist()
    company_names = car_data["company"].dropna().astype(str).unique().tolist()

    # Sort the choices so they are easier to find.
    car_names.sort()
    company_names.sort()

    return car_names, company_names


@st.cache_resource
# Load the saved prediction model.
def load_prediction_model(model_file_path):
    with open(model_file_path, "rb") as file:
        prediction_model = pickle.load(file)
    return prediction_model


# Stop the app with a helpful message if the CSV is missing or cannot be read.
if not data_file_path.exists():
    st.error("Cleaned car.csv was not found. Put it in the same folder as this app.")
    st.stop()

try:
    car_names, company_names = load_car_choices(str(data_file_path))
except Exception as error:
    st.error(f"Could not read the car choices: {error}")
    st.stop()


# The questions inside a form. The app runs the prediction when
# the user clicks the button.
with st.form("car_details"):
    selected_car = st.selectbox("Car model", car_names)
    selected_company = st.selectbox("Company", company_names)

    left_column, right_column = st.columns(2)

    with left_column:
        selected_year = st.number_input(
            "Year", min_value=1980, max_value=2100, value=2019, step=1
        )

    with right_column:
        selected_kilometres = st.number_input(
            "Kilometres driven",
            min_value=0,
            max_value=2_000_000,
            value=100,
            step=100,
        )

    selected_fuel_type = st.selectbox(
        "Fuel type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
    )

    estimate_clicked = st.form_submit_button(
        "Estimate price", type="primary", use_container_width=True
    )


# Make and show a prediction after the user submits the form.
if estimate_clicked:
    if not model_file_path.exists():
        st.error(
            "The model file was not found. Put "
            "LinearRegressionModel.pkl beside this app to make predictions."
        )
        st.info("The model should use these columns, in this order: " + ", ".join(input_columns))
    else:
        try:
            model = load_prediction_model(str(model_file_path))

            # Create one row of data in the same column order used to train the model.
            car_details = pd.DataFrame(
                [[
                    selected_car,
                    selected_company,
                    int(selected_year),
                    int(selected_kilometres),
                    selected_fuel_type,
                ]],
                columns=input_columns,
            )

            price = model.predict(car_details)[0]
            st.success(f"Estimated price: ₹{price:,.0f}")
        except Exception as error:
            st.error(f"Could not estimate the price: {error}")
            st.caption(
                "Check that the model file is trusted and expects the same input "
                "columns in the same order."
            )
