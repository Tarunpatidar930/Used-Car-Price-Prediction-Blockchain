import streamlit as st
import pandas as pd
import joblib
import hashlib
import json
import os
from datetime import datetime


# ==========================================
# 1. LOAD TRAINED MODEL
# ==========================================

model = joblib.load("car_price_model.pkl")


# ==========================================
# 2. SIMPLE BLOCKCHAIN FUNCTIONS
# ==========================================

BLOCKCHAIN_FILE = "blockchain.json"


def create_hash(data):
    data_string = json.dumps(data, sort_keys=True)
    return hashlib.sha256(data_string.encode()).hexdigest()


def load_blockchain():

    if os.path.exists(BLOCKCHAIN_FILE):

        with open(BLOCKCHAIN_FILE, "r") as file:
            return json.load(file)

    # First block
    genesis_block = {
        "block": 0,
        "time": str(datetime.now()),
        "car": "Genesis Block",
        "price": 0,
        "previous_hash": "0"
    }

    genesis_block["hash"] = create_hash(genesis_block)

    blockchain = [genesis_block]

    save_blockchain(blockchain)

    return blockchain


def save_blockchain(blockchain):

    with open(BLOCKCHAIN_FILE, "w") as file:
        json.dump(blockchain, file, indent=4)


def add_block(car_details, price):

    blockchain = load_blockchain()

    previous_hash = blockchain[-1]["hash"]

    new_block = {

        "block": len(blockchain),

        "time": str(datetime.now()),

        "car": car_details,

        "price": float(price),

        "previous_hash": previous_hash
    }

    new_block["hash"] = create_hash(new_block)

    blockchain.append(new_block)

    save_blockchain(blockchain)

    return new_block


# Load blockchain
blockchain = load_blockchain()


# ==========================================
# 3. PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)


# ==========================================
# 4. TITLE
# ==========================================

st.title("🚗 Used Car Price Predictor")

st.write(
    "Enter your car details to estimate its selling price."
)

st.divider()


# ==========================================
# 5. USER INPUTS
# ==========================================

col1, col2 = st.columns(2)


with col1:

    brand = st.selectbox(
        "Brand",
        [
            "Maruti",
            "Hyundai",
            "Honda",
            "Toyota",
            "Tata",
            "Mahindra",
            "Ford",
            "Volkswagen"
        ]
    )

    model_name = st.text_input(
        "Car Model",
        "City"
    )

    year = st.number_input(
        "Manufacturing Year",
        min_value=1990,
        max_value=2026,
        value=2021
    )

    kilometers = st.number_input(
        "Kilometers Driven",
        min_value=0,
        max_value=500000,
        value=35000
    )


with col2:

    fuel = st.selectbox(
        "Fuel Type",
        [
            "Petrol",
            "Diesel",
            "CNG",
            "Electric"
        ]
    )

    transmission = st.selectbox(
        "Transmission",
        [
            "Manual",
            "Automatic"
        ]
    )

    engine = st.number_input(
        "Engine CC",
        min_value=500,
        max_value=6000,
        value=1498
    )

    mileage = st.number_input(
        "Mileage",
        min_value=1.0,
        max_value=100.0,
        value=18.4
    )

    owner = st.selectbox(
        "Previous Owners",
        [0, 1, 2, 3, 4]
    )


# ==========================================
# 6. PREDICT BUTTON
# ==========================================

st.divider()

predict_button = st.button(
    "🔮 Predict Car Price",
    use_container_width=True
)


# ==========================================
# 7. PREDICTION
# ==========================================

if predict_button:

    # Create input for ML model

    new_car = pd.DataFrame({

        "Brand": [brand],

        "Model": [model_name],

        "Year": [year],

        "Kilometers_Driven": [kilometers],

        "Fuel_Type": [fuel],

        "Transmission": [transmission],

        "Engine": [engine],

        "Mileage": [mileage],

        "Owner": [owner]
    })


    # ======================================
    # ML PREDICTION
    # ======================================

    prediction = model.predict(new_car)

    price = prediction[0]


    # ======================================
    # SHOW PREDICTION
    # ======================================

    st.success(
        "Prediction completed successfully! 🎉"
    )

    st.metric(
        label="Estimated Selling Price",
        value=f"₹ {price:.2f} Lakh"
    )


    # ======================================
    # BLOCKCHAIN RECORD
    # ======================================

    car_details = {

        "Brand": brand,

        "Model": model_name,

        "Year": int(year),

        "Kilometers": int(kilometers),

        "Fuel": fuel,

        "Transmission": transmission,

        "Engine": int(engine),

        "Mileage": float(mileage),

        "Owner": int(owner)
    }


    new_block = add_block(
        car_details,
        price
    )


    # ======================================
    # BLOCKCHAIN MESSAGE
    # ======================================

    st.success(
        "🔗 Prediction saved in Blockchain!"
    )

    st.write(
        f"**Block Number:** {new_block['block']}"
    )

    st.write(
        f"**Hash:** `{new_block['hash']}`"
    )

    st.write(
        f"**Previous Hash:** `{new_block['previous_hash']}`"
    )


# ==========================================
# 8. VIEW BLOCKCHAIN
# ==========================================

st.divider()

st.subheader("🔗 Blockchain Records")


blockchain = load_blockchain()

st.write(
    f"Total Blocks: **{len(blockchain)}**"
)


with st.expander("View Blockchain"):

    for block in blockchain:

        st.write(
            f"### Block {block['block']}"
        )

        st.write(
            f"**Time:** {block['time']}"
        )

        st.write(
            f"**Price:** ₹ {block['price']:.2f} Lakh"
        )

        st.write(
            f"**Previous Hash:** `{block['previous_hash']}`"
        )

        st.write(
            f"**Hash:** `{block['hash']}`"
        )

        st.divider()


