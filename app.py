"""
Crop Recommendation Web Application Backend
Uses the trained scikit-learn pipeline (StandardScaler + GaussianNB)
and LabelEncoder to perform real ML inferences on user inputs.
"""

import os
import json
import warnings
from pathlib import Path
from flask import Flask, request, jsonify, render_template

# Suppress unpickling version warnings between sklearn 1.6 and 1.8
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "crop_pipeline.joblib"
ENCODER_PATH = BASE_DIR / "label_encoder.joblib"
METADATA_PATH = BASE_DIR / "metadata.json"

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load model artifacts
if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
    raise FileNotFoundError("Model or LabelEncoder file missing from directory.")

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Crop agronomic metadata for rich UI feedback
CROP_DETAILS = {
    "apple": {
        "title": "Apple (Malus domestica)",
        "icon": "🍎",
        "category": "Fruit / Horticultural",
        "soil_type": "Well-drained, loamy to sandy loam",
        "season": "Temperate / High Altitude",
        "growth_period": "120 - 180 days",
        "key_notes": "Requires cooler temperatures and moderate rainfall with well-aerated soil rich in organic matter."
    },
    "banana": {
        "title": "Banana (Musa acuminata)",
        "icon": "🍌",
        "category": "Fruit / Cash Crop",
        "soil_type": "Rich, deep, loose loamy soil",
        "season": "Tropical / Warm & Humid",
        "growth_period": "9 - 12 months",
        "key_notes": "Needs continuous high moisture, high potassium (K), and warm humid weather without cold snaps."
    },
    "blackgram": {
        "title": "Black Gram (Urad)",
        "icon": "🌱",
        "category": "Pulse / Legume",
        "soil_type": "Loam to heavy clay with good drainage",
        "season": "Kharif & Zaid",
        "growth_period": "70 - 90 days",
        "key_notes": "Nitrogen-fixing legume requiring moderate moisture and warm growing conditions."
    },
    "chickpea": {
        "title": "Chickpea (Gram / Chana)",
        "icon": "🫘",
        "category": "Pulse / Legume",
        "soil_type": "Deep sandy loam to clay loam",
        "season": "Rabi (Winter season)",
        "growth_period": "90 - 120 days",
        "key_notes": "Grows best in moderate cool climate with modest moisture; excessively wet conditions harm yield."
    },
    "coconut": {
        "title": "Coconut (Cocos nucifera)",
        "icon": "🥥",
        "category": "Plantation / Oilseed",
        "soil_type": "Laterite, coastal sandy, alluvial soil",
        "season": "Tropical Coastal",
        "growth_period": "Perennial (All Year)",
        "key_notes": "Prefers warm humid coastal climates with consistent rainfall and adequate sunshine."
    },
    "coffee": {
        "title": "Coffee (Coffea arabica / robusta)",
        "icon": "☕",
        "category": "Plantation / Beverage",
        "soil_type": "Deep, porous, slightly acidic volcanic loam",
        "season": "Highland Monsoon",
        "growth_period": "Perennial / 8-9 months berry cycle",
        "key_notes": "Requires high rainfall, shaded canopy, high humidity and cool hill station elevations."
    },
    "cotton": {
        "title": "Cotton (Gossypium)",
        "icon": "☁️",
        "category": "Fiber / Cash Crop",
        "soil_type": "Deep black cotton soils (regur) or alluvial",
        "season": "Kharif",
        "growth_period": "150 - 180 days",
        "key_notes": "Needs warm days, moderate rainfall during vegetation, and dry bright weather during boll ripening."
    },
    "grapes": {
        "title": "Grapes (Vitis vinifera)",
        "icon": "🍇",
        "category": "Fruit / Horticultural",
        "soil_type": "Sandy loam to clay loam with high drainage",
        "season": "Sub-tropical / Mediterranean",
        "growth_period": "120 - 150 days",
        "key_notes": "Requires low humidity during berry maturation, good potassium reserve, and controlled irrigation."
    },
    "jute": {
        "title": "Jute (Golden Fiber)",
        "icon": "🌾",
        "category": "Fiber Crop",
        "soil_type": "New alluvial soil (river basins)",
        "season": "Warm Wet Monsoon",
        "growth_period": "120 - 150 days",
        "key_notes": "Requires warm tropical temperatures (24-37°C), heavy rainfall (>150mm), and high humidity (>70%)."
    },
    "kidneybeans": {
        "title": "Kidney Beans (Rajma)",
        "icon": "🫘",
        "category": "Pulse / Legume",
        "soil_type": "Fertile, loamy, well-aerated soil",
        "season": "Rabi in plains / Kharif in hills",
        "growth_period": "90 - 120 days",
        "key_notes": "Needs cool to warm temperature and moderate moisture. Cannot tolerate waterlogging."
    },
    "lentil": {
        "title": "Lentil (Masoor)",
        "icon": "🥣",
        "category": "Pulse / Legume",
        "soil_type": "Light loams to alluvial soils",
        "season": "Rabi (Cold weather crop)",
        "growth_period": "100 - 120 days",
        "key_notes": "Cool temperature during vegetative phase and warm weather at maturity; highly resilient to modest moisture."
    },
    "maize": {
        "title": "Maize (Corn)",
        "icon": "🌽",
        "category": "Cereal / Grain",
        "soil_type": "Deep, fertile, well-drained loamy soil",
        "season": "Kharif / Spring",
        "growth_period": "90 - 110 days",
        "key_notes": "Fast growing cereal requiring substantial nitrogen (N), balanced P & K, and steady warm weather."
    },
    "mango": {
        "title": "Mango (Mangifera indica)",
        "icon": "🥭",
        "category": "Fruit / Horticultural",
        "soil_type": "Deep, well-drained alluvial or lateritic soil",
        "season": "Tropical to Sub-tropical",
        "growth_period": "Perennial tree",
        "key_notes": "Thrives in warm sunny weather; needs dry weather during flowering and moderate rainfall during fruit setting."
    },
    "mothbeans": {
        "title": "Moth Beans (Matki)",
        "icon": "🌾",
        "category": "Pulse / Drought-Hardy",
        "soil_type": "Sandy to sandy loam",
        "season": "Kharif (Arid)",
        "growth_period": "75 - 90 days",
        "key_notes": "Extremely drought-tolerant legume; thrives in arid dry conditions with low rainfall."
    },
    "mungbean": {
        "title": "Mung Bean (Green Gram)",
        "icon": "🫛",
        "category": "Pulse / Legume",
        "soil_type": "Well-drained loamy to sandy loam",
        "season": "Kharif & Summer",
        "growth_period": "60 - 75 days",
        "key_notes": "Short duration pulse crop, enriches soil nitrogen, adaptable to warm dry climates."
    },
    "muskmelon": {
        "title": "Muskmelon (Cucumis melo)",
        "icon": "🍈",
        "category": "Cucurbit / Fruit",
        "soil_type": "Sandy loam river beds with organic manure",
        "season": "Summer (Zaid)",
        "growth_period": "80 - 100 days",
        "key_notes": "Requires warm, dry weather, low humidity, and high sunlight to develop high sugar sweetness."
    },
    "orange": {
        "title": "Orange (Citrus sinensis)",
        "icon": "🍊",
        "category": "Fruit / Citrus",
        "soil_type": "Well-drained light loamy or alluvial soil",
        "season": "Sub-tropical / Winter bloom",
        "growth_period": "Perennial tree",
        "key_notes": "Sensitive to waterlogging; requires moderate temperature and dry ripening conditions."
    },
    "papaya": {
        "title": "Papaya (Carica papaya)",
        "icon": "🪴",
        "category": "Fruit / Quick-Yielding",
        "soil_type": "Rich, porous, well-drained sandy loam",
        "season": "Tropical & Sub-tropical",
        "growth_period": "9 - 12 months",
        "key_notes": "Very sensitive to standing water; demands warm sunny weather and balanced potassium levels."
    },
    "pigeonpeas": {
        "title": "Pigeon Peas (Arhar / Tur)",
        "icon": "🌿",
        "category": "Pulse / Legume",
        "soil_type": "Deep loamy to clay soils",
        "season": "Kharif",
        "growth_period": "150 - 180 days",
        "key_notes": "Deep-rooting legume with high drought resilience; improves soil aeration and fixes nitrogen."
    },
    "pomegranate": {
        "title": "Pomegranate (Punica granatum)",
        "icon": "🫐",
        "category": "Fruit / Semi-Arid",
        "soil_type": "Light, loamy, well-drained soil",
        "season": "Semi-arid / Sub-tropical",
        "growth_period": "Perennial shrub",
        "key_notes": "Tolerates hot dry summers and cool winters; highly prized in water-conscious agro-zones."
    },
    "rice": {
        "title": "Rice (Oryza sativa)",
        "icon": "🍚",
        "category": "Cereal / Staple Grain",
        "soil_type": "Clayey loam, alluvial soil with low permeability",
        "season": "Kharif (Monsoon)",
        "growth_period": "100 - 150 days",
        "key_notes": "Water-intensive staple crop requiring high temperatures, heavy rainfall (>180mm), and high humidity (>80%)."
    },
    "watermelon": {
        "title": "Watermelon (Citrullus lanatus)",
        "icon": "🍉",
        "category": "Cucurbit / Fruit",
        "soil_type": "Sandy loam, alluvial riverbeds",
        "season": "Summer (Zaid)",
        "growth_period": "80 - 100 days",
        "key_notes": "Requires high sunshine, warm temperature (24-30°C), low humidity, and sandy well-drained soil."
    }
}

# Real-world benchmark presets derived from dataset conditions
PRESETS = [
    {
        "id": "rice_paddy",
        "name": "Monsoon Rice Paddy",
        "description": "High moisture, heavy rainfall, high nitrogen in warm tropical valley",
        "icon": "🍚",
        "values": {
            "N": 90,
            "P": 42,
            "K": 43,
            "temperature": 23.5,
            "humidity": 82.5,
            "ph": 6.5,
            "rainfall": 230.0
        }
    },
    {
        "id": "coffee_estate",
        "name": "Highland Coffee Estate",
        "description": "High rainfall, cool hill elevation, high phosphorus and potassium",
        "icon": "☕",
        "values": {
            "N": 105,
            "P": 25,
            "K": 30,
            "temperature": 25.5,
            "humidity": 65.0,
            "ph": 6.8,
            "rainfall": 170.0
        }
    },
    {
        "id": "cotton_black_soil",
        "name": "Deccan Cotton Plains",
        "description": "High nitrogen, warm climate, moderate rainfall with high potassium",
        "icon": "☁️",
        "values": {
            "N": 118,
            "P": 45,
            "K": 20,
            "temperature": 24.0,
            "humidity": 80.0,
            "ph": 6.9,
            "rainfall": 80.0
        }
    },
    {
        "id": "apple_orchard",
        "name": "Temperate Apple Orchard",
        "description": "Cool temperate climate, moderate humidity, high P & K reserves",
        "icon": "🍎",
        "values": {
            "N": 20,
            "P": 135,
            "K": 198,
            "temperature": 22.0,
            "humidity": 92.0,
            "ph": 6.0,
            "rainfall": 110.0
        }
    },
    {
        "id": "chickpea_rabi",
        "name": "Arid Chickpea Field",
        "description": "Modest nitrogen, dry air, low rainfall, winter growing condition",
        "icon": "🫘",
        "values": {
            "N": 38,
            "P": 68,
            "K": 80,
            "temperature": 18.5,
            "humidity": 16.5,
            "ph": 7.3,
            "rainfall": 75.0
        }
    },
    {
        "id": "watermelon_summer",
        "name": "Summer Watermelon Basin",
        "description": "Warm summer sun, neutral pH, moderate nitrogen and low moisture",
        "icon": "🍉",
        "values": {
            "N": 98,
            "P": 18,
            "K": 50,
            "temperature": 26.0,
            "humidity": 85.0,
            "ph": 6.5,
            "rainfall": 50.0
        }
    }
]


def assess_parameters(features: dict) -> dict:
    """Analyze soil and climatic parameters for agronomic insights."""
    n = features["N"]
    p = features["P"]
    k = features["K"]
    temp = features["temperature"]
    humidity = features["humidity"]
    ph = features["ph"]
    rainfall = features["rainfall"]

    # Soil pH status
    if ph < 5.5:
        ph_status = "Strongly Acidic (may limit phosphorus availability)"
    elif ph < 6.5:
        ph_status = "Slightly Acidic (favorable for most horticultural crops)"
    elif ph <= 7.5:
        ph_status = "Neutral (optimal nutrient uptake zone)"
    elif ph <= 8.5:
        ph_status = "Moderately Alkaline (may require sulfur or gypsum)"
    else:
        ph_status = "Strongly Alkaline (salinity risk)"

    # NPK Balance
    fertility_level = "Balanced"
    if n > 100:
        n_desc = "High Nitrogen (promotes vigorous vegetative foliage)"
    elif n < 40:
        n_desc = "Low Nitrogen (favorable for nitrogen-fixing legumes)"
    else:
        n_desc = "Moderate Nitrogen (balanced fertility)"

    # Moisture / Climate
    if rainfall > 180:
        water_regime = "High Rainfall / Wetland Zone"
    elif rainfall >= 90:
        water_regime = "Moderate Rainfall / Semi-Humid Zone"
    else:
        water_regime = "Low Rainfall / Semi-Arid Zone"

    if temp > 30:
        climate_zone = "Hot Tropical"
    elif temp >= 20:
        climate_zone = "Warm Temperate / Sub-tropical"
    else:
        climate_zone = "Cool Temperate"

    return {
        "ph_status": ph_status,
        "n_desc": n_desc,
        "water_regime": water_regime,
        "climate_zone": climate_zone,
        "humidity_zone": "High Humidity" if humidity > 70 else ("Moderate Humidity" if humidity >= 40 else "Low Humidity (Arid)")
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/metadata", methods=["GET"])
def get_metadata():
    """Return model specifications, feature boundaries, and registered classes."""
    return jsonify({
        "status": "success",
        "model_name": metadata.get("model", "GaussianNB Pipeline"),
        "features": metadata.get("features", []),
        "feature_ranges": metadata.get("feature_ranges", {}),
        "classes": metadata.get("classes", []),
        "accuracy": {
            "cv": round(metadata.get("cv_accuracy", 0.0) * 100, 2),
            "test": round(metadata.get("test_accuracy", 0.0) * 100, 2)
        },
        "crop_details": CROP_DETAILS,
        "presets": PRESETS
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """Execute machine learning inference through the saved pipeline."""
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "No JSON payload provided"}), 400

        # Validate required features
        features = metadata.get("features", ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
        input_values = {}
        for feat in features:
            if feat not in data:
                return jsonify({"status": "error", "message": f"Missing required parameter: '{feat}'"}), 400
            try:
                val = float(data[feat])
                input_values[feat] = val
            except (ValueError, TypeError):
                return jsonify({"status": "error", "message": f"Invalid numeric value for '{feat}'"}), 400

        # Construct DataFrame with identical column names fitted to the pipeline
        df = pd.DataFrame([input_values])

        # Execute scikit-learn pipeline inference
        probabilities = model.predict_proba(df)[0]
        classes = label_encoder.classes_

        # Sort all class probabilities in descending order
        sorted_indices = np.argsort(probabilities)[::-1]
        primary_crop = classes[sorted_indices[0]]
        primary_prob = float(probabilities[sorted_indices[0]] * 100)

        # Build top 5 recommendation breakdown
        top_recommendations = []
        for idx in sorted_indices[:5]:
            crop_name = classes[idx]
            prob = float(probabilities[idx] * 100)
            details = CROP_DETAILS.get(crop_name, {})
            top_recommendations.append({
                "crop": crop_name,
                "title": details.get("title", crop_name.capitalize()),
                "icon": details.get("icon", "🌾"),
                "category": details.get("category", "Agricultural Crop"),
                "probability": round(prob, 2),
                "season": details.get("season", "Seasonal"),
                "soil_type": details.get("soil_type", "Standard Soil")
            })

        # Calculate agronomic assessment
        assessment = assess_parameters(input_values)
        primary_details = CROP_DETAILS.get(primary_crop, {
            "title": primary_crop.capitalize(),
            "icon": "🌾",
            "category": "Agricultural Crop",
            "soil_type": "Loam to clay loam",
            "season": "Seasonal",
            "growth_period": "90 - 120 days",
            "key_notes": "Well-suited for the given soil and environmental conditions."
        })

        return jsonify({
            "status": "success",
            "prediction": {
                "recommended_crop": primary_crop,
                "confidence": round(primary_prob, 2),
                "details": primary_details,
                "top_recommendations": top_recommendations,
                "agronomic_assessment": assessment,
                "inputs": input_values
            }
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    # Start local server on port 5000
    app.run(host="127.0.0.1", port=5000, debug=True)
