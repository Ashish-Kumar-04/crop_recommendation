"""
AgroAI - Precision Agriculture & Crop Recommendation System
Streamlit Frontend Application (Light Blue Theme)
"""

import json
import textwrap
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Suppress unpickling version warnings between sklearn versions
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

# --- Path Configurations & Artifact Loading ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "crop_pipeline.joblib"
ENCODER_PATH = BASE_DIR / "label_encoder.joblib"
METADATA_PATH = BASE_DIR / "metadata.json"


@st.cache_resource
def load_ml_artifacts():
    """Load and cache the trained ML pipeline, label encoder, and metadata."""
    if not MODEL_PATH.exists() or not ENCODER_PATH.exists() or not METADATA_PATH.exists():
        st.error("Missing ML artifact files (crop_pipeline.joblib, label_encoder.joblib, or metadata.json).")
        st.stop()

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, encoder, meta


model, label_encoder, metadata = load_ml_artifacts()

# --- Comprehensive Agronomic Knowledge Base ---
CROP_DETAILS = {
    "apple": {
        "title": "Apple (Malus domestica)",
        "icon": "🍎",
        "category": "Fruit / Horticultural",
        "soil_type": "Well-drained, loamy to sandy loam",
        "season": "Temperate / High Altitude",
        "growth_period": "120 - 180 days",
        "key_notes": "Requires cooler temperatures (15-24°C) and moderate rainfall with well-aerated soil rich in organic matter."
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

PRESETS = {
    "Monsoon Rice Paddy 🍚": {
        "N": 90.0,
        "P": 42.0,
        "K": 43.0,
        "temperature": 23.5,
        "humidity": 82.5,
        "ph": 6.5,
        "rainfall": 230.0,
        "description": "High moisture, heavy monsoon rainfall, high nitrogen in warm tropical valley."
    },
    "Highland Coffee Estate ☕": {
        "N": 105.0,
        "P": 25.0,
        "K": 30.0,
        "temperature": 25.5,
        "humidity": 65.0,
        "ph": 6.8,
        "rainfall": 170.0,
        "description": "High rainfall, cool hill elevation, high phosphorus and potassium."
    },
    "Deccan Cotton Plains ☁️": {
        "N": 118.0,
        "P": 45.0,
        "K": 20.0,
        "temperature": 24.0,
        "humidity": 80.0,
        "ph": 6.9,
        "rainfall": 80.0,
        "description": "High nitrogen, warm climate, moderate rainfall with high potassium in black soil."
    },
    "Temperate Apple Orchard 🍎": {
        "N": 20.0,
        "P": 135.0,
        "K": 198.0,
        "temperature": 22.0,
        "humidity": 92.0,
        "ph": 6.0,
        "rainfall": 110.0,
        "description": "Cool temperate climate, moderate humidity, high P & K reserves in loamy soil."
    },
    "Arid Chickpea Field 🫘": {
        "N": 38.0,
        "P": 68.0,
        "K": 80.0,
        "temperature": 18.5,
        "humidity": 16.5,
        "ph": 7.3,
        "rainfall": 75.0,
        "description": "Modest nitrogen, dry air, low rainfall, winter growing condition."
    },
    "Summer Watermelon Basin 🍉": {
        "N": 98.0,
        "P": 18.0,
        "K": 50.0,
        "temperature": 26.0,
        "humidity": 85.0,
        "ph": 6.5,
        "rainfall": 50.0,
        "description": "Warm summer sun, neutral pH, moderate nitrogen and low moisture in riverbeds."
    }
}


def assess_parameters(features: dict) -> dict:
    """Analyze soil and climatic parameters for agronomic insights."""
    n = features["N"]
    temp = features["temperature"]
    humidity = features["humidity"]
    ph = features["ph"]
    rainfall = features["rainfall"]

    if ph < 5.5:
        ph_status = "Strongly Acidic (may restrict phosphorus availability)"
        ph_color = "#ef4444"
    elif ph < 6.5:
        ph_status = "Slightly Acidic (favorable for most horticultural crops)"
        ph_color = "#0284c7"
    elif ph <= 7.5:
        ph_status = "Neutral (optimal nutrient uptake zone)"
        ph_color = "#0369a1"
    elif ph <= 8.5:
        ph_status = "Moderately Alkaline (may require sulfur or gypsum)"
        ph_color = "#0284c7"
    else:
        ph_status = "Strongly Alkaline (salinity risk)"
        ph_color = "#ef4444"

    if n > 100:
        n_desc = "High Nitrogen (promotes vigorous vegetative canopy)"
    elif n < 40:
        n_desc = "Low Nitrogen (suitable for nitrogen-fixing legumes)"
    else:
        n_desc = "Moderate Nitrogen (balanced vegetative fertility)"

    if rainfall > 180:
        water_regime = "High Rainfall / Wetland Zone"
    elif rainfall >= 90:
        water_regime = "Moderate Rainfall / Semi-Humid Zone"
    else:
        water_regime = "Low Rainfall / Semi-Arid Zone"

    if temp > 30:
        climate_zone = "Hot Tropical Climate"
    elif temp >= 20:
        climate_zone = "Warm Temperate / Sub-tropical"
    else:
        climate_zone = "Cool Temperate Climate"

    if humidity > 70:
        humidity_zone = "High Humidity (>70%)"
    elif humidity >= 40:
        humidity_zone = "Moderate Humidity (40-70%)"
    else:
        humidity_zone = "Low Humidity / Arid Air (<40%)"

    return {
        "ph_status": ph_status,
        "ph_color": ph_color,
        "n_desc": n_desc,
        "water_regime": water_regime,
        "climate_zone": climate_zone,
        "humidity_zone": humidity_zone
    }


# --- Page Configuration & Styling (Completely hide Sidebar & Light Blue Theme) ---
st.set_page_config(
    page_title="AgroAI | Precision Crop Recommendation",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom High-Aesthetic Light Blue Modern Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    /* Completely remove sidebar and toggle */
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"],
    section[data-testid="stSidebar"] {
        display: none !important;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #f0f7ff !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #0c4a6e !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3.5rem;
        max-width: 1240px;
    }

    /* App Header Banner - Light Blue Oceanic / Sky Gradient */
    .hero-banner {
        background: linear-gradient(135deg, #075985 0%, #0284c7 50%, #38bdf8 100%);
        border-radius: 18px;
        padding: 2rem 2.2rem;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(2, 132, 199, 0.28);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
        color: #ffffff !important;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    .hero-title span {
        background: linear-gradient(120deg, #bae6fd, #e0f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-sub {
        font-size: 1.02rem;
        color: #f0f9ff !important;
        margin: 0.4rem 0 0 0;
        opacity: 0.95;
    }

    .hero-tags {
        display: flex;
        gap: 0.65rem;
        margin-top: 0.8rem;
        flex-wrap: wrap;
    }

    .hero-tag {
        background: rgba(255, 255, 255, 0.22);
        backdrop-filter: blur(8px);
        padding: 0.3rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #f0f9ff;
        border: 1px solid rgba(255, 255, 255, 0.35);
    }

    /* Result Hero Card (Light Blue Modern) */
    .result-hero {
        background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 60%, #e0f2fe 100%);
        border: 2px solid #7dd3fc;
        border-radius: 18px;
        padding: 1.8rem;
        box-shadow: 0 8px 24px rgba(2, 132, 199, 0.12);
        margin-bottom: 1.4rem;
    }

    /* Prominent Crop Name Hero Banner */
    .crop-name-header-row {
        display: flex;
        align-items: center;
        gap: 1.2rem;
        padding: 0.8rem 1.2rem;
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        border-radius: 14px;
        color: #ffffff;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.25);
    }

    .crop-name-large-icon {
        font-size: 3.5rem;
        line-height: 1;
    }

    .crop-name-label {
        font-size: 0.75rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #bae6fd;
    }

    .crop-name-main {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff !important;
        margin: 0;
        line-height: 1.15;
    }

    .crop-scientific-sub {
        font-size: 0.95rem;
        color: #e0f2fe;
        font-weight: 500;
        margin-top: 0.2rem;
    }

    .confidence-badge {
        display: inline-flex;
        align-items: center;
        background: #0284c7;
        color: white;
        padding: 0.32rem 0.9rem;
        border-radius: 9999px;
        font-size: 0.88rem;
        font-weight: 700;
    }

    .badge-category {
        display: inline-flex;
        align-items: center;
        background: #e0f2fe;
        color: #0369a1;
        padding: 0.32rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #bae6fd;
        margin-left: 0.4rem;
    }

    /* Metric Tiles Grid */
    .grid-spec {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.85rem;
        margin: 1.2rem 0;
    }

    .spec-item {
        background: #ffffff;
        padding: 0.85rem 1rem;
        border-radius: 12px;
        border: 1px solid #bae6fd;
        box-shadow: 0 2px 6px rgba(2, 132, 199, 0.04);
    }

    .spec-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        font-weight: 700;
        color: #0284c7;
        margin-bottom: 0.2rem;
    }

    .spec-value {
        font-size: 0.98rem;
        font-weight: 600;
        color: #0f172a;
    }

    /* Agronomic Guidance Box */
    .guidance-box {
        background: #ffffff;
        padding: 0.95rem 1.15rem;
        border-radius: 10px;
        color: #1e293b;
        font-size: 0.93rem;
        line-height: 1.5;
        margin-top: 1rem;
        border: 1px solid #bae6fd;
        border-left: 5px solid #0284c7;
    }

    /* Diagnostic Cards */
    .diag-card {
        padding: 0.85rem 1rem;
        border-radius: 12px;
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        margin-bottom: 0.6rem;
    }

    /* Encyclopedia Cards */
    .encyclo-card {
        background: #ffffff;
        border: 1px solid #bae6fd;
        border-radius: 14px;
        padding: 1.2rem;
        height: 100%;
        box-shadow: 0 2px 6px rgba(2, 132, 199, 0.05);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .encyclo-card:hover {
        transform: translateY(-2px);
        border-color: #38bdf8;
    }

    /* Primary Buttons - Sky Blue Gradient */
    button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35) !important;
        font-weight: 700 !important;
        padding: 0.65rem 1.5rem !important;
        border-radius: 10px !important;
        color: white !important;
    }
    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
    }

    /* Tabs Styling */
    div[data-baseweb="tab-list"] {
        gap: 8px;
    }
    button[data-baseweb="tab"] {
        border-radius: 8px 8px 0 0 !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Initialize Session State for Inputs ---
DEFAULT_PARAMS = {
    "N": 90.0,
    "P": 42.0,
    "K": 43.0,
    "temperature": 23.5,
    "humidity": 82.5,
    "ph": 6.5,
    "rainfall": 230.0
}

for key, val in DEFAULT_PARAMS.items():
    if key not in st.session_state:
        st.session_state[key] = val

if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None


def run_prediction():
    """Execute inference using pipeline."""
    input_dict = {
        "N": float(st.session_state["N"]),
        "P": float(st.session_state["P"]),
        "K": float(st.session_state["K"]),
        "temperature": float(st.session_state["temperature"]),
        "humidity": float(st.session_state["humidity"]),
        "ph": float(st.session_state["ph"]),
        "rainfall": float(st.session_state["rainfall"])
    }

    df_input = pd.DataFrame([input_dict])
    probs = model.predict_proba(df_input)[0]
    classes = label_encoder.classes_

    # Sort descending
    sorted_idx = np.argsort(probs)[::-1]
    primary_crop = classes[sorted_idx[0]]
    primary_prob = float(probs[sorted_idx[0]] * 100)

    top_recs = []
    for idx in sorted_idx[:5]:
        cname = classes[idx]
        p = float(probs[idx] * 100)
        c_detail = CROP_DETAILS.get(cname, {})
        top_recs.append({
            "crop": cname,
            "title": c_detail.get("title", cname.capitalize()),
            "icon": c_detail.get("icon", "🌾"),
            "category": c_detail.get("category", "Field Crop"),
            "probability": round(p, 2),
            "season": c_detail.get("season", "Seasonal"),
            "soil_type": c_detail.get("soil_type", "Standard Soil")
        })

    assessment = assess_parameters(input_dict)
    primary_details = CROP_DETAILS.get(primary_crop, {
        "title": primary_crop.capitalize(),
        "icon": "🌾",
        "category": "Agricultural Crop",
        "soil_type": "Loam to clay loam",
        "season": "Seasonal",
        "growth_period": "90 - 120 days",
        "key_notes": "Well-suited for the given soil and environmental conditions."
    })

    st.session_state["prediction_result"] = {
        "recommended_crop": primary_crop,
        "confidence": round(primary_prob, 2),
        "details": primary_details,
        "top_recommendations": top_recs,
        "assessment": assessment,
        "inputs": input_dict
    }


# --- Hero Banner (Light Blue Theme) ---
hero_html = textwrap.dedent("""
<div class="hero-banner">
    <div class="hero-title">
        🌱 AgroAI <span>Crop Recommendation Engine</span>
    </div>
    <p class="hero-sub">
        Precision Agriculture Decision Support: Input your soil nutrients (NPK) and local weather conditions to recommend the optimal high-yielding crop.
    </p>
    <div class="hero-tags">
        <span class="hero-tag">🧪 Soil Nutrients (NPK)</span>
        <span class="hero-tag">🌤️ Weather & Climate Factors</span>
        <span class="hero-tag">🌾 22 Crop Varieties</span>
    </div>
</div>
""").strip()
st.markdown(hero_html, unsafe_allow_html=True)


# --- Tabs Organization ---
tab_rec, tab_encyclo = st.tabs([
    "🌾 Crop Recommendation",
    "📚 22 Crop Directory"
])


# ==============================================================================
# TAB 1: CROP RECOMMENDATION
# ==============================================================================
with tab_rec:
    # Quick preset bar directly on the page
    preset_box = st.container()
    with preset_box:
        col_pres1, col_pres2 = st.columns([3.5, 1])
        with col_pres1:
            preset_options = ["Custom Farm Input"] + list(PRESETS.keys())
            selected_preset = st.selectbox(
                "⚡ Quick Fill with Benchmark Field Conditions:",
                options=preset_options,
                index=0,
                help="Select a benchmark agricultural field to auto-populate the inputs."
            )
        with col_pres2:
            st.write("")
            st.write("")
            if st.button("🔄 Reset Inputs", use_container_width=True):
                for k, v in DEFAULT_PARAMS.items():
                    st.session_state[k] = v
                st.session_state["prediction_result"] = None
                st.rerun()

        if selected_preset != "Custom Farm Input":
            p = PRESETS[selected_preset]
            for k in DEFAULT_PARAMS.keys():
                st.session_state[k] = float(p[k])
            st.info(f"**Loaded Preset ({selected_preset}):** {p['description']}")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    col_inputs, col_results = st.columns([1.1, 1.4], gap="large")

    with col_inputs:
        st.markdown("### 🧪 Soil & Environmental Inputs")
        st.caption("Enter your soil test readings and local climate observations:")

        # --- Section A: Soil Nutrients (NPK) ---
        st.markdown("##### 🌿 Soil Nutrients (NPK)")
        col_n, col_p, col_k = st.columns(3)
        with col_n:
            st.session_state["N"] = st.number_input(
                "Nitrogen (N)",
                min_value=0.0,
                max_value=140.0,
                value=float(st.session_state["N"]),
                step=1.0,
                help="Range: 0 to 140. Leaf growth and vegetative development."
            )
        with col_p:
            st.session_state["P"] = st.number_input(
                "Phosphorus (P)",
                min_value=5.0,
                max_value=145.0,
                value=float(st.session_state["P"]),
                step=1.0,
                help="Range: 5 to 145. Root growth and flower/seed production."
            )
        with col_k:
            st.session_state["K"] = st.number_input(
                "Potassium (K)",
                min_value=5.0,
                max_value=205.0,
                value=float(st.session_state["K"]),
                step=1.0,
                help="Range: 5 to 205. Disease resistance and overall plant health."
            )

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        # --- Section B: Climatic & Environmental Parameters ---
        st.markdown("##### 🌤️ Climate & Weather Parameters")
        col_temp, col_hum = st.columns(2)
        with col_temp:
            st.session_state["temperature"] = st.slider(
                "Temperature (°C)",
                min_value=8.0,
                max_value=45.0,
                value=float(st.session_state["temperature"]),
                step=0.5
            )
        with col_hum:
            st.session_state["humidity"] = st.slider(
                "Relative Humidity (%)",
                min_value=10.0,
                max_value=100.0,
                value=float(st.session_state["humidity"]),
                step=0.5
            )

        col_ph, col_rain = st.columns(2)
        with col_ph:
            st.session_state["ph"] = st.slider(
                "Soil pH",
                min_value=3.5,
                max_value=10.0,
                value=float(st.session_state["ph"]),
                step=0.1
            )
            current_ph = st.session_state["ph"]
            if current_ph < 6.0:
                ph_lbl = "Acidic Soil"
                ph_bg = "#fee2e2"
                ph_fg = "#991b1b"
            elif current_ph <= 7.5:
                ph_lbl = "Neutral / Optimal Soil"
                ph_bg = "#e0f2fe"
                ph_fg = "#0369a1"
            else:
                ph_lbl = "Alkaline Soil"
                ph_bg = "#fef3c7"
                ph_fg = "#92400e"
            st.markdown(
                f"<div style='font-size:0.8rem; font-weight:700; color:{ph_fg}; background:{ph_bg}; "
                f"padding:2px 8px; border-radius:6px; display:inline-block; margin-top:-6px; margin-bottom:8px;'>"
                f"pH {current_ph:.1f}: {ph_lbl}</div>",
                unsafe_allow_html=True
            )

        with col_rain:
            st.session_state["rainfall"] = st.slider(
                "Rainfall (mm)",
                min_value=20.0,
                max_value=300.0,
                value=float(st.session_state["rainfall"]),
                step=1.0
            )

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        if st.button("🌱 Recommend Optimal Crop", type="primary", use_container_width=True):
            run_prediction()

    # --- Right Column: Crop Recommendation Results ---
    with col_results:
        st.markdown("### 🎯 Recommended Crop & Agronomic Insights")

        # Auto-compute initial recommendation if not set
        if st.session_state["prediction_result"] is None:
            run_prediction()

        res = st.session_state["prediction_result"]
        details = res["details"]
        assessment = res["assessment"]
        top_recs = res["top_recommendations"]
        crop_clean_name = res["recommended_crop"].capitalize()

        # Prominent Result Card with CLEAR CROP NAME
        card_html = (
            f'<div class="result-hero">'
            f'<div class="crop-name-header-row">'
            f'<div class="crop-name-large-icon">{details["icon"]}</div>'
            f'<div>'
            f'<div class="crop-name-label">Recommended Crop</div>'
            f'<h1 class="crop-name-main">Crop: {crop_clean_name}</h1>'
            f'<div class="crop-scientific-sub">Variety / Scientific: <strong>{details["title"]}</strong></div>'
            f'</div>'
            f'</div>'
            f'<div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.8rem;">'
            f'<div>'
            f'<span class="confidence-badge">✓ {res["confidence"]:.1f}% Confidence Match</span>'
            f'<span class="badge-category">{details["category"]}</span>'
            f'</div>'
            f'<div style="font-size: 0.95rem; font-weight: 700; color: #0284c7;">'
            f'Status: Prime Cultivation Match'
            f'</div>'
            f'</div>'
            f'<div class="grid-spec">'
            f'<div class="spec-item"><div class="spec-label">🌱 Crop Name</div><div class="spec-value">{crop_clean_name}</div></div>'
            f'<div class="spec-item"><div class="spec-label">🗓️ Best Season</div><div class="spec-value">{details["season"]}</div></div>'
            f'<div class="spec-item"><div class="spec-label">⏳ Growth Cycle</div><div class="spec-value">{details["growth_period"]}</div></div>'
            f'<div class="spec-item"><div class="spec-label">🪨 Optimal Soil</div><div class="spec-value">{details["soil_type"]}</div></div>'
            f'<div class="spec-item"><div class="spec-label">💧 Water Regime</div><div class="spec-value">{assessment["water_regime"]}</div></div>'
            f'</div>'
            f'<div class="guidance-box"><strong>💡 Agronomic Cultivation Notes:</strong><br>{details["key_notes"]}</div>'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

        # Top 5 Alternative Crop Likelihoods
        st.markdown("##### 📊 Top Candidate Crops Comparison")
        chart_data = pd.DataFrame({
            "Crop Name": [f"{r['icon']} {r['crop'].capitalize()}" for r in top_recs],
            "Match Probability (%)": [r["probability"] for r in top_recs]
        }).set_index("Crop Name")
        st.bar_chart(chart_data, color="#0284c7", horizontal=True)

        # Field Health Diagnostics
        with st.expander("🔬 Soil & Environmental Health Diagnostics", expanded=True):
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                diag_1 = (
                    f'<div class="diag-card">'
                    f'<strong style="color: {assessment["ph_color"]};">🧪 Soil pH Assessment</strong><br>'
                    f'<span style="font-size:0.9rem; color:#334155;">{assessment["ph_status"]}</span>'
                    f'</div>'
                    f'<div class="diag-card">'
                    f'<strong style="color: #0284c7;">🌱 Nitrogen (N) Dynamics</strong><br>'
                    f'<span style="font-size:0.9rem; color:#334155;">{assessment["n_desc"]}</span>'
                    f'</div>'
                )
                st.markdown(diag_1, unsafe_allow_html=True)
            with col_d2:
                diag_2 = (
                    f'<div class="diag-card">'
                    f'<strong style="color: #0284c7;">🌧️ Precipitation & Moisture</strong><br>'
                    f'<span style="font-size:0.9rem; color:#334155;">{assessment["water_regime"]} &bull; {assessment["humidity_zone"]}</span>'
                    f'</div>'
                    f'<div class="diag-card">'
                    f'<strong style="color: #0369a1;">☀️ Thermal Growing Zone</strong><br>'
                    f'<span style="font-size:0.9rem; color:#334155;">{assessment["climate_zone"]} ({res["inputs"]["temperature"]:.1f}°C)</span>'
                    f'</div>'
                )
                st.markdown(diag_2, unsafe_allow_html=True)


# ==============================================================================
# TAB 2: CROP ENCYCLOPEDIA & DIRECTORY
# ==============================================================================
with tab_encyclo:
    st.markdown("### 📚 Comprehensive 22-Crop Directory")
    st.write(
        "Explore agricultural requirements, growing seasons, and soil preferences for all 22 supported crops:"
    )

    col_filter1, col_filter2 = st.columns([1.2, 2])
    with col_filter1:
        category_options = ["All Categories", "Fruit / Horticultural", "Pulse / Legume", "Cereal / Grain", "Fiber", "Plantation"]
        selected_cat = st.selectbox("Filter by Category:", category_options)
    with col_filter2:
        search_query = st.text_input("🔍 Search Crop Name or Keyword:", placeholder="e.g., Rice, Chickpea, drought, tropical...")

    filtered_crops = {}
    for crop_id, cdata in CROP_DETAILS.items():
        if selected_cat != "All Categories":
            cat_keyword = selected_cat.split("/")[0].strip().lower()
            if cat_keyword not in cdata["category"].lower():
                continue

        if search_query:
            query = search_query.lower()
            content = f"{crop_id} {cdata['title']} {cdata['category']} {cdata['soil_type']} {cdata['season']} {cdata['key_notes']}".lower()
            if query not in content:
                continue

        filtered_crops[crop_id] = cdata

    st.caption(f"Showing **{len(filtered_crops)}** of 22 crops")

    # Render Crop Cards in 3-column layout without 4-space markdown indent
    crop_cols = st.columns(3)
    for i, (cid, cdata) in enumerate(filtered_crops.items()):
        col_target = crop_cols[i % 3]
        with col_target:
            enc_html = (
                f'<div class="encyclo-card">'
                f'<div style="font-size: 2.2rem; line-height: 1; margin-bottom: 0.3rem;">{cdata["icon"]}</div>'
                f'<h4 style="margin: 0 0 0.2rem 0; color: #0369a1; font-size: 1.15rem;">{cdata["title"]}</h4>'
                f'<div style="display:inline-block; font-size:0.75rem; font-weight:700; color:#0369a1; background:#e0f2fe; padding:2px 8px; border-radius:9999px; margin-bottom:0.6rem;">{cdata["category"]}</div>'
                f'<div style="font-size:0.85rem; color:#334155; margin-bottom:0.3rem;"><strong>🗓️ Season:</strong> {cdata["season"]}</div>'
                f'<div style="font-size:0.85rem; color:#334155; margin-bottom:0.3rem;"><strong>⏳ Growth:</strong> {cdata["growth_period"]}</div>'
                f'<div style="font-size:0.85rem; color:#334155; margin-bottom:0.5rem;"><strong>🪨 Soil:</strong> {cdata["soil_type"]}</div>'
                f'<div style="font-size:0.82rem; color:#64748b; line-height:1.4; border-top:1px dashed #bae6fd; padding-top:0.5rem;">{cdata["key_notes"]}</div>'
                f'</div>'
                f'<div style="height: 14px;"></div>'
            )
            st.markdown(enc_html, unsafe_allow_html=True)


# --- Footer ---
st.markdown("---")
current_year = datetime.now().year
footer_html = (
    f'<div style="text-align: center; color: #0369a1; font-size: 0.9rem; padding: 1rem 0; font-weight: 500;">'
    f'Designed with <span style="color: #ef4444; font-size: 1.1rem;">&hearts;</span> by <strong>Ashish Kumar</strong> &bull; &copy; {current_year} All copyrights reserved'
    f'</div>'
)
st.markdown(footer_html, unsafe_allow_html=True)
