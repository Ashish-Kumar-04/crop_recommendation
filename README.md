# 🌱 AgroAI - Precision Crop Recommendation System

A modern precision agriculture decision support system powered by Scikit-Learn machine learning and an interactive **Streamlit** frontend.

AgroAI analyzes soil chemistry (Nitrogen, Phosphorus, Potassium) and local weather metrics (Temperature, Humidity, Soil pH, Rainfall) to recommend the optimal crop for maximum harvest yield.

---

## 🚀 Quick Start (Streamlit Frontend)

To launch the Streamlit web application:

```bash
streamlit run streamlit_app.py
```

*(You can also run `streamlit run app.py` — it will automatically dispatch to the Streamlit app!)*

The application will be accessible at:
```
http://localhost:8501
```

---

## ✨ Features & Capabilities

1. **🌾 Crop Recommendation Studio**:
   - **Interactive Soil Nutrients (NPK)**: Precision number inputs and sliders for Nitrogen (0-140), Phosphorus (5-145), and Potassium (5-205).
   - **Climatic Parameters**: Real-time sliders for Temperature (°C), Relative Humidity (%), Soil pH (with dynamic acidity/alkalinity classification), and Rainfall (mm).
   - **⚡ 1-Click Agricultural Field Presets**: Quick-load scenarios including *Monsoon Rice Paddy*, *Highland Coffee Estate*, *Deccan Cotton Plains*, *Temperate Apple Orchard*, *Arid Chickpea Field*, and *Summer Watermelon Basin*.
   - **🏆 Primary Crop Recommendation Hero Card**:
     - Visual crop badge & scientific classification.
     - Prediction confidence percentage.
     - Recommended growing season, ideal soil type, growth cycle duration, and water regime.
     - Agronomic farming guidance and management notes.
   - **📊 Top-5 Candidate Crop Likelihoods**: Horizontal probability distribution breakdown for alternative crop considerations.
   - **🔬 Soil & Climate Health Diagnostics**:
     - Soil pH status (acidic / neutral / alkaline nutrient uptake zone).
     - Nitrogen vegetative foliage assessment.
     - Precipitation / water regime categorization.
     - Thermal and atmospheric moisture zones.

2. **📚 22-Crop Agronomic Directory**:
   - Interactive search and category filter (Cereals, Fruits, Pulses, Fiber, Cash Crops).
   - Comprehensive profiles for all 22 supported crops with soil specs, seasonality, and characteristics.

---

## 🏗️ Technical Architecture

- **Machine Learning Pipeline**: `sklearn.pipeline.Pipeline(StandardScaler(), GaussianNB())`
- **Encoder**: `sklearn.preprocessing.LabelEncoder` (22 agricultural classes)
- **Frontend**: Streamlit 1.58+ with custom responsive Emerald/Slate styling
- **Alternative Flask API**: `python app.py` (serves the Flask REST endpoints on port 5000)
