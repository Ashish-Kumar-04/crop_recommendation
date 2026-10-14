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
- **Frontend**: Streamlit 1.58+ with custom responsive Light Blue theme
- **Alternative Flask API**: `python app.py` (serves the Flask REST endpoints on port 5000)

---

## 🌐 Deploy to Streamlit Community Cloud (Free & 1-Click)

1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "Configure requirements and deployment files for Streamlit Cloud"
   git push origin main
   ```

2. **Sign in to Streamlit Community Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.

3. **Deploy the App**:
   - Click **"Create app"** (or **"New app"**).
   - Select your repository: `Ashish-Kumar-04/crop_recommendation` (or your fork/repo).
   - Set **Branch**: `main`.
   - Set **Main file path**: `streamlit_app.py` (or `app.py`).
   - Click **"Deploy!"**.

Streamlit Cloud will automatically detect `requirements.txt`, install dependencies, load the ML pipeline artifacts, and give you a live public URL (e.g., `https://crop-recommendation.streamlit.app`).
