/**
 * Crop Recommendation Application Logic
 * Pure manual input fields, real ML inference via Flask API,
 * dynamic probability animation, and report copying.
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const form = document.getElementById('predictionForm');
  const submitBtn = document.getElementById('submitBtn');
  const clearBtn = document.getElementById('clearBtn');
  const copySummaryBtn = document.getElementById('copySummaryBtn');
  const toastMsg = document.getElementById('toastMsg');

  // Result UI Elements
  const initialState = document.getElementById('initialState');
  const loadingState = document.getElementById('loadingState');
  const resultContent = document.getElementById('resultContent');
  const statusTag = document.getElementById('statusTag');

  // Hero Card Fields
  const heroCropIcon = document.getElementById('heroCropIcon');
  const heroCropCategory = document.getElementById('heroCropCategory');
  const heroConfidenceScore = document.getElementById('heroConfidenceScore');
  const heroCropName = document.getElementById('heroCropName');
  const heroCropNotes = document.getElementById('heroCropNotes');

  // Specs
  const specSoil = document.getElementById('specSoil');
  const specSeason = document.getElementById('specSeason');
  const specDuration = document.getElementById('specDuration');
  const specPhStatus = document.getElementById('specPhStatus');

  // Result Lists
  const distributionList = document.getElementById('distributionList');
  const insightsChips = document.getElementById('insightsChips');

  // State
  let latestPredictionData = null;

  // Features list
  const features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'];

  /* ==========================================================================
     Clear Inputs Button
     ========================================================================== */
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      features.forEach((feat) => {
        const input = document.getElementById(`input_${feat}`);
        if (input) input.value = '';
      });
      // Reset view to initial state
      resultContent.classList.add('hidden');
      loadingState.classList.add('hidden');
      initialState.classList.remove('hidden');
      statusTag.textContent = 'Ready';
      statusTag.classList.remove('active');
    });
  }

  /* ==========================================================================
     Form Submit & Inference Request
     ========================================================================== */
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    executePrediction();
  });

  async function executePrediction() {
    // Validate and build payload
    const payload = {};
    for (const feat of features) {
      const input = document.getElementById(`input_${feat}`);
      if (!input || input.value.trim() === '' || isNaN(parseFloat(input.value))) {
        showToast(`Please enter a valid number for ${feat}`);
        if (input) input.focus();
        return;
      }
      payload[feat] = parseFloat(input.value);
    }

    // Set UI to loading state
    initialState.classList.add('hidden');
    resultContent.classList.add('hidden');
    loadingState.classList.remove('hidden');
    submitBtn.classList.add('loading');
    submitBtn.disabled = true;
    statusTag.textContent = 'Processing...';
    statusTag.classList.remove('active');

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.message || 'Server error occurred during prediction.');
      }

      const resData = await response.json();
      if (resData.status === 'success') {
        latestPredictionData = resData.prediction;
        displayResults(resData.prediction);
      } else {
        throw new Error(resData.message || 'Prediction calculation failed');
      }
    } catch (err) {
      console.error(err);
      initialState.classList.remove('hidden');
      initialState.querySelector('h3').textContent = 'Prediction Error';
      initialState.querySelector('p').textContent = err.message;
      statusTag.textContent = 'Error';
    } finally {
      loadingState.classList.add('hidden');
      submitBtn.classList.remove('loading');
      submitBtn.disabled = false;
    }
  }

  /* ==========================================================================
     Display Real ML Prediction Results
     ========================================================================== */
  function displayResults(data) {
    const details = data.details || {};
    const assessment = data.agronomic_assessment || {};

    // Hero details
    heroCropIcon.textContent = details.icon || '🌾';
    heroCropCategory.textContent = details.category || 'Agricultural Crop';
    heroCropName.textContent = details.title || data.recommended_crop.toUpperCase();
    heroConfidenceScore.textContent = `${data.confidence.toFixed(1)}%`;
    heroCropNotes.textContent = details.key_notes || 'Optimal match for supplied conditions.';

    // Specifications
    specSoil.textContent = details.soil_type || 'Loam / Clay loam';
    specSeason.textContent = details.season || 'Seasonal';
    specDuration.textContent = details.growth_period || '90 - 120 days';
    specPhStatus.textContent = assessment.ph_status || 'Balanced pH';

    // Top Candidates Probability Distribution
    distributionList.innerHTML = '';
    (data.top_recommendations || []).forEach((item, index) => {
      const row = document.createElement('div');
      row.className = 'prob-row';
      const pct = item.probability.toFixed(1);
      row.innerHTML = `
        <div class="prob-meta">
          <span class="prob-crop-name">
            <span>${item.icon || '🌱'}</span>
            <strong>${item.title || item.crop}</strong>
            <span style="font-size:0.75rem; color:var(--text-muted); font-weight:400;">(${item.category})</span>
          </span>
          <span class="prob-score">${pct}%</span>
        </div>
        <div class="prob-track">
          <div class="prob-bar" style="width: 0%"></div>
        </div>
      `;
      distributionList.appendChild(row);

      // Trigger animated width on next tick
      requestAnimationFrame(() => {
        setTimeout(() => {
          const bar = row.querySelector('.prob-bar');
          if (bar) bar.style.width = `${Math.max(pct, 1)}%`;
        }, 40 * index);
      });
    });

    // Agronomic Diagnostics Chips
    insightsChips.innerHTML = '';
    const diagnosticPoints = [
      { icon: '🧪', label: 'Soil Fertility', text: assessment.n_desc },
      { icon: '⚖️', label: 'Soil Acidity (pH)', text: assessment.ph_status },
      { icon: '🌧️', label: 'Water Regime', text: assessment.water_regime },
      { icon: '🌡️', label: 'Climate Zone', text: `${assessment.climate_zone} (${assessment.humidity_zone})` }
    ];

    diagnosticPoints.forEach((point) => {
      if (point.text) {
        const item = document.createElement('div');
        item.className = 'insight-chip';
        item.innerHTML = `
          <span class="insight-icon">${point.icon}</span>
          <div class="insight-text">
            <strong>${point.label}:</strong> ${point.text}
          </div>
        `;
        insightsChips.appendChild(item);
      }
    });

    // Status Badge
    statusTag.textContent = 'Predicted';
    statusTag.classList.add('active');

    // Reveal Result Card
    resultContent.classList.remove('hidden');
  }

  /* ==========================================================================
     Copy Summary to Clipboard
     ========================================================================== */
  if (copySummaryBtn) {
    copySummaryBtn.addEventListener('click', () => {
      if (!latestPredictionData) return;

      const crop = latestPredictionData.recommended_crop.toUpperCase();
      const conf = latestPredictionData.confidence;
      const inputs = latestPredictionData.inputs;

      const summaryText = `[AgroAI Crop Recommendation Report]
Recommended Crop: ${crop} (Confidence: ${conf}%)

Field Parameters:
- Nitrogen (N): ${inputs.N}
- Phosphorus (P): ${inputs.P}
- Potassium (K): ${inputs.K}
- Temperature: ${inputs.temperature} °C
- Humidity: ${inputs.humidity} %
- Soil pH: ${inputs.ph}
- Rainfall: ${inputs.rainfall} mm

Agronomic Details:
- Ideal Soil: ${latestPredictionData.details?.soil_type || 'N/A'}
- Season: ${latestPredictionData.details?.season || 'N/A'}
- Notes: ${latestPredictionData.details?.key_notes || 'Optimal match.'}`;

      navigator.clipboard.writeText(summaryText)
        .then(() => showToast('Summary copied to clipboard!'))
        .catch(() => showToast('Failed to copy.'));
    });
  }

  function showToast(msg) {
    if (!toastMsg) return;
    toastMsg.textContent = msg;
    setTimeout(() => {
      toastMsg.textContent = '';
    }, 3000);
  }

  // Initial execution with default form values
  executePrediction();
});
