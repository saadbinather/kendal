from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

app = Flask(__name__, static_folder='.')
CORS(app)

# --- CONFIGURATION ---
MODEL_PATH = 'model.pkl'
COLUMNS_PATH = 'feature_columns.pkl'

model = None
feature_columns = None

# --- RISK CONFIGURATION ---
RISK_CONFIG = {
    'Low': {
        'label_en': 'Low Risk', 'label_ur': 'کم خطرہ', 
        'color': '#28a745', 
        'action_en': 'Monitor local weather updates.', 
        'action_ur': 'مقامی موسم کی صورتحال پر نظر رکھیں۔'
    },
    'Medium': {
        'label_en': 'Medium Risk', 'label_ur': 'معتدل خطرہ', 
        'color': '#ffc107', 
        'action_en': 'Secure outdoor items and stay alert.', 
        'action_ur': 'باہر موجود اشیاء کو محفوظ کریں اور الرٹ رہیں۔'
    },
    'High': {
        'label_en': 'High Risk', 'label_ur': 'زیادہ خطرہ', 
        'color': '#fd7e14', 
        'action_en': 'Prepare emergency kits and check evacuation routes.', 
        'action_ur': 'ہنگامی کٹس تیار کریں اور انخلاء کے راستوں کی جانچ کریں۔'
    },
    'Critical': {
        'label_en': 'Critical Risk', 'label_ur': 'انتہائی خطرناک', 
        'color': '#dc3545', 
        'action_en': 'IMMEDIATE EVACUATION RECOMMENDED!', 
        'action_ur': 'فوری انخلاء کی سفارش کی جاتی ہے!'
    }
}

DISTRICT_LOOKUP = {
    'Nowshera': {'elev': 285, 'press': 1005, 'hum': 75, 'temp': 28, 'water': 45.2, 'terrain': 2},
    'Sindh_District': {'elev': 109, 'press': 1002, 'hum': 80, 'temp': 33, 'water': 88.7, 'terrain': 3},
    'Balochistan_District': {'elev': 610, 'press': 998, 'hum': 45, 'temp': 25, 'water': 12.1, 'terrain': 1},
    'KP_District': {'elev': 285, 'press': 1004, 'hum': 70, 'temp': 26, 'water': 38.4, 'terrain': 2},
    'Jacobabad': {'elev': 57, 'press': 1001, 'hum': 55, 'temp': 36, 'water': 67.3, 'terrain': 3},
}

DISTRICT_UI_MAP = {
    'Sindh District': 'Sindh_District',
    'Balochistan District': 'Balochistan_District',
    'KP District': 'KP_District',
    'Nowshera': 'Nowshera',
    'Jacobabad': 'Jacobabad',
}

POPULATION = {
    'Nowshera': 1500000, 'Sindh_District': 3200000, 'Balochistan_District': 2100000,
    'KP_District': 2800000, 'Jacobabad': 1100000,
}

# --- CORE FUNCTIONS ---

def load_model_assets():
    global model, feature_columns
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(COLUMNS_PATH):
            model = joblib.load(MODEL_PATH)
            feature_columns = joblib.load(COLUMNS_PATH)
            print("✅ Model & Feature Columns loaded successfully.")
            return True
        else:
            print("❌ Error: model.pkl or feature_columns.pkl missing.")
            return False
    except Exception as e:
        print(f"❌ Loading Error: {e}")
        return False

def prepare_features(ui_data):
    rain = float(ui_data['rainfall'] or 0)
    dist_name = ui_data['district']
    dist_key = DISTRICT_UI_MAP.get(dist_name, dist_name)
    
    # Safety fallback if district not in lookup
    d = DISTRICT_LOOKUP.get(dist_key, DISTRICT_LOOKUP['Sindh_District'])
    
    date_obj = datetime.strptime(ui_data['date'], '%Y-%m-%d')
    soil_map = {'Dry': 0.2, 'Moist': 0.5, 'Saturated': 0.9}
    soil_val = soil_map.get(ui_data['soil'], 0.5)

    feat = {
        'precipitation': rain, 'precip_lag1': rain, 'precip_3day_avg': rain,
        'soil_moisture': soil_val, 'soil_3day_avg': max(0, soil_val - 0.1),
        'water_area_lag1': d['water'] if ui_data['water'] == 'Yes' else 0.0,
        'water_change_lag1': 5.0 if ui_data['water'] == 'Yes' else 0.0,
        'pressure': d['press'], 'humidity': d['hum'], 'temperature': d['temp'],
        'month': date_obj.month, 'day_of_year': date_obj.timetuple().tm_yday,
        'is_monsoon': 1 if 6 <= date_obj.month <= 9 else 0,
        'saturation_impact': rain * soil_val,
        'elevation_risk': rain / (d['elev'] + 1),
        'terrain_danger_score': d['terrain'],
        'loc_Nowshera': 1 if dist_key == 'Nowshera' else 0,
        'loc_Sindh_District': 1 if dist_key == 'Sindh_District' else 0,
        'loc_Balochistan_District': 1 if dist_key == 'Balochistan_District' else 0,
        'loc_KP_District': 1 if dist_key == 'KP_District' else 0,
        'loc_Jacobabad': 1 if dist_key == 'Jacobabad' else 0,
    }
    
    df_row = pd.DataFrame([feat])
    return df_row.reindex(columns=feature_columns, fill_value=0)

def classify_risk(prob):
    if prob < 0.25: return 'Low'
    elif prob < 0.50: return 'Medium'
    elif prob < 0.75: return 'High'
    else: return 'Critical'

# --- ROUTES ---

@app.route('/')
def serve_index():
    """Serves the frontend index.html from the root directory."""
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded on server'}), 500
    
    try:
        data = request.get_json(force=True)
        X = prepare_features(data)
        
        # 1. Model Prediction
        proba = model.predict_proba(X)[0]
        p = float(proba[1]) if len(proba) > 1 else float(proba[0])
        
        # 2. Heuristic Overrides (Scenario Cards)
        rain_val = float(data['rainfall'] or 0)
        if rain_val > 250: p = max(p, 0.94)
        if data['soil'] == 'Saturated' and rain_val > 100: p = max(p, 0.85)
        if data['water'] == 'Yes': p = max(p, 0.88)
        if data['water'] == 'No' and rain_val < 20 and data['soil'] == 'Dry': p = min(p, 0.10)

        # 3. Response Construction
        risk_level = classify_risk(p)
        config = RISK_CONFIG[risk_level]
        dist_key = DISTRICT_UI_MAP.get(data['district'], 'Sindh_District')
        pop_base = POPULATION.get(dist_key, 1000000)
        
        pop_mult = {'Low': 0, 'Medium': 0.05, 'High': 0.18, 'Critical': 0.42}
        
        return jsonify({
            'status': 'success',
            'risk_level': risk_level,
            'label_en': config['label_en'],
            'label_ur': config['label_ur'],
            'color': config['color'],
            'action_en': config['action_en'],
            'action_ur': config['action_ur'],
            'probability': round(p, 4),
            'confidence_pct': round(p * 100, 1) if p > 0.5 else round((1-p)*100, 1),
            'population_at_risk': int(pop_base * pop_mult[risk_level]),
            'historical_match': 1 if p > 0.6 else 0 # Mock ground truth for UI
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

# --- MAIN ---

if __name__ == '__main__':
    load_model_assets()
    # Deployment environment uses PORT, local uses 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)