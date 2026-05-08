from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'model.pkl'
COLUMNS_PATH = 'feature_columns.pkl'

model = None
feature_columns = None

DISTRICT_LOOKUP = {
    'Nowshera':             {'elevation': 285,  'pressure': 1005, 'humidity': 75,
                             'temperature': 28, 'wind_speed': 12, 'water_mean': 45.2,
                             'terrain_danger_score': 2},
    'Sindh_District':       {'elevation': 109,  'pressure': 1002, 'humidity': 80,
                             'temperature': 33, 'wind_speed': 10, 'water_mean': 88.7,
                             'terrain_danger_score': 3},
    'Balochistan_District': {'elevation': 610,  'pressure': 998,  'humidity': 45,
                             'temperature': 25, 'wind_speed': 15, 'water_mean': 12.1,
                             'terrain_danger_score': 1},
    'KP_District':          {'elevation': 285,  'pressure': 1004, 'humidity': 70,
                             'temperature': 26, 'wind_speed': 13, 'water_mean': 38.4,
                             'terrain_danger_score': 2},
    'Jacobabad':            {'elevation': 57,   'pressure': 1001, 'humidity': 55,
                             'temperature': 36, 'wind_speed': 9,  'water_mean': 67.3,
                             'terrain_danger_score': 3},
}

# UI label → internal key
DISTRICT_UI_MAP = {
    'Sindh District':       'Sindh_District',
    'Balochistan District': 'Balochistan_District',
    'KP District':          'KP_District',
    'Nowshera':             'Nowshera',
    'Jacobabad':            'Jacobabad',
}

POPULATION = {
    'Nowshera':             1500000,
    'Sindh_District':       3200000,
    'Balochistan_District': 2100000,
    'KP_District':          2800000,
    'Jacobabad':            1100000,
}

SOIL_MOISTURE_MAP = {'Dry': 0.2,  'Moist': 0.5,  'Saturated': 0.9}
SOIL_3DAY_MAP     = {'Dry': 0.15, 'Moist': 0.4,  'Saturated': 0.8}


def load_model():
    global model, feature_columns
    missing = [p for p in [MODEL_PATH, COLUMNS_PATH] if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"ERROR: '{p}' not found.")
        print("Run the last cell of Team_Kendal.ipynb to export model.pkl and feature_columns.pkl.")
        return False
    try:
        model = joblib.load(MODEL_PATH)
        feature_columns = joblib.load(COLUMNS_PATH)
        print(f"Model loaded successfully. Expecting {len(feature_columns)} features.")
        return True
    except Exception as e:
        print(f"ERROR loading model files: {e}")
        return False


def prepare_features(ui_data):
    rainfall   = float(ui_data['rainfall'])
    date_str   = ui_data['date']
    district   = DISTRICT_UI_MAP.get(ui_data['district'], ui_data['district'])
    soil       = ui_data['soil']
    water      = ui_data['water']

    d         = DISTRICT_LOOKUP[district]
    elevation = d['elevation']

    date_obj   = datetime.strptime(date_str, '%Y-%m-%d')
    month      = date_obj.month
    day_of_year = date_obj.timetuple().tm_yday
    is_monsoon = 1 if month in [6, 7, 8, 9] else 0

    soil_moisture = SOIL_MOISTURE_MAP[soil]
    soil_3day_avg = SOIL_3DAY_MAP[soil]

    water_area_lag1    = d['water_mean'] if water == 'Yes' else 0.0
    water_change_lag1  = 5.0            if water == 'Yes' else 0.0

    saturation_impact = rainfall * soil_moisture
    elevation_risk    = rainfall / (elevation + 1)
    monsoon_intensity = is_monsoon * rainfall  # precip_7day_avg simulated as rainfall

    feature_dict = {
        'precipitation':        rainfall,
        'precip_lag1':          rainfall,
        'precip_lag2':          rainfall,
        'precip_3day_avg':      rainfall,
        'precip_7day_avg':      rainfall,
        'soil_moisture':        soil_moisture,
        'soil_3day_avg':        soil_3day_avg,
        'soil_lag1':            soil_moisture,
        'soil_lag2':            soil_moisture,
        'water_area_lag1':      water_area_lag1,
        'water_change_lag1':    water_change_lag1,
        'temperature':          d['temperature'],
        'humidity':             d['humidity'],
        'pressure':             d['pressure'],
        'wind_speed':           d['wind_speed'],
        'month':                month,
        'day_of_year':          day_of_year,
        'is_monsoon':           is_monsoon,
        'saturation_impact':    saturation_impact,
        'elevation_risk':       elevation_risk,
        'monsoon_intensity':    monsoon_intensity,
        'terrain_danger_score': d['terrain_danger_score'],
        'loc_Nowshera':             1 if district == 'Nowshera'             else 0,
        'loc_Sindh_District':       1 if district == 'Sindh_District'       else 0,
        'loc_Balochistan_District': 1 if district == 'Balochistan_District' else 0,
        'loc_KP_District':          1 if district == 'KP_District'          else 0,
        'loc_Jacobabad':            1 if district == 'Jacobabad'            else 0,
    }

    df = pd.DataFrame([feature_dict])
    # Align to exact column order the model was trained on; fill any gap with 0
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df


def classify_risk(prob):
    if prob < 0.25:   return 'Low'
    elif prob < 0.50: return 'Medium'
    elif prob < 0.75: return 'High'
    else:             return 'Critical'


def calc_population_at_risk(district_key, risk_level):
    pop  = POPULATION.get(district_key, 0)
    mult = {'Low': 0, 'Medium': 0.08, 'High': 0.22, 'Critical': 0.45}
    return int(pop * mult[risk_level])


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or feature_columns is None:
        return jsonify({'error': 'Model not loaded. Run the notebook export cell first.'}), 503

    try:
        data = request.get_json(force=True)

        required = ['rainfall', 'date', 'district', 'soil', 'water']
        for field in required:
            if field not in data or data[field] == '':
                return jsonify({'error': f'Missing required field: {field}'}), 400

        if data['district'] not in DISTRICT_UI_MAP:
            return jsonify({'error': f"Unknown district: {data['district']}"}), 400
        if data['soil'] not in SOIL_MOISTURE_MAP:
            return jsonify({'error': f"Unknown soil value: {data['soil']}"}), 400

        X            = prepare_features(data)
        proba        = model.predict_proba(X)[0]
        flood_prob   = float(proba[1]) if len(proba) > 1 else float(proba[0])
        risk_level   = classify_risk(flood_prob)
        confidence   = round(float(np.max(proba)) * 100, 1)

        district_key = DISTRICT_UI_MAP[data['district']]
        population   = calc_population_at_risk(district_key, risk_level)

        return jsonify({
            'risk_level':        risk_level,
            'probability':       round(flood_prob, 4),
            'confidence_pct':    confidence,
            'population_at_risk': population,
        })

    except KeyError as e:
        return jsonify({'error': f'Invalid input value: {e}'}), 400
    except Exception:
        return jsonify({'error': 'Prediction failed. Check server logs for details.'}), 500


if __name__ == '__main__':
    if load_model():
        app.run(port=5000, debug=False)
    else:
        print("\nServer did not start. Fix the errors above and try again.")
