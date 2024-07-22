from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and label encoder
try:
    model_path = r'C:\Users\Abhishek\Downloads\Major Project Disease Predictor Backend\disease_prediction_model.pkl'
    label_encoder_path = r'C:\Users\Abhishek\Downloads\Major Project Disease Predictor Backend\label_encoder.pkl'
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)
except Exception as e:
    raise Exception(f"Error loading model or label encoder: {e}")

# Define feature names (assuming you have them from the training data)
symptoms = [
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering", "chills", "joint_pain", 
    "stomach_pain", "acidity", "ulcers_on_tongue", "muscle_wasting", "vomiting", "burning_micturition", 
    "spotting_urination", "fatigue", "weight_gain", "anxiety", "cold_hands_and_feets", "mood_swings", 
    "weight_loss", "restlessness", "lethargy", "patches_in_throat", "irregular_sugar_level", "cough", 
    "high_fever", "sunken_eyes", "breathlessness", "sweating", "dehydration", "indigestion", "headache", 
    "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "pain_behind_the_eyes", "back_pain", 
    "constipation", "abdominal_pain", "diarrhoea", "mild_fever", "yellow_urine", "yellowing_of_eyes", 
    "acute_liver_failure", "fluid_overload", "swelling_of_stomach", "swelled_lymph_nodes", "malaise", 
    "blurred_and_distorted_vision", "c:\$Recycle.Bin\S-1-5-21-3830242582-982210628-1531626654-1001\$R63E9MY.pkl", "throat_irritation", "redness_of_eyes", "sinus_pressure", 
    "runny_nose", "congestion", "chest_pain", "weakness_in_limbs", "fast_heart_rate", 
    "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool", "irritation_in_anus", "neck_pain", 
    "dizziness", "cramps", "bruising", "obesity", "swollen_legs", "swollen_blood_vessels", "puffy_face_and_eyes", 
    "enlarged_thyroid", "brittle_nails", "swollen_extremeties", "excessive_hunger", "extra_marital_contacts", 
    "drying_and_tingling_lips", "slurred_speech", "knee_pain", "hip_joint_pain", "muscle_weakness", 
    "stiff_neck", "swelling_joints", "movement_stiffness", "spinning_movements", "loss_of_balance", 
    "unsteadiness", "weakness_of_one_body_side", "loss_of_smell", "bladder_discomfort", 
    "foul_smell_of_urine", "continuous_feel_of_urine", "passage_of_gases", "internal_itching", 
    "toxic_look_(typhos)", "depression", "irritability", "muscle_pain", "altered_sensorium", 
    "red_spots_over_body", "belly_pain", "abnormal_menstruation", "dischromic_patches", 
    "watering_from_eyes", "increased_appetite", "polyuria", "family_history", "mucoid_sputum", 
    "rusty_sputum", "lack_of_concentration", "visual_disturbances", "receiving_blood_transfusion", 
    "receiving_unsterile_injections", "coma", "stomach_bleeding", "distention_of_abdomen", 
    "history_of_alcohol_consumption", "fluid_overload", "blood_in_sputum", "prominent_veins_on_calf", 
    "palpitations", "painful_walking", "pus_filled_pimples", "blackheads", "scurring", 
    "skin_peeling", "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails", 
    "blister", "red_sore_around_nose", "yellow_crust_ooze"
]

symptom_dict = {index: symptom for index, symptom in enumerate(symptoms)}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        symptoms_provided = data.get('symptoms', [])

        if not symptoms_provided:
            return jsonify({'error': 'No symptoms provided'}), 400

        # Create a zero vector for all features
        input_vector = np.zeros(len(symptoms))

        # Set the vector entries to 1 for the provided symptoms
        for symptom in symptoms_provided:
            if symptom in symptoms:
                index = symptoms.index(symptom)
                input_vector[index] = 1
            else:
                return jsonify({'error': f'Invalid symptom: {symptom}'}), 400

        # Reshape the vector for prediction
        input_vector = input_vector.reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_vector)
        predicted_disease = label_encoder.inverse_transform(prediction)
        
        return jsonify({'predicted_disease': predicted_disease[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
