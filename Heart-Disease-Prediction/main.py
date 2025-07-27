from flask import Flask, render_template, request  # Importing libraries
import pickle  # To load the model
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)  # Creating app

# Load the logistic regression model
model = pickle.load(open("Logistic_regression_model.pkl", "rb"))

@app.route('/', methods=['GET'])  # Render the homepage
def home():
    return render_template('index.html')  # Push UI or HTML code

# Start Preprocessing
standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])  # Collect inputs
def predict():
    try:
        if request.method == 'POST':
            # Collecting form data
            sop = int(request.form['sop'])
            thal_normal = request.form['thal_normal']
            if thal_normal == 'normal':
                thal_normal = 2
                thal_reversible_defect = 1
                thal_fixed_defect = 0
            else:
                thal_normal = 0
                thal_reversible_defect = 2
                thal_fixed_defect = 1

            resting_bp = float(request.form['resting_bp'])
            cpt = int(request.form['cpt'])
            major_vessels = int(request.form['major_vessels'])
            fasting_blood_sugar = int(request.form['fasting_blood_sugar'])
            ekg_result = int(request.form['ekg_result'])
            serum_cholesterol = float(request.form['serum_cholesterol'])
            oldpeak_st_depression = float(request.form['oldpeak_st_depression'])
            sex = int(request.form['sex'])
            age = float(request.form['age'])
            max_heart_rate = float(request.form['max_heart_rate'])
            exercise_induced_angina = int(request.form['exercise_induced_angina'])

            # Prepare input for prediction
            input_data = np.array([[sop, thal_normal, resting_bp, cpt, major_vessels,
                                    fasting_blood_sugar, ekg_result, serum_cholesterol,
                                    oldpeak_st_depression, sex, age, max_heart_rate,
                                    exercise_induced_angina]]).reshape(1, 13)

            # Make prediction
            prediction = model.predict(input_data)
            output = round(prediction[0], 2)

            # Generate output message
            if output == 0:
                pred_message = "The Patient Has No Heart Disease"
            else:
                pred_message = "The Patient Has Heart Disease"

            return render_template('index.html', pred=pred_message)

    except Exception as e:
        return render_template('index.html', pred=f"An error occurred: {str(e)}")

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)  # Run the app in debug mode
