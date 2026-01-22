from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
with open('reg_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scalar.pkl', 'rb') as f1:
    scaler = pickle.load(f1)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        # Get inputs from form
        age = float(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        duration = float(request.form['duration'])
        heart_rate = float(request.form['heart_rate'])
        body_temp = float(request.form['body_temp'])
        gender = request.form['gender']

        # Convert gender to numeric
        gender_male = 1 if gender == 'male' else 0

        # Scale numerical features
        scaled_features = scaler.transform([[age, height, weight, duration, heart_rate, body_temp]])

        # Combine scaled + unscaled
        final_input = np.append(scaled_features, [[gender_male]], axis=1)
        print(final_input)
        # Predict
        prediction = round(model.predict(final_input)[0], 2)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)