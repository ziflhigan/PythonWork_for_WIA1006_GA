from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Print the form data for debugging
    print(request.form)

    # Get user input from the form
    user_input = [float(val) for val in request.form.values()]

    # Convert the user input into a NumPy array and reshape it
    user_input_array = np.array(user_input).reshape(1, -1)
    user_input_array_scaled = scaler.transform(user_input_array)

    # Predict the probability of the presence of heart disease using the trained model
    probabilities = model.predict_proba(user_input_array_scaled)
    heart_disease_probability = probabilities[0][1] * 100

    # Return the probability result as a string
    return f"The probability of the presence of heart disease is {heart_disease_probability:.2f}%"


if __name__ == '__main__':
    app.run(debug=True)
