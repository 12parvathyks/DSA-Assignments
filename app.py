import warnings
from flask import Flask, render_template, request
import pickle
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder

# Suppress sklearn warning for version mismatch
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

model = None
onehot_encoder = None
selected_columns = None

try:
    # Load model
    with open('model_reg.pkl', 'rb') as f:
        model = pickle.load(f)

    # Load OneHotEncoder
    with open('onehot_encoder.pkl', 'rb') as f:
        onehot_encoder = pickle.load(f)

    # Load selected columns
    with open('selected_columns.pkl', 'rb') as f:
        selected_columns = pickle.load(f)

except Exception as e:
    print(f"Error loading model or encoder: {str(e)}")

cities = ['Ahmedabad', 'Aizawl', 'Amaravati', 'Amritsar', 'Bengaluru', 'Bhopal', 'Brajrajnagar',
          'Chandigarh', 'Chennai', 'Coimbatore', 'Delhi', 'Ernakulam', 'Gurugram', 'Guwahati',
          'Hyderabad', 'Jaipur', 'Jorapokhar', 'Kochi', 'Lucknow', 'Mumbai', 'Patna', 'Shillong',
          'Talcher', 'Thiruvananthapuram', 'Visakhapatnam']

@app.route('/')
def home():
    return render_template('index.html', cities=cities)

@app.route('/predict', methods=['POST'])
def predict():
    global model, onehot_encoder, selected_columns

    if request.method == 'POST':
        try:
            if model is None or onehot_encoder is None or selected_columns is None:
                return render_template('error.html', error="Model or encoder not loaded. Please try again later.")

            PM25 = float(request.form['PM25'])
            City = request.form['City']
            PM10 = float(request.form['PM10'])
            NO = float(request.form['NO'])
            NO2 = float(request.form['NO2'])
            NOx = float(request.form['NOx'])
            CO = float(request.form['CO'])
            SO2 = float(request.form['SO2'])
            Toluene = float(request.form['Toluene'])

            # Encode city
            city_encoded = onehot_encoder.transform(np.array([[City]]))

            # Convert city_encoded to dense array if it's sparse
            if isinstance(city_encoded, csr_matrix):
                city_encoded = city_encoded.toarray()

            # Prepare input for prediction
            value = np.array([[PM25, PM10, NO, NO2, NOx, CO, SO2, Toluene]])

            # Concatenate city_encoded and value
            input_data = np.concatenate([city_encoded, value], axis=1)

            # Make prediction
            my_prediction = model.predict(input_data)

            # Return result template
            return render_template('result.html', prediction=my_prediction[0], city=City)

        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            return render_template('error.html', error=error_message)

@app.route('/fill_null_aqi', methods=['GET'])
def fill_null_aqi():
    global model, selected_columns

    if model is None or selected_columns is None:
        return render_template('error.html', error="Model or selected columns not loaded. Please try again later.")

    try:
        # Load data
        df = pd.read_csv('city_day.csv')

        # Preprocess data
        df = preprocess_data(df)

        # Fill null AQI values
        predicted_aqi = fill_null_aqi(df, model, selected_columns)

        # Update original dataframe with predicted AQI values
        df.loc[df['AQI'].isna(), 'AQI'] = predicted_aqi

        # Save updated dataframe as CSV
        df.to_csv('city_day_predicted.csv', index=False)

        return "Null AQI values filled and saved successfully."

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return render_template('error.html', error=error_message)

if __name__ == '__main__':
    app.run(debug=True)
