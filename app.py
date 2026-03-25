from flask import Flask, render_template, request
import pickle
import numpy as np
app = Flask(__name__)
model = pickle.load(open('car_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    present_price = float(request.form['Present_Price'])
    kms_driven = int(request.form['Kms_Driven'])
    fuel_type = int(request.form['Fuel_Type'])
    seller_type = int(request.form['Seller_Type'])
    transmission = int(request.form['Transmission'])
    owner = int(request.form['Owner'])
    car_age = int(request.form['Car_Age'])

    
    features = np.array([[present_price, kms_driven, fuel_type, seller_type, transmission, owner, car_age]])
    
    
    prediction = model.predict(features)
    
    
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text=f"Estimated Selling Price: ₹{output} Lakhs")

if __name__ == "__main__":
    app.run(debug=True)
