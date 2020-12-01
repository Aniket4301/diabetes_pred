from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('modelForPrediction.sav', 'rb'))

dataset = pd.read_csv('diabetes.csv')

set_x = dataset.iloc[:,[1,2,5,7]].values

from sklearn.preprocessing import MinMaxScaler
s = MinMaxScaler(feature_range=(0, 1))
data_scaled = s.fit_transform(set_x)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict",methods=['POST'])

def predict():

    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)

    if prediction ==1:
        pred = "you have Diabetes"
    elif prediction ==0:
        pred = "you don't have Diabetes"

    output = pred


    return render_template('index.html', prediction_text='{}'.format(output))



if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
    app.run(debug=True) # running the app