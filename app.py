from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('joseph5.h5')  # Load the trained LSTM model

# Load the dataset and convert the timestamp column
data= pd.read_csv('modified.csv')
data['Gmt time'] = pd.to_datetime(data['Gmt time']).dt.date.apply(lambda x: int(datetime.timestamp(datetime.combine(x, datetime.min.time()))))


array = data.to_numpy(dtype=float)

scaler = MinMaxScaler()  # Create a scaler object for normalizing the data
scaler.fit(array)  # Fit the scaler on the entire dataset

@app.route('/')
def index():
    return render_template('index2.html')


@app.route('/predict')
def predict():
    return render_template('result.html')


@app.route('/predict-high-low-prices', methods=['POST'])
def predict_high_low_prices():
    data = request.form.to_dict()
    print(data.keys())
    date_time_str = data['Gmt time']
    date_time_obj = datetime.strptime(date_time_str, '%Y-%m-%d').date()
    time_steps = 24  # set the number of time steps
    df = pd.DataFrame.from_dict({k:v for k,v in data.items() if k!='Gmt time'}, orient='index').T  # convert dictionary to DataFrame, excluding the 'Gmt time' column
    df['datetime'] = pd.to_datetime(df['Gmt time'])  # convert datetime column to datetime type
    df = df.set_index('datetime')  # set datetime column as index
    input_data = df.loc[date_time_obj - timedelta(hours=time_steps-1):date_time_obj]
    input_data = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns, index=input_data.index)
    X_input = np.array(input_data).reshape((1, time_steps, input_data.shape[1]))
    y_pred = model.predict(X_input)[0]
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
    return render_template('result.html', high_price=y_pred[0], low_price=y_pred[1])



if __name__ == '__main__':
    app.run(debug=True)
