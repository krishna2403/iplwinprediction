# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# Load the Random Forest CLassifier model
filename = 'score-prediction-model.pkl'
regressor = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    temp_array = list()
    
    if request.method == 'POST':
        
        
            
            
        overs = float(request.form['overs'])
        runs = int(request.form['runs'])
        wickets = int(request.form['wickets'])
        striker_sr = float(request.form['striker_sr'])
        nstriker_sr = float(request.form['nstriker_sr'])
        striker_score = float(request.form['striker_score'])
        nstriker_score = float(request.form['nstriker_score'])
        runs_in_prev_5 = int(request.form['runs_in_prev_5'])
        wickets_in_prev_5 = int(request.form['wickets_in_prev_5'])
        
        temp_array = temp_array + [overs, runs, wickets, striker_sr,nstriker_sr,striker_score,nstriker_score,runs_in_prev_5, wickets_in_prev_5]
        
        data = np.array([temp_array])
        my_prediction = int(regressor.predict(data)[0])
              
        return render_template('result.html', lower_limit = my_prediction-10, upper_limit = my_prediction+5)



if __name__ == '__main__':
	app.run(debug=True)