# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


filename = 'score-prediction-model.pkl'
filename1 = 'win-probability-model.pkl'
regressor = pickle.load(open(filename, 'rb'))
regressor1 = pickle.load(open(filename1, 'rb'))
data0 = pd.read_csv('data.csv')
data1 = pd.read_csv('data1.csv')
strike_rates = pd.read_csv('strike_rates.csv')
        

app = Flask(__name__)

@app.route('/')
def home():

    return render_template('home.html')

@app.route('/score', methods=['GET','POST'])
def score():
    temp_array = list()

    if request.method == 'GET':
        return render_template('predict_score.html')
    if request.method == 'POST':
        
        
            
            
        overs = float(request.form['overs'])
        runs = int(request.form['runs'])
        wickets = int(request.form['wickets'])
        striker_sr = strike_rates[request.form['striker']][0]
        nstriker_sr = strike_rates[request.form['nstriker']][0]

        striker_score = float(request.form['striker_score'])
        nstriker_score = float(request.form['nstriker_score'])
        runs_in_prev_5 = int(request.form['runs_in_prev_5'])
        wickets_in_prev_5 = int(request.form['wickets_in_prev_5'])
        
        temp_array = temp_array +   [overs, runs, wickets, striker_sr,nstriker_sr,striker_score,nstriker_score,runs_in_prev_5, wickets_in_prev_5]

        encoded_df = data0.copy()
        encoded_df.drop(labels='Matchid', axis=True, inplace=True)
        X = encoded_df.drop(labels='Total', axis=1)
        y = encoded_df['Total']

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)


        
        data = np.array([temp_array])
        my_prediction = int(regressor.predict(sc.transform(data))[0])
        print(my_prediction)
        return render_template('score_result.html', lower_limit = my_prediction-5, upper_limit = my_prediction+5)

@app.route('/win', methods=['GET','POST'])
def win():
    temp_array = list()

    if request.method == 'GET':
        return render_template('predict_win.html')
    
    if request.method == 'POST':
        
        
            
            
        overs = float(request.form['overs'])
        runs = int(request.form['runs'])
        wickets = int(request.form['wickets'])
        striker_sr = strike_rates[request.form['striker']][0]
        nstriker_sr = strike_rates[request.form['nstriker']][0]
        striker_score = float(request.form['striker_score'])
        nstriker_score = float(request.form['nstriker_score'])
        runs_in_prev_5 = int(request.form['runs_in_prev_5'])
        wickets_in_prev_5 = int(request.form['wickets_in_prev_5'])
        rem_runs = int(request.form['rem_runs'])
        
        temp_array = temp_array + [overs, runs, wickets, striker_sr,nstriker_sr,striker_score,nstriker_score,runs_in_prev_5, wickets_in_prev_5,rem_runs]

        encoded_df = data1.copy()
        encoded_df.drop(labels='Matchid', axis=True, inplace=True)
        X = encoded_df.drop(labels='is_Winner', axis=1)
        y = encoded_df['is_Winner']

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)



        #data = np.array([temp_array])
        my_prediction = regressor1.predict_proba(sc.transform(np.array([temp_array])))[0][1]
        
              
        return render_template('win_result.html', probability = int(my_prediction*100))


if __name__ == '__main__':
    app.run(debug=True)