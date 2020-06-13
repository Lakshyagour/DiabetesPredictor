from flask import Flask,render_template, jsonify, request, Markup, json, session, redirect, url_for
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
app=Flask(__name__)
app.debug=True
app.secret_key = 'aqwertyuiop1234567890'
model = pickle.load(open('diabetesClassifier.pkl','rb'))
df=pd.read_csv('clean_df.csv',index_col='Unnamed: 0')
X=df.drop('Outcome',1)
scalar=StandardScaler()
scalar.fit(X)
@app.route('/',methods=['GET', 'POST'])
def fill_form():
    if request.method=='POST':
        preg = int(request.form["pregnancies"])
        glucose = int(request.form["glucose"])
        bp= int(request.form["bloodPressure"])
        skin=int(request.form["skinThickness"])
        insulin=int(request.form["insulin"])
        bmi=float(request.form["bmi"])
        dbf=float(request.form["dpf"])
        age = int(request.form["age"])
        data=[preg,glucose,bp,skin,insulin,bmi,dbf,age]
        print(data)
        X_test=scalar.transform([data])
        print(X_test)
        y=model.predict(X_test)
        print(y)
        return redirect(url_for('result',data=y))
    else:
        return render_template('bootstrap.html')
@app.route('/result')
def result():
    data = request.args.get('data', None)
    return f"<H1>{data[1]}</H1>"

if __name__=='__main__':
    app.run()