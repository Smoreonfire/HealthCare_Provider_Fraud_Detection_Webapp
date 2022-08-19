from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pickle


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    pred = ""
    if request.method == "POST":
        TotalTeDiagCode = request.form["TotalTeDiagCode"]
        TotalTeProcCode = request.form["TotalTeProcCode"]
        MaxHospitalDays = request.form["MaxHospitalDays"]
        MaxDiagCodeNumPerClaim = request.form["MaxDiagCodeNumPerClaim"]
        MedianHospitalDays = request.form["MedianHospitalDays"]
        ClmsperBene = request.form["ClmsperBene"]
        uniqBeneCount = request.form["uniqBeneCount"]
        MajorRace = request.form["MajorRace"]
        MeanLowFreqDiagCodeNumPerClaim = request.form["MeanLowFreqDiagCodeNumPerClaim"]
        InClmsPct = request.form["InClmsPct"]	
        TotalInscClaimAmtReimbursed = request.form["TotalInscClaimAmtReimbursed"]
	
        X = np.array([[float(TotalTeDiagCode), float(TotalTeProcCode), float(MaxHospitalDays), float(MaxDiagCodeNumPerClaim), float(MedianHospitalDays),float(ClmsperBene), float(uniqBeneCount),float(MajorRace), float(MeanLowFreqDiagCodeNumPerClaim), float(InClmsPct), float(TotalInscClaimAmtReimbursed)]])
        pred = model.predict_proba(X)[0][1]
    
    return render_template("index.html", pred=pred)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
