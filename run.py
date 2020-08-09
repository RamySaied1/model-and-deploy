import json
import os
from flask import Flask,jsonify,request,render_template
from flask_cors import CORS
from predection import binary_predictor

app = Flask(__name__)
CORS(app)
@app.route("/class/",methods=['POST'])
def return_class():
    data = request.get_json(force=True)
    try:
        prediction = binary_predictor.predict(list(data.values()))
        output = prediction[0]
        if (output==0):
        
            prediction={'output':'no'}
        else:
            prediction={'output':'yes'}

        return jsonify(prediction),200
    except Exception as e:
        return jsonify({"error ":e}),404


@app.route("/",methods=['GET'])
def default():
    return "<p> microservice for wide pot task !</p>"

if __name__ == "__main__":
    app.run() 