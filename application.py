import joblib
import numpy as np
from config.paths_config import MODEL_OUTPUT_PATH
from flask import Flask, request, render_template

app = Flask(__name__)

loaded_model = joblib.load(MODEL_OUTPUT_PATH)

@app.route('/', methods=['GET', "POST"])
def home():
    if request.method == "POST":
        Steel_Plate_Thickness= float(request.form['Steel_Plate_Thickness'])
        Length_of_Conveyer= float(request.form['Length_of_Conveyer'])
        Edges_Index= float(request.form['Edges_Index'])
        Minimum_of_Luminosity= float(request.form['Minimum_of_Luminosity'])
        Edges_Y_Index= float(request.form['Edges_Y_Index'])
        Square_Index= float(request.form['Square_Index'])
        features = np.array([[Steel_Plate_Thickness, Length_of_Conveyer, Edges_Index,
                              Minimum_of_Luminosity, Edges_Y_Index, Square_Index,]])
        
        prediction = loaded_model.predict(features)[0]
        print(prediction)
        return render_template("index.html", prediction=prediction)

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)