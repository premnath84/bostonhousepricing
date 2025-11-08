import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

## Load the models
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


## Crete routes for app
@app.route('/')
def home():
    return render_template('home_main.html')


# For api testing using Postman
@app.route ('/predict_api_test', methods = ['POST'])
def predict_api_test():
    data = request.json ['data']
    # data = request.json
    # print(data)
    # print (data.values)
    # print(np.array(list(data.values())).reshape(1, -1))

    new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    output= regmodel.predict(new_data)
    # print(output[0])

    return jsonify(output[0])

# From 1 - as per home.html
@app.route('/form_api')
def form_api():
    return render_template('home.html')
# This api gets opened when form in home.html is submitted
@app.route('/predict_api', methods = ['POST'])
def predict_api():

    # print("raw form values:")
    # print(request.form.values())
    # print(list(request.form.values()))
    # print(dict(request.form))

    # Get data from form submission
    data = [float (x) for x in request.form.values()]
    print(data)

    # transform data and run ML model code
    final_input = scaler.transform (np.array(data).reshape(1, -1))
    print (final_input)
    output = regmodel.predict(final_input)[0]

    # render output back to webpage
    return render_template ("home.html", prediction_text = "Then House price prediction is {}".format(output))

#------------------------------------
# From 1a - as per home1a.html
# this is futureproof as feature sequence is handles by backend
@app.route('/form_api1a')
def form_api1a():
    return render_template('home1a.html')

# Define the exact feature order expected by your model
FEATURE_ORDER = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

@app.route('/predict_api1a', methods=['POST'])
def predict_api1a():

    # Extract and convert to float
    data = [float(request.form[feature]) for feature in FEATURE_ORDER]
    print(data)

    # transform data and run ML model code
    final_input = scaler.transform (np.array(data).reshape(1, -1))
    print (final_input)
    output = regmodel.predict(final_input)[0]

    # render output back to webpage
    return render_template ("home1a.html", prediction_text = "Then House price prediction is {}".format(output))



#------------------------------------
#------------------------------------
# Form 2 - as per home2.html
@app.route('/form_api2')
def form_api2():
    return render_template('home2.html')


@app.route('/predict_api2', methods=['POST'])
def predict_api2():
    try:
        # Get data from frontend form submission
        data = [float(x) for x in request.form.values()]
        
        # transform data and run ML model code
        final_input = scaler.transform(np.array(data).reshape(1, -1))
        print(final_input)
        output = regmodel.predict(final_input)[0]
        
        # render output back to webpage
        return jsonify({
            'prediction': round(output, 2),
            'status': 'success'
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 400

#------------------------------------
#------------------------------------
# Form 3 - as per home3.html
@app.route('/form_api3')
def form_api3():
    return render_template('home3.html')


@app.route('/predict_api3', methods=['POST'])
def predict_api3():
    try:

         # Get data from frontend
        data = request.get_json()
        print("api3 data:")
        print(data)

         # Convert form data to float values (adjust field names as needed)
        feature_values = [float(data[key]) for key in data.keys()]

         # Use your existing ML code
        final_input = scaler.transform(np.array(feature_values).reshape(1, -1))
        print(final_input)
        output = regmodel.predict(final_input)[0]
        
        # Return prediction as JSON
        return jsonify({
            'prediction': round(output, 2),
            'status': 'success'
        })


    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

#------------------------------------
#------------------------------------
# Form 4 - as per home4.html
@app.route('/form_api4')
def form_api4():
    return render_template('home4.html')



@app.route('/predict_api4', methods=['POST'])
def predict_api4():
    try:

         # Get data from frontend
        data = request.get_json()
        print(f"api4 data: {data}")

         # Convert form data to float values (adjust field names as needed)
        feature_values = [data["data"][key] for key in data["data"].keys()]

         # Use your existing ML code
        final_input = scaler.transform(np.array(feature_values).reshape(1, -1))
        print(final_input)
        output = regmodel.predict(final_input)[0]
        
        # Return prediction as JSON
        return jsonify({
            'prediction': round(output, 2),
            'status': 'success'
        })


    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400


#------------------------------------
#------------------------------------
# Form 5 - as per home5.html
@app.route('/form_api5')
def form_api5():
    return render_template('home5.html')



@app.route('/predict_api5', methods=['POST'])
def predict_api5():
    try:

         # Get data from frontend
        data = request.get_json()
        print(f"api5 data: {data}")

         # Convert form data to float values (adjust field names as needed)
        feature_values = [data["data"][key] for key in data["data"].keys()]

         # Use your existing ML code
        final_input = scaler.transform(np.array(feature_values).reshape(1, -1))
        print(final_input)
        output = regmodel.predict(final_input)[0]
        
        # Return prediction as JSON
        return jsonify({
            'prediction': round(output, 2),
            'status': 'success'
        })


    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400





#------------------------------------

# if __name__ == "__main__":
#     app.run (debug=True)

#------------------------------------
# Disable this if deployment is done directly github repo and not using Dockerimage
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

