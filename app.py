from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)
# load the data transform object
# tf = pickle.load(open('transform.pkl', 'rb'))
# load the trained model object
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # For rendering results on HTML GUI

    int_features = [float(x) for x in request.form.values()]

    '''use transform object to transform the input'''
    # data = tf.transform(int_features)
    
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text='Result {}'.format(output))


@app.route('/predict_api',methods=['POST'])
def predict_api():
    """ For direct API calls trought request"""
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
