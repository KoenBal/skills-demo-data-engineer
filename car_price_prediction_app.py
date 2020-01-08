from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import joblib

app = Flask(__name__)
api = Api(app)

model = joblib.load('lgbr_cars.model')


class MakePrediction(Resource):
    @staticmethod
    def post():
        posted_data = request.get_json()
        if 'single_input' in posted_data:
            single_input = posted_data['single_input']

        prediction = model.predict([single_input])[0]

        return jsonify({
            'car_price_prediction': prediction
        })

# Create the predict endpoint
api.add_resource(MakePrediction, '/predict')

if __name__ == '__main__':
    # set debug to False when deploying to production
    app.run(debug=True, host='0.0.0.0')
