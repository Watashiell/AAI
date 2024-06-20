from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import logging
from chat import predict_class, get_response, intents

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s', handlers=[logging.StreamHandler()])

@app.route("/", methods=["GET"])
def index_get():
    return render_template("base.html")

@app.route("/chat", methods=["POST"])
def chat():
    text = request.get_json().get("message")
    app.logger.debug(f"Received message: {text}")
    if not text:
        return jsonify({"response": "No input provided."})

    # Predict the class of the input text
    intents_list = predict_class(text)
    app.logger.debug(f"Predicted intents: {intents_list}")
    response = get_response(intents_list, intents)
    
    message = {"response": response}
    app.logger.debug(f"Response message: {message}")
    return jsonify(message)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
