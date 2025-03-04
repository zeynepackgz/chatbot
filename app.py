from flask import Flask, request, jsonify, render_template
from chatbot import predict_class, get_response, intents
from flask_cors import CORS

app = Flask(__name__, template_folder='templates')
CORS(app, resources={r"/chat": {"origins": "*"}})

@app.route("/")
def index():
    return render_template("chat.html")

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()

        prediction = predict_class(user_message)
        response = get_response(prediction, intents)

        # JSON yanıtını düzenleme
        response_data = {
            "intent": prediction[0]["intent"],
            "probability": prediction[0]["probability"],
            "response": response if isinstance(response, dict) else {"text": response}
        }

        return jsonify(response_data)

    except Exception as e:
        app.logger.error(f"Hata: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "response": {"text": "Bir hata oluştu, lütfen tekrar deneyin."}
        }), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)


