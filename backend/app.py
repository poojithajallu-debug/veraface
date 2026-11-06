from flask import Flask, render_template
from flask_cors import CORS

from api.predict import predict_bp

app = Flask(__name__)

CORS(app)

app.register_blueprint(predict_bp)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return {
        "success": True,
        "message": "VeraFace backend is healthy"
    }


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )