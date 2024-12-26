# # Importing libraries
# import warnings
# warnings.filterwarnings("ignore")
# from flask import Flask, render_template, request
# from werkzeug.utils import secure_filename
# from src.predict import build
# import os

# # Flask app
# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route("/result", methods=["GET", "POST"])
# def result():
#     if request.method == 'POST':
#         f = request.files.get('file')
#         if f:
#             save_path = os.path.join("upload", secure_filename(f.filename))
#             f.save(save_path)
#             result = build(save_path)
#             result_text = f"Your Predicted result is {result}"
#             print(result_text)
#             return render_template("result.html", result=result_text)
#     return render_template("index.html")

# # Main function app run
# if __name__ == "__main__":
#     app.run(host="127.0.0.1", port=8080, debug=True)
# Importing libraries

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.models import load_model
from src.predict import build

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
MODEL_PATH = "./models/resv152.h5"
model = load_model(MODEL_PATH)

# Allowed extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route("/result", methods=["GET", "POST"])
def result():
    if request.method == 'POST':
        # Check if the file is part of the request
        f = request.files.get('file')
        if f and allowed_file(f.filename):
            # Save file securely
            filename = secure_filename(f.filename)
            save_path = os.path.join("upload", filename)
            os.makedirs("upload", exist_ok=True)  # Ensure the upload directory exists
            f.save(save_path)

            # Use `build` function for prediction
            result = build(save_path, model)

            # Render result
            result_text = f"Your Predicted result is {result}"
            print(result_text)
            return render_template("result.html", result=result_text)
        else:
            error = "Please upload a valid JPG or JPEG file."
            return render_template("index.html", error=error)
    return render_template("index.html")

# Main function to run the app
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
