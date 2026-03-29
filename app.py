
from flask import Flask, render_template, request
from model import Translator
import os

app = Flask(__name__)

translator = Translator()

@app.route("/", methods=["GET", "POST"])
def index():
    translation = ""
    input_text = ""

    if request.method == "POST":
        input_text = request.form.get("input_text", "")
        if input_text.strip():
            translation = translator.translate(input_text)

    return render_template("index.html", translation=translation, input_text=input_text)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
