from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    message = request.form.get("message", "").strip()

    if not message:
        response = "Please enter something."
    else:
        response = f"OurPlatform received: {message}"

    return render_template(
        "index.html",
        response=response
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )