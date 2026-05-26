from flask import Flask
import json

app = Flask(__name__)

with open("routing_output.json") as f:
    dummy_output = json.load(f)

@app.route("/api/route", methods=["POST"])
def route():
    return dummy_output

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
