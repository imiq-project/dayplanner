from flask import Flask, render_template, request, jsonify
from typing import List
import re
import hmac
import hashlib
import os
import jsonschema
import math

app = Flask(__name__)
KEY = os.environ["APP_SECRET"]


def distance(start: List[float], end: List[float]):
    return math.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)


def interpolate_coords(start: list, end: list):
    num_points = int(distance(start, end) * 1000) + 1
    if num_points == 1:
        return start
    lat = start[0]
    lon = start[1]
    lat_step = (end[0] - start[0]) / (num_points - 1)
    lon_step = (end[1] - start[1]) / (num_points - 1)
    points = []
    for _ in range(num_points):
        points.append([lat, lon])
        lat += lat_step
        lon += lon_step

    return points

def calc_hmac(code: str) -> str:
    return hmac.new(KEY.encode(), code.encode(), hashlib.sha256).hexdigest()

# TODO: rate limit
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    code = data.get("code", "")

    if not re.fullmatch(r"\d{3}-\d{3}-\d{3}", code):
        return (
            jsonify({"error": "Invalid code. Must be in the format XXX-XXX-XXX."}),
            400,
        )

    if code == "123-456-789":
        token = calc_hmac(code)
        return jsonify({"message": "Login successful!", "token": token})
    else:
        return jsonify({"error": "Unauthorized"}), 401


@app.route("/route", methods=["POST"])
def route():
    data = request.get_json()
    schema = {
        "type": "object",
        "properties": {
            "start": {"$ref": "#/$refs/coords"},
            "destination": {"$ref": "#/$refs/coords"},
            "profile": {"type": "object"},
            "token": {"type": "string"},
        },
        "$refs": {
            "coords": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "number"},
            }
        },
        "required": ["start", "destination", "profile"],
    }
    try:
        jsonschema.validate(data, schema)
    except jsonschema.exceptions.ValidationError as e:
        return jsonify({"error": e.message}, 400)
    
    return jsonify({"points": interpolate_coords(data["start"], data["destination"])})

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=80)
