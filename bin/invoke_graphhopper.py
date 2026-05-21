import requests
import json

url = "http://localhost:4000/api/graphhopper/route"

params = {
    "point": [
        "52.10340121158378, 11.612077698233252",  # Uni Klinik
        "52.13054700114789, 11.642084679359717",  # Allee Center
    ],
    "profile": "car",
    "locale": "en",
    "calc_points": "true",
    "points_encoded": "false",
}

response = requests.get(url, params=params)
response.raise_for_status()
data = response.json()

print("Distance (meters):", data["paths"][0]["distance"])
print("Time (ms):", data["paths"][0]["time"])

# Route coordinates
coordinates = data["paths"][0]["points"]["coordinates"]

print("First 5 coordinates:")
for coord in coordinates[:5]:
    print(coord)
