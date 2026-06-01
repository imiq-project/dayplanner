import requests

response = requests.post(
    "http://localhost:4000/api/routing/ranked-routes",
    json={
        "start": {"lat": 52.10340121158378, "lon": 11.612077698233252},  # Uni Klinik
        "stop": {"lat": 52.13054700114789, "lon": 11.642084679359717},  # Allee Center
        "cognitive_passport": {},
    },
)

print(response.json())
response.raise_for_status()
