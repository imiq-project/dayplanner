import requests

response = requests.post(
    "http://localhost:4000/api/route",
    json={
        "origin": [52.10340121158378, 11.612077698233252],  # Uni Klinik
        "destination": [52.13054700114789, 11.642084679359717],  # Allee Center
        "agent": "altruistic",
        "pois": True,
    },
)

response.raise_for_status()
print(response.json())
