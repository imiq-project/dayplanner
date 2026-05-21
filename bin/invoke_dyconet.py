#! /usr/bin/env python3
import requests
import json

response = requests.post(
    "http://localhost:4000/api/dyconet",
    json={
        "needs": {
            "pro_env": .7,
            "physical": .3,
            "privacy": .3,
            "autonomy": 0,
            "hedonism": 0,
            "cost": .3,
            "speed": .9,
            "safety": .8,
            "comfort": .2,
        },
        "valences": {
            "car": 0.15,
            "bike": 0.15,
            "pt": 0.15,
            "walk": .55,
        },
        "stressors": {
            "rain": 0,
            "crowding": .7,
            "darkness": .7,
            "traffic": .7,
            "temperature": 0,
        },
        "tolerances": {
            "rain": 0.1,
            "crowding": 0.1,
            "darkness": 0.1,
            "traffic": 0.1,
            "temperature": 0.1,
        },
    },
)

response.raise_for_status()
print(json.dumps(response.json(), indent=4))
