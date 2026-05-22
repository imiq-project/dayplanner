# IMIQ dayplanner app

## API

### GraphHopper

```python
import requests

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
```

### Dyconet (dyconet)

```python
import requests

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
```

Output:

```json
{
    "confidence": 0.1391478031873703,
    "probabilities": {
        "bike": 0.44986045360565186,
        "car": 0.2243480086326599,
        "pt": 0.18502913415431976,
        "walk": 0.14076241850852966
    }
}
```
