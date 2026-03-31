import requests

API_URL = "https://api.example.com/data"
API_KEY = "your_api_key_here"

def get_data(endpoint: str) -> dict:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.get(f"{API_URL}/{endpoint}", headers=headers)
    response.raise_for_status()  # raises error if status != 200
    return response.json()

# Usage
data = get_data("users")
print(data)
