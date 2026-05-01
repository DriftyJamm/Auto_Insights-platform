import requests

def get_crypto_price():
    url = "https://api.coindesk.com/v1/bpi/currentprice.json"

    try:
        response = requests.get(url, timeout=5)

        # Check if request was successful
        if response.status_code != 200:
            return None

        data = response.json()

        # Extract price safely
        price_str = data.get("bpi", {}).get("USD", {}).get("rate", None)

        if price_str:
            # Convert "67,123.45" → 67123.45
            price = float(price_str.replace(",", ""))
            return price

        return None

    except Exception:
        return None
    