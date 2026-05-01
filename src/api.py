import requests

def get_crypto_price():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"

    try:
        data = requests.get(url, timeout=5).json()
        return data["bitcoin"]["usd"]
    except:
        return None
