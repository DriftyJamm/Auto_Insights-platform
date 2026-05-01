import requests

def get_crypto_price():
    import requests
    url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"

    try:
        data = requests.get(url, timeout=5).json()
        return float(data["price"])
    except:
        return None
    
