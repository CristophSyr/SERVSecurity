import urllib.request, urllib.error, json

url = "https://servsecurity.metered.live/api/v1/turn/credential?secretKey=brf9aRf8UHS1_fI-OnS00FLPIDXvB0LQejTS1GpmpSNMpASB"
body = json.dumps({"expiryInSeconds": 86400, "label": "servsecurity"}).encode("utf-8")
req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")

try:
    resp = urllib.request.urlopen(req)
    print("SUCCESS:", resp.read().decode())
except urllib.error.HTTPError as e:
    print(f"Status: {e.code}")
    print(f"Body: {e.read().decode()}")
