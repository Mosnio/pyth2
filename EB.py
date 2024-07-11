import requests
import time

# Create a session
session = requests.Session()

# First request URL and headers
url1 = 'https://api.getgrass.io/retrieveDevice?input=%7B%22deviceId%22:%2208d56b85-f0ee-5760-a9c5-765934402000%22%7D'
headers1 = {
    'authority': 'api.getgrass.io',
    'accept': 'application/json, text/plain, */*',
    'accept-language': 'en-IN,en-GB;q=0.9,en-US;q=0.8,en;q=0.7',
    'authorization': 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IkJseGtPeW9QaWIwMlNzUlpGeHBaN2JlSzJOSEJBMSJ9.eyJ1c2VySWQiOiIzZWVmMzljYi1mZTkwLTRjYWQtOTY2My0yODZiYjRlNWZmYmMiLCJlbWFpbCI6Im1pa2VtZXNoYWsyNzNAZ21haWwuY29tIiwic2NvcGUiOiJTRUxMRVIiLCJpYXQiOjE3MjA3MDg0MDYsIm5iZiI6MTcyMDcwODQwNiwiZXhwIjoxNzUxODEyNDA2LCJhdWQiOiJ3eW5kLXVzZXJzIiwiaXNzIjoiaHR0cHM6Ly93eW5kLnMzLmFtYXpvbmF3cy5jb20vcHVibGljIn0.SG_EsPLdto94kH1RIYhOJABuSV358kMvIbi1i_4rkXcTROwqbU53NdhhJsMdRA4twXVzP_1wcVpb0IrBW6oGYaovDagF3CAZjyKgGpNU51jiSq7pQjhtGXdNcf76zOWXgyGxBdCSiUs_qHiLhb4dxjGrzeBr4B5iv3M35GC39XB-9rdXF0q01mv6MzCTVxOnAyAzrdmejQZTBiv3fdkkz9Zi2V2U8hpEVl0etXl8zLfQuRM6a7DmxNl6qzILhmDPns7Zyf5TNTRWWhvJEtMdxBgha6NKqMxbb4DJhExRx1UQATJla8RRnayGJ54IW3i--w5acQWwLLlQQxWS9S_SRw',
    'cookie': 'token=eyJhbGciOiJ7C12%7C0%7Cw.clarity.ms%2Fcollecttoken=eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IjJmUEYtV05SbDdFbWYyRmktdVlUX2V2SVRlUSJ9.eyJhdWQiOiI0NjU0NWI0Zi1lNjNkLTQyNzAtYmJjMy02NGQ1N2E3YzhkNzciLCJleHAiOjM1OTk0ODExMjIsImlhdCI6MTcwNzMyMTEyMiwiaXNzIjoiYWNtZS5jb20iLCJzdWIiOiIzZWVmMzljYi1mZTkwLTRjYWQtOTY2My0yODZiYjRlNWZmYmMiLCJqdGkiOiJmMDIzOWUwMC0wMDkzLTQyMTMtYjc0Ny1hMjcwMjZlNGEyZjUiLCJhdXRoZW50aWNhdGlvblR5cGUiOiJQQVNTV09SRCIsImVtYWlsIjoibWlrZW1lc2hhazI3M0BnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwicHJlZmVycmVkX3VzZXJuYW1lIjoiTWVzaGFrIiwiYXBwbGljYXRpb25JZCI6IjQ2NTQ1YjRmLWU2M2QtNDI3MC1iYmMzLTY0ZDU3YTdjOGQ3NyIsInJvbGVzIjpbInNlbGxlciJdLCJhdXRoX3RpbWUiOjE3MDczMjExMjIsInRpZCI6ImZhM2ZlNTBlLTc3ZDItMGE0Mi03MmNiLTg0NGI2YzAzYmFjMiJ9.PH6zBeiXEIG8jUebYr3z13iywEeIy4pyR3gKBF8jxF9K19n17TPb6v_ZicwWuh12i8t6A_yrLyWWu5M_6JZRd_0_O0bubpKk7Tju5pe1nCs59PFC1xI4fj14qDT9MpTfc7HB1WKZE-p-0t9VRYjzMYoijcpHVXU6KWwUVqj0Q402xgBKhg-edTiuni4dbdK-TOK25Z0PpJ0VngwvNWwbS_NERefYrSrSyQx5lZ3v4IO3DWZ5NKsd4K7PKKpLTnRro7gK2wgQrwtJVIFJOCXjv1k37UiqJu3iZl_51sPLRKSmAWQdWwh0D1CadkK1YU4Psm-DRC27vpCn2iXBFg0-Aw; _clck=xt3kdb%7C2%7Cfnd%7C0%7C1498; token=eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IkJseGtPeW9QaWIwMlNzUlpGeHBaN2JlSzJOSEJBMSJ9.eyJ1c2VySWQiOiIzZWVmMzljYi1mZTkwLTRjYWQtOTY2My0yODZiYjRlNWZmYmMiLCJlbWFpbCI6Im1pa2VtZXNoYWsyNzNAZ21haWwuY29tIiwic2NvcGUiOiJTRUxMRVIiLCJpYXQiOjE3MjA3MDg0MDYsIm5iZiI6MTcyMDcwODQwNiwiZXhwIjoxNzUxODEyNDA2LCJhdWQiOiJ3eW5kLXVzZXJzIiwiaXNzIjoiaHR0cHM6Ly93eW5kLnMzLmFtYXpvbmF3cy5jb20vcHVibGljIn0.SG_EsPLdto94kH1RIYhOJABuSV358kMvIbi1i_4rkXcTROwqbU53NdhhJsMdRA4twXVzP_1wcVpb0IrBW6oGYaovDagF3CAZjyKgGpNU51jiSq7pQjhtGXdNcf76zOWXgyGxBdCSiUs_qHiLhb4dxjGrzeBr4B5iv3M35GC39XB-9rdXF0q01mv6MzCTVxOnAyAzrdmejQZTBiv3fdkkz9Zi2V2U8hpEVl0etXl8zLfQuRM6a7DmxNl6qzILhmDPns7Zyf5TNTRWWhvJEtMdxBgha6NKqMxbb4DJhExRx1UQATJla8RRnayGJ54IW3i--w5acQWwLLlQQxWS9S_SRw; _clsk=mchxkf%7C1720708638741%7C12%7C0%7Cw.clarity.ms%2Fcollectum";v="124"',
    'sec-ch-ua-mobile': '?1',
    'sec-ch-ua-platform': '"Android"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'none',
    'user-agent': 'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36',
}

# Make the first request
response1 = session.get(url1, headers=headers1,verify=False)
print(response1.status_code)
print(response1.json())

# Second request URL and headers
url2 = 'https://api.bigdatacloud.net/data/client-ip'
headers2 = {
    'authority': 'api.bigdatacloud.net',
    'accept': 'application/json, text/plain, */*',
    'accept-language': 'en-IN,en-GB;q=0.9,en-US;q=0.8,en;q=0.7',
    'sec-ch-ua': '"Not-A.Brand";v="99", "Chromium";v="124"',
    'sec-ch-ua-mobile': '?1',
    'sec-ch-ua-platform': '"Android"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'none',
    'user-agent': 'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36',
}

# Loop to make the second request repeatedly
while True:
    response2 = session.get(url2, headers=headers2)
    print(response2.status_code)
    print(response2.json())
    
    # Wait for some time before making the next request
    time.sleep(15)  # Sleep for 5 seconds (adjust as needed)
