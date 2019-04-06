import requests

with open("test4.jpg", 'rb') as f:
    r = requests.post("http://localhost:5000/api/v1/setState", 
                        files={'file':f})

    print r
