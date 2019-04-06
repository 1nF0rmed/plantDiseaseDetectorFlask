import requests
import sys

with open(sys.argv[1], 'rb') as f:
    r = requests.post("http://3.19.68.99:5000/api/v1/setState", 
                        files={'file':f})

    print r
