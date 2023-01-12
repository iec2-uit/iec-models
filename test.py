import requests
import json
import os

name = "iCassava-2019-Dataset"
r = requests.get("https://server-datasets-noqxfy4uf-nvhieu-04.vercel.app/datasets")
data = r.json()
for d in data:
    if(d['name'] == name):
        link = d['link']
print(link)