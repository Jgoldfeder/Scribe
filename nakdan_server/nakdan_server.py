import requests
import json 
def tag(text):
    results = []
    headers = {'Content-Type': 'text/plain;charset=utf-8'}
    params = {
    "task" : "nakdan",
    "genre" :"rabbinic",
    "data" : text,
    "addmorph" : True,
    "newjson" : True,
    "matchpartial" : True,  
    "keepmetagim" : True ,
    "keepqq" :True,
    }
    r = requests.post("https://nakdan-3-0.loadbalancer.dicta.org.il/addnikud",headers=headers,json=params)
    r.encoding= "UTF-8"
    tags = json.loads(r.text)
    for data in tags:
        if len(data["options"]) == 0:
            continue
        word = data["word"]
        mask = int(data["options"][0]["morph"])
        results.append((word,mask))
    return results


    