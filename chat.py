import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_words, tokenize
import googlemaps
import html

gmaps = googlemaps.Client(key='AIzaSyBhblSnjcYWqGu8pkFVo-3wNcMNdPD1Gbs')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"  
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Bot"

def get_response(msg):
   # Tokenize sentence and calculate bag of words
    sentence = tokenize(msg)
    X = bag_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                if tag == "product_refund":
                    address = gmaps.geocode('840 Academy Way, Kelowna') #hardcoded the address just for test, in real app address would be known from customer order info
                    latlong = address[0]['geometry']['location']
                    postOffices = gmaps.places_nearby(location=latlong, type="post_office", rank_by="distance")
                    nearestPost = postOffices['results'][0]['geometry']['location']                   
                    directions_result = gmaps.directions(latlong, nearestPost)
                    directionsStr = ''
                    for i in range(0,len(directions_result[0]['legs'][0]['steps'])):
                        directionsStr = directionsStr+(html.unescape(directions_result[0]['legs'][0]['steps'][i]['html_instructions']).replace('<b>','').replace('</b>','').replace('/<wbr/>','/').replace('<div style="font-size:0.9em">',' ').replace('</div>','')+'\n')
                    return directionsStr + random.choice(intent['responses'])
                return random.choice(intent['responses'])
  
    return "I do not understand..."


            

