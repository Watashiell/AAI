import json
import pickle
import random
import numpy as np
from flask import Flask, request, render_template
from keras.optimizers import SGD
from tensorflow.keras.models import load_model
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Initialize the stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Define custom objects for loading the model
custom_objects = {'SGD': SGD}

# Load the model and necessary files
model = load_model('aset/chatbotmodel.h5', compile=False,custom_objects=custom_objects)
with open("aset/intents.json") as f:
    intents = json.load(f)
words = pickle.load(open("aset/words.pkl", "rb"))
classes = pickle.load(open("aset/classes.pkl", "rb"))

bot_name = "Sam"

def clean_up_sentence(sentence):
    sentence_words = word_tokenize(sentence)
    sentence_words = [stemmer.stem(word) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.75
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if not intents_list:
        return "Maaf, saya tidak mengerti apa yang Anda maksud."
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])


if __name__ == "__main__":
    print("Hi!!. Saya adalah QuakeBot, assisten virtual yang siap menjawab pertanyaan yang anda miliki mengenai Gempa Bumi. Silahkan bertanya!! ")
    while True:
        message = input("| Kamu: ")
        ints = predict_class(message)
        res = get_response(ints, intents)
        print(res)

        # Check if the predicted intent is "thanks"
        if ints and ints[0]['intent'] == "terima_kasih":
            print("Terima kasih")
            break