import os
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')  # Bu satırı ekleyin

lemmatizer = WordNetLemmatizer()

# JSON dosyasını yükle
current_dir = os.path.dirname(os.path.abspath(__file__))
intents_path = os.path.join(current_dir, 'intents.json')

try:
    with open(intents_path, encoding='utf-8') as f:
        intents = json.load(f)
except FileNotFoundError:
    print(f"Hata: '{intents_path}' dosyası bulunamadı. Dosyanın mevcut olduğundan ve doğru yerde olduğundan emin olun.")
    exit()

words = pickle.load(open(os.path.join(current_dir, 'words.pkl'), 'rb'))
classes = pickle.load(open(os.path.join(current_dir, 'classes.pkl'), 'rb'))
model = load_model(os.path.join(current_dir, 'chatbot_model.h5'), compile=False)


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)


def get_response(ints, intents_json):
    try:
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']

        for intent in list_of_intents:
            if intent['tag'] == tag:
                # Özel yanıt tipleri kontrolü
                response = intent['responses'][0]

                # Uygulama listesi için
                if 'apps' in response and 'footer' in response:
                    return {
                        'type': 'apps',
                        'text': response['text'],
                        'apps': response['apps'],
                        'footer': response['footer']
                    }

                # Form linkleri için
                elif 'links' in response:
                    return {
                        'type': 'links',
                        'text': response['text'],
                        'links': response['links']
                    }

                # Standart metin yanıtları
                else:
                    return {
                        'type': 'plain',
                        'text': random.choice(intent['responses'])
                    }

        # Eşleşme bulunamazsa
        return {
            'type': 'plain',
            'text': "Üzgünüm, bunu anlayamadım."
        }

    except Exception as e:
        print(f"Hata: {str(e)}")
        return {
            'type': 'error',
            'text': "Bir hata oluştu, lütfen tekrar deneyin."
        }



def predict_class(sentence):
    try:
        bow = bag_of_words(sentence)
        res = model.predict(np.array([bow]), verbose=0)[0]  # verbose=0 ile gereksiz çıktıları engelle
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})

        if not return_list:
            return_list.append({'intent': 'default_fallback', 'probability': '0.8'})

        return return_list
    except Exception as e:
        print(f"Tahmin hatası: {str(e)}")
        return [{'intent': 'default_fallback', 'probability': '0.8'}]


if __name__ == "__main__":
    print("Merhaba! Ben BERA. Sana nasıl yardımcı olabilirim?")
    while True:
        message = input("You: ")
        ints = predict_class(message)
        res = get_response(ints, intents)
        print(res)
