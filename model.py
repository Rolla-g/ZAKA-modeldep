
import pickle
import json
import re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Translator:
    def __init__(self):
        # Load trained model
        self.model = load_model("translation_model.h5")

        # Load tokenizers
        with open("en_tokenizer.pkl", "rb") as f:
            self.en_tokenizer = pickle.load(f)

        with open("fr_tokenizer.pkl", "rb") as f:
            self.fr_tokenizer = pickle.load(f)

        # Load config values
        with open("translation_config.json", "r") as f:
            config = json.load(f)

        self.max_en_len = config["max_en_len"]
        self.max_fr_len = config["max_fr_len"]

        # Reverse mapping for decoding French tokens
        self.index_to_fr_word = {
            index: word for word, index in self.fr_tokenizer.word_index.items()
        }

    def clean_text(self, text):
        text = str(text).lower().strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-zA-ZÀ-ÿ0-9?.!,¿'’\s-]", "", text)
        text = re.sub(r"\s+([?.!,¿])", r"\1", text)
        return text

    def decode_prediction(self, pred_sequence):
        words = []

        for idx in pred_sequence:
            if idx == 0:
                continue

            word = self.index_to_fr_word.get(idx, "")

            if word == "startseq":
                continue
            if word == "endseq":
                break

            words.append(word)

        return " ".join(words)

    def translate(self, sentence):
        sentence = self.clean_text(sentence)

        seq = self.en_tokenizer.texts_to_sequences([sentence])
        seq = pad_sequences(seq, maxlen=self.max_en_len, padding="post")

        pred = self.model.predict(seq, verbose=0)
        pred_ids = np.argmax(pred[0], axis=1)

        return self.decode_prediction(pred_ids)
