import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re



import tensorflow as tf
from tensorflow.keras.layers import Layer

class AttentionLayer(Layer):
    def __init__(self, **kwargs):   # ✅ THIS IS THE FIX
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='context_vector',
            shape=(input_shape[-1],),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        score = tf.nn.tanh(tf.tensordot(x, self.W, axes=[2, 0]) + self.b)
        attention_weights = tf.nn.softmax(
            tf.tensordot(score, self.u, axes=[2, 0]), axis=1
        )
        context_vector = tf.reduce_sum(
            attention_weights[..., tf.newaxis] * x, axis=1
        )
        return context_vector





# Load model
model = load_model(
    "disease_prediction_model.h5",
    custom_objects={"AttentionLayer": AttentionLayer}
)

# Load preprocessing
with open("preprocessing.pkl", "rb") as f:
    data = pickle.load(f)

tokenizer = data["tokenizer"]
label_encoder = data["label_encoder"]

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_disease(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=150)

    pred_probs = model.predict(padded)
    pred_class = np.argmax(pred_probs, axis=1)[0]

    return label_encoder.inverse_transform([pred_class])[0]

# Test loop
while True:
    user_input = input("Enter symptoms: ")
    if user_input.lower() == "exit":
        break

    result = predict_disease(user_input)
    print(f"Predicted Disease: {result}")