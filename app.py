"""
app.py - Flask backend for AI Medical Diagnosis Dashboard
Voice layer is fully offline via voice/voice_client.py (Vosk + pyttsx3 + argostranslate).
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import os
import pickle
import re
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer

from fusion import normalize, get_top_prediction, get_top_k
from gradcam import get_gradcam_heatmap, overlay_heatmap

# app = Flask(__name__)
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)


###############
THRESHOLD = 0.65  # start here, we may tune later
###############

# ── Custom Attention Layer ─────────────────────────────────────────────────

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], input_shape[-1]),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True,
        )
        self.u = self.add_weight(
            name="context_vector",
            shape=(input_shape[-1],),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        score = tf.nn.tanh(tf.tensordot(x, self.W, axes=[2, 0]) + self.b)
        attention_weights = tf.nn.softmax(
            tf.tensordot(score, self.u, axes=[2, 0]), axis=1
        )
        context_vector = tf.reduce_sum(
            attention_weights[..., tf.newaxis] * x, axis=1
        )
        return context_vector


# ── Load Models ───────────────────────────────────────────────────────────

nlp_model = load_model(
    "model/disease_prediction_model.h5",
    custom_objects={"AttentionLayer": AttentionLayer},
)

with open("model/preprocessing.pkl", "rb") as f:
    _data = pickle.load(f)

tokenizer = _data["tokenizer"]
label_encoder = _data["label_encoder"]

image_model = load_model("model/skin_cancer_cnn.h5")

# Force model build for Grad-CAM
dummy_input = tf.zeros((1, 128, 128, 3))
_ = image_model(dummy_input, training=False)

# Warm-up pass to avoid first-inference latency
image_model.predict(np.zeros((1, 128, 128, 3)))

# Auto-detect last Conv2D layer for Grad-CAM
LAST_CONV_LAYER = None
for layer in reversed(image_model.layers):
    if "conv" in layer.name.lower() and "pool" not in layer.name.lower():
        LAST_CONV_LAYER = layer.name
        break
print(f"[INFO] GradCAM layer: {LAST_CONV_LAYER}")


# ── Text Processing ───────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_symptoms(text: str) -> dict:
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=150)
    pred_probs = nlp_model.predict(padded)[0]
    return {
        label_encoder.inverse_transform([i])[0]: float(pred_probs[i])
        for i in range(len(pred_probs))
    }


# ── Image Processing ──────────────────────────────────────────────────────
def predict_image(img_array: np.ndarray) -> dict:
    prediction = image_model.predict(img_array)[0][0]

    print("Raw prediction:", prediction)

    return {
        "Malignant (Skin Cancer)": float(prediction),
        "Benign": float(1 - prediction),
    }
# def predict_image(img_array: np.ndarray) -> dict:
#     prediction = image_model.predict(img_array)[0][0]

#     print("Raw prediction:", prediction)   # 👈 RIGHT HERE

#     if prediction > 0.5:
#         return {
#             "Malignant (Skin Cancer)": float(prediction),
#             "Benign": float(1 - prediction),
#         }
#     else:
#         return {
#             "Benign": float(1 - prediction),
#             "Malignant (Skin Cancer)": float(prediction),
#         }



# def run_gradcam(filepath: str, img_array: np.ndarray) -> dict:
#     print("[DEBUG] run_gradcam called")
#     original_url = "/" + filepath.replace("\\", "/")
#     fallback = {"original": original_url, "heatmap": original_url, "overlay": original_url}

#     if LAST_CONV_LAYER is None:
#         return fallback

#     heatmap_path = os.path.join(STATIC_DIR, "heatmap.jpg")
#     overlay_path = os.path.join(STATIC_DIR, "overlay.jpg")

#     try:
#         heatmap = get_gradcam_heatmap(image_model, img_array, LAST_CONV_LAYER)

#         heatmap_resized = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_LINEAR)
#         heatmap_uint8 = np.uint8(255 * heatmap_resized)
#         heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
#         cv2.imwrite(heatmap_path, heatmap_colored)

#         orig_img = cv2.imread(filepath)
#         if orig_img is None:
#             raise ValueError(f"Cannot read image: {filepath}")
#         orig_img = cv2.resize(orig_img, (224, 224))
#         overlay_img = cv2.addWeighted(orig_img, 0.55, heatmap_colored, 0.45, 0)
#         cv2.imwrite(overlay_path, overlay_img)

#         return {
#             "original": original_url,
#             "heatmap": "/static/heatmap.jpg",
#             "overlay": "/static/overlay.jpg",
#         }
#     except Exception as e:
#         print(f"[GradCAM] ERROR: {e}")
#         return fallback
def run_gradcam(filepath: str, img_array: np.ndarray) -> dict:
    print("[DEBUG] run_gradcam called")

    filename = os.path.basename(filepath)

    original_url = f"/static/{filename}"
    heatmap_url = "/static/heatmap.jpg"
    overlay_url = "/static/overlay.jpg"

    fallback = {
        "original": original_url,
        "heatmap": original_url,
        "overlay": original_url
    }

    if LAST_CONV_LAYER is None:
        return fallback

    heatmap_path = os.path.join(STATIC_DIR, "heatmap.jpg")
    overlay_path = os.path.join(STATIC_DIR, "overlay.jpg")

    try:
        # Generate heatmap
        heatmap = get_gradcam_heatmap(
            image_model,
            img_array,
            LAST_CONV_LAYER
        )

        # Resize heatmap
        heatmap_resized = cv2.resize(
            heatmap,
            (224, 224),
            interpolation=cv2.INTER_LINEAR
        )

        # Convert to colored heatmap
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )

        # Save heatmap properly
        cv2.imwrite(
            heatmap_path,
            heatmap_colored
        )

        # Read original image
        orig_img = cv2.imread(filepath)

        if orig_img is None:
            raise ValueError(f"Cannot read image: {filepath}")

        orig_img = cv2.resize(orig_img, (224, 224))

        # Create overlay
        overlay_img = cv2.addWeighted(
            orig_img,
            0.6,
            heatmap_colored,
            0.4,
            0
        )

        # Save overlay properly
        cv2.imwrite(
            overlay_path,
            overlay_img
        )

        return {
            "original": original_url,
            "heatmap": heatmap_url,
            "overlay": overlay_url,
        }

    except Exception as e:
        print(f"[GradCAM ERROR] {e}")
        return fallback



################################################
################################################
def get_suggestions(disease: str, risk: str = "Low") -> list:
    suggestions_db = {

        "Peptic ulcer disease": [
            "Avoid spicy and acidic foods",
            "Eat smaller, frequent meals",
            "Avoid NSAIDs like ibuprofen",
            "Consult a gastroenterologist"
        ],

        "Diabetes": [
            "Monitor blood sugar regularly",
            "Reduce sugar and refined carbs",
            "Exercise daily (30 mins)",
            "Follow prescribed medication strictly"
        ],

        "Drug Reaction": [
            "Stop the suspected medication immediately",
            "Consult a doctor before taking alternatives",
            "Watch for swelling or breathing difficulty",
            "Seek urgent care if symptoms worsen"
        ],

        "Gastroesophageal reflux disease": [
            "Avoid oily and spicy food",
            "Do not lie down after eating",
            "Eat smaller meals",
            "Elevate head while sleeping"
        ],

        "Allergy": [
            "Avoid known allergens",
            "Take antihistamines if prescribed",
            "Keep environment clean",
            "Consult doctor for severe reactions"
        ],

        "Urinary tract infection": [
            "Drink plenty of water",
            "Do not hold urine for long",
            "Maintain hygiene",
            "Consult doctor for antibiotics"
        ],

        "Malaria": [
            "Seek immediate medical treatment",
            "Take prescribed antimalarial drugs",
            "Avoid mosquito exposure",
            "Stay hydrated"
        ],

        "Jaundice": [
            "Avoid fatty foods",
            "Drink plenty of fluids",
            "Get liver function tested",
            "Consult a doctor immediately"
        ],

        "Cervical spondylosis": [
            "Maintain proper posture",
            "Avoid long screen time",
            "Do neck exercises",
            "Use ergonomic chair/pillow"
        ],

        "Migraine": [
            "Rest in a quiet dark room",
            "Avoid triggers (stress, caffeine)",
            "Stay hydrated",
            "Maintain sleep schedule"
        ],

        "Hypertension": [
            "Reduce salt intake",
            "Exercise regularly",
            "Avoid stress",
            "Monitor BP regularly"
        ],

        "Bronchial Asthma": [
            "Avoid dust and smoke",
            "Use inhaler as prescribed",
            "Monitor breathing",
            "Seek help during attacks"
        ],

        "Acne": [
            "Wash face regularly",
            "Avoid oily products",
            "Do not pop pimples",
            "Maintain healthy diet"
        ],

        "Arthritis": [
            "Exercise gently",
            "Maintain healthy weight",
            "Use hot/cold therapy",
            "Consult doctor for pain relief"
        ],

        "Dimorphic Hemorrhoids": [
            "Increase fiber intake",
            "Drink lots of water",
            "Avoid straining during bowel movement",
            "Consult doctor if bleeding occurs"
        ],

        "Pneumonia": [
            "Seek medical attention immediately",
            "Take prescribed antibiotics",
            "Get proper rest",
            "Stay hydrated"
        ],

        "Common Cold": [
            "Rest well",
            "Drink warm fluids",
            "Use steam inhalation",
            "Take OTC medicines if needed"
        ],

        "Fungal infection": [
            "Keep affected area dry",
            "Use antifungal creams",
            "Avoid sharing personal items",
            "Maintain hygiene"
        ],

        "Dengue": [
            "Drink fluids (very important)",
            "Avoid NSAIDs",
            "Monitor platelet count",
            "Seek immediate care if symptoms worsen"
        ],

        "Impetigo": [
            "Keep area clean",
            "Avoid scratching",
            "Use prescribed antibiotic cream",
            "Avoid close contact"
        ],

        "Chicken pox": [
            "Avoid scratching",
            "Use calamine lotion",
            "Stay isolated",
            "Rest and hydrate"
        ],

        "Typhoid": [
            "Take antibiotics as prescribed",
            "Drink boiled/clean water",
            "Avoid outside food",
            "Maintain hygiene"
        ],

        "Varicose Veins": [
            "Avoid standing for long periods",
            "Elevate legs while resting",
            "Exercise regularly",
            "Wear compression stockings"
        ],

        "Psoriasis": [
            "Keep skin moisturized",
            "Avoid triggers like stress",
            "Use medicated creams",
            "Consult dermatologist"
        ],

        "Malignant (Skin Cancer)": [
            "⚠️ Consult a dermatologist immediately",
            "Avoid sun exposure",
            "Monitor skin changes",
            "Biopsy may be required"
        ],

        "Benign": [
            "Condition appears non-cancerous",
            "Monitor for changes",
            "Maintain hygiene",
            "Routine checkup if needed"
        ]
    }

    default_suggestions = [
        "Stay hydrated",
        "Maintain a healthy diet",
        "Consult a doctor if symptoms persist"
    ]

    suggestions = suggestions_db.get(disease, default_suggestions)

    # Add risk-based urgency
    if risk == "High":
        suggestions = ["⚠️ Seek medical attention immediately"] + suggestions

    return suggestions
################################################
def translate_suggestions_to_hindi(suggestions):
    hindi_map = {

        "⚠️ Seek medical attention immediately":
            "तुरंत चिकित्सा सहायता लें",

        "Avoid spicy and acidic foods":
            "मसालेदार और खट्टे भोजन से बचें",

        "Eat smaller, frequent meals":
            "थोड़ा-थोड़ा और बार-बार भोजन करें",

        "Avoid NSAIDs like ibuprofen":
            "आइबुप्रोफेन जैसी दवाओं से बचें",

        "Consult a gastroenterologist":
            "गैस्ट्रोएंटेरोलॉजिस्ट से सलाह लें",

        "Monitor blood sugar regularly":
            "ब्लड शुगर नियमित रूप से जांचें",

        "Reduce sugar and refined carbs":
            "चीनी और रिफाइंड कार्बोहाइड्रेट कम करें",

        "Exercise daily (30 mins)":
            "रोज़ाना 30 मिनट व्यायाम करें",

        "Follow prescribed medication strictly":
            "डॉक्टर द्वारा दी गई दवाइयाँ नियमित लें",

        "Stop the suspected medication immediately":
            "संदिग्ध दवा तुरंत बंद करें",

        "Consult a doctor before taking alternatives":
            "दूसरी दवा लेने से पहले डॉक्टर से सलाह लें",

        "Watch for swelling or breathing difficulty":
            "सूजन या सांस लेने में कठिनाई पर ध्यान दें",

        "Seek urgent care if symptoms worsen":
            "लक्षण बढ़ने पर तुरंत इलाज करवाएं",

        "Avoid oily and spicy food":
            "तेल और मसालेदार भोजन से बचें",

        "Do not lie down after eating":
            "खाना खाने के बाद तुरंत न लेटें",

        "Eat smaller meals":
            "कम मात्रा में भोजन करें",

        "Elevate head while sleeping":
            "सोते समय सिर ऊँचा रखें",

        "Avoid known allergens":
            "एलर्जी पैदा करने वाली चीजों से बचें",

        "Take antihistamines if prescribed":
            "डॉक्टर द्वारा दी गई एंटीहिस्टामिन दवा लें",

        "Keep environment clean":
            "अपने आसपास सफाई रखें",

        "Consult doctor for severe reactions":
            "गंभीर प्रतिक्रिया होने पर डॉक्टर से संपर्क करें",

        "Drink plenty of water":
            "अधिक पानी पिएं",

        "Do not hold urine for long":
            "लंबे समय तक पेशाब न रोकें",

        "Maintain hygiene":
            "साफ-सफाई बनाए रखें",

        "Consult doctor for antibiotics":
            "एंटीबायोटिक के लिए डॉक्टर से सलाह लें",

        "Seek immediate medical treatment":
            "तुरंत चिकित्सा उपचार लें",

        "Take prescribed antimalarial drugs":
            "डॉक्टर द्वारा दी गई मलेरिया की दवाइयाँ लें",

        "Avoid mosquito exposure":
            "मच्छरों से बचाव करें",

        "Stay hydrated":
            "शरीर में पानी की कमी न होने दें",

        "Avoid fatty foods":
            "तैलीय भोजन से बचें",

        "Drink plenty of fluids":
            "अधिक तरल पदार्थ पिएं",

        "Get liver function tested":
            "लिवर की जांच करवाएं",

        "Consult a doctor immediately":
            "तुरंत डॉक्टर से संपर्क करें",

        "Maintain proper posture":
            "सही बैठने और खड़े होने की मुद्रा रखें",

        "Avoid long screen time":
            "लंबे समय तक स्क्रीन न देखें",

        "Do neck exercises":
            "गर्दन के व्यायाम करें",

        "Use ergonomic chair/pillow":
            "आरामदायक कुर्सी और तकिए का उपयोग करें",

        "Rest in a quiet dark room":
            "शांत और अंधेरे कमरे में आराम करें",

        "Avoid triggers (stress, caffeine)":
            "तनाव और कैफीन जैसी चीजों से बचें",

        "Maintain sleep schedule":
            "सोने का नियमित समय बनाए रखें",

        "Reduce salt intake":
            "नमक कम खाएं",

        "Exercise regularly":
            "नियमित व्यायाम करें",

        "Avoid stress":
            "तनाव से बचें",

        "Monitor BP regularly":
            "ब्लड प्रेशर नियमित जांचें",

        "Avoid dust and smoke":
            "धूल और धुएं से बचें",

        "Use inhaler as prescribed":
            "डॉक्टर के अनुसार इनहेलर का उपयोग करें",

        "Monitor breathing":
            "सांस लेने पर ध्यान दें",

        "Seek help during attacks":
            "अटैक आने पर तुरंत मदद लें",

        "Wash face regularly":
            "चेहरा नियमित रूप से धोएं",

        "Avoid oily products":
            "तेल वाले उत्पादों से बचें",

        "Do not pop pimples":
            "मुहांसों को न फोड़ें",

        "Maintain healthy diet":
            "स्वस्थ आहार लें",

        "Exercise gently":
            "हल्का व्यायाम करें",

        "Maintain healthy weight":
            "स्वस्थ वजन बनाए रखें",

        "Use hot/cold therapy":
            "गर्म और ठंडी सिकाई करें",

        "Consult doctor for pain relief":
            "दर्द के लिए डॉक्टर से सलाह लें",

        "Increase fiber intake":
            "फाइबर युक्त भोजन बढ़ाएं",

        "Drink lots of water":
            "भरपूर पानी पिएं",

        "Avoid straining during bowel movement":
            "मल त्याग के समय ज़ोर न लगाएं",

        "Consult doctor if bleeding occurs":
            "खून आने पर डॉक्टर से संपर्क करें",

        "Take prescribed antibiotics":
            "डॉक्टर द्वारा दी गई एंटीबायोटिक लें",

        "Get proper rest":
            "पूरा आराम करें",

        "Rest well":
            "अच्छी तरह आराम करें",

        "Drink warm fluids":
            "गर्म तरल पदार्थ पिएं",

        "Use steam inhalation":
            "भाप लें",

        "Take OTC medicines if needed":
            "ज़रूरत पड़ने पर सामान्य दवाइयाँ लें",

        "Keep affected area dry":
            "प्रभावित जगह को सूखा रखें",

        "Use antifungal creams":
            "एंटीफंगल क्रीम का उपयोग करें",

        "Avoid sharing personal items":
            "अपनी निजी चीजें साझा न करें",

        "Drink fluids (very important)":
            "अधिक तरल पदार्थ पिएं, यह बहुत ज़रूरी है",

        "Avoid NSAIDs":
            "एनएसएआईडी दवाओं से बचें",

        "Monitor platelet count":
            "प्लेटलेट्स की जांच करवाएं",

        "Seek immediate care if symptoms worsen":
            "लक्षण बढ़ने पर तुरंत डॉक्टर से संपर्क करें",

        "Keep area clean":
            "प्रभावित जगह साफ रखें",

        "Avoid scratching":
            "खुजली करने से बचें",

        "Use prescribed antibiotic cream":
            "डॉक्टर द्वारा दी गई एंटीबायोटिक क्रीम लगाएं",

        "Avoid close contact":
            "करीबी संपर्क से बचें",

        "Use calamine lotion":
            "कैलामाइन लोशन लगाएं",

        "Stay isolated":
            "अलग रहें",

        "Rest and hydrate":
            "आराम करें और पर्याप्त पानी पिएं",

        "Drink boiled/clean water":
            "उबला या साफ पानी पिएं",

        "Avoid outside food":
            "बाहर का खाना न खाएं",

        "Avoid standing for long periods":
            "लंबे समय तक खड़े न रहें",

        "Elevate legs while resting":
            "आराम करते समय पैरों को ऊँचा रखें",

        "Wear compression stockings":
            "कम्प्रेशन स्टॉकिंग्स पहनें",

        "Keep skin moisturized":
            "त्वचा को मॉइस्चराइज रखें",

        "Avoid triggers like stress":
            "तनाव जैसी चीजों से बचें",

        "Use medicated creams":
            "औषधीय क्रीम का उपयोग करें",

        "Consult dermatologist":
            "त्वचा विशेषज्ञ से सलाह लें",

        "⚠️ Consult a dermatologist immediately":
            "तुरंत त्वचा विशेषज्ञ से संपर्क करें",

        "Avoid sun exposure":
            "धूप से बचें",

        "Monitor skin changes":
            "त्वचा में बदलाव पर ध्यान दें",

        "Biopsy may be required":
            "बायोप्सी की आवश्यकता हो सकती है",

        "Condition appears non-cancerous":
            "स्थिति कैंसर जैसी नहीं लग रही",

        "Monitor for changes":
            "बदलावों पर नज़र रखें",

        "Routine checkup if needed":
            "ज़रूरत होने पर नियमित जांच करवाएं",

        "⚠️ Models disagree":
            "दोनों मॉडल के परिणाम अलग हैं",

        "Consult a doctor for accurate diagnosis":
            "सही निदान के लिए डॉक्टर से सलाह लें",

        "Do not rely solely on AI output":
            "केवल एआई परिणामों पर निर्भर न रहें"
    }

    translated = []

    for s in suggestions:
        translated.append(hindi_map.get(s, s))

    return translated

#################################


def build_voice_response(
    disease,
    confidence,
    suggestions,
    mode="general"
):
    confidence_pct = int(confidence * 100)

    # top_suggestions = suggestions[:3]
    top_suggestions = translate_suggestions_to_hindi(
        suggestions[:4]
    )

    intro = ""

    if mode == "image_only":
        intro = "इमेज विश्लेषण के आधार पर"

    elif mode == "symptom_only":
        intro = "दिए गए लक्षणों के आधार पर"

    elif mode == "both_agree":
        intro = "इमेज और लक्षण दोनों के विश्लेषण के आधार पर"

    else:
        intro = "एआई विश्लेषण के अनुसार"

    return f"""
{intro} आपको {disease} हो सकता है।

विश्वास स्तर लगभग {confidence_pct} प्रतिशत है।

कृपया ध्यान दें कि यह अंतिम चिकित्सा निदान नहीं है।

सही सलाह और उपचार के लिए डॉक्टर से संपर्क करें।

मुख्य सुझाव:
{". ".join(top_suggestions)}
"""
################################################

# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    symptoms = request.form.get("symptoms", "") or ""
    file = request.files.get("image")

    has_image = file is not None and file.filename != ""
    has_text = symptoms.strip() != ""

    if not has_image and not has_text:
        return jsonify({"error": "Please provide image or symptoms"}), 400

    image_preds = None
    symptom_preds = None
    img_array = None
    filepath = None
    images = None

    if has_image:
        # filepath = os.path.join("static", file.filename)
        filepath = os.path.join(STATIC_DIR, file.filename)

        file.save(filepath)

        img = cv2.imread(filepath)
        if img is None:
            return jsonify({"error": "Invalid image file"}), 400

        # 🔥 FIX: Convert BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_resized = cv2.resize(img, (128, 128))
        img_array = np.expand_dims(img_resized / 255.0, axis=0)
        image_preds = predict_image(img_array)
        images = run_gradcam(filepath, img_array)

    if has_text:
        symptom_preds = predict_symptoms(symptoms)

    # Case 1: Image only
    if has_image and not has_text:
        final_preds = normalize(image_preds)
        result = get_top_prediction(final_preds)
        top3 = get_top_k(final_preds)


        ###########
        confidence = float(final_preds[result])

        risk = "Low"
        if confidence > 0.7:
            risk = "High"
        elif confidence > 0.4:
            risk = "Medium"

        suggestions = get_suggestions(result, risk)

        hindi_response = build_voice_response(
            result,
            confidence,
            suggestions,
            "image_only"
        )


        ###########
        return jsonify({
            "mode": "image_only",
            "primary_disease": result,
            "confidence": float(final_preds[result]),
            "top_predictions": top3,
            "images": images,
            "suggestions": suggestions,   # 👈 THIS LINE IS NEW

            "hindi_response": hindi_response
        })

    # Case 2: Symptoms only
    if has_text and not has_image:
        final_preds = normalize(symptom_preds)
        result = get_top_prediction(final_preds)
        top3 = get_top_k(final_preds)

         # ✅ NEW
        confidence = float(final_preds[result])

        risk = "Low"
        if confidence > 0.7:
            risk = "High"
        elif confidence > 0.4:
            risk = "Medium"



        suggestions = get_suggestions(result, risk)

        hindi_response = build_voice_response(
            result,
            confidence,
            suggestions,
            "symptom_only"
        )

       

        return jsonify({
            "mode": "symptom_only",
            "primary_disease": result,
            "confidence": float(final_preds[result]),
            "top_predictions": top3,
            "suggestions": suggestions,   # 👈 NEW

            "hindi_response": hindi_response
        })
    

    # Case 3: Both
    img_norm = normalize(image_preds)
    sym_norm = normalize(symptom_preds)

    img_result = get_top_prediction(img_norm)
    sym_result = get_top_prediction(sym_norm)

    img_conf = float(img_norm[img_result])
    sym_conf = float(sym_norm[sym_result])

    # ✅ If both agree
    if img_result == sym_result:
        final_result = img_result
        confidence = (img_conf + sym_conf) / 2
        mode = "both_agree"

    # ❗ If they disagree (conflict)
    

    else:

        hindi_response = """
        इमेज विश्लेषण और लक्षण विश्लेषण के परिणाम अलग-अलग हैं।

        कृपया सही जांच और उपचार के लिए डॉक्टर से संपर्क करें।

        केवल एआई परिणामों पर पूरी तरह निर्भर न रहें।
        """

        return jsonify({
            "mode": "conflict",
            "image_prediction": img_result,
            "symptom_prediction": sym_result,
            "image_confidence": img_conf,
            "symptom_confidence": sym_conf,
            "top_predictions": get_top_k(sym_norm),
            "images": images,
            "suggestions": [
                "⚠️ Models disagree",
                "Consult a doctor for accurate diagnosis",
                "Do not rely solely on AI output"
            ],

            "hindi_response": hindi_response
        })

    # ✅ Risk calculation
    risk = "Low"
    if confidence > 0.7:
        risk = "High"
    elif confidence > 0.4:
        risk = "Medium"

    suggestions = get_suggestions(final_result, risk)

    hindi_response = build_voice_response(
        final_result,
        confidence,
        suggestions,
        "both_agree"
    )

    return jsonify({
        "mode": mode,
        "primary_disease": final_result,
        "confidence": confidence,
        "top_predictions": get_top_k(sym_norm),
        "images": images,
        "suggestions": suggestions,

        "hindi_response": hindi_response
    })


# ── Offline voice endpoint ────────────────────────────────────────────────
# Called by voice/voice_client.py — accepts JSON {"symptoms": "..."}
# symptoms must be in ENGLISH (voice_client.py handles translation from Hindi)

@app.route("/voice-predict", methods=["POST"])
def voice_predict():
    """
    Lightweight endpoint for the offline voice client.

    Request  JSON : {"symptoms": "fever headache cough"}   ← English text
    Response JSON : same structure as /predict (symptom_only mode)
    """
    body = request.get_json(silent=True) or {}
    symptoms = body.get("symptoms", "").strip()

    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    # Clean + predict
    preds = predict_symptoms(symptoms)
    final_preds = normalize(preds)
    result = get_top_prediction(final_preds)
    top3 = get_top_k(final_preds)

    return jsonify({
        "mode": "symptom_only",
        "primary_disease": result,
        "confidence": float(final_preds[result]),
        "top_predictions": top3,
    })



############################





# ── Run ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True)