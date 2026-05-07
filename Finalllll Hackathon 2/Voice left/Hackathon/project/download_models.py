import gdown
import os

os.makedirs("model", exist_ok=True)

files = {
    "model/disease_prediction_model.h5": "1kVri61LgeJzgh_aZG2-HH83E45am0rGB",
    "model/skin_cancer_cnn.h5": "1npshfwh7RFqwtsz0OUaOzSqAtDSPoArS",
    "model/preprocessing.pkl": "1iZiQw_1sjyB7kHF4DFhg3lmFbF4gX6Oh",
}

for path, file_id in files.items():
    if not os.path.exists(path):
        print(f"Downloading {path}...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", path, quiet=False)

print("All models ready!")