import cv2
import numpy as np
import pandas as pd
import pytesseract
from transformers import pipeline
from ultralytics import YOLO
import joblib  # For loading trained models
from xgboost import XGBClassifier
from deepface.DeepFace import represent
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cosine

# 🔹 Set Tesseract OCR Path (Update this if needed)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Users\karth\OneDrive\Documents\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = r"/opt/homebrew/bin/tesseract"

# 🔹 Load Models
print("Loading models...")
caption_generator = pipeline("summarization", model="facebook/bart-large-cnn")
genre_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
character_detector = YOLO("yolov8_comic_characters_detect.pt")
xgboost_model = joblib.load("xgboost_spoiler_classifier.pkl")  # Load trained XGBoost model
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")  # Load TF-IDF vectorizer
genre_encoder = joblib.load("genre_encoder.pkl")  # Load genre encoder

# Define genre labels (Ensure consistency with training data)
genre_labels = ["Sports", "Crime", "Action", "Fantasy", "Sci-Fi", "Romance", "Horror", "Comedy", "Drama", "Mystery", "Superhero"]

# 🔹 Feature Extraction Functions
def extract_text_from_image(image_path):
    """Extract text from an image using Tesseract OCR."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return ""
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text.strip()

def generate_caption(text):
    """Generate a short caption using BART summarization."""
    if not text or text.strip() == "":
        return "No caption available"
    
    try:
        summary = caption_generator(text, max_length=50, min_length=10, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return "Error generating caption"

def predict_genre(text):
    """Predict comic genre using BART-based zero-shot classification."""
    if not text.strip():
        return "Unknown"

    result = genre_classifier(text, genre_labels)
    predicted_genre = result["labels"][0]

    # Ensure predicted genre is within allowed genres
    if predicted_genre not in genre_labels:
        predicted_genre = "Unknown"

    return predicted_genre

def get_face_embedding(face):
    """Extract face embeddings using DeepFace."""
    try:
        face_embedding = represent(face, model_name="Facenet", enforce_detection=False)
        return face_embedding[0]['embedding']
    except Exception as e:
        print(f"Error extracting face embedding: {e}")
        return None

def detect_unique_characters(image_path):
    """Detect unique characters in a comic page image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return 0
    
    results = character_detector(image)
    face_embeddings = []

    # Extract detected characters
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            face = image[y1:y2, x1:x2]
            if face.size == 0:
                continue
            
            embedding = get_face_embedding(face)
            if embedding is not None:
                face_embeddings.append(embedding)
    
    if not face_embeddings:
        print("Unique characters detected: 0")
        return 0

    # Cluster face embeddings to identify unique characters
    clustering = DBSCAN(metric=cosine, eps=0.5, min_samples=1)
    labels = clustering.fit_predict(face_embeddings)
    
    unique_characters = len(set(labels))
    print(f"Unique characters detected: {unique_characters}")
    return unique_characters

# 🔹 Spoiler Prediction Function
def predict_spoiler(image_path):
    """Predict whether a comic panel contains a spoiler."""
    print(f"Processing image: {image_path}...")

    # Step 1: Extract text
    extracted_text = extract_text_from_image(image_path)
    print(f"Extracted Text: {extracted_text}")

    # Step 2: Generate Caption
    caption = generate_caption(extracted_text)
    print(f"Generated Caption: {caption}")

    # Step 3: Predict Genre
    genre = predict_genre(extracted_text)
    print(f"Predicted Genre: {genre}")

    # Step 4: Count Unique Characters
    unique_character_count = detect_unique_characters(image_path)
    print(f"Unique Character Count: {unique_character_count}")

    # Step 5: Encode Genre
    if genre in genre_encoder.classes_:
        genre_encoded = genre_encoder.transform([genre])[0]
    else:
        genre_encoded = -1  # Assign a default value for unknown genres

    # Step 6: Convert Caption to TF-IDF Features
    caption_features = tfidf_vectorizer.transform([caption]).toarray()  # Shape (1, 500)

    # Step 7: Fix Dimension Mismatch
    numeric_features = np.array([[unique_character_count, genre_encoded]])  # Shape (1, 2)
    X_input = np.hstack((numeric_features, caption_features))  # Shape (1, 502)

    # Step 8: Ensure Correct Shape Before Prediction
    X_input = np.array(X_input).reshape(1, -1)  # Ensure 2D array shape (1, 502)

    # Step 9: Predict Spoiler
    prediction = xgboost_model.predict(X_input)[0]

    # Step 10: Interpret Prediction
    # label_mapping = {idx: label for idx, label in enumerate(xgboost_model.classes_)}
    # spoiler_label = label_mapping.get(prediction, "Unknown")
    label_mapping = {0: "Unknown", 1: "Non-Spoiler", 2: "Spoiler"}
    print(label_mapping.get(prediction, "Unknown"))

    # print(f"Predicted Spoiler Label: {spoiler_label}")
    # return spoiler_label


# 🔹 Run the Prediction
if __name__ == "__main__":
    # test_image = r"C:\Users\karth\OneDrive\Desktop\mpp\tes.jpg"

    test_image = r'/Users/sj/Documents/MinorProject22/002a6441-5424-4f66-8998-d74a052b92ec (1).jpg'
    # test_image = r'/Users/sj/Documents/MinorProject22/0adacd53-c788-470a-8c38-b68449d504a9.jpg'

    #test_image = r"C:\Users\karth\OneDrive\Desktop\mpp\Screenshot 2025-05-23 125850.png"
    # test_image = r"cpp\002a6441-5424-4f66-8998-d74a052b92ec.jpg"  # Change to your test image path
    # test_image = r"C:\Users\karth\OneDrive\Desktop\Screenshot 2025-04-19 143841.png"  # Change to your test image path
    # test_image = r"cpp\48ea2bac-b410-487c-a522-970dd2a2e135.jpg"  # Change to your test image path
    # test_image = r"C:\Users\karth\OneDrive\Desktop\RCO013_1559819589.jpg"  # Change to your test image path
    predict_spoiler(test_image)



