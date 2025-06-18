# 📚 ComicSpoilerDetection

This project detects **spoilers in comic panels** using a combination of Natural Language Processing (NLP) and Computer Vision. It leverages **TF-IDF + XGBoost** for text classification and **YOLOv8** for visual character recognition.

---

## 🔍 Project Structure

ComicSpoilerDetection/
├── data/ # Dataset files (CSV)
├── models/ # Trained ML and YOLO models (.pkl, .pt)
├── images/ # Sample/test comic panel images
├── src/ # Source code (e.g., demo2.py)
├── README.md
├── requirements.txt
└── .gitignore


---

## 🚀 Run on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/djcode0718/ComicSpoilerDetection/blob/main/notebooks/ComicSpoilerDetection.ipynb)


---

## 🧠 Technologies Used

- Python 3.x
- YOLOv8 (Ultralytics)
- OpenCV
- pandas, numpy, matplotlib
- scikit-learn
- XGBoost
- Pickle for model saving/loading

---

## 🛠️ How to Use

1. Clone the repository:
   git clone https://github.com/your-username/ComicSpoilerDetection.git
   cd ComicSpoilerDetection

2. Install dependencies:
   pip install -r requirements.txt

3. Run the demo script:
   python src/demo2.py



📂 Contents
| Folder/File        | Description                                                           |
| ------------------ | --------------------------------------------------------------------- |
| `data/`            | Contains the spoiler dataset (`character_counts.csv`)                 |
| `models/`          | Pretrained models: TF-IDF vectorizer, genre encoder, classifier, YOLO |
| `images/`          | Test comic panels                                                     |
| `src/`             | Main processing and detection script (`demo2.py`)                     |
| `requirements.txt` | Project dependencies                                                  |
| `README.md`        | Project overview                                                      |



📌 Notes:
This project is educational and for experimentation only.

Consider using Git LFS or cloud storage for large model files.


🧪 Optional Check:
You can test these locally by creating a virtual environment:

   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

Then run your script to make sure no ModuleNotFoundError occurs.

