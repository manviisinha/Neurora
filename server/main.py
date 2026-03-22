import easyocr
import numpy as np
from PIL import Image
import io
import re
import cv2
import difflib
import sqlite3
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize EasyOCR reader (Load once)
reader = easyocr.Reader(['en'], gpu=False)

DB_PATH = os.path.join(os.path.dirname(__file__), "medicines.db")
CSV_PATH = os.path.join(os.path.dirname(__file__), "A_Z_medicines_dataset_of_India.csv")

def init_db():
    """Build the SQLite database from CSV if it doesn't exist."""
    if os.path.exists(DB_PATH):
        print("Database already exists.")
        return

    print("Building medicines database (this may take a minute)...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE medicines (
                        name TEXT, 
                        clean_name TEXT, 
                        uses TEXT)''')
    cursor.execute('CREATE INDEX idx_clean_name ON medicines(clean_name)')
    
    import csv
    with open(CSV_PATH, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        to_db = []
        seen = set()
        for row in reader:
            if len(row) > 7:
                name = row[1]
                comp1 = row[7].strip()
                comp2 = row[8].strip() if len(row) > 8 else ""
                composition = f"{comp1}, {comp2}".strip(", ")
                
                clean_name = re.sub(r'\s+(Tablet|Capsule|Syrup|Injection|Duo|Liquid|Drops|Suspension|mg|gm|mcg)\b.*', '', name, flags=re.IGNORECASE).strip()
                clean_lower = clean_name.lower()
                
                if clean_lower and clean_lower not in seen:
                    to_db.append((clean_name, clean_lower, composition))
                    seen.add(clean_lower)
        
        cursor.executemany("INSERT INTO medicines VALUES (?, ?, ?)", to_db)
    
    conn.commit()
    conn.close()
    print("Database built successfully!")

init_db()

STOPWORDS = {"doctor", "patient", "hospital", "clinic", "date", "morning", "afternoon", "evening", "night", "male", "female", "age", "weight", "name", "sex"}

def preprocess_image(image):
    img = np.array(image)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    img = cv2.bilateralFilter(img, 11, 85, 85)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)
    return img

def query_medicine(word):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Try exact match
    cursor.execute("SELECT name, uses FROM medicines WHERE clean_name = ?", (word,))
    res = cursor.fetchone()
    if res:
        conn.close()
        return res

    # 2. Try fuzzy match (only for words > 3 chars)
    if len(word) >= 4:
        first_char = word[0]
        cursor.execute("SELECT name, clean_name, uses FROM medicines WHERE clean_name LIKE ?", (f"{first_char}%",))
        targets = cursor.fetchall() # This only pulls a subset into memory
        
        matches = difflib.get_close_matches(word, [t[1] for t in targets], n=1, cutoff=0.75)
        if matches:
            for t in targets:
                if t[1] == matches[0]:
                    conn.close()
                    return (t[0], t[2])
                    
    conn.close()
    return None

def extract_medicines(text):
    text = re.sub(r'\b8\s*([a-zA-Z])', r'S\1', text, flags=re.IGNORECASE)
    clean_text = re.sub(r'[^a-zA-Z0-9\s/.-]', ' ', text)
    words = clean_text.split()
    
    medicines = []
    seen_names = set()
    
    for word in words:
        word_lower = word.lower()
        if len(word_lower) < 3 or word_lower in STOPWORDS:
            continue
            
        res = query_medicine(word_lower)
        if res and res[0] not in seen_names:
            medicines.append({"name": res[0], "uses": res[1], "dose": "As directed"})
            seen_names.add(res[0])
            
    return medicines

@app.get("/health")
async def health():
    return {"status": "ok", "db_found": os.path.exists(DB_PATH)}

@app.post("/upload-prescription")
async def upload_prescription(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        preprocessed_img = preprocess_image(image)
        result = reader.readtext(preprocessed_img, detail=0)
        extracted_text = " ".join(result)
        medicines = extract_medicines(extracted_text)
        return {"status": "success", "raw_text": extracted_text, "medicines": medicines}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
