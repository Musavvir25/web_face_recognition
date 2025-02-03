from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import csv  # Built-in, no installation needed
import base64
import pandas as pd
import os
import datetime
from supabase import create_client, Client

app = Flask(__name__)

# Supabase Credentials
SUPABASE_URL = "https://tgbpkywgajahjpvjmmtz.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRnYnBreXdnYWphaGpwdmptbXR6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzg0NzczMzIsImV4cCI6MjA1NDA1MzMzMn0.HPBUJs8Ip0BqgEggbloThGp6NX0EQJEWeRa30fORhV0"

# Connect to Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# File Paths
STUDENT_CSV = r"D:\RTendace\StudentDetails\StudentDetails.csv"
TRAINER_FILE = r"D:\RTendace\TrainingImageLabel\Trainner.yml"

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load LBPH Face Recognizer
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_FILE)
except Exception as e:
    print("Error loading recognizer:", e)
    recognizer = None

# Load student details from CSV
if os.path.isfile(STUDENT_CSV):
    student_df = pd.read_csv(STUDENT_CSV)
else:
    student_df = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.json.get('image')
    if not data:
        return jsonify({'error': 'No image data provided'}), 400

    try:
        # Decode base64 image
        _, encoded = data.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        face_results = []
        for (x, y, w, h) in faces:
            face_roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            
            recognized_id, confidence = recognizer.predict(face_roi) if recognizer else (None, 100)
            recognized_name = "Unknown"
            
            if confidence < 55 and student_df is not None and recognized_id in student_df['ID'].values:
                recognized_name = student_df.loc[student_df['ID'] == recognized_id, 'NAME'].values[0]
                store_attendance(recognized_id, recognized_name, "CSE101")

            face_results.append({
                'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h),
                'recognized_id': recognized_id,
                'recognized_name': recognized_name,
                'confidence': confidence
            })

        return jsonify({'status': 'success', 'face_count': len(faces), 'faces': face_results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def store_attendance(student_id, student_name, course_code):
    now = datetime.datetime.now()
    date_today = now.strftime('%Y-%m-%d')
    time_now = now.strftime('%H:%M:%S')

    data = {
        "student_id": student_id,
        "student_name": student_name,
        "course_code": course_code,
        "date": date_today,
        "time": time_now
    }

    supabase.table("attendance").insert(data).execute()

if __name__ == '__main__':
    app.run(debug=True)
