############################################# IMPORTING ################################################
import tkinter as tk
from tkinter import ttk, messagebox as mess
from tkinter import simpledialog as tsd
import os
import cv2
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time

############################################# FUNCTIONS ################################################

def assure_path_exists(path):
    directory = os.path.dirname(path)
    if directory == "":
        directory = path
    if not os.path.exists(directory):
        os.makedirs(directory)





def tick():
    time_string = time.strftime('%H:%M:%S')
    clock.config(text=time_string)
    clock.after(200, tick)

def contact():
    mess.showinfo("Contact Us", "Please contact us at: 'xxxxxxxxxxxxx@gmail.com'")

def check_haarcascadefile():
    haarcascade_path = os.path.join(os.getcwd(), "haarcascade_frontalface_default.xml")
    if not os.path.isfile(haarcascade_path):
        mess.showerror("Missing File", "haarcascade_frontalface_default.xml not found!")
        window.destroy()

def save_pass():
    assure_path_exists("TrainingImageLabel")
    file_path = os.path.join("TrainingImageLabel", "psd.txt")
    
    if os.path.isfile(file_path):
        with open(file_path, "r") as tf:
            key = tf.read()
    else:
        new_pas = tsd.askstring("New Password", "Enter new password:", show='*')
        if not new_pas:
            mess.showwarning("Error", "Password not set!")
            return
        with open(file_path, "w") as tf:
            tf.write(new_pas)
        mess.showinfo("Success", "New password registered!")
        return

    op = old.get()
    newp = new.get()
    nnewp = nnew.get()

    if op != key:
        mess.showerror("Error", "Incorrect old password!")
        return
    if newp != nnewp:
        mess.showerror("Error", "New passwords don't match!")
        return

    with open(file_path, "w") as txf:
        txf.write(newp)
    mess.showinfo("Success", "Password changed successfully!")
    master.destroy()

def change_pass():
    global master, old, new, nnew
    master = tk.Toplevel()
    master.title("Change Password")
    master.geometry("400x200")
    master.configure(bg='white')

    tk.Label(master, text="Old Password:", bg='white').place(x=10, y=10)
    old = tk.Entry(master, show='*', width=25)
    old.place(x=150, y=10)

    tk.Label(master, text="New Password:", bg='white').place(x=10, y=50)
    new = tk.Entry(master, show='*', width=25)
    new.place(x=150, y=50)

    tk.Label(master, text="Confirm New:", bg='white').place(x=10, y=90)
    nnew = tk.Entry(master, show='*', width=25)
    nnew.place(x=150, y=90)

    tk.Button(master, text="Save", command=save_pass, width=10).place(x=100, y=130)
    tk.Button(master, text="Cancel", command=master.destroy, width=10).place(x=200, y=130)

def psw():
    file_path = os.path.join("TrainingImageLabel", "psd.txt")
    if os.path.isfile(file_path):
        with open(file_path, "r") as tf:
            key = tf.read()
    else:
        key = tsd.askstring("New Password", "Set new password:", show='*')
        if not key:
            mess.showwarning("Error", "Password required!")
            return
        with open(file_path, "w") as tf:
            tf.write(key)
    
    password = tsd.askstring("Password", "Enter password:", show='*')
    if password == key:
        TrainImages()
    elif password is None:
        return
    else:
        mess.showerror("Error", "Incorrect password!")

def clear():
    txt.delete(0, 'end')
    message1.config(text="1)Take Images  >>>  2)Save Profile")

def clear2():
    txt2.delete(0, 'end')
    message1.config(text="1)Take Images  >>>  2)Save Profile")

def TakeImages():
    global txt, txt2, message1

    # Validate ID and Name Inputs
    try:
        Id = int(txt.get().strip())  # Ensure ID is an integer
    except ValueError:
        mess.showerror("Error", "ID must be a numeric value!")
        return

    name = txt2.get().strip()
    if not name.replace(' ', '').isalpha():
        mess.showerror("Error", "Name must contain only letters!")
        return

    # Directories and CSV Setup
    columns = ['SERIAL NO.', 'ID', 'NAME']
    student_dir = "StudentDetails"
    training_dir = "TrainingImage"
    assure_path_exists(student_dir)
    assure_path_exists(training_dir)

    # Read/Initialize CSV
    csv_path = os.path.join(student_dir, "StudentDetails.csv")
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        serial = df.size // 3 + 1  # Calculate next serial number
    else:
        serial = 0
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(columns)

    # DNN Face Detector Setup
    dnn_config = "deploy.prototxt"
    dnn_model = "res10_300x300_ssd_iter_140000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(dnn_config, dnn_model)

    # Capture Faces
    cam = cv2.VideoCapture(0)
    sampleNum = 0

    while True:
        ret, img = cam.read()
        if not ret:
            break

        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w, endX), min(h, endY)

                face = gray[startY:endY, startX:endX]
                face = cv2.resize(face, (200, 200))

                # Save face with ID in filename
                img_name = os.path.join(training_dir, f"{name}.{serial}.{Id}.{sampleNum}.jpg")
                cv2.imwrite(img_name, face)
                cv2.rectangle(img, (startX, startY), (endX, endY), (255, 0, 0), 2)
                sampleNum += 1

        cv2.imshow('Capturing Faces', img)
        if cv2.waitKey(100) == ord('q') or sampleNum >= 100:
            break

    cam.release()
    cv2.destroyAllWindows()

    # Update CSV
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([serial, Id, name])

    message1.config(text=f"Registered: {name} (ID: {Id})")



def TrainImages():
    check_haarcascadefile()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    faces, ids = getImagesAndLabels("TrainingImage")
    if not ids:
        mess.showerror("Error", "No training data found!")
        return

    recognizer.train(faces, np.array(ids))
    recognizer.save(os.path.join("TrainingImageLabel", "Trainner.yml"))
    message1.config(text="Profile saved successfully!")
    message.config(text=f"Total Registrations: {len(set(ids))}")

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []

    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        img_numpy = np.array(pilImage, 'uint8')
        
        # Extract ID from filename (name.serial.id.sample.jpg)
        try:
            Id = int(os.path.split(imagePath)[1].split('.')[2])
        except (IndexError, ValueError):
            continue
        
        faces.append(img_numpy)
        ids.append(Id)
    
    return faces, ids

def TrackImages():
    global tv

    # Load Recognizer and Student Data
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(os.path.join("TrainingImageLabel", "Trainner.yml"))
    student_csv = os.path.join("StudentDetails", "StudentDetails.csv")
    df = pd.read_csv(student_csv)

    # DNN Face Detector
    dnn_config = r"D:\RTendace\Face Recognition Based Attendance Monitoring System\deploy.prototxt"
    dnn_model = r"D:\RTendace\Face Recognition Based Attendance Monitoring System\res10_300x300_ssd_iter_140000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(dnn_config, dnn_model)

    # Attendance Setup
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    attendance_list = []
    recorded_ids = set()

    while True:
        ret, im = cam.read()
        if not ret:
            break

        h, w = im.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(im, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w, endX), min(h, endY)

                face_roi = im[startY:endY, startX:endX]
                gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (200, 200))

                # Predict ID and Confidence
                Id_pred, confidence = recognizer.predict(gray)
                cv2.putText(im, f"Conf: {confidence:.2f}", (startX, startY-10), font, 0.5, (0, 255, 0), 1)

                if confidence < 55:  # Lower confidence threshold for better recognition
                    student_info = df[df['ID'] == Id_pred]
                    if not student_info.empty:
                        student_name = student_info.iloc[0]['NAME']
                        student_id = student_info.iloc[0]['ID']

                        if student_id not in recorded_ids:
                            ts = time.time()
                            date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                            attendance_list.append([student_id, student_name, date, timeStamp])
                            recorded_ids.add(student_id)

                        cv2.putText(im, student_name, (startX, endY), font, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(im, "Unknown", (startX, endY), font, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(im, "Unknown", (startX, endY), font, 1, (0, 0, 255), 2)

                cv2.rectangle(im, (startX, startY), (endX, endY), (0, 255, 0), 2)

        cv2.imshow('Attendance System', im)
        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    # Save Attendance
    if attendance_list:
        date_str = datetime.datetime.now().strftime('%d-%m-%Y')
        csv_path = os.path.join("Attendance", f"Attendance_{date_str}.csv")
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for entry in attendance_list:
                writer.writerow(entry)
                tv.insert('', 'end', values=(entry[0], entry[1], f"{entry[2]} {entry[3]}"))
    else:
        mess.showinfo("No Attendance", "No faces recognized today!")
######################################## GUI SETUP ###########################################

window = tk.Tk()
window.title("Attendance System")
window.geometry("1280x720")
window.configure(bg='#262523')

# Date and Clock
date_label = tk.Label(window, bg='#262523', fg='white', 
                     font=('arial', 20))
date_label.place(relx=0.35, rely=0.05)

clock = tk.Label(window, bg='#262523', fg='white', 
                font=('arial', 20))
clock.place(relx=0.52, rely=0.05)
tick()

# Frames
frame_left = tk.Frame(window, bg='#00aeff')
frame_left.place(relx=0.1, rely=0.2, relwidth=0.4, relheight=0.7)

frame_right = tk.Frame(window, bg='#00aeff')
frame_right.place(relx=0.52, rely=0.2, relwidth=0.37, relheight=0.7)

# TreeView
tv = ttk.Treeview(frame_left, columns=('id', 'name', 'time'), show='headings')
tv.heading('id', text='ID')
tv.heading('name', text='Name')
tv.heading('time', text='Date/Time')
tv.column('id', width=80)
tv.column('name', width=120)
tv.column('time', width=150)
tv.place(relx=0.05, rely=0.1, relwidth=0.9, relheight=0.8)

# Input Fields
tk.Label(frame_right, text="ID:", bg='#00aeff').place(relx=0.1, rely=0.1)
txt = tk.Entry(frame_right, width=25)
txt.place(relx=0.3, rely=0.1)

tk.Label(frame_right, text="Name:", bg='#00aeff').place(relx=0.1, rely=0.2)
txt2 = tk.Entry(frame_right, width=25)
txt2.place(relx=0.3, rely=0.2)
# Status message label for feedback (insert this right after the input fields)
message1 = tk.Label(frame_right, text="", bg='#00aeff', fg='white', font=('arial', 12))
message1.place(relx=0.1, rely=0.3)
message = tk.Label(frame_right, text="", bg='#00aeff', fg='white', font=('arial', 12))
message.place(relx=0.1, rely=0.35)


# Buttons
btn_frame = tk.Frame(frame_right, bg='#00aeff')
btn_frame.place(relx=0.1, rely=0.4, relwidth=0.8, relheight=0.5)

tk.Button(btn_frame, text="Take Images", command=TakeImages, width=20).pack(pady=5)
tk.Button(btn_frame, text="Train Model", command=psw, width=20).pack(pady=5)
tk.Button(btn_frame, text="Take Attendance", command=TrackImages, width=20).pack(pady=5)
tk.Button(btn_frame, text="Clear", command=clear, width=20).pack(pady=5)
tk.Button(btn_frame, text="Quit", command=window.destroy, width=20).pack(pady=5)

# Menu
menubar = tk.Menu(window)
help_menu = tk.Menu(menubar, tearoff=0)
help_menu.add_command(label="Change Password", command=change_pass)
help_menu.add_command(label="Contact", command=contact)
help_menu.add_separator()
help_menu.add_command(label="Exit", command=window.destroy)
menubar.add_cascade(label="Help", menu=help_menu)
window.config(menu=menubar)

window.mainloop()