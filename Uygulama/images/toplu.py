# -- coding: utf-8 --
import cv2
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
from sklearn.cluster import KMeans
import threading
import datetime
import pickle
import os
import time
from deepface import DeepFace
from deepface.commons import functions

# Renk veri kümesi (Büyük Harf ile yazıldı, sabit olduğu için)
COLOR_DATASET = {
    'Siyah': [(20, 20, 20), (50, 50, 50)],
    'Kahverengi': [(101, 67, 33), (150, 100, 60)],
    'Sarı': [(200, 180, 50), (255, 220, 100)],
    'Kızıl': [(150, 50, 50), (200, 80, 80)],
    'Gri': [(100, 100, 100), (180, 180, 180)],
    'Beyaz': [(200, 200, 200), (255, 255, 255)]
}

# Yüz Tanıma Sabitleri
KNOWN_FACES_DB = "known_faces.pkl" # Bilinen yüzlerin kaydedileceği dosya
FACE_RECOGNITION_TOLERANCE = 0.4 # Eşleşme toleransı (düşük değer = daha katı eşleşme)

# Global Değişkenler
running = False # Kamera döngüsünün çalışıp çalışmadığını kontrol eder
cap = None # Kamera nesnesi
label = None # Tkinter label for video feed
btn = None # Tkinter button to toggle camera
name_entry = None # Tkinter entry for face name
enroll_button = None # Tkinter button to enroll face
status_label = None # Tkinter label for status messages

# Bilinen yüzleri ve embedding'lerini saklayan dictionary
# Format: {'İsim': [embedding1, embedding2, ...], ...}
known_faces = {}

# --- Veritabanı Yükleme/Kaydetme Fonksiyonları ---
def load_known_faces(filename=KNOWN_FACES_DB):
    """Bilinen yüz veritabanını dosyadan yükler."""
    global known_faces
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                known_faces = pickle.load(f)
            print(f"Bilinen yüzler '{filename}' dosyasından yüklendi. Toplam {len(known_faces)} kişi.")
        except Exception as e:
            print(f"Veritabanı yüklenirken hata oluştu: {e}")
            known_faces = {} # Hata olursa boş başlat
    else:
        print(f"'{filename}' veritabanı dosyası bulunamadı. Yeni veritabanı oluşturuluyor.")
        known_faces = {}

def save_known_faces(filename=KNOWN_FACES_DB):
    """Bilinen yüz veritabanını dosyaya kaydeder."""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(known_faces, f)
        print(f"Bilinen yüzler '{filename}' dosyasına kaydedildi.")
    except Exception as e:
        print(f"Veritabanı kaydedilirken hata oluştu: {e}")

# --- Yüz Tanıma Yardımcı Fonksiyonları ---
def get_face_embedding(img):
    """DeepFace kullanarak yüz embedding'i çıkarır."""
    try:
        # DeepFace'in beklediği formatta ön işleme yap
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_img = functions.preprocess_face(img=img_rgb, target_size=(152, 152), 
                                           enforce_detection=False, detector_backend='opencv')
        if face_img.shape[1:3] == (152, 152): # Geçerli bir yüz bulundu
            embedding = DeepFace.represent(img_path=face_img, model_name='Facenet', enforce_detection=False)
            return embedding[0]["embedding"]
        return None
    except Exception as e:
        print(f"Embedding çıkarılırken hata: {e}")
        return None

def recognize_face(face_embedding):
    """Verilen embedding'i bilinen yüzlerle karşılaştırır ve ismi döndürür."""
    if not known_faces or face_embedding is None:
        return "Tanımlanmamış"

    # Bilinen tüm embedding'leri ve isimlerini ayrı listelere al
    known_encodings = []
    known_names = []
    
    for name, embeddings in known_faces.items():
        for emb in embeddings:
            known_encodings.append(emb)
            known_names.append(name)

    if not known_encodings:
        return "Tanımlanmamış"

    # DeepFace'in distance metriğini kullanarak karşılaştırma
    distances = []
    for known_emb in known_encodings:
        distance = functions.find_distance(known_emb, face_embedding)
        distances.append(distance)

    min_distance = min(distances)
    best_match_index = distances.index(min_distance)
    
    if min_distance < FACE_RECOGNITION_TOLERANCE:
        return known_names[best_match_index]
    else:
        return "Tanımlanmamış"

# --- Mevcut Analiz Fonksiyonları ---

# Baskın renk tespiti
def detect_dominant_color(image, k=3):
    if image is None or image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        return (0, 0, 0)
        
    h, w, channels = image.shape
    if channels == 1:
        return (0,0,0)
        
    pixels = image.reshape(h * w, channels)
    
    if pixels.shape[0] < k:
         k = pixels.shape[0]
         if k < 1:
             return (0,0,0)
    
    try:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42) 
        kmeans.fit(pixels)
        counts = np.bincount(kmeans.labels_)
        dominant = kmeans.cluster_centers_[np.argmax(counts)]
        return tuple(dominant.astype(int))
    except Exception as e:
        print(f"KMeans hatası: {e}")
        return (0,0,0)

# Renk sınıflandırma
def classify_color(bgr_color):
    if len(bgr_color) != 3:
        return "Belirsiz (Hata)"

    rgb = (bgr_color[2], bgr_color[1], bgr_color[0])
    min_dist = float('inf')
    color_name = "Belirsiz"
    
    color_centers = {name: np.mean(shades, axis=0) for name, shades in COLOR_DATASET.items()}

    for name, center in color_centers.items():
        try:
            dist = np.linalg.norm(np.array(rgb) - center)
            if dist < min_dist:
                min_dist = dist
                color_name = name
        except Exception as e:
             print(f"Renk sınıflandırma mesafesi hatası: {e}")
             continue
             
    return color_name

# --- Tkinter Fonksiyonları ---
def toggle():
    """Kamerayı açar veya kapatır."""
    global running, cap
    if not running:
        running = True
        btn.config(text="Kamerayı Kapat")
        if cap is None or not cap.isOpened():
             cap = cv2.VideoCapture(0)
             if not cap.isOpened():
                  print("Kamera başlatılamadı.")
                  running = False
                  btn.config(text="Kamerayı Aç")
                  status_label.config(text="Hata: Kamera başlatılamadı", fg="red")
                  return
        
        thread = threading.Thread(target=camera_loop)
        thread.daemon = True
        thread.start()
        status_label.config(text="Kamera Açık", fg="green")

    else:
        running = False
        btn.config(text="Kamerayı Aç")
        status_label.config(text="Kamera Kapalı", fg="orange")

def enroll_face():
    """Giriş kutusundaki isimle mevcut yüzü veritabanına kaydeder."""
    name = name_entry.get().strip()
    if not name:
        status_label.config(text="Lütfen bir isim girin.", fg="red")
        return

    if not running or cap is None or not cap.isOpened():
         status_label.config(text="Kayıt için önce kamerayı açın.", fg="orange")
         return

    ret, frame = cap.read()
    if not ret:
        status_label.config(text="Kareden veri alınamadı.", fg="red")
        return
        
    # DeepFace ile yüz tespiti
    try:
        face_embedding = get_face_embedding(frame)
        if face_embedding is None:
            status_label.config(text="Yüz bulunamadı veya embedding çıkarılamadı.", fg="orange")
            return

        # Embedding'i veritabanına ekle
        if name not in known_faces:
            known_faces[name] = []
            
        known_faces[name].append(face_embedding)

        # Veritabanını kaydet
        save_known_faces()

        status_label.config(text=f"'{name}' adlı yüz başarıyla kaydedildi.", fg="blue")
        name_entry.delete(0, END)
    except Exception as e:
        status_label.config(text=f"Hata: {str(e)}", fg="red")

# --- Kamera İşleme Döngüsü ---
def camera_loop():
    global running, cap, label, status_label
    
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml") 

    while running:
        if cap is None or not cap.isOpened():
             running = False
             break
             
        ret, frame = cap.read()
        if not ret:
            print("Uyarı: Kameradan kare alınamadı.")
            continue

        # DeepFace ile yüz analizi
        try:
            # Yüz tespiti ve analizi
            analysis = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False, silent=True)
            
            if isinstance(analysis, list) and len(analysis) > 0:
                # Tüm tespit edilen yüzler için işlem yap
                for face_info in analysis:
                    # Yüz konum bilgisi
                    x = face_info['region']['x']
                    y = face_info['region']['y']
                    w = face_info['region']['w']
                    h = face_info['region']['h']
                    
                    # Yüz bölgesini al
                    face_roi = frame[y:y+h, x:x+w]
                    
                    # Yüz tanıma
                    face_embedding = get_face_embedding(face_roi)
                    name = recognize_face(face_embedding)
                    
                    # Duygu analizi (DeepFace'ten alınan)
                    emotion = face_info['dominant_emotion']
                    
                    # Yüz çevresine dikdörtgen çiz
                    box_color = (255, 255, 0) if name == "Tanımlanmamış" else (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
                    
                    # Bilgileri ekrana yazdır
                    cv2.putText(frame, f"Isim: {name}", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                    cv2.putText(frame, f"Duygu: {emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    
                    # Saç rengi analizi (orijinal fonksiyonumuzu kullanıyoruz)
                    hair_y1 = max(0, y - int(h * 0.4))
                    hair_y2 = y
                    hair_x1 = x
                    hair_x2 = x + w
                    
                    hair_y1 = max(0, hair_y1)
                    hair_y2 = min(frame.shape[0], hair_y2)
                    hair_x1 = max(0, hair_x1)
                    hair_x2 = min(frame.shape[1], frame.shape[1])
                    
                    hair_roi = frame[hair_y1:hair_y2, hair_x1:hair_x2]
                    
                    if hair_roi.size > 0 and hair_roi.shape[0] > 0 and hair_roi.shape[1] > 0:
                        hair_color = detect_dominant_color(hair_roi)
                        hair_name = classify_color(hair_color)
                        cv2.putText(frame, f"Saç: {hair_name}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    else:
                         cv2.putText(frame, "Saç: Tespit Edilemedi", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    
                    # Göz rengi analizi
                    roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if face_roi.size > 0 and face_roi.shape[2] > 1 else None
                    
                    eye_name = "Tespit Edilemedi"
                    if roi_gray is not None and roi_gray.shape[0] > 0 and roi_gray.shape[1] > 0:
                        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
                        
                        if len(eyes) > 0:
                            ex, ey, ew, eh = eyes[0]
                            eye_x1 = x + ex
                            eye_y1 = y + ey
                            eye_x2 = x + ex + ew
                            eye_y2 = y + ey + eh
                            
                            eye_roi = frame[eye_y1:eye_y2, eye_x1:eye_x2]
                            
                            if eye_roi.size > 0 and eye_roi.shape[0] > 0 and eye_roi.shape[1] > 0 and eye_roi.shape[2] > 1:
                                eye_color = detect_dominant_color(eye_roi)
                                eye_name = classify_color(eye_color)
                    cv2.putText(frame, f"Göz: {eye_name}", (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)
                    
                    # Fotoğrafı kaydet
                    if face_roi.size > 0 and face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
                        try:
                            safe_name = name.replace(" ", "_").replace("Tanımlanmamış", "Unknown")
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                            filename = f"faces/{safe_name}.jpg"
                            os.makedirs(os.path.dirname(filename), exist_ok=True)
                            cv2.imwrite(filename, face_roi)
                        except Exception as e:
                            print(f"Yüz fotoğrafı kaydedilirken hata oluştu: {e}")
        except Exception as e:
            print(f"DeepFace analiz hatası: {e}")

        # Görüntüyü Tkinter'a aktar
        img_bgr = frame
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)
        
        img_width, img_height = img.size
        display_width = 960
        display_height = 540
        
        if img_width > 0 and img_height > 0:
            ratio = min(display_width / img_width, display_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            if new_width > 0 and new_height > 0:
                 img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            else:
                 img = Image.new('RGB', (display_width, display_height), color = (0, 0, 0))

        imgtk = ImageTk.PhotoImage(image=img)
        label.config(image=imgtk)
        label.image = imgtk
        label.update()

    # Döngü bittiğinde kaynakları serbest bırak
    if cap is not None and cap.isOpened():
        cap.release()
        cap = None
    cv2.destroyAllWindows()

# --- Arayüz ---
app = Tk()
app.title("DeepFace ile Yüz, Saç, Göz ve Duygu Analizi + Tanıma")
app.geometry("1024x700")

# Video akışını gösterecek label
label = Label(app)
label.pack(pady=5)

# İsim girişi ve Kayıt düğmesi için Frame
control_frame = Frame(app)
control_frame.pack(pady=5)

name_label = Label(control_frame, text="Kayıt Edilecek İsim:", font="Arial 12")
name_label.pack(side=LEFT, padx=5)

name_entry = Entry(control_frame, font="Arial 12", width=20)
name_entry.pack(side=LEFT, padx=5)

enroll_button = Button(control_frame, text="Bu Yüzü Kaydet", font="Arial 12", command=enroll_face)
enroll_button.pack(side=LEFT, padx=5)

# Kamera Aç/Kapat düğmesi
btn = Button(app, text="Kamerayı Aç", font="Arial 14", command=toggle)
btn.pack(pady=5)

# Durum mesajları için label
status_label = Label(app, text="Uygulama Başlatıldı", font="Arial 12", fg="black")
status_label.pack(pady=5)

# Uygulama başlatıldığında bilinen yüzleri yükle
load_known_faces()

# Tkinter ana döngüsü
app.mainloop()

# Program kapatılırken kaynakları serbest bırak
running = False
time.sleep(0.1)
if cap is not None and cap.isOpened():
    cap.release()
    cap = None
cv2.destroyAllWindows()
print("Program kapatıldı.")