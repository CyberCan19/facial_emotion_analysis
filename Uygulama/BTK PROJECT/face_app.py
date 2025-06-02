import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from deepface import DeepFace
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import threading
import time
import logging
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import sqlite3
import os
from PIL import Image, ImageTk

# Loglama ayarları
logging.basicConfig(
    filename='face_analysis.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class FaceAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gelişmiş Yüz Analizi Uygulaması")
        self.center_window(1000, 700)
        
        self.data_list = []
        self.stop_event = threading.Event()
        self.camera_thread = None
        self.cap = None
        self.is_camera_active = False
        
        # Veritabanı bağlantısı
        self.db_connection = sqlite3.connect('face_analysis.db')
        self.create_db_tables()
        
        self.setup_ui()
        self.update_display()
        
    def center_window(self, width, height):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = int((screen_width / 2) - (width / 2))
        y = int((screen_height / 2) - (height / 2))
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def create_db_tables(self):
        cursor = self.db_connection.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                gender TEXT,
                hair_color TEXT,
                eye_color TEXT,
                emotion TEXT,
                age INTEGER,
                clothing_color TEXT
            )
        ''')
        self.db_connection.commit()

    def setup_ui(self):
        # Ana çerçeveler
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        control_frame = ttk.LabelFrame(main_frame, text="Kontrol Paneli")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        display_frame = ttk.LabelFrame(main_frame, text="Analiz Sonuçları")
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Kontrol paneli butonları
        buttons = [
            ("📷 Resim Seç ve Analiz Et", self.open_image),
            ("🎥 Kamerayı Başlat/Durdur", self.toggle_camera),
            ("📊 İstatistikleri Göster", self.show_statistics),
            ("🔍 Verileri Filtrele", self.open_filter_dialog),
            ("💾 CSV'ye Kaydet", self.save_dataset),
            ("🗃️ Veritabanına Kaydet", self.save_to_db),
            ("📈 Grafik Oluştur", self.show_pie_chart),
            ("🧹 Verileri Temizle", self.clear_data)
        ]

        for text, command in buttons:
            btn = ttk.Button(control_frame, text=text, command=command)
            btn.pack(fill=tk.X, pady=3)

        # Görüntü önizleme alanı
        self.preview_label = ttk.Label(control_frame)
        self.preview_label.pack(pady=10)

        # Analiz sonuçları görüntüleme
        self.data_display = tk.Text(
            display_frame,
            height=20,
            width=70,
            wrap=tk.WORD,
            font=("Arial", 10),
            bg="white"
        )
        self.data_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(display_frame, command=self.data_display.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.data_display.config(yscrollcommand=scrollbar.set)

        # Durum çubuğu
        self.status_var = tk.StringVar()
        self.status_var.set("Hazır")
        status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def detect_dominant_color(self, image, k=3):
        if image.size == 0:
            return (0, 0, 0)
            
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).reshape((-1, 3))
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(img)
        counts = np.bincount(kmeans.labels_)
        dominant_color = kmeans.cluster_centers_[np.argmax(counts)].astype(int)
        return tuple(dominant_color)

    def get_eye_color_name(self, rgb):
        r, g, b = rgb

        color_ranges = [
            ((80, 40, 0), (255, 255, 50), "Kahverengi"),
            ((100, 60, 0), (255, 255, 40), "Ela"),
            ((180, 140, 0), (255, 255, 70), "Kehribar"),
            ((0, 0, 130), (100, 150, 255), "Mavi"),
            ((0, 130, 0), (120, 255, 100), "Yeşil"),
            ((120, 120, 120), (255, 255, 255), "Gri"),
            ((150, 0, 0), (255, 80, 80), "Kırmızı")
        ]

        for (lower, upper, color_name) in color_ranges:
            if all(lower[i] <= rgb[i] <= upper[i] for i in range(3)):
                return color_name
        return "Bilinmiyor"

    def get_hair_color_name(self, rgb):
        r, g, b = rgb
        brightness = (r + g + b) / 3

        if r > 100 and g > 50 and b < 50:
            color = "Kızıl"
        elif r > 190 and g > 170 and b > 120:
            color = "Sarışın"
        elif r > 60 and g > 40 and b > 20:
            color = "Kahverengi"
        elif r < 50 and g < 50 and b < 50:
            color = "Siyah"
        else:
            color = "Bilinmiyor"

        dyed = "(Boyalı olabilir)" if brightness > 170 or brightness < 40 else ""
        return f"{color} {dyed}".strip()

    def extract_hair_color(self, image, face):
        x, y, w, h = face
        region = image[max(0, y - int(h * 0.6)):y, x:x + w]
        return self.detect_dominant_color(region)

    def extract_eye_color(self, image, face):
        x, y, w, h = face
        region = image[y + int(h * 0.2):y + int(h * 0.4), x + int(w * 0.2):x + int(w * 0.8)]
        return self.detect_dominant_color(region)

    def extract_clothing_color(self, image, face):
        x, y, w, h = face
        region = image[y + h:y + h + int(h * 0.5), x:x + w]
        return self.detect_dominant_color(region)

    def analyze_faces(self, image):
        results = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Daha gelişmiş yüz tespiti
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for face in faces:
            x, y, w, h = face
            roi = image[y:y + h, x:x + w]

            try:
                analysis = DeepFace.analyze(
                    roi,
                    actions=["emotion", "gender", "age"],
                    enforce_detection=False,
                    silent=True
                )
                emotion = analysis[0]["dominant_emotion"]
                gender = analysis[0]["dominant_gender"]
                age = int(analysis[0]["age"])
            except Exception as e:
                logging.error(f"DeepFace analiz hatası: {str(e)}")
                emotion = "Tespit Edilemedi"
                gender = "Bilinmiyor"
                age = 0

            hair_rgb = self.extract_hair_color(image, face)
            hair_color = self.get_hair_color_name(hair_rgb)

            eye_rgb = self.extract_eye_color(image, face)
            eye_color = self.get_eye_color_name(eye_rgb)

            clothing_rgb = self.extract_clothing_color(image, face)
            clothing_color = f"RGB({clothing_rgb[0]}, {clothing_rgb[1]}, {clothing_rgb[2]})"

            results.append({
                "Cinsiyet": gender,
                "Yaş": age,
                "Saç Rengi": hair_color,
                "Göz Rengi": eye_color,
                "Duygu": emotion,
                "Kıyafet Rengi": clothing_color,
                "RGB": (hair_rgb, eye_rgb, clothing_rgb)
            })

            # Görsel işaretleme
            self.draw_analysis_results(image, face, results[-1])

        return image, results

    def draw_analysis_results(self, image, face, result):
        x, y, w, h = face
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Bilgileri görsele ekle
        info_texts = [
            f"Cinsiyet: {result['Cinsiyet']}",
            f"Yaş: {result['Yaş']}",
            f"Duygu: {result['Duygu']}",
            f"Saç: {result['Saç Rengi']}",
            f"Göz: {result['Göz Rengi']}"
        ]
        
        for i, text in enumerate(info_texts):
            cv2.putText(
                image, text, (x, y - 10 - (i * 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
            )

    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Resimler", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            try:
                image = cv2.imread(file_path)
                if image is None:
                    raise ValueError("Geçersiz resim dosyası")
                    
                analyzed, results = self.analyze_faces(image)
                self.data_list.extend(results)
                
                # Önizleme göster
                self.show_image_preview(analyzed)
                
                self.status_var.set(f"Analiz tamamlandı - {len(results)} yüz tespit edildi")
                logging.info(f"Resim analizi tamamlandı: {file_path}")
                
            except Exception as e:
                messagebox.showerror("Hata", f"Resim işlenirken hata oluştu: {str(e)}")
                logging.error(f"Resim işleme hatası: {str(e)}")

    def show_image_preview(self, image):
        # OpenCV görüntüsünü PIL formatına çevir
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image)
        img.thumbnail((300, 300))  # Boyutlandır
        
        # Tkinter için uygun formata çevir
        imgtk = ImageTk.PhotoImage(image=img)
        self.preview_label.config(image=imgtk)
        self.preview_label.image = imgtk  # Referansı sakla

    def toggle_camera(self):
        if self.is_camera_active:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        if not self.is_camera_active:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.status_var.set("Kamera açılamadı!")
                logging.error("Kamera açılamadı")
                return
                
            self.stop_event.clear()
            self.is_camera_active = True
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            self.status_var.set("Kamera aktif - Analiz yapılıyor...")
            logging.info("Kamera başlatıldı")

    def stop_camera(self):
        if self.is_camera_active:
            self.stop_event.set()
            self.is_camera_active = False
            if self.cap:
                self.cap.release()
            self.status_var.set("Kamera durduruldu")
            logging.info("Kamera durduruldu")

    def camera_loop(self):
        while not self.stop_event.is_set() and self.is_camera_active:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            analyzed, results = self.analyze_faces(frame)
            self.data_list.extend(results)
            
            # Önizleme göster
            self.show_camera_preview(analyzed)
            
            # Her 10 frame'de bir veriyi güncelle
            if len(self.data_list) % 10 == 0:
                self.root.event_generate("<<UpdateDisplay>>")
                
            time.sleep(0.1)  # CPU kullanımını azalt

        self.is_camera_active = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.event_generate("<<UpdateDisplay>>")

    def show_camera_preview(self, image):
        # Bu fonksiyon kamera görüntüsünü gerçek zamanlı göstermek için
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(image)
            img.thumbnail((300, 300))
            
            imgtk = ImageTk.PhotoImage(image=img)
            self.preview_label.config(image=imgtk)
            self.preview_label.image = imgtk
        except Exception as e:
            logging.error(f"Kamera önizleme hatası: {str(e)}")

    def update_display(self, event=None):
        self.data_display.delete(1.0, tk.END)
        
        if not self.data_list:
            self.data_display.insert(tk.END, "Henüz analiz verisi yok...\n")
            return
            
        for i, d in enumerate(self.data_list[-20:], 1):  # Son 20 kaydı göster
            text = (f"Kayıt #{i}\n"
                   f"Cinsiyet: {d['Cinsiyet']}\n"
                   f"Yaş: {d['Yaş']}\n"
                   f"Saç Rengi: {d['Saç Rengi']}\n"
                   f"Göz Rengi: {d['Göz Rengi']}\n"
                   f"Duygu: {d['Duygu']}\n"
                   f"Kıyafet Rengi: {d['Kıyafet Rengi']}\n"
                   "-----------------------------\n")
            self.data_display.insert(tk.END, text)
            
        self.data_display.see(tk.END)  # En sona kaydır

    def show_statistics(self):
        if not self.data_list:
            messagebox.showinfo("Bilgi", "Analiz verisi yok!")
            return
            
        df = pd.DataFrame(self.data_list)
        stats_window = tk.Toplevel(self.root)
        stats_window.title("İstatistikler")
        
        # Temel istatistikler
        gender_stats = df['Cinsiyet'].value_counts()
        emotion_stats = df['Duygu'].value_counts()
        age_stats = df['Yaş'].describe()
        
        # Yaş dağılımı
        age_text = (f"Yaş İstatistikleri:\n"
                   f"Ortalama: {age_stats['mean']:.1f}\n"
                   f"Min: {age_stats['min']}\n"
                   f"Max: {age_stats['max']}\n"
                   f"Standart Sapma: {age_stats['std']:.1f}\n")
        
        # Cinsiyet dağılımı
        gender_text = "Cinsiyet Dağılımı:\n"
        for gender, count in gender_stats.items():
            gender_text += f"{gender}: {count} (%{count/len(df)*100:.1f})\n"
            
        # Duygu dağılımı
        emotion_text = "Duygu Dağılımı:\n"
        for emotion, count in emotion_stats.items():
            emotion_text += f"{emotion}: {count} (%{count/len(df)*100:.1f})\n"
        
        # Tüm istatistikleri birleştir
        stats_text = age_text + "\n" + gender_text + "\n" + emotion_text
        
        # İstatistikleri göster
        stats_display = tk.Text(stats_window, wrap=tk.WORD, width=60, height=20)
        stats_display.pack(padx=10, pady=10)
        stats_display.insert(tk.END, stats_text)
        stats_display.config(state=tk.DISABLED)

    def open_filter_dialog(self):
        if not self.data_list:
            messagebox.showinfo("Bilgi", "Filtrelemek için veri yok!")
            return
            
        filter_win = tk.Toplevel(self.root)
        filter_win.title("Veri Filtreleme")
        
        # Filtre seçenekleri
        ttk.Label(filter_win, text="Cinsiyet:").grid(row=0, column=0, padx=5, pady=5)
        gender_var = tk.StringVar()
        gender_combo = ttk.Combobox(
            filter_win,
            textvariable=gender_var,
            values=["Tümü", "Man", "Woman"]
        )
        gender_combo.grid(row=0, column=1, padx=5, pady=5)
        gender_combo.current(0)
        
        ttk.Label(filter_win, text="Duygu:").grid(row=1, column=0, padx=5, pady=5)
        emotion_var = tk.StringVar()
        emotion_combo = ttk.Combobox(
            filter_win,
            textvariable=emotion_var,
            values=["Tümü", "happy", "sad", "angry", "surprise", "fear", "neutral"]
        )
        emotion_combo.grid(row=1, column=1, padx=5, pady=5)
        emotion_combo.current(0)
        
        ttk.Label(filter_win, text="Yaş Aralığı:").grid(row=2, column=0, padx=5, pady=5)
        age_frame = ttk.Frame(filter_win)
        age_frame.grid(row=2, column=1, padx=5, pady=5)
        
        min_age_var = tk.StringVar(value="0")
        max_age_var = tk.StringVar(value="100")
        
        ttk.Entry(age_frame, textvariable=min_age_var, width=5).pack(side=tk.LEFT)
        ttk.Label(age_frame, text=" - ").pack(side=tk.LEFT)
        ttk.Entry(age_frame, textvariable=max_age_var, width=5).pack(side=tk.LEFT)
        
        def apply_filter():
            gender = gender_var.get() if gender_var.get() != "Tümü" else None
            emotion = emotion_var.get() if emotion_var.get() != "Tümü" else None
            
            try:
                min_age = int(min_age_var.get())
                max_age = int(max_age_var.get())
            except ValueError:
                messagebox.showerror("Hata", "Geçersiz yaş aralığı!")
                return
                
            filtered = [
                d for d in self.data_list
                if (not gender or d['Cinsiyet'] == gender) and
                   (not emotion or d['Duygu'] == emotion) and
                   (min_age <= d['Yaş'] <= max_age)
            ]
            
            if not filtered:
                messagebox.showinfo("Sonuç", "Filtreye uygun veri bulunamadı!")
                return
                
            # Filtrelenmiş veriyi göster
            self.show_filtered_results(filtered)
            filter_win.destroy()
            
        ttk.Button(filter_win, text="Uygula", command=apply_filter).grid(row=3, columnspan=2, pady=10)

    def show_filtered_results(self, filtered_data):
        result_win = tk.Toplevel(self.root)
        result_win.title("Filtrelenmiş Sonuçlar")
        
        text = tk.Text(result_win, wrap=tk.WORD, width=80, height=20)
        scroll = ttk.Scrollbar(result_win, command=text.yview)
        text.config(yscrollcommand=scroll.set)
        
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(fill=tk.BOTH, expand=True)
        
        for i, d in enumerate(filtered_data, 1):
            text.insert(tk.END,
                       f"Kayıt #{i}\n"
                       f"Cinsiyet: {d['Cinsiyet']}\n"
                       f"Yaş: {d['Yaş']}\n"
                       f"Saç Rengi: {d['Saç Rengi']}\n"
                       f"Göz Rengi: {d['Göz Rengi']}\n"
                       f"Duygu: {d['Duygu']}\n"
                       f"Kıyafet Rengi: {d['Kıyafet Rengi']}\n"
                       "-----------------------------\n")
        
        text.config(state=tk.DISABLED)

    def save_dataset(self):
        if not self.data_list:
            messagebox.showwarning("Uyarı", "Kaydedilecek veri yok!")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Dosyaları", "*.csv")]
        )
        
        if file_path:
            try:
                df = pd.DataFrame(self.data_list)
                # RGB bilgilerini kaydetme
                df['Saç RGB'] = df['RGB'].apply(lambda x: f"{x[0][0]},{x[0][1]},{x[0][2]}")
                df['Göz RGB'] = df['RGB'].apply(lambda x: f"{x[1][0]},{x[1][1]},{x[1][2]}")
                df['Kıyafet RGB'] = df['RGB'].apply(lambda x: f"{x[2][0]},{x[2][1]},{x[2][2]}")
                df.drop('RGB', axis=1, inplace=True)
                
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
                self.status_var.set(f"Veri başarıyla kaydedildi: {os.path.basename(file_path)}")
                logging.info(f"CSV dosyası kaydedildi: {file_path}")
                messagebox.showinfo("Başarılı", "Veri başarıyla CSV dosyasına kaydedildi!")
            except Exception as e:
                messagebox.showerror("Hata", f"Dosya kaydedilirken hata oluştu: {str(e)}")
                logging.error(f"CSV kaydetme hatası: {str(e)}")

    def save_to_db(self):
        if not self.data_list:
            messagebox.showwarning("Uyarı", "Kaydedilecek veri yok!")
            return
            
        try:
            cursor = self.db_connection.cursor()
            
            for data in self.data_list:
                cursor.execute('''
                    INSERT INTO analysis_data 
                    (gender, hair_color, eye_color, emotion, age, clothing_color)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    data['Cinsiyet'],
                    data['Saç Rengi'],
                    data['Göz Rengi'],
                    data['Duygu'],
                    data['Yaş'],
                    data['Kıyafet Rengi']
                ))
            
            self.db_connection.commit()
            self.status_var.set(f"{len(self.data_list)} kayıt veritabanına kaydedildi")
            logging.info(f"{len(self.data_list)} kayıt veritabanına kaydedildi")
            messagebox.showinfo("Başarılı", "Veriler başarıyla veritabanına kaydedildi!")
        except Exception as e:
            messagebox.showerror("Hata", f"Veritabanına kaydedilirken hata oluştu: {str(e)}")
            logging.error(f"Veritabanı kayıt hatası: {str(e)}")

    def show_pie_chart(self):
        if not self.data_list:
            messagebox.showwarning("Uyarı", "Grafik oluşturmak için veri yok!")
            return
            
        chart_window = tk.Toplevel(self.root)
        chart_window.title("Veri Dağılımları")
        
        notebook = ttk.Notebook(chart_window)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        df = pd.DataFrame(self.data_list)
        
        # Cinsiyet dağılımı
        gender_frame = ttk.Frame(notebook)
        gender_fig = Figure(figsize=(6, 4), dpi=100)
        gender_ax = gender_fig.add_subplot(111)
        df['Cinsiyet'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=gender_ax)
        gender_ax.set_title('Cinsiyet Dağılımı')
        gender_canvas = FigureCanvasTkAgg(gender_fig, master=gender_frame)
        gender_canvas.draw()
        gender_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        notebook.add(gender_frame, text="Cinsiyet")
        
        # Duygu dağılımı
        emotion_frame = ttk.Frame(notebook)
        emotion_fig = Figure(figsize=(6, 4), dpi=100)
        emotion_ax = emotion_fig.add_subplot(111)
        df['Duygu'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=emotion_ax)
        emotion_ax.set_title('Duygu Dağılımı')
        emotion_canvas = FigureCanvasTkAgg(emotion_fig, master=emotion_frame)
        emotion_canvas.draw()
        emotion_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        notebook.add(emotion_frame, text="Duygu")
        
        # Yaş histogramı
        age_frame = ttk.Frame(notebook)
        age_fig = Figure(figsize=(6, 4), dpi=100)
        age_ax = age_fig.add_subplot(111)
        df['Yaş'].plot(kind='hist', bins=20, ax=age_ax)
        age_ax.set_title('Yaş Dağılımı')
        age_ax.set_xlabel('Yaş')
        age_ax.set_ylabel('Kişi Sayısı')
        age_canvas = FigureCanvasTkAgg(age_fig, master=age_frame)
        age_canvas.draw()
        age_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        notebook.add(age_frame, text="Yaş")

    def clear_data(self):
        if messagebox.askyesno("Onay", "Tüm analiz verilerini silmek istediğinize emin misiniz?"):
            self.data_list = []
            self.status_var.set("Veriler temizlendi")
            logging.info("Analiz verileri temizlendi")

    def on_closing(self):
        self.stop_camera()
        if self.db_connection:
            self.db_connection.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceAnalysisApp(root)
    
    # Özel olay bağlantıları
    root.bind("<<UpdateDisplay>>", app.update_display)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    root.mainloop()