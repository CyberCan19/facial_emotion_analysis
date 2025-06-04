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
from tkinter import font as tkfont

# Loglama ayarları
logging.basicConfig(
    filename='face_analysis.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ModernFaceAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Face Analyzer Pro")
        self.root.geometry("1200x800")
        self.center_window(1200, 800)
        
        # Stil ayarları
        self.setup_styles()
        
        # Uygulama verileri
        self.data_list = []
        self.stop_event = threading.Event()
        self.camera_thread = None
        self.cap = None
        self.is_camera_active = False
        self.current_frame = None
        
        # Veritabanı bağlantısı
        self.db_connection = sqlite3.connect('face_analysis.db', check_same_thread=False)
        self.create_db_tables()
        
        # UI oluştur
        self.setup_ui()
        self.update_display()
        
    def center_window(self, width, height, window=None):
        window = window if window else self.root
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x = int((screen_width / 2) - (width / 2))
        y = int((screen_height / 2) - (height / 2))
        window.geometry(f"{width}x{height}+{x}+{y}")

    def setup_styles(self):
        # Modern renk paleti
        self.bg_color = "#2c3e50"
        self.sidebar_color = "#34495e"
        self.accent_color = "#3498db"
        self.highlight_color = "#2980b9"
        self.text_color = "#ecf0f1"
        self.button_color = "#16a085"
        self.button_hover = "#1abc9c"
        self.card_color = "#3d566e"
        
        # Genel stil ayarları
        style = ttk.Style()
        style.theme_use('clam')
        
        # Frame stilleri
        style.configure('TFrame', background=self.bg_color)
        style.configure('Sidebar.TFrame', background=self.sidebar_color)
        style.configure('Card.TFrame', background=self.card_color, relief=tk.RAISED, borderwidth=2)
        
        # Label stilleri
        style.configure('TLabel', background=self.bg_color, foreground=self.text_color)
        style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'), background=self.sidebar_color)
        style.configure('Sidebar.TLabel', background=self.sidebar_color, foreground=self.text_color)
        style.configure('Card.TLabel', background=self.card_color, foreground=self.text_color)
        
        # Button stilleri
        style.configure('TButton', background=self.button_color, foreground=self.text_color, 
                      font=('Helvetica', 10), borderwidth=1)
        style.map('TButton', 
                 background=[('active', self.button_hover), ('pressed', self.highlight_color)],
                 foreground=[('active', self.text_color), ('pressed', self.text_color)])
        
        # Entry ve Combobox stilleri
        style.configure('TEntry', fieldbackground="#ecf0f1")
        style.configure('TCombobox', fieldbackground="#ecf0f1")
        
        # Notebook stilleri
        style.configure('TNotebook', background=self.bg_color)
        style.configure('TNotebook.Tab', background=self.sidebar_color, foreground=self.text_color,
                      padding=[10, 5], font=('Helvetica', 10))
        style.map('TNotebook.Tab', 
                 background=[('selected', self.accent_color), ('active', self.highlight_color)],
                 foreground=[('selected', self.text_color), ('active', self.text_color)])

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
        # Ana konteyner
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Sidebar (sol panel)
        self.sidebar = ttk.Frame(self.main_container, width=250, style='Sidebar.TFrame')
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.sidebar.pack_propagate(False)
        
        # Başlık
        title_label = ttk.Label(
            self.sidebar, 
            text="AI Face Analyzer", 
            style='Title.TLabel'
        )
        title_label.pack(pady=20, padx=10)
        
        # Logo/ikon
        self.setup_logo()
        
        # Butonlar
        self.setup_sidebar_buttons()
        
        # Önizleme alanı
        self.setup_preview_area()
        
        # Ana içerik alanı (sağ panel)
        self.content = ttk.Frame(self.main_container)
        self.content.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Sekmeli arayüz
        self.notebook = ttk.Notebook(self.content)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Analiz sonuçları sekmesi
        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text="Analiz Sonuçları")
        self.setup_results_display()
        
        # Veri görselleştirme sekmesi
        self.visualization_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.visualization_tab, text="Veri Görselleştirme")
        
        # Durum çubuğu
        self.setup_status_bar()
        
        # Güncelleme olayı bağlantısı
        self.root.bind("<<UpdateDisplay>>", self.update_display)

    def setup_logo(self):
        logo_frame = ttk.Frame(self.sidebar, style='Card.TFrame')
        logo_frame.pack(pady=10, padx=10, fill=tk.X)
        
        logo_label = ttk.Label(
            logo_frame, 
            text="🧑‍💻", 
            font=('Helvetica', 48), 
            background=self.card_color, 
            foreground=self.accent_color
        )
        logo_label.pack(pady=10)
        
        version_label = ttk.Label(
            logo_frame, 
            text="v2.1.0", 
            style='Card.TLabel',
            font=('Helvetica', 8)
        )
        version_label.pack(pady=5)

    def setup_sidebar_buttons(self):
        button_frame = ttk.Frame(self.sidebar)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        buttons = [
            ("📷 Resim Yükle", self.open_image, "#3498db"),
            ("🎥 Kamera Aç/Kapat", self.toggle_camera, "#e74c3c"),
            ("📊 İstatistikler", self.show_statistics, "#9b59b6"),
            ("🔍 Veri Filtrele", self.open_filter_dialog, "#f39c12"),
            ("📈 Grafikler", self.show_pie_chart, "#2ecc71"),
            ("💾 Veriyi Dışa Aktar", self.save_dataset, "#1abc9c"),
            ("🗃️ Veritabanına Kaydet", self.save_to_db, "#34495e"),
            ("🧹 Verileri Temizle", self.clear_data, "#e74c3c")
        ]
        
        for text, command, color in buttons:
            btn = tk.Button(
                button_frame,
                text=text,
                command=command,
                bg=color,
                fg=self.text_color,
                activebackground=self.highlight_color,
                activeforeground=self.text_color,
                relief=tk.FLAT,
                font=('Helvetica', 10, 'bold'),
                padx=10,
                pady=8,
                bd=0,
                highlightthickness=0
            )
            btn.pack(fill=tk.X, pady=3, ipady=3)
            btn.bind("<Enter>", lambda e, b=btn: b.config(bg=self.highlight_color))
            btn.bind("<Leave>", lambda e, b=btn, c=color: b.config(bg=c))

    def setup_preview_area(self):
        preview_frame = ttk.Frame(self.sidebar, style='Card.TFrame')
        preview_frame.pack(fill=tk.X, padx=10, pady=20)
        
        preview_title = ttk.Label(
            preview_frame, 
            text="Önizleme", 
            style='Card.TLabel',
            font=('Helvetica', 12, 'bold')
        )
        preview_title.pack(pady=(5, 10))
        
        self.preview_label = ttk.Label(preview_frame, background=self.card_color)
        self.preview_label.pack(pady=5, padx=5, fill=tk.X)
        
        # Kamera kontrol butonları
        cam_btn_frame = ttk.Frame(preview_frame)
        cam_btn_frame.pack(pady=5)
        
        self.cam_btn = ttk.Button(
            cam_btn_frame,
            text="Kamerayı Başlat",
            command=self.toggle_camera,
            style='TButton'
        )
        self.cam_btn.pack(side=tk.LEFT, padx=2)
        
        self.snap_btn = ttk.Button(
            cam_btn_frame,
            text="Fotoğraf Çek",
            command=self.take_snapshot,
            style='TButton'
        )
        self.snap_btn.pack(side=tk.LEFT, padx=2)
        self.snap_btn.state(['disabled'])

    def setup_results_display(self):
        # Sonuçlar için kart görünümü
        self.results_canvas = tk.Canvas(self.results_tab, bg=self.bg_color, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.results_tab, orient="vertical", command=self.results_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.results_canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.results_canvas.configure(
                scrollregion=self.results_canvas.bbox("all")
            )
        )
        
        self.results_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.results_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.results_canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Fare tekerleği ile kaydırma
        self.results_canvas.bind_all("<MouseWheel>", 
            lambda event: self.results_canvas.yview_scroll(int(-1*(event.delta/120)), "units"))

    def setup_status_bar(self):
        self.status_var = tk.StringVar()
        self.status_var.set("Hazır")
        
        status_bar = tk.Frame(self.root, bg=self.sidebar_color, height=25)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        status_label = tk.Label(
            status_bar,
            textvariable=self.status_var,
            bg=self.sidebar_color,
            fg=self.text_color,
            font=('Helvetica', 9)
        )
        status_label.pack(side=tk.LEFT, padx=10)
        
        # Saat göstergesi
        self.time_var = tk.StringVar()
        self.update_clock()
        time_label = tk.Label(
            status_bar,
            textvariable=self.time_var,
            bg=self.sidebar_color,
            fg=self.text_color,
            font=('Helvetica', 9)
        )
        time_label.pack(side=tk.RIGHT, padx=10)

    def update_clock(self):
        current_time = time.strftime("%H:%M:%S")
        self.time_var.set(current_time)
        self.root.after(1000, self.update_clock)

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
                self.root.event_generate("<<UpdateDisplay>>")
                
            except Exception as e:
                messagebox.showerror("Hata", f"Resim işlenirken hata oluştu: {str(e)}")
                logging.error(f"Resim işleme hatası: {str(e)}")

    def show_image_preview(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image)
        img.thumbnail((300, 300))
        
        imgtk = ImageTk.PhotoImage(image=img)
        self.preview_label.config(image=imgtk)
        self.preview_label.image = imgtk

    def toggle_camera(self):
        if self.is_camera_active:
            self.stop_camera()
            self.cam_btn.config(text="Kamerayı Başlat")
            self.snap_btn.state(['disabled'])
        else:
            self.start_camera()
            self.cam_btn.config(text="Kamerayı Durdur")
            self.snap_btn.state(['!disabled'])

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

    def take_snapshot(self):
        if self.current_frame is not None and self.is_camera_active:
            analyzed, results = self.analyze_faces(self.current_frame.copy())
            self.data_list.extend(results)
            self.show_image_preview(analyzed)
            self.root.event_generate("<<UpdateDisplay>>")
            self.status_var.set(f"Fotoğraf çekildi - {len(results)} yüz tespit edildi")

    def camera_loop(self):
        while not self.stop_event.is_set() and self.is_camera_active:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            self.current_frame = frame.copy()
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
        self.root.event_generate("<<UpdateDisplay>>")

    def show_camera_preview(self, image):
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
        # Önceki sonuçları temizle
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        if not self.data_list:
            empty_label = ttk.Label(
                self.scrollable_frame,
                text="Henüz analiz verisi yok...\nLütfen bir resim yükleyin veya kamerayı başlatın.",
                style='TLabel',
                font=('Helvetica', 10, 'italic')
            )
            empty_label.pack(pady=50)
            return
            
        for i, d in enumerate(self.data_list[-20:], 1):  # Son 20 kaydı göster
            card_frame = ttk.Frame(
                self.scrollable_frame,
                style='Card.TFrame',
                padding=10
            )
            card_frame.pack(fill=tk.X, pady=5, padx=5)
            
            # Kart başlığı
            title_label = ttk.Label(
                card_frame,
                text=f"Analiz Kaydı #{i}",
                style='Card.TLabel',
                font=('Helvetica', 12, 'bold')
            )
            title_label.pack(anchor=tk.W)
            
            # Veri alanları
            fields = [
                f"👤 Cinsiyet: {d['Cinsiyet']}",
                f"🎂 Yaş: {d['Yaş']}",
                f"💇 Saç Rengi: {d['Saç Rengi']}",
                f"👁️ Göz Rengi: {d['Göz Rengi']}",
                f"😊 Duygu: {d['Duygu']}",
                f"👕 Kıyafet Rengi: {d['Kıyafet Rengi']}"
            ]
            
            for field in fields:
                field_label = ttk.Label(
                    card_frame,
                    text=field,
                    style='Card.TLabel',
                    font=('Helvetica', 10)
                )
                field_label.pack(anchor=tk.W, padx=10, pady=2)
            
            # Ayırıcı çizgi
            separator = ttk.Separator(card_frame, orient='horizontal')
            separator.pack(fill=tk.X, pady=5)

    def show_statistics(self):
        if not self.data_list:
            messagebox.showinfo("Bilgi", "Analiz verisi yok!")
            return
            
        stats_window = tk.Toplevel(self.root)
        stats_window.title("İstatistikler")
        stats_window.geometry("800x600")
        self.center_window(800, 600, stats_window)
        
        # Notebook ile farklı istatistikler
        notebook = ttk.Notebook(stats_window)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        df = pd.DataFrame(self.data_list)
        
        # Genel istatistikler
        general_frame = ttk.Frame(notebook)
        notebook.add(general_frame, text="Genel")
        
        # Kart stilinde istatistikler
        cards = [
            ("Toplam Analiz", len(df), "#3498db"),
            ("Ortalama Yaş", f"{df['Yaş'].mean():.1f}", "#2ecc71"),
            ("Erkek/Kadın Oranı", 
             f"{len(df[df['Cinsiyet'] == 'Man'])}/{len(df[df['Cinsiyet'] == 'Woman'])}", 
             "#9b59b6"),
            ("En Yaygın Duygu", df['Duygu'].mode()[0], "#e74c3c")
        ]
        
        for i, (title, value, color) in enumerate(cards):
            card = tk.Frame(
                general_frame,
                bg=color,
                bd=0,
                highlightthickness=0,
                relief=tk.RAISED
            )
            card.grid(row=i//2, column=i%2, padx=10, pady=10, sticky="nsew")
            
            title_label = tk.Label(
                card,
                text=title,
                bg=color,
                fg="white",
                font=('Helvetica', 12, 'bold')
            )
            title_label.pack(pady=(10, 0))
            
            value_label = tk.Label(
                card,
                text=str(value),
                bg=color,
                fg="white",
                font=('Helvetica', 24, 'bold')
            )
            value_label.pack(pady=(0, 10))
            
            general_frame.grid_columnconfigure(i%2, weight=1)
        
        general_frame.grid_rowconfigure(0, weight=1)
        general_frame.grid_rowconfigure(1, weight=1)
        
        # Detaylı istatistikler
        detailed_frame = ttk.Frame(notebook)
        notebook.add(detailed_frame, text="Detaylar")
        
        # Cinsiyet dağılımı
        gender_frame = ttk.LabelFrame(detailed_frame, text="Cinsiyet Dağılımı")
        gender_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        gender_stats = df['Cinsiyet'].value_counts()
        for gender, count in gender_stats.items():
            percent = count / len(df) * 100
            ttk.Label(
                gender_frame,
                text=f"{gender}: {count} kişi (%{percent:.1f})",
                style='TLabel'
            ).pack(anchor=tk.W, padx=10, pady=5)
            
            # Progress bar
            progress = ttk.Progressbar(
                gender_frame,
                orient=tk.HORIZONTAL,
                length=200,
                mode='determinate',
                value=percent
            )
            progress.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Duygu dağılımı
        emotion_frame = ttk.LabelFrame(detailed_frame, text="Duygu Dağılımı")
        emotion_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        emotion_stats = df['Duygu'].value_counts()
        for emotion, count in emotion_stats.items():
            percent = count / len(df) * 100
            ttk.Label(
                emotion_frame,
                text=f"{emotion}: {count} kişi (%{percent:.1f})",
                style='TLabel'
            ).pack(anchor=tk.W, padx=10, pady=5)
            
            progress = ttk.Progressbar(
                emotion_frame,
                orient=tk.HORIZONTAL,
                length=200,
                mode='determinate',
                value=percent
            )
            progress.pack(fill=tk.X, padx=10, pady=(0, 10))

    def open_filter_dialog(self):
        if not self.data_list:
            messagebox.showinfo("Bilgi", "Filtrelemek için veri yok!")
            return
            
        filter_win = tk.Toplevel(self.root)
        filter_win.title("Veri Filtreleme")
        self.center_window(400, 300, filter_win)
        
        # Filtre seçenekleri
        ttk.Label(filter_win, text="Cinsiyet:").grid(row=0, column=0, padx=5, pady=5)
        gender_var = tk.StringVar()
        gender_combo = ttk.Combobox(
            filter_win,
            textvariable=gender_var,
            values=["Tümü", "Man", "Woman"],
            state="readonly"
        )
        gender_combo.grid(row=0, column=1, padx=5, pady=5)
        gender_combo.current(0)
        
        ttk.Label(filter_win, text="Duygu:").grid(row=1, column=0, padx=5, pady=5)
        emotion_var = tk.StringVar()
        emotion_combo = ttk.Combobox(
            filter_win,
            textvariable=emotion_var,
            values=["Tümü", "happy", "sad", "angry", "surprise", "fear", "neutral"],
            state="readonly"
        )
        emotion_combo.grid(row=1, column=1, padx=5, pady=5)
        emotion_combo.current(0)
        
        ttk.Label(filter_win, text="Yaş Aralığı:").grid(row=2, column=0, padx=5, pady=5)
        age_frame = ttk.Frame(filter_win)
        age_frame.grid(row=2, column=1, padx=5, pady=5)
        
        min_age_var = tk.IntVar(value=0)
        max_age_var = tk.IntVar(value=100)
        
        ttk.Entry(age_frame, textvariable=min_age_var, width=5).pack(side=tk.LEFT)
        ttk.Label(age_frame, text=" - ").pack(side=tk.LEFT)
        ttk.Entry(age_frame, textvariable=max_age_var, width=5).pack(side=tk.LEFT)
        
        def apply_filter():
            gender = gender_var.get() if gender_var.get() != "Tümü" else None
            emotion = emotion_var.get() if emotion_var.get() != "Tümü" else None
            min_age = min_age_var.get()
            max_age = max_age_var.get()
            
            if min_age > max_age:
                messagebox.showerror("Hata", "Minimum yaş maksimum yaştan büyük olamaz!")
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
        result_win.geometry("600x400")
        self.center_window(600, 400, result_win)
        
        canvas = tk.Canvas(result_win)
        scrollbar = ttk.Scrollbar(result_win, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        for i, d in enumerate(filtered_data, 1):
            card_frame = ttk.Frame(
                scrollable_frame,
                style='Card.TFrame',
                padding=10
            )
            card_frame.pack(fill=tk.X, pady=5, padx=5)
            
            title_label = ttk.Label(
                card_frame,
                text=f"Filtrelenmiş Kayıt #{i}",
                style='Card.TLabel',
                font=('Helvetica', 12, 'bold')
            )
            title_label.pack(anchor=tk.W)
            
            fields = [
                f"👤 Cinsiyet: {d['Cinsiyet']}",
                f"🎂 Yaş: {d['Yaş']}",
                f"💇 Saç Rengi: {d['Saç Rengi']}",
                f"👁️ Göz Rengi: {d['Göz Rengi']}",
                f"😊 Duygu: {d['Duygu']}",
                f"👕 Kıyafet Rengi: {d['Kıyafet Rengi']}"
            ]
            
            for field in fields:
                field_label = ttk.Label(
                    card_frame,
                    text=field,
                    style='Card.TLabel',
                    font=('Helvetica', 10)
                )
                field_label.pack(anchor=tk.W, padx=10, pady=2)
            
            separator = ttk.Separator(card_frame, orient='horizontal')
            separator.pack(fill=tk.X, pady=5)

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
        chart_window.geometry("800x600")
        self.center_window(800, 600, chart_window)
        
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
            self.update_display()

    def on_closing(self):
        self.stop_camera()
        if self.db_connection:
            self.db_connection.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernFaceAnalysisApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
