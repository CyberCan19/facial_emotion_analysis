import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from deepface import DeepFace
import os

class DeepFaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DeepFace: Yüz Analizi ve Tanıma")

        # Görsel alanı
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        # Butonlar
        self.select_btn = tk.Button(root, text="Resim Seç", command=self.select_image)
        self.select_btn.pack(pady=5)

        self.analyze_btn = tk.Button(root, text="Yüz Analizi", command=self.analyze_image, state=tk.DISABLED)
        self.analyze_btn.pack(pady=5)

        self.recognize_btn = tk.Button(root, text="Yüz Tanıma", command=self.recognize_face, state=tk.DISABLED)
        self.recognize_btn.pack(pady=5)

        # Sonuç metin kutusu
        self.result_text = tk.Text(root, height=15, width=70)
        self.result_text.pack(pady=10)

        # Seçilen resim yolu
        self.img_path = None

        # Tanınacak yüzlerin bulunduğu klasör
        self.known_faces_folder = "images"

    def select_image(self):
        filetypes = [("Görüntü Dosyaları", "*.jpg *.jpeg *.png")]
        path = filedialog.askopenfilename(title="Resim Seç", filetypes=filetypes)

        if path:
            self.img_path = path
            self.show_image(path)
            self.analyze_btn.config(state=tk.NORMAL)
            self.recognize_btn.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)

    def show_image(self, path):
        img = Image.open(path)
        img.thumbnail((400, 400))
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk

    def analyze_image(self):
        if not self.img_path:
            messagebox.showerror("Hata", "Lütfen önce bir resim seçin.")
            return

        try:
            results = DeepFace.analyze(
                img_path=self.img_path,
                actions=['age', 'gender', 'race', 'emotion'],
                enforce_detection=False
            )

            face = results[0] if isinstance(results, list) else results

            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "--- Yüz Özellikleri ---\n")
            self.result_text.insert(tk.END, f"Yaş: {face['age']}\n")
            self.result_text.insert(tk.END, f"Cinsiyet: {face['gender']}\n")
            self.result_text.insert(tk.END, f"Irk: {face['dominant_race']}\n")
            self.result_text.insert(tk.END, f"Duygu: {face['dominant_emotion']}\n\n")

        except Exception as e:
            messagebox.showerror("Analiz Hatası", f"Hata oluştu:\n{str(e)}")

    def recognize_face(self):
        if not self.img_path:
            messagebox.showerror("Hata", "Lütfen önce bir resim seçin.")
            return

        try:
            self.result_text.insert(tk.END, "--- Yüz Tanıma Sonucu ---\n")
            matched = False

            for filename in os.listdir(self.known_faces_folder):
                known_path = os.path.join(self.known_faces_folder, filename)

                if known_path == self.img_path:
                    continue
                if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                result = DeepFace.verify(
                    img1_path=self.img_path,
                    img2_path=known_path,
                    enforce_detection=False
                )

                if result["verified"]:
                    self.result_text.insert(tk.END, f"Eşleşen Kişi: {filename} ✅\n")
                    matched = True
                    break

            if not matched:
                self.result_text.insert(tk.END, "Eşleşme bulunamadı ❌\n")

        except Exception as e:
            messagebox.showerror("Tanıma Hatası", f"Hata oluştu:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DeepFaceApp(root)
    root.mainloop()
