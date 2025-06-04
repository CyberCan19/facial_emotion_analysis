import os
import subprocess
import sys
import venv

VENV_DIR = "env"
REQUIREMENTS = [
    "deepface>=0.0.79",
    "opencv-python>=4.5.0",
    "scikit-learn>=1.0.0",
    "pandas>=1.3.0",
    "numpy>=1.21.0",
    "matplotlib>=3.4.0",
    "Pillow>=9.0.0",
    "tf-keras"
]

APP_FILENAME = "face_app.py"  # Ana uygulamanın dosya adı

def set_execution_policy():
    if os.name == "nt":  # Sadece Windows'ta geçerli
        try:
            print("🔐 PowerShell güvenlik politikası güncelleniyor...")
            subprocess.run([
                "powershell", 
                "-Command", 
                "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force"
            ], check=True)
            print("✅ ExecutionPolicy başarıyla güncellendi.")
        except subprocess.CalledProcessError:
            print("⚠️ ExecutionPolicy ayarlanamadı. Lütfen PowerShell'i yönetici olarak çalıştırarak manuel olarak aşağıdaki komutu girin:\n")
            print("Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser")

def create_virtual_env():
    if not os.path.exists(VENV_DIR):
        print("🔧 Sanal ortam oluşturuluyor...")
        venv.create(VENV_DIR, with_pip=True)
        print("✅ Sanal ortam oluşturuldu.")
    else:
        print("ℹ️ Sanal ortam zaten var, atlanıyor.")

def install_requirements():
    print("📦 Gerekli kütüphaneler yükleniyor...")
    pip_executable = os.path.join(VENV_DIR, "Scripts", "pip") if os.name == "nt" else os.path.join(VENV_DIR, "bin", "pip")
    subprocess.check_call([pip_executable, "install", "--upgrade", "pip"])
    subprocess.check_call([pip_executable, "install"] + REQUIREMENTS)
    print("✅ Kütüphaneler yüklendi.")

def run_app():
    print("🚀 Uygulama başlatılıyor...")
    python_executable = os.path.join(VENV_DIR, "Scripts", "python") if os.name == "nt" else os.path.join(VENV_DIR, "bin", "python")
    subprocess.call([python_executable, APP_FILENAME])

if __name__ == "__main__":
    set_execution_policy()
    create_virtual_env()
    install_requirements()
    run_app()
