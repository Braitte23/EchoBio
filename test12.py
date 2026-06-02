import cv2
import numpy as np
import os
import pickle
import socket
import base64
import threading
import time
from datetime import datetime
import queue

from flask import Flask, request, render_template_string, jsonify, send_from_directory
import customtkinter as ctk
from PIL import Image

# ==========================================
# CONFIGURACIÓN DE TEMA Y RUTAS
# ==========================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BASE_PATH = "C:/Users/bryan/fotos" if os.name == 'nt' else "/home/braitte/Desktop/fotos"
DB_PATH = os.path.join(BASE_PATH, "faces_db.pkl")
UNKNOWN_PATH = os.path.join(BASE_PATH, "unknown")
THUMBS_PATH = os.path.join(BASE_PATH, "thumbs")
LOG_FILE = os.path.join(BASE_PATH, "access_log.txt")

for p in [BASE_PATH, UNKNOWN_PATH, THUMBS_PATH]:
    os.makedirs(p, exist_ok=True)

THRESHOLD = 0.45
CAMERA_INDEX = "rtsp://192.168.100.15:8554/cam"

db = {}
db_names = []
db_embeddings = np.empty((0, 512))

app_instance = None
gui_queue = queue.Queue(maxsize=2)
last_unknown_time = 0

# ==========================================
# UTILIDADES
# ==========================================
def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

# ==========================================
# MOTOR IA (INSIGHTFACE)
# ==========================================
from insightface.app import FaceAnalysis

face_app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
face_app.prepare(ctx_id=0, det_size=(640, 640))

ia_lock = threading.Lock()

def align_face(img, kps, size=160):
    src = np.array(kps, dtype=np.float32)
    dst = np.array([[38.29, 51.69], [73.53, 51.50], [56.02, 71.73], [41.54, 92.36], [70.72, 92.20]], dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, dst)
    return cv2.warpAffine(img, M, (size, size))

def rebuild_embeddings():
    global db_names, db_embeddings, db
    if os.path.exists(DB_PATH):
        try:
            with open(DB_PATH, "rb") as f:
                db = pickle.load(f)
        except:
            pass
    names, embeds = [], []
    for name, emb_list in db.items():
        for emb in emb_list:
            embeds.append(np.array(emb).reshape(-1))
            names.append(name)
    db_embeddings = np.vstack(embeds) if embeds else np.empty((0, 512))
    db_names = names

def update_db(name, embeddings, faces):
    global db
    name = name.strip()
    db.setdefault(name, []).extend(embeddings)
    if faces and len(faces) > 0:
        cv2.imwrite(os.path.join(THUMBS_PATH, f"{name}.jpg"), faces[0])
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)
    rebuild_embeddings()

rebuild_embeddings()

# ==========================================
# HILO LECTOR RTSP
# ==========================================
class RTSPStreamer:
    def __init__(self, src):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|max_delay;500000"
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.ret = False
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            if self.cap.isOpened():
                try:
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        self.ret = ret
                        self.frame = frame
                    else:
                        time.sleep(0.01)
                except Exception:
                    time.sleep(0.05)
            else:
                time.sleep(1)

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()

# ==========================================
# APP ESCRITORIO (SOLO MONITOREO + LOGS)
# ==========================================
class VernaimApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        global app_instance
        app_instance = self

        self.title("VERNAiM - Monitor v1.0")
        self.geometry("900x550")

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar minimalista
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(self.sidebar, text="VERNAiM", font=("Segoe UI", 24, "bold")).pack(pady=(20, 0))
        ctk.CTkLabel(self.sidebar, text="IDENTITY GATEWAY", font=("Segoe UI", 8), text_color="#22d3ee").pack(pady=(0, 10))

        ip = get_ip()
        ctk.CTkLabel(self.sidebar, text="PANEL WEB:", font=("Segoe UI", 10, "bold"), text_color="#94a3b8").pack(pady=(20, 2))
        ctk.CTkLabel(self.sidebar, text=f"[{ip}](http://{ip}:5000)", font=("Consolas", 11), text_color="#22d3ee").pack(pady=(0, 20))

        ctk.CTkLabel(self.sidebar, text="ACTIVIDAD:", font=("Segoe UI", 10, "bold"), text_color="#94a3b8").pack(pady=(10, 5))
        self.log_box = ctk.CTkTextbox(self.sidebar, width=200, height=280, font=("Consolas", 10))
        self.log_box.pack(pady=5, padx=10)

        # Panel principal de video
        self.main_view = ctk.CTkFrame(self, fg_color="#020617")
        self.main_view.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.video_label = ctk.CTkLabel(self.main_view, text="INICIANDO CÁMARA...")
        self.video_label.pack(expand=True, fill="both")

        self.rtsp_stream = None
        self.is_monitoring = True

        self.update_gui_frame()
        threading.Thread(target=self.run_camera, daemon=True).start()

    def log(self, msg):
        try:
            self.log_box.insert("end", f"[{datetime.now().strftime('%H:%M')}] {msg}\n")
            self.log_box.see("end")
        except:
            pass

    def update_gui_frame(self):
        if self.is_monitoring:
            try:
                while True:
                    img_ctk = gui_queue.get_nowait()
                    self.video_label.configure(image=img_ctk, text="")
                    self.video_label.image = img_ctk
                    gui_queue.task_done()
            except queue.Empty:
                pass
        self.after(16, self.update_gui_frame)

    def run_camera(self):
        self.rtsp_stream = RTSPStreamer(CAMERA_INDEX)
        last_known_log_time = 0

        while self.is_monitoring:
            ret, frame = self.rtsp_stream.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            proc_frame = frame.copy()

            with ia_lock:
                faces = face_app.get(proc_frame)

            for face in faces:
                name, color = "DESCONOCIDO", (0, 0, 255)
                if len(db_embeddings) > 0:
                    sims = (db_embeddings / np.linalg.norm(db_embeddings, axis=1, keepdims=True)) @ (
                        face.embedding / np.linalg.norm(face.embedding))
                    if np.max(sims) > THRESHOLD:
                        name, color = db_names[np.argmax(sims)], (0, 255, 0)

                if name == "DESCONOCIDO":
                    msg = handle_event(name, proc_frame, face)
                    if msg:
                        self.log(msg)
                else:
                    if time.time() - last_known_log_time > 15:
                        self.log(f"Acceso: {name}")
                        last_known_log_time = time.time()

                x, y, x2, y2 = face.bbox.astype(int)
                cv2.rectangle(proc_frame, (x, y), (x2, y2), color, 2)
                cv2.putText(proc_frame, name, (x, y - 10), 0, 0.6, color, 2)

            img = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            img_ctk = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(650, 450))

            if gui_queue.full():
                try:
                    gui_queue.get_nowait()
                except queue.Empty:
                    pass
            gui_queue.put(img_ctk)

# ==========================================
# SERVIDOR WEB COMPLETO (FLASK)
# ==========================================
app_web = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VERNAiM // Control Panel</title>
    <link href="[fonts.googleapis.com](https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Share+Tech+Mono&display=swap)" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #09090b;
            --bg-secondary: #11111b;
            --bg-card: #1e1e2e;
            --accent-cyan: #00f0ff;
            --accent-green: #39ff14;
            --accent-red: #ff0055;
            --accent-purple: #a855f7;
            --text-main: #e2e8f0;
            --text-muted: rgba(226, 232, 240, 0.5);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: var(--bg-primary);
            color: var(--text-main);
            font-family: 'Share Tech Mono', monospace;
            min-height: 100vh;
        }
        
        /* NAV */
        .nav {
            background: var(--bg-secondary);
            border-bottom: 1px solid rgba(0, 240, 255, 0.2);
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }
        .nav h1 {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.3rem;
            color: var(--accent-cyan);
            letter-spacing: 2px;
        }
        .nav-links {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        .nav-btn {
            padding: 10px 18px;
            background: transparent;
            border: 1px solid var(--accent-cyan);
            color: var(--accent-cyan);
            font-family: inherit;
            font-size: 0.85rem;
            cursor: pointer;
            text-transform: uppercase;
            transition: all 0.2s;
            border-radius: 4px;
        }
        .nav-btn:hover, .nav-btn.active {
            background: var(--accent-cyan);
            color: #000;
        }
        .nav-btn.green { border-color: var(--accent-green); color: var(--accent-green); }
        .nav-btn.green:hover, .nav-btn.green.active { background: var(--accent-green); color: #000; }
        .nav-btn.purple { border-color: var(--accent-purple); color: var(--accent-purple); }
        .nav-btn.purple:hover, .nav-btn.purple.active { background: var(--accent-purple); color: #000; }
        
        /* CONTENEDOR PRINCIPAL */
        .container {
            max-width: 900px;
            margin: 20px auto;
            padding: 0 15px;
        }
        
        /* SECCIONES */
        .section { display: none; }
        .section.active { display: block; }
        
        .section-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.1rem;
            color: var(--accent-cyan);
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px dashed rgba(0, 240, 255, 0.3);
        }
        
        /* CARDS */
        .card {
            background: var(--bg-card);
            border: 1px solid rgba(0, 240, 255, 0.15);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 15px;
        }
        
        /* FORMULARIO REGISTRO */
        .form-group { margin-bottom: 18px; }
        label {
            display: block;
            margin-bottom: 8px;
            font-size: 0.85rem;
            color: var(--text-muted);
            text-transform: uppercase;
        }
        input[type="text"] {
            width: 100%;
            padding: 12px;
            background: #000;
            border: 1px solid rgba(0, 240, 255, 0.4);
            color: #fff;
            font-family: inherit;
            font-size: 1rem;
            border-radius: 4px;
            outline: none;
        }
        input[type="text"]:focus {
            border-color: var(--accent-cyan);
            box-shadow: 0 0 10px rgba(0, 240, 255, 0.3);
        }
        
        .media-box {
            width: 100%;
            height: 260px;
            background: #000;
            border: 1px dashed rgba(226, 232, 240, 0.2);
            margin-bottom: 15px;
            display: flex;
            justify-content: center;
            align-items: center;
            position: relative;
            overflow: hidden;
            border-radius: 4px;
        }
        video, .preview-img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: none;
        }
        
        .tabs { display: flex; margin-bottom: 15px; border-bottom: 1px solid rgba(0, 240, 255, 0.2); }
        .tab-btn {
            flex: 1;
            padding: 10px;
            background: transparent;
            border: none;
            color: var(--text-muted);
            font-family: inherit;
            font-size: 0.9rem;
            cursor: pointer;
            text-transform: uppercase;
            transition: all 0.2s;
        }
        .tab-btn.active {
            color: var(--accent-cyan);
            border-bottom: 2px solid var(--accent-cyan);
        }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        
        .file-label {
            padding: 12px;
            background: rgba(0,0,0,0.4);
            border: 1px dashed var(--accent-cyan);
            border-radius: 4px;
            cursor: pointer;
            display: block;
            text-align: center;
            color: var(--accent-cyan);
            transition: all 0.2s;
        }
        .file-label:hover { background: rgba(0, 240, 255, 0.1); }
        input[type="file"] { display: none; }
        
        .btn {
            width: 100%;
            padding: 14px;
            font-family: 'Orbitron', sans-serif;
            font-size: 0.95rem;
            font-weight: bold;
            text-transform: uppercase;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-top: 10px;
            transition: all 0.2s;
        }
        .btn-cyan { background: var(--accent-cyan); color: #000; }
        .btn-green { background: var(--accent-green); color: #000; }
        .btn-red { background: var(--accent-red); color: #fff; }
        .btn-small {
            padding: 8px 14px;
            font-size: 0.75rem;
            width: auto;
            margin: 3px;
        }
        
        #response-log {
            padding: 12px;
            background: rgba(0,0,0,0.6);
            border-left: 3px solid var(--accent-cyan);
            font-size: 0.85rem;
            display: none;
            margin-top: 15px;
            text-align: center;
        }
        
        /* USUARIOS GRID */
        .users-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
            gap: 15px;
        }
        .user-card {
            background: var(--bg-card);
            border: 1px solid rgba(0, 240, 255, 0.15);
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }
        .user-thumb {
            width: 100px;
            height: 100px;
            object-fit: cover;
            border-radius: 50%;
            border: 2px solid var(--accent-cyan);
            margin-bottom: 10px;
        }
        .user-name {
            font-size: 0.95rem;
            font-weight: bold;
            margin-bottom: 10px;
            color: var(--text-main);
        }
        .no-thumb {
            width: 100px;
            height: 100px;
            background: #333;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 10px auto;
            font-size: 2rem;
            color: var(--text-muted);
        }
        
        /* CLIPS */
        .clip-item {
            display: flex;
            align-items: center;
            background: var(--bg-card);
            border: 1px solid rgba(0, 240, 255, 0.15);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 12px;
            flex-wrap: wrap;
            gap: 15px;
        }
        .clip-thumb {
            width: 100px;
            height: 100px;
            object-fit: cover;
            border-radius: 6px;
            border: 1px solid var(--accent-cyan);
        }
        .clip-info {
            flex: 1;
            min-width: 150px;
        }
        .clip-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
        }
        
        /* MODAL */
        .modal-overlay {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.85);
            display: none;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }
        .modal-overlay.active { display: flex; }
        .modal-content {
            background: var(--bg-secondary);
            border: 1px solid var(--accent-cyan);
            border-radius: 10px;
            padding: 25px;
            max-width: 500px;
            width: 90%;
            text-align: center;
        }
        .modal-content img {
            max-width: 100%;
            max-height: 350px;
            border-radius: 6px;
            margin-bottom: 15px;
        }
        .modal-content input[type="text"] {
            margin: 15px 0;
        }
        
        .empty-state {
            text-align: center;
            padding: 40px;
            color: var(--text-muted);
        }
    </style>
</head>
<body>
    <nav class="nav">
        <h1>VERNAiM</h1>
        <div class="nav-links">
            <button class="nav-btn green active" onclick="showSection('register')">Registrar</button>
            <button class="nav-btn" onclick="showSection('users')">Usuarios</button>
            <button class="nav-btn purple" onclick="showSection('clips')">Clips</button>
        </div>
    </nav>
    
    <div class="container">
        <!-- SECCIÓN REGISTRO -->
        <div id="section-register" class="section active">
            <h2 class="section-title">REGISTRO BIOMÉTRICO</h2>
            <div class="card">
                <div class="form-group">
                    <label>Nombre Completo</label>
                    <input type="text" id="subject-name" placeholder="Ej: Bryan Perez" autocomplete="off">
                </div>
                <div class="tabs">
                    <button class="tab-btn active" onclick="switchTab('cam-tab')">Cámara</button>
                    <button class="tab-btn" onclick="switchTab('file-tab')">Subir Foto</button>
                </div>
                <div id="cam-tab" class="tab-content active">
                    <div class="media-box">
                        <span id="cam-placeholder" style="color:var(--text-muted);">CÁMARA APAGADA</span>
                        <video id="video" autoplay playsinline></video>
                        <img id="cam-preview" class="preview-img" alt="Preview">
                    </div>
                    <button class="btn btn-cyan" id="btn-init-cam" onclick="initCamera()">Encender Cámara</button>
                    <button class="btn btn-green" id="btn-capture" style="display:none;" onclick="registerViaCamera()">Capturar y Registrar</button>
                </div>
                <div id="file-tab" class="tab-content">
                    <div class="media-box">
                        <span id="file-placeholder" style="color:var(--text-muted);">NINGÚN ARCHIVO</span>
                        <img id="file-preview" class="preview-img" alt="Preview File">
                    </div>
                    <label class="file-label" for="file-input">Seleccionar Imagen</label>
                    <input type="file" id="file-input" accept="image/*" onchange="previewFile()">
                    <button class="btn btn-green" onclick="registerViaFile()">Registrar Archivo</button>
                </div>
                <canvas id="canvas" style="display:none;"></canvas>
                <div id="response-log"></div>
            </div>
        </div>
        
        <!-- SECCIÓN USUARIOS -->
        <div id="section-users" class="section">
            <h2 class="section-title">USUARIOS REGISTRADOS</h2>
            <div id="users-container" class="users-grid"></div>
        </div>
        
        <!-- SECCIÓN CLIPS -->
        <div id="section-clips" class="section">
            <h2 class="section-title">DETECCIONES RECIENTES</h2>
            <div id="clips-container"></div>
        </div>
    </div>
    
    <!-- MODALS -->
    <div id="modal-photo" class="modal-overlay" onclick="closeModal('modal-photo')">
        <div class="modal-content" onclick="event.stopPropagation()">
            <img id="modal-photo-img" src="" alt="Foto">
            <button class="btn btn-cyan" onclick="closeModal('modal-photo')">Cerrar</button>
        </div>
    </div>
    
    <div id="modal-rename" class="modal-overlay" onclick="closeModal('modal-rename')">
        <div class="modal-content" onclick="event.stopPropagation()">
            <h3 style="margin-bottom:15px; color:var(--accent-cyan);">RENOMBRAR USUARIO</h3>
            <input type="text" id="rename-input" placeholder="Nuevo nombre">
            <input type="hidden" id="rename-old">
            <button class="btn btn-green" onclick="confirmRename()">Guardar</button>
            <button class="btn btn-cyan" onclick="closeModal('modal-rename')" style="margin-top:10px;">Cancelar</button>
        </div>
    </div>
    
    <div id="modal-delete" class="modal-overlay" onclick="closeModal('modal-delete')">
        <div class="modal-content" onclick="event.stopPropagation()">
            <h3 style="margin-bottom:15px; color:var(--accent-red);">¿ELIMINAR USUARIO?</h3>
            <p id="delete-user-name" style="margin-bottom:20px;"></p>
            <input type="hidden" id="delete-target">
            <button class="btn btn-red" onclick="confirmDelete()">Sí, Eliminar</button>
            <button class="btn btn-cyan" onclick="closeModal('modal-delete')" style="margin-top:10px;">Cancelar</button>
        </div>
    </div>
    
    <div id="modal-register-clip" class="modal-overlay" onclick="closeModal('modal-register-clip')">
        <div class="modal-content" onclick="event.stopPropagation()">
            <h3 style="margin-bottom:15px; color:var(--accent-green);">REGISTRAR DESDE CLIP</h3>
            <img id="clip-reg-img" src="" alt="Clip" style="max-height:200px;">
            <input type="text" id="clip-reg-name" placeholder="Nombre de la persona">
            <input type="hidden" id="clip-reg-file">
            <button class="btn btn-green" onclick="confirmClipRegister()">Registrar</button>
            <button class="btn btn-cyan" onclick="closeModal('modal-register-clip')" style="margin-top:10px;">Cancelar</button>
        </div>
    </div>

    <script>
        let stream = null;
        let activeTab = 'cam-tab';
        let base64FileString = null;
        
        function showSection(name) {
            document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
            document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
            document.getElementById('section-' + name).classList.add('active');
            event.target.classList.add('active');
            
            if (name === 'users') loadUsers();
            if (name === 'clips') loadClips();
        }
        
        function switchTab(tabId) {
            activeTab = tabId;
            document.querySelectorAll('.tab-btn').forEach((b, i) => {
                b.classList.toggle('active', (i === 0 && tabId === 'cam-tab') || (i === 1 && tabId === 'file-tab'));
            });
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            
            if (tabId !== 'cam-tab' && stream) {
                stream.getTracks().forEach(track => track.stop());
                document.getElementById('video').style.display = 'none';
                document.getElementById('cam-placeholder').style.display = 'block';
                document.getElementById('btn-init-cam').style.display = 'block';
                document.getElementById('btn-capture').style.display = 'none';
                document.getElementById('cam-preview').style.display = 'none';
                stream = null;
            }
        }
        
        async function initCamera() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
                const video = document.getElementById('video');
                video.srcObject = stream;
                document.getElementById('cam-placeholder').style.display = 'none';
                document.getElementById('cam-preview').style.display = 'none';
                video.style.display = 'block';
                document.getElementById('btn-init-cam').style.display = 'none';
                document.getElementById('btn-capture').style.display = 'block';
            } catch (err) {
                alert("Error al acceder a la cámara.");
            }
        }
        
        function previewFile() {
            const file = document.getElementById('file-input').files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onloadend = function() {
                base64FileString = reader.result;
                document.getElementById('file-placeholder').style.display = 'none';
                const imgPrev = document.getElementById('file-preview');
                imgPrev.src = reader.result;
                imgPrev.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
        
        function showLog(msg, type) {
            const log = document.getElementById('response-log');
            log.style.display = 'block';
            log.style.borderLeftColor = type === 'success' ? 'var(--accent-green)' : type === 'error' ? 'var(--accent-red)' : 'var(--accent-cyan)';
            log.style.color = type === 'success' ? 'var(--accent-green)' : type === 'error' ? 'var(--accent-red)' : '#fff';
            log.innerText = msg;
            setTimeout(() => { log.style.display = 'none'; }, 3000);
        }
        
        function sendRegister(name, base64Str, callback) {
            showLog("PROCESANDO...", "info");
            fetch('/register_web', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name, image: base64Str })
            })
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    showLog("ÉXITO: " + data.message, "success");
                    document.getElementById('subject-name').value = '';
                    if (callback) callback();
                } else {
                    showLog("ERROR: " + data.message, "error");
                }
            })
            .catch(() => showLog("ERROR: Desconexión", "error"));
        }
        
        function registerViaCamera() {
            const name = document.getElementById('subject-name').value.trim();
            if (!name) { alert("Ingrese el nombre."); return; }
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
            const dataUrl = canvas.toDataURL('image/jpeg');
            video.style.display = 'none';
            const camPrev = document.getElementById('cam-preview');
            camPrev.src = dataUrl;
            camPrev.style.display = 'block';
            sendRegister(name, dataUrl, () => {
                camPrev.style.display = 'none';
                video.style.display = 'block';
            });
        }
        
        function registerViaFile() {
            const name = document.getElementById('subject-name').value.trim();
            if (!name) { alert("Ingrese el nombre."); return; }
            if (!base64FileString) { alert("Seleccione una imagen."); return; }
            sendRegister(name, base64FileString, () => {
                document.getElementById('file-preview').style.display = 'none';
                document.getElementById('file-placeholder').style.display = 'block';
                document.getElementById('file-input').value = '';
                base64FileString = null;
            });
        }
        
        // USUARIOS
        function loadUsers() {
            fetch('/api/users')
            .then(res => res.json())
            .then(data => {
                const container = document.getElementById('users-container');
                if (data.users.length === 0) {
                    container.innerHTML = '<div class="empty-state">No hay usuarios registrados</div>';
                    return;
                }
                container.innerHTML = data.users.map(u => `
                    <div class="user-card">
                        ${u.thumb ? `<img class="user-thumb" src="/thumbs/${u.name}.jpg?t=${Date.now()}" alt="${u.name}">` : `<div class="no-thumb">?</div>`}
                        <div class="user-name">${u.name}</div>
                        <button class="btn btn-small btn-cyan" onclick="openRename('${u.name}')">Renombrar</button>
                        <button class="btn btn-small btn-red" onclick="openDelete('${u.name}')">Eliminar</button>
                    </div>
                `).join('');
            });
        }
        
        function openRename(name) {
            document.getElementById('rename-old').value = name;
            document.getElementById('rename-input').value = '';
            document.getElementById('modal-rename').classList.add('active');
        }
        
        function confirmRename() {
            const oldName = document.getElementById('rename-old').value;
            const newName = document.getElementById('rename-input').value.trim();
            if (!newName) { alert("Ingrese un nombre."); return; }
            fetch('/api/rename_user', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ old_name: oldName, new_name: newName })
            })
            .then(res => res.json())
            .then(data => {
                closeModal('modal-rename');
                loadUsers();
            });
        }
        
        function openDelete(name) {
            document.getElementById('delete-target').value = name;
            document.getElementById('delete-user-name').innerText = name;
            document.getElementById('modal-delete').classList.add('active');
        }
        
        function confirmDelete() {
            const name = document.getElementById('delete-target').value;
            fetch('/api/delete_user', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name })
            })
            .then(res => res.json())
            .then(data => {
                closeModal('modal-delete');
                loadUsers();
            });
        }
        
        // CLIPS
        function loadClips() {
            fetch('/api/clips')
            .then(res => res.json())
            .then(data => {
                const container = document.getElementById('clips-container');
                if (data.clips.length === 0) {
                    container.innerHTML = '<div class="empty-state">No hay detecciones recientes</div>';
                    return;
                }
                container.innerHTML = data.clips.map(c => `
                    <div class="clip-item">
                        <img class="clip-thumb" src="/unknown/${c.img}?t=${Date.now()}" alt="Clip">
                        <div class="clip-info">
                            <div style="font-weight:bold; margin-bottom:5px;">${c.timestamp}</div>
                            ${c.video ? `<div style="font-size:0.8rem; color:var(--text-muted);">Video: ${c.video}</div>` : ''}
                        </div>
                        <div class="clip-actions">
                            <button class="btn btn-small btn-cyan" onclick="viewPhoto('/unknown/${c.img}')">Ver</button>
                            <button class="btn btn-small btn-green" onclick="openClipRegister('${c.img}')">Registrar</button>
                            <button class="btn btn-small btn-red" onclick="deleteClip('${c.img}', '${c.video || ''}')">Borrar</button>
                        </div>
                    </div>
                `).join('');
            });
        }
        
        function viewPhoto(url) {
            document.getElementById('modal-photo-img').src = url + '?t=' + Date.now();
            document.getElementById('modal-photo').classList.add('active');
        }
        
        function openClipRegister(imgFile) {
            document.getElementById('clip-reg-img').src = '/unknown/' + imgFile + '?t=' + Date.now();
            document.getElementById('clip-reg-file').value = imgFile;
            document.getElementById('clip-reg-name').value = '';
            document.getElementById('modal-register-clip').classList.add('active');
        }
        
        function confirmClipRegister() {
            const name = document.getElementById('clip-reg-name').value.trim();
            const imgFile = document.getElementById('clip-reg-file').value;
            if (!name) { alert("Ingrese un nombre."); return; }
            fetch('/api/register_from_clip', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name, img_file: imgFile })
            })
            .then(res => res.json())
            .then(data => {
                closeModal('modal-register-clip');
                if (data.success) {
                    alert("Registrado con éxito: " + name);
                    loadClips();
                } else {
                    alert("Error: " + data.message);
                }
            });
        }
        
        function deleteClip(img, vid) {
            if (!confirm("¿Eliminar este clip?")) return;
            fetch('/api/delete_clip', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ img: img, vid: vid })
            })
            .then(() => loadClips());
        }
        
        function closeModal(id) {
            document.getElementById(id).classList.remove('active');
        }
    </script>
</body>
</html>
"""

@app_web.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app_web.route("/thumbs/<path:filename>")
def serve_thumb(filename):
    return send_from_directory(THUMBS_PATH, filename)

@app_web.route("/unknown/<path:filename>")
def serve_unknown(filename):
    return send_from_directory(UNKNOWN_PATH, filename)

@app_web.route("/register_web", methods=["POST"])
def register_web():
    try:
        data = request.get_json()
        name = data.get("name", "").strip()
        img_b64 = data.get("image", "")

        if not name or not img_b64:
            return jsonify({"success": False, "message": "Faltan parámetros."})

        header, encoded = img_b64.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"success": False, "message": "Error al decodificar imagen."})

        h, w = img.shape[:2]
        with ia_lock:
            face_app.prepare(ctx_id=0, det_size=(w - (w % 32), h - (h % 32)))
            faces = face_app.get(img)
            face_app.prepare(ctx_id=0, det_size=(640, 640))

        if not faces:
            return jsonify({"success": False, "message": "No se detectó ningún rostro."})

        face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        update_db(name, [face.embedding], [align_face(img, face.kps)])

        if app_instance:
            app_instance.log(f"WEB: {name} registrado")

        return jsonify({"success": True, "message": f"'{name}' registrado."})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@app_web.route("/api/users")
def api_users():
    rebuild_embeddings()
    users = []
    for name in db.keys():
        thumb_exists = os.path.exists(os.path.join(THUMBS_PATH, f"{name}.jpg"))
        users.append({"name": name, "thumb": thumb_exists})
    return jsonify({"users": users})

@app_web.route("/api/rename_user", methods=["POST"])
def api_rename_user():
    data = request.get_json()
    old_name = data.get("old_name", "").strip()
    new_name = data.get("new_name", "").strip()
    if old_name in db:
        db[new_name] = db.pop(old_name)
        old_p = os.path.join(THUMBS_PATH, f"{old_name}.jpg")
        new_p = os.path.join(THUMBS_PATH, f"{new_name}.jpg")
        if os.path.exists(old_p):
            os.rename(old_p, new_p)
        with open(DB_PATH, "wb") as f:
            pickle.dump(db, f)
        rebuild_embeddings()
        if app_instance:
            app_instance.log(f"Renombrado: {old_name} → {new_name}")
    return jsonify({"success": True})

@app_web.route("/api/delete_user", methods=["POST"])
def api_delete_user():
    data = request.get_json()
    name = data.get("name", "").strip()
    if name in db:
        del db[name]
    p = os.path.join(THUMBS_PATH, f"{name}.jpg")
    if os.path.exists(p):
        os.remove(p)
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)
    rebuild_embeddings()
    if app_instance:
        app_instance.log(f"Eliminado: {name}")
    return jsonify({"success": True})

@app_web.route("/api/clips")
def api_clips():
    files = sorted([f for f in os.listdir(UNKNOWN_PATH) if f.endswith(".jpg")], reverse=True)
    clips = []
    for f_img in files:
        timestamp = f_img.replace("FACE_", "").replace(".jpg", "").replace("_", " ")
        v_file = f"CLIP_{f_img.replace('FACE_', '').replace('.jpg', '')}.mp4"
        video_exists = os.path.exists(os.path.join(UNKNOWN_PATH, v_file))
        clips.append({
            "img": f_img,
            "video": v_file if video_exists else None,
            "timestamp": timestamp
        })
    return jsonify({"clips": clips})

@app_web.route("/api/register_from_clip", methods=["POST"])
def api_register_from_clip():
    data = request.get_json()
    name = data.get("name", "").strip()
    img_file = data.get("img_file", "")
    img_path = os.path.join(UNKNOWN_PATH, img_file)

    if not name or not os.path.exists(img_path):
        return jsonify({"success": False, "message": "Datos inválidos."})

    img = cv2.imread(img_path)
    if img is None:
        return jsonify({"success": False, "message": "No se pudo leer la imagen."})

    h, w = img.shape[:2]
    with ia_lock:
        face_app.prepare(ctx_id=0, det_size=(w - (w % 32), h - (h % 32)))
        faces = face_app.get(img)
        face_app.prepare(ctx_id=0, det_size=(640, 640))

    if not faces:
        return jsonify({"success": False, "message": "No se detectó rostro."})

    face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    update_db(name, [face.embedding], [align_face(img, face.kps)])

    # Eliminar clip
    v_file = f"CLIP_{img_file.replace('FACE_', '').replace('.jpg', '')}.mp4"
    for f in [img_file, v_file]:
        p = os.path.join(UNKNOWN_PATH, f)
        if os.path.exists(p):
            os.remove(p)

    if app_instance:
        app_instance.log(f"Clip → {name}")

    return jsonify({"success": True, "message": f"'{name}' registrado desde clip."})

@app_web.route("/api/delete_clip", methods=["POST"])
def api_delete_clip():
    data = request.get_json()
    img = data.get("img", "")
    vid = data.get("vid", "")
    for f in [img, vid]:
        p = os.path.join(UNKNOWN_PATH, f)
        if os.path.exists(p):
            os.remove(p)
    return jsonify({"success": True})

# ==========================================
# GESTIÓN DE EVENTOS
# ==========================================
def handle_event(name, frame, face):
    global last_unknown_time
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if name == "DESCONOCIDO":
        if time.time() - last_unknown_time > 4:
            last_unknown_time = time.time()
            cv2.imwrite(os.path.join(UNKNOWN_PATH, f"FACE_{ts}.jpg"), align_face(frame, face.kps))

            def rec():
                out = cv2.VideoWriter(
                    os.path.join(UNKNOWN_PATH, f"CLIP_{ts}.mp4"),
                    cv2.VideoWriter_fourcc(*'mp4v'), 20,
                    (frame.shape[1], frame.shape[0])
                )
                c_v = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_FFMPEG)
                for _ in range(100):
                    r, f = c_v.read()
                    if r:
                        out.write(f)
                out.release()
                c_v.release()

            threading.Thread(target=rec, daemon=True).start()
            return "ALERTA: Desconocido (Clip guardado)"
    return None

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    threading.Thread(
        target=lambda: app_web.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False),
        daemon=True
    ).start()
    VernaimApp().mainloop()
