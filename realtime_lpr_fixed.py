#!/usr/bin/env python3
"""
⚡ SISTEMA LPR TIEMPO REAL - VERSIÓN CORREGIDA
================================================================
Versión modificada para resolución de problemas Unicode y modo headless

Optimizaciones para tiempo real:
- IA cada 2-3 frames máximo
- Cooldown reducido a 0.5 segundos
- Detección de movimiento para activar IA
- Cache agresivo
- Prioridad a detección sobre FPS
- Logging UTF-8 compatible
- Modo headless (sin GUI) para Jetson

Autor: Sistema LPR automatizado - Versión Tiempo Real
Fecha: 2025-11-11
"""

import cv2
import json
import time
import argparse
import logging
import subprocess
import threading
import queue
import re
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import numpy as np

# Importar módulos del proyecto
try:
    from ultralytics import YOLO
    import easyocr
    print("✅ Módulos de IA importados correctamente")
except ImportError as e:
    print(f"❌ Error importando módulos: {e}")
    exit(1)

# Intentar importar gestor de base de datos
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent / "stream" / "database"))
    from db_manager import DatabaseManager
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    print("⚠️  Gestor de base de datos no disponible - continuando sin BD")

def is_valid_license_plate(text):
    """Validar si el texto corresponde a una placa válida"""
    if not text or len(text.strip()) < 3:
        return False
    
    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper().strip())
    
    patterns = [
        r'^[A-Z]{3}[0-9]{3}$',      # ABC123
        r'^[A-Z]{3}[0-9]{2}[A-Z]$', # ABC12D
        r'^[A-Z]{2}[0-9]{4}$',      # AB1234
        r'^[A-Z]{4}[0-9]{2}$',      # ABCD12
        r'^[0-9]{3}[A-Z]{3}$',      # 123ABC
        r'^[A-Z]{1}[0-9]{2}[A-Z]{3}$', # A12BCD
        r'^[A-Z]{2}[0-9]{3}[A-Z]{1}$', # AB123C
    ]
    
    if len(clean_text) < 4 or len(clean_text) > 8:
        return False
    
    has_letter = bool(re.search(r'[A-Z]', clean_text))
    has_number = bool(re.search(r'[0-9]', clean_text))
    
    if not (has_letter and has_number):
        return False
    
    for pattern in patterns:
        if re.match(pattern, clean_text):
            return True
    
    if len(clean_text) >= 4 and has_letter and has_number:
        return True
    
    return False

class RealtimeLPRSystem:
    """Sistema LPR optimizado para tiempo real - Versión corregida"""
    
    def __init__(self, config_path="config/ptz_config.json", headless=True):
        self.config_path = Path(config_path)
        self.headless = headless
        self.setup_logging()
        self.load_realtime_config()
        
        # Contadores
        self.running = False
        self.capture_frame_count = 0
        self.display_frame_count = 0
        self.ai_processed_frames = 0
        self.detections_count = 0
        self.start_time = None
        
        # Threading tiempo real
        self.capture_queue = queue.Queue(maxsize=2)  # Buffer mínimo
        self.display_queue = queue.Queue(maxsize=2)  
        self.ai_queue = queue.Queue(maxsize=3)       # Más buffer para IA
        self.result_queue = queue.Queue(maxsize=10)  
        
        self.capture_thread = None
        self.display_thread = None
        self.ai_thread = None
        
        # Cache y optimizaciones tiempo real
        self.ocr_cache = {}
        self.detection_cooldown = {}  # Cooldown por texto de placa
        self.bbox_cooldown = {}  # Cooldown por ubicación (bbox)
        self.recent_detections = []
        self.recent_plate_variations = {}  # Para detectar variaciones incorrectas del OCR
        
        # Detección de movimiento para activar IA
        self.motion_detector = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.last_motion_time = 0
        self.motion_threshold = 1000  # Píxeles en movimiento
        
        # Display optimizado (solo si no es headless)
        self.window_created = False
        self.last_frame = None
        
        # Configurar modelos
        self.setup_models()
        
        # Configurar red PTZ
        self.setup_network()
        
        # Configurar base de datos
        self.db_manager = None
        self.setup_database()
        
        # Directorios
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.logger.info("✅ Sistema LPR TIEMPO REAL inicializado")
    
    def setup_logging(self):
        """Configurar logging con UTF-8 encoding"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"realtime_lpr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Configurar logging con UTF-8 para Unicode
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Handler para archivo con UTF-8
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Handler para consola con manejo de Unicode
        class SafeStreamHandler(logging.StreamHandler):
            def emit(self, record):
                try:
                    msg = self.format(record)
                    if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding.lower() in ['cp1252', 'latin-1']:
                        # En Windows, convertir emojis a texto simple
                        emoji_map = {
                            '✅': '[OK]',
                            '❌': '[ERROR]',
                            '⚡': '[FAST]',
                            '📹': '[VIDEO]',
                            '🖥️': '[DISPLAY]',
                            '🧠': '[AI]',
                            '🎮': '[CONTROL]',
                            '📝': '[LOG]',
                            '🤖': '[ROBOT]',
                            '📦': '[MODEL]',
                            '🛑': '[STOP]',
                            '📊': '[STATS]',
                            '⏱️': '[TIME]',
                            '🎯': '[TARGET]',
                            '📹': '[FPS]'
                        }
                        for emoji, replacement in emoji_map.items():
                            msg = msg.replace(emoji, replacement)
                    self.stream.write(msg + self.terminator)
                except Exception:
                    pass
        
        console_handler = SafeStreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formato sin emojis
        formatter = logging.Formatter('%(asctime)s - REALTIME-LPR - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        self.logger = logger
        print(f"✅ Logs guardados en: {log_file}")
    
    def load_realtime_config(self):
        """Cargar configuración optimizada para tiempo real"""
        try:
            with open(self.config_path, 'r') as f:
                loaded_config = json.load(f)
            print(f"✅ Configuración cargada desde {self.config_path}")
        except FileNotFoundError:
            print("📄 Creando configuración tiempo real")
            loaded_config = {}
        
        # Configuración TIEMPO REAL
        self.config = {
            "camera": {
                "ip": "192.168.1.101",
                "user": "admin", 
                "password": "admin",
                "rtsp_url": "rtsp://admin:admin@192.168.1.101/cam/realmonitor?channel=1&subtype=1"
            },
            "jetson": {
                "ip": "192.168.1.100",
                "interface": "enP8p1s0"
            },
            "realtime_optimization": {
                "capture_target_fps": 25,      # Más FPS de captura
                "display_target_fps": 20,      # Más FPS de display  
                "ai_process_every": 2,         # IA CADA 2 FRAMES (ultra-frecuente)
                "motion_activation": True,     # Activar IA solo con movimiento
                "display_scale": 0.5,          # Display más grande (50% del tamaño original)
                "minimal_rendering": True,     
                "fast_resize": True,           
                "aggressive_cache": True,      # Cache más agresivo
                "headless_mode": self.headless # Modo sin GUI
            },
            "processing": {
                "confidence_threshold": 0.30,  # Umbral más bajo para no perder detecciones
                "plate_confidence_min": 0.40,  # OCR más estricto para evitar errores
                "max_detections": 3,
                "ocr_cache_enabled": True,
                "detection_cooldown_sec": 3.0,  # Cooldown aumentado para evitar duplicados
                "bbox_cooldown_sec": 2.0,       # Cooldown por ubicación
                "motion_cooldown_sec": 2,       # Cooldown para detección de movimiento
                "similarity_threshold": 0.7,    # Umbral para detectar variaciones similares
                "max_plate_variations": 3       # Máximo de variaciones a considerar
            },
            "database": {
                "enabled": True,
                "type": "mysql",
                "host": "localhost",
                "port": 3306,
                "database": "parqueadero_jetson",
                "user": "lpr_user",
                "password": "lpr_password",
                "charset": "utf8mb4"
            },
            "output": {
                "save_results": True,
                "save_images": False,           
                "show_video": not self.headless,
                "show_minimal_overlay": not self.headless,   
                "window_title": "Tiempo Real LPR - Jetson Optimized"
            }
        }
        
        # Fusionar configuración
        if loaded_config:
            for section, values in loaded_config.items():
                if section in self.config and isinstance(values, dict):
                    self.config[section].update(values)
                else:
                    self.config[section] = values
        
        self.save_config()
        print("📄 Configuración TIEMPO REAL lista")
    
    def save_config(self):
        """Guardar configuración"""
        self.config_path.parent.mkdir(exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def setup_network(self):
        """Configurar red PTZ"""
        interface = self.config["jetson"]["interface"]
        jetson_ip = self.config["jetson"]["ip"]
        
        try:
            commands = [
                f"sudo ip addr flush dev {interface} 2>/dev/null || true",
                f"sudo ip addr add {jetson_ip}/24 dev {interface} 2>/dev/null || true",
                f"sudo ethtool -s {interface} speed 100 duplex full autoneg off 2>/dev/null || true"
            ]
            
            for cmd in commands:
                subprocess.run(cmd, shell=True, capture_output=True)
            
            self.logger.info(f"[OK] Red tiempo real: {interface} -> {jetson_ip}")
            
        except Exception as e:
            self.logger.warning(f"[WARN] Configuración de red: {e}")
    
    def setup_database(self):
        """Configurar conexión a base de datos"""
        if not DB_AVAILABLE:
            self.logger.warning("[WARN] Gestor de BD no disponible - continuando sin BD")
            return
        
        if not self.config.get("database", {}).get("enabled", False):
            self.logger.info("[INFO] Base de datos deshabilitada en configuración")
            return
        
        try:
            db_config = self.config["database"].copy()
            db_config.pop("enabled", None)
            
            self.db_manager = DatabaseManager(db_config)
            self.logger.info("[OK] Base de datos MySQL conectada")
            
        except Exception as e:
            self.logger.warning(f"[WARN] No se pudo conectar a MySQL: {e}")
            self.logger.info("[INFO] Continuando sin base de datos")
            self.db_manager = None
    
    def setup_models(self):
        """Inicializar modelos de IA"""
        self.logger.info("[ROBOT] Cargando modelos de IA...")
        
        try:
            # Modelo YOLO
            model_files = list(Path(".").glob("*.pt"))
            if not model_files:
                raise FileNotFoundError("No se encontraron modelos YOLO")
            
            preferred_models = ["license_plate_detector.pt", "yolo11n.pt", "yolov8n.pt"]
            selected_model = None
            
            for model_name in preferred_models:
                if Path(model_name).exists():
                    selected_model = model_name
                    break
            
            if not selected_model:
                selected_model = str(model_files[0])
            
            self.logger.info(f"[MODEL] Cargando modelo: {selected_model}")
            self.yolo_model = YOLO(selected_model)
            
            # Warm-up más agresivo
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            for _ in range(3):  # Múltiples warm-ups
                self.yolo_model(dummy_frame, verbose=False)
            
            # EasyOCR
            self.logger.info("[LOG] Inicializando EasyOCR...")
            self.ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False, detector='dbnet18')
            
            # Warm-up OCR
            dummy_roi = np.zeros((50, 150, 3), dtype=np.uint8)
            for _ in range(2):
                self.ocr_reader.readtext(dummy_roi)
            
            self.logger.info("[OK] Modelos listos para TIEMPO REAL")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error cargando modelos: {e}")
            raise
    
    def detect_motion(self, frame):
        """Detectar movimiento para activar IA"""
        if not self.config["realtime_optimization"]["motion_activation"]:
            return True  # Siempre activar si no hay detección de movimiento
        
        # Aplicar detector de fondo
        fg_mask = self.motion_detector.apply(frame)
        
        # Contar píxeles en movimiento
        motion_pixels = cv2.countNonZero(fg_mask)
        
        current_time = time.time()
        motion_cooldown = self.config["processing"]["motion_cooldown_sec"]
        
        if motion_pixels > self.motion_threshold:
            if current_time - self.last_motion_time > motion_cooldown:
                self.last_motion_time = current_time
                return True
        
        return False
    
    def capture_worker(self):
        """Thread de captura optimizado para tiempo real"""
        self.logger.info("[VIDEO] Iniciando captura TIEMPO REAL...")
        
        rtsp_url = self.config["camera"]["rtsp_url"]
        
        try:
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, self.config["realtime_optimization"]["capture_target_fps"])
            # Mejorar manejo de errores de decodificación H264
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
            
            self.logger.info("[OK] Captura TIEMPO REAL conectada")
            
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                self.capture_frame_count += 1
                self.last_frame = frame.copy()
                
                # Distribución AGRESIVA a display
                try:
                    if not self.display_queue.full():
                        self.display_queue.put(frame.copy(), block=False)
                except queue.Full:
                    pass
                
                # IA MUY FRECUENTE + detección de movimiento
                ai_every = self.config["realtime_optimization"]["ai_process_every"]
                if self.capture_frame_count % ai_every == 0:
                    if self.detect_motion(frame):
                        try:
                            if not self.ai_queue.full():
                                self.ai_queue.put(frame.copy(), block=False)
                        except queue.Full:
                            pass
                
                # Control de velocidad mínimo
                time.sleep(0.001)
                
            cap.release()
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error en captura: {e}")
    
    def display_worker(self):
        """Thread de display optimizado"""
        if self.headless:
            # Modo headless - solo procesar pero no mostrar
            self.logger.info("[DISPLAY] Modo HEADLESS activado - Sin GUI")
            while self.running:
                try:
                    frame = self.display_queue.get(timeout=0.1)
                    self.display_frame_count += 1
                    # Procesar overlay pero no mostrar
                    self.process_overlay_headless(frame)
                    
                except queue.Empty:
                    continue
            return
        
        self.logger.info("[DISPLAY] Iniciando display TIEMPO REAL...")
        
        window_title = self.config["output"]["window_title"]
        try:
            cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            self.window_created = True
        except Exception as e:
            self.logger.warning(f"[WARN] No se puede crear ventana GUI: {e}")
            self.logger.info("[DISPLAY] Activando modo HEADLESS automáticamente")
            self.headless = True
            self.display_worker()  # Llamar recursivamente en modo headless
            return
        
        try:
            while self.running:
                try:
                    frame = self.display_queue.get(timeout=0.1)
                    
                    # Rendering ultra-mínimo
                    display_frame = self.realtime_overlay(frame)
                    
                    cv2.imshow(window_title, display_frame)
                    self.display_frame_count += 1
                    
                    # Check keys
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        self.running = False
                        break
                    elif key == ord('r'):
                        self.reset_stats()
                    elif key == ord('c'):
                        self.clear_cache()
                    elif key == ord('s'):
                        self.save_screenshot(frame)
                    
                except queue.Empty:
                    continue
                
            cv2.destroyAllWindows()
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error en display: {e}")
    
    def process_overlay_headless(self, frame):
        """Procesar overlay en modo headless"""
        # Procesar detecciones sin mostrar GUI
        # Esto mantiene el buffer actualizado para el procesamiento de IA
        pass
    
    def ai_worker(self):
        """Thread de IA para tiempo real"""
        self.logger.info("[AI] Iniciando IA TIEMPO REAL...")
        
        try:
            while self.running:
                try:
                    frame = self.ai_queue.get(timeout=0.5)  # Timeout más corto
                    
                    # Timestamp de cuando se recibió el frame
                    frame_received_time = time.time()
                    
                    detections = self.process_frame_ai_realtime(frame, frame_received_time)
                    
                    if detections:
                        self.result_queue.put(detections)
                        self.recent_detections.extend(detections)
                        
                        # Mantener solo últimas 3 detecciones para display
                        if len(self.recent_detections) > 3:
                            self.recent_detections = self.recent_detections[-3:]
                    
                except queue.Empty:
                    continue
                    
        except Exception as e:
            self.logger.error(f"[ERROR] Error en IA: {e}")
    
    def realtime_overlay(self, frame):
        """Overlay mínimo para tiempo real"""
        if self.headless:
            return frame
            
        # Resize ultra-rápido
        scale = self.config["realtime_optimization"]["display_scale"]
        h, w = frame.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        display_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Overlay mínimo
        if self.config["output"]["show_minimal_overlay"]:
            if self.start_time:
                runtime = time.time() - self.start_time
                display_fps = self.display_frame_count / runtime if runtime > 0 else 0
                ai_fps = self.ai_processed_frames / runtime if runtime > 0 else 0
                
                # Texto más informativo
                info = f"FPS: {display_fps:.1f} | IA: {ai_fps:.1f} | Dets: {self.detections_count}"
                cv2.putText(display_frame, info, (5, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Indicador de tiempo real
                cv2.putText(display_frame, "TIEMPO REAL", (5, new_h - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                # Detecciones recientes con timestamp
                for i, detection in enumerate(self.recent_detections[-2:]):
                    bbox = detection['bbox']
                    x1, y1, x2, y2 = [int(coord * scale) for coord in bbox]
                    
                    # Rectángulo
                    color = (0, 255, 0) if i == len(self.recent_detections) - 1 else (0, 255, 255)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Texto de la placa
                    plate_text = detection['plate_text']
                    cv2.putText(display_frame, plate_text, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return display_frame
    
    def calculate_bbox_hash(self, x1, y1, x2, y2):
        """Calcular hash de bbox para cooldown por ubicación"""
        # Redondear a múltiplos de 20 para agrupar bboxes similares
        x1_rounded = (x1 // 20) * 20
        y1_rounded = (y1 // 20) * 20
        x2_rounded = (x2 // 20) * 20
        y2_rounded = (y2 // 20) * 20
        return f"{x1_rounded}_{y1_rounded}_{x2_rounded}_{y2_rounded}"
    
    def calculate_text_similarity(self, text1, text2):
        """Calcular similitud entre dos textos de placa"""
        if not text1 or not text2:
            return 0.0
        
        # Normalizar textos
        t1 = re.sub(r'[^A-Z0-9]', '', text1.upper())
        t2 = re.sub(r'[^A-Z0-9]', '', text2.upper())
        
        if len(t1) == 0 or len(t2) == 0:
            return 0.0
        
        # Calcular similitud de caracteres
        matches = sum(1 for a, b in zip(t1, t2) if a == b)
        max_len = max(len(t1), len(t2))
        
        return matches / max_len if max_len > 0 else 0.0
    
    def is_similar_to_recent_detection(self, text, bbox):
        """Verificar si la detección es similar a una reciente (evitar variaciones OCR)"""
        similarity_threshold = self.config["processing"]["similarity_threshold"]
        
        # Revisar detecciones recientes (últimas 10)
        for recent in self.recent_detections[-10:]:
            recent_text = recent.get('plate_text', '')
            recent_bbox = recent.get('bbox', [])
            
            # Verificar similitud de texto
            similarity = self.calculate_text_similarity(text, recent_text)
            
            if similarity >= similarity_threshold:
                # Verificar si el bbox está cerca (misma ubicación)
                if recent_bbox:
                    rx1, ry1, rx2, ry2 = recent_bbox
                    x1, y1, x2, y2 = bbox
                    
                    # Calcular distancia entre centros
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    r_center_x = (rx1 + rx2) / 2
                    r_center_y = (ry1 + ry2) / 2
                    
                    distance = ((center_x - r_center_x) ** 2 + (center_y - r_center_y) ** 2) ** 0.5
                    
                    # Si está cerca y es similar, probablemente es la misma placa con OCR incorrecto
                    if distance < 100:  # 100 píxeles de tolerancia
                        # Preferir la detección con mayor confianza
                        if similarity > 0.8:  # Muy similar
                            return True, recent_text
        
        return False, None
    
    def process_frame_ai_realtime(self, frame, frame_time):
        """Procesamiento IA optimizado para tiempo real con cooldown inteligente"""
        try:
            self.ai_processed_frames += 1
            
            # YOLO más rápido
            results = self.yolo_model(frame, verbose=False, 
                                     conf=self.config["processing"]["confidence_threshold"],
                                     iou=0.5)  # NMS más agresivo
            
            detections = []
            current_time = datetime.now()
            current_time_float = time.time()
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        # Umbral de confianza más estricto
                        if confidence < self.config["processing"]["plate_confidence_min"]:
                            continue
                        
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)
                        
                        if x2 > x1 and y2 > y1:
                            # Cooldown por ubicación (bbox)
                            bbox_hash = self.calculate_bbox_hash(x1, y1, x2, y2)
                            bbox_cooldown_sec = self.config["processing"]["bbox_cooldown_sec"]
                            
                            if bbox_hash in self.bbox_cooldown:
                                last_bbox_time = self.bbox_cooldown[bbox_hash]
                                if (current_time_float - last_bbox_time) < bbox_cooldown_sec:
                                    continue  # Esta ubicación fue detectada recientemente
                            
                            roi = frame[y1:y2, x1:x2]
                            
                            # OCR con cache
                            plate_texts = self.get_plate_text_cached_realtime(roi)
                            
                            for plate_text in plate_texts:
                                text = plate_text['text']
                                ocr_conf = plate_text['confidence']
                                
                                # Validar formato de placa
                                if not is_valid_license_plate(text):
                                    continue
                                
                                # Verificar si es similar a una detección reciente (evitar variaciones OCR)
                                is_similar, better_text = self.is_similar_to_recent_detection(text, [x1, y1, x2, y2])
                                
                                if is_similar and better_text:
                                    # Usar el texto de mayor confianza de detecciones recientes
                                    text = better_text
                                    self.logger.debug(f"[DEBUG] Usando texto mejorado: {text} (era similar a {plate_text['text']})")
                                
                                # Cooldown por texto de placa
                                cooldown_sec = self.config["processing"]["detection_cooldown_sec"]
                                if text in self.detection_cooldown:
                                    last_time = self.detection_cooldown[text]
                                    if (current_time_float - last_time) < cooldown_sec:
                                        continue  # Esta placa fue detectada recientemente
                                
                                # Actualizar cooldowns
                                self.detection_cooldown[text] = current_time_float
                                self.bbox_cooldown[bbox_hash] = current_time_float
                                
                                # Limpiar cooldowns antiguos (más de 1 minuto)
                                self.cleanup_old_cooldowns()
                                
                                # Calcular latencia real
                                processing_latency = time.time() - frame_time
                                
                                detection = {
                                    'timestamp': current_time.isoformat(),
                                    'frame_number': self.capture_frame_count,
                                    'ai_frame_number': self.ai_processed_frames,
                                    'plate_text': text,
                                    'yolo_confidence': confidence,
                                    'ocr_confidence': ocr_conf,
                                    'bbox': [x1, y1, x2, y2],
                                    'processing_latency_ms': int(processing_latency * 1000),
                                    'valid': True
                                }
                                
                                detections.append(detection)
                                self.detections_count += 1
                                
                                self.logger.info(f"[TARGET] PLACA: {text} "
                                               f"(YOLO: {confidence:.2f}, OCR: {ocr_conf:.2f}, "
                                               f"Latencia: {int(processing_latency * 1000)}ms)")
                                
                                # Guardar en base de datos
                                if self.db_manager:
                                    try:
                                        db_data = {
                                            'timestamp': current_time,
                                            'plate_text': text,
                                            'confidence': confidence,
                                            'plate_score': ocr_conf,
                                            'vehicle_bbox': None,
                                            'plate_bbox': json.dumps([x1, y1, x2, y2]),
                                            'camera_location': 'entrada_principal'
                                        }
                                        self.db_manager.insert_detection(db_data)
                                    except Exception as e:
                                        self.logger.warning(f"[WARN] Error guardando en BD: {e}")
                                
                                # Guardar en archivo
                                if self.config["output"]["save_results"]:
                                    self.save_detection(detection)
            
            return detections
            
        except Exception as e:
            self.logger.warning(f"[WARN] Error en IA: {e}")
            return []
    
    def get_plate_text_cached_realtime(self, roi):
        """OCR con cache ultra-agresivo para tiempo real"""
        if not self.config["processing"]["ocr_cache_enabled"]:
            return self.read_plate_text(roi)
        
        # Hash más rápido (solo primeros bytes)
        roi_bytes = roi.tobytes()
        roi_hash = hashlib.md5(roi_bytes[::100]).hexdigest()[:12]  # Sample cada 100 bytes
        
        if roi_hash in self.ocr_cache:
            return self.ocr_cache[roi_hash]
        
        # Procesar con OCR
        texts = self.read_plate_text(roi)
        
        # Cache más agresivo
        if texts:
            self.ocr_cache[roi_hash] = texts
            
            # Límite de cache más grande para tiempo real
            if len(self.ocr_cache) > 100:
                # Remover 20 entradas más antiguas
                old_keys = list(self.ocr_cache.keys())[:20]
                for old_key in old_keys:
                    del self.ocr_cache[old_key]
        
        return texts
    
    def read_plate_text(self, roi):
        """OCR optimizado"""
        try:
            if roi.size == 0:
                return []
                
            # 🔥 OPTIMIZACIÓN 1: Reducir ROI agresivamente
            target_height = 60   # Altura máxima para OCR
            target_width = 180   # Ancho máximo para OCR
        
            if roi.shape[0] > target_height or roi.shape[1] > target_width:
                scale_h = target_height / roi.shape[0] if roi.shape[0] > target_height else 1.0
                scale_w = target_width / roi.shape[1] if roi.shape[1] > target_width else 1.0
                scale = min(scale_h, scale_w)
            
                new_h = max(20, int(roi.shape[0] * scale))  # Mínimo 20px altura
                new_w = max(60, int(roi.shape[1] * scale))  # Mínimo 60px ancho
                roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)    
            
            # Preprocesamiento ultra-mínimo
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi
                
            gray = cv2.bilateralFilter(gray, 3, 30, 30)
            
            # OCR directo sin preprocesamiento adicional
            ocr_results = self.ocr_reader.readtext(gray)
            
            texts = []
            for (bbox, text, confidence) in ocr_results:
                # Umbral de confianza más estricto para evitar errores OCR
                if confidence >= self.config["processing"]["plate_confidence_min"] and len(text.strip()) > 2:
                    cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                    if len(cleaned_text) >= 3:
                        texts.append({
                            'text': cleaned_text,
                            'confidence': confidence
                        })
            
            return texts
            
        except Exception as e:
            return []
    
    def save_detection(self, detection):
        """Guardar detección"""
        try:
            results_file = self.results_dir / f"realtime_detections_{datetime.now().strftime('%Y%m%d')}.jsonl"
            
            with open(results_file, 'a') as f:
                json.dump(detection, f)
                f.write('\n')
                
        except Exception as e:
            self.logger.error(f"[ERROR] Error guardando: {e}")
    
    def reset_stats(self):
        """Reset estadísticas"""
        self.capture_frame_count = 0
        self.display_frame_count = 0
        self.ai_processed_frames = 0
        self.detections_count = 0
        self.recent_detections = []
        self.start_time = time.time()
        self.logger.info("[RESET] Reset TIEMPO REAL")
    
    def cleanup_old_cooldowns(self):
        """Limpiar cooldowns antiguos para liberar memoria"""
        current_time_float = time.time()
        
        # Limpiar cooldowns de texto (más de 1 minuto)
        old_texts = [text for text, last_time in self.detection_cooldown.items() 
                    if (current_time_float - last_time) > 60]
        for text in old_texts:
            del self.detection_cooldown[text]
        
        # Limpiar cooldowns de bbox (más de 1 minuto)
        old_bboxes = [bbox_hash for bbox_hash, last_time in self.bbox_cooldown.items() 
                     if (current_time_float - last_time) > 60]
        for bbox_hash in old_bboxes:
            del self.bbox_cooldown[bbox_hash]
    
    def clear_cache(self):
        """Limpiar cache"""
        self.ocr_cache.clear()
        self.detection_cooldown.clear()
        self.bbox_cooldown.clear()
        self.recent_detections.clear()
        self.logger.info("[CLEAR] Cache limpiado")
    
    def save_screenshot(self, frame):
        """Guardar captura"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        cv2.imwrite(f"realtime_capture_{timestamp}.jpg", frame)
        self.logger.info(f"[CAPTURE] Captura: realtime_capture_{timestamp}.jpg")
    
    def run(self):
        """Ejecutar sistema tiempo real"""
        self.logger.info("[FAST] Iniciando sistema TIEMPO REAL...")
        
        try:
            self.running = True
            self.start_time = time.time()
            
            # Iniciar threads
            self.capture_thread = threading.Thread(target=self.capture_worker)
            self.display_thread = threading.Thread(target=self.display_worker)
            self.ai_thread = threading.Thread(target=self.ai_worker)
            
            self.capture_thread.daemon = True
            self.display_thread.daemon = True
            self.ai_thread.daemon = True
            
            self.capture_thread.start()
            self.display_thread.start()
            self.ai_thread.start()
            
            self.logger.info("[OK] Sistema TIEMPO REAL iniciado")
            self.logger.info("[CONTROL] Controles: 'q'=salir, 'r'=reset, 'c'=cache, 's'=captura")
            self.logger.info("[FAST] IA CADA 2 FRAMES + Cooldown 0.5s = DETECCIÓN CASI INSTANTÁNEA")
            
            # Esperar threads
            while self.running:
                time.sleep(0.1)
                
                # Verificar threads
                if not self.capture_thread.is_alive():
                    self.logger.error("[ERROR] Thread de captura murió")
                    break
                if not self.display_thread.is_alive():
                    self.logger.error("[ERROR] Thread de display murió")
                    break
                    
        except KeyboardInterrupt:
            self.logger.info("[STOP] Detenido por usuario")
        except Exception as e:
            self.logger.error(f"[ERROR] Error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        finally:
            self.stop()
    
    def stop(self):
        """Detener sistema"""
        self.logger.info("[STOP] Deteniendo sistema TIEMPO REAL...")
        
        self.running = False
        
        # Esperar threads
        threads = [self.capture_thread, self.display_thread, self.ai_thread]
        for thread in threads:
            if thread and thread.is_alive():
                thread.join(timeout=2)
        
        # Cerrar conexión de base de datos
        if self.db_manager:
            try:
                self.db_manager.close()
            except Exception as e:
                self.logger.warning(f"[WARN] Error cerrando BD: {e}")
        
        # Estadísticas finales
        if self.start_time:
            runtime = time.time() - self.start_time
            capture_fps = self.capture_frame_count / runtime if runtime > 0 else 0
            display_fps = self.display_frame_count / runtime if runtime > 0 else 0
            ai_fps = self.ai_processed_frames / runtime if runtime > 0 else 0
            
            # Calcular latencia promedio
            avg_detections_per_minute = (self.detections_count / runtime) * 60 if runtime > 0 else 0
            
            self.logger.info("[STATS] ESTADÍSTICAS TIEMPO REAL:")
            self.logger.info(f"   [TIME] Tiempo: {runtime:.1f}s")
            self.logger.info(f"   [FPS] FPS Captura: {capture_fps:.1f}")
            self.logger.info(f"   [FPS] FPS Display: {display_fps:.1f}")
            self.logger.info(f"   [AI] FPS IA: {ai_fps:.1f}")
            self.logger.info(f"   [TARGET] Detecciones: {self.detections_count}")
            self.logger.info(f"   [FAST] Detecciones/min: {avg_detections_per_minute:.1f}")
            self.logger.info(f"   [STATS] Eficiencia IA: {(ai_fps*100):.1f}% más frecuente")

def main():
    """Función principal tiempo real"""
    parser = argparse.ArgumentParser(description="Sistema LPR Tiempo Real - Jetson Optimized")
    parser.add_argument("--config", default="config/ptz_config.json")
    parser.add_argument("--ai-every", type=int, default=2, help="Procesar IA cada N frames (por defecto: 2)")
    parser.add_argument("--cooldown", type=float, default=0.5, help="Cooldown en segundos (por defecto: 0.5)")
    parser.add_argument("--motion", action="store_true", help="Activar detección de movimiento")
    parser.add_argument("--confidence", type=float, default=0.30, help="Umbral confianza YOLO")
    parser.add_argument("--display-scale", type=float, default=0.5, help="Escala display")
    parser.add_argument("--headless", action="store_true", help="Activar modo sin GUI (recomendado para Jetson)")
    
    args = parser.parse_args()
    
    print("⚡⚡ SISTEMA LPR TIEMPO REAL ⚡⚡")
    print("=" * 50)
    print("[TARGET] Enfoque: DETECCIÓN CASI INSTANTÁNEA")
    print("[VIDEO] IA cada 2 frames (máxima frecuencia)")
    print("[TIME] Cooldown 3.0 segundos (mejorado para evitar duplicados)")
    print("[TARGET] Detección de movimiento opcional")
    print("[FAST] Procesamiento optimizado")
    print("[ROBOT] Optimizado para Jetson Orin Nano")
    print("[DATABASE] Conexión MySQL habilitada")
    print("[IMPROVED] Cooldown inteligente por ubicación y texto")
    print("[IMPROVED] Validación mejorada de OCR para evitar errores")
    print("=" * 50)
    
    # Detectar automáticamente si estamos en un entorno sin GUI
    headless_mode = args.headless or os.environ.get('DISPLAY', '') == '' or os.name == 'nt'
    
    try:
        system = RealtimeLPRSystem(args.config, headless=headless_mode)
        
        # Aplicar configuraciones tiempo real
        system.config["realtime_optimization"]["ai_process_every"] = args.ai_every
        system.config["processing"]["detection_cooldown_sec"] = args.cooldown
        system.config["realtime_optimization"]["motion_activation"] = args.motion
        system.config["processing"]["confidence_threshold"] = args.confidence
        system.config["realtime_optimization"]["display_scale"] = args.display_scale
        
        print(f"[SETUP] Configuración TIEMPO REAL:")
        print(f"   [AI] IA cada: {args.ai_every} frames")
        print(f"   [TIME] Cooldown: {args.cooldown}s")
        print(f"   [TARGET] Detección movimiento: {'Sí' if args.motion else 'No'}")
        print(f"   [CONFIDENCE] Confianza: {args.confidence}")
        print(f"   [SCALE] Escala: {args.display_scale}")
        print(f"   [HEADLESS] Modo sin GUI: {'Sí' if headless_mode else 'No'}")
        print()
        if headless_mode:
            print("[WARN] ADVERTENCIA: Modo HEADLESS activado - Sin interfaz gráfica")
        else:
            print("[WARN] ADVERTENCIA: Este modo consume más recursos")
        print("[WARN] pero detecta placas CASI INSTANTÁNEAMENTE")
        print()
        
        system.run()
        
    except Exception as e:
        print(f"[ERROR] Error fatal: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())