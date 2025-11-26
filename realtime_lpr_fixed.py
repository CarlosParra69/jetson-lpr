#!/usr/bin/env python3
"""
SISTEMA LPR TIEMPO REAL - VERSIÓN BETA 1.0.0
================================================================

Optimizaciones para tiempo real:
- IA cada 2-3 frames máximo
- Cooldown reducido a 0.5 segundos
- Detección de movimiento para activar IA
- Cache agresivo
- Prioridad a detección sobre FPS
- Logging UTF-8 compatible
- Modo headless (sin GUI) para Jetson

Autor: SENA - Tecnologias Virtuales CIMM (Centro Industrial de Mantenimiento y Manufactura)
Desarrollo: Carlos Parra
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

# Intentar importar controlador PTZ
try:
    from ptz_controller import PTZController
    PTZ_AVAILABLE = True
except ImportError:
    PTZ_AVAILABLE = False
    print("⚠️  Controlador PTZ no disponible - continuando sin control PTZ")

def is_valid_license_plate(text):
    """
    Validar si el texto corresponde a una placa válida colombiana
    Validación más estricta para evitar reconocimientos falsos
    """
    if not text or len(text.strip()) < 3:
        return False
    
    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper().strip())
    
    # Validar longitud (placas colombianas típicamente 6 caracteres)
    if len(clean_text) < 5 or len(clean_text) > 7:
        return False
    
    # Debe tener al menos 2 letras y 2 números
    letters = len(re.findall(r'[A-Z]', clean_text))
    numbers = len(re.findall(r'[0-9]', clean_text))
    
    if letters < 2 or numbers < 2:
        return False
    
    # Patrones específicos de placas colombianas (más estrictos)
    patterns = [
        r'^[A-Z]{3}[0-9]{3}$',      # ABC123 (formato estándar)
        r'^[A-Z]{3}[0-9]{2}[A-Z]$', # ABC12D (formato alternativo)
        r'^[A-Z]{2}[0-9]{4}$',      # AB1234 (formato antiguo)
        r'^[A-Z]{2}[0-9]{3}[A-Z]$', # AB123C (formato mixto)
    ]
    
    # Verificar contra patrones específicos
    for pattern in patterns:
        if re.match(pattern, clean_text):
            return True
    
    # Validación adicional: no debe tener solo números o solo letras
    if letters == 0 or numbers == 0:
        return False
    
    # Validación de formato: debe alternar o tener estructura razonable
    # Rechazar patrones sospechosos como "VU54" (muy corto, formato raro)
    if len(clean_text) == 4:
        # Placas de 4 caracteres deben tener al menos 2 números y 2 letras
        if not (letters >= 2 and numbers >= 2):
            return False
        # Rechazar si empieza con solo letras seguidas (ej: VU54 -> VU5 sería mejor)
        if re.match(r'^[A-Z]{2}[0-9]{2}$', clean_text):
            # Formato VU54 es válido pero necesita más validación
            return True
    
    # Para placas de 5-7 caracteres, validar estructura
    if len(clean_text) >= 5:
        # Debe tener mezcla razonable de letras y números
        letter_ratio = letters / len(clean_text)
        number_ratio = numbers / len(clean_text)
        
        # Las placas colombianas típicamente tienen 40-60% letras y 40-60% números
        if letter_ratio < 0.3 or letter_ratio > 0.7:
            return False
        if number_ratio < 0.3 or number_ratio > 0.7:
            return False
    
    return False  # Por defecto, rechazar si no cumple patrones específicos

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
        self.recent_detections = []  # Detecciones recientes con timestamp
        self.recent_plate_variations = {}  # Para detectar variaciones incorrectas del OCR
        self.display_detections = []  # Detecciones para mostrar (con expiración)
        # Funcionalidad de detección mejorada eliminada - mostrar detecciones inmediatamente
        
        # Sistema de agrupación inteligente de detecciones
        self.detection_groups = {}  # Agrupa detecciones similares por vehículo
        self.group_timeout_sec = 0.2  # Tiempo muy reducido para procesar grupos más rápido (0.2s)
        self.group_lock = threading.Lock()  # Lock para grupos de detecciones
        self.processing_groups = set()  # Grupos que están siendo procesados para evitar duplicados
        self.active_detections = []  # Detecciones activas para mostrar inmediatamente
        
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
        
        # Configurar control PTZ
        self.ptz_controller = None
        self.setup_ptz()
        
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
                "capture_target_fps": 30,      # FPS aumentado para más frames
                "display_target_fps": 30,      # FPS de display aumentado para fluidez
                "ai_process_every": 3,         # IA cada 3 frames (balance velocidad/precisión)
                "motion_activation": False,    # Desactivar detección de movimiento para procesar siempre
                "display_scale": 0.6,          # Display más grande para mejor visualización
                "minimal_rendering": True,     
                "fast_resize": True,           
                "aggressive_cache": True,      # Cache más agresivo
                "headless_mode": self.headless, # Modo sin GUI
                "skip_frame_processing": False # No saltar frames en display
            },
            "processing": {
                "confidence_threshold": 0.40,  # Umbral aumentado para evitar falsos positivos
                "plate_confidence_min": 0.50,  # OCR aumentado para reconocimiento más preciso
                "max_detections": 3,            # Reducido para evitar detecciones múltiples falsas
                "ocr_cache_enabled": True,
                "detection_cooldown_sec": 2.0,  # Cooldown aumentado para evitar duplicados
                "bbox_cooldown_sec": 1.5,       # Cooldown por ubicación aumentado
                "motion_cooldown_sec": 1,       # Cooldown para detección de movimiento
                "similarity_threshold": 0.75,   # Umbral aumentado para agrupar solo variaciones muy similares
                "max_plate_variations": 3,      # Menos variaciones para evitar errores
                "max_detection_distance_m": 30.0,  # Distancia máxima aumentada para máximo alcance (30 metros)
                "min_plate_width_px": 25,        # Ancho mínimo muy reducido para detectar placas muy lejanas
                "min_plate_height_px": 10,       # Altura mínima muy reducida para detectar placas muy lejanas
                "distance_filter_enabled": True,  # Habilitar filtro de distancia pero con rango amplio
                "detection_display_timeout_sec": 15.0,  # Tiempo aumentado para ver múltiples detecciones
                "enhanced_detection_enabled": False,   # Deshabilitado - mostrar detecciones inmediatamente
                "colombian_plate_optimization": True,  # Optimización específica para placas colombianas
                "color_detection_enabled": False,      # Deshabilitar filtro de color (más permisivo)
                "preprocess_aggressive": True,         # Preprocesamiento agresivo
                "min_roi_width_for_ocr": 60,          # Ancho mínimo de ROI para OCR (reducido para mayor alcance)
                "min_roi_height_for_ocr": 20           # Altura mínima de ROI para OCR (reducido para mayor alcance)
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
            "ptz": {
                "enabled": True,
                "auto_focus": True,  # Enfocar automáticamente en placas detectadas
                "zoom_level": 0.7,  # Nivel de zoom al detectar placa (0.0 a 1.0)
                "restore_after_sec": 3.0,  # Segundos antes de restaurar posición
                "movement_speed": 0.6  # Velocidad de movimiento (0.0 a 1.0)
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
            # Usar sudo con contraseña automática (proyecto)
            sudo_password = "proyecto"
            commands = [
                f"echo '{sudo_password}' | sudo -S ip addr flush dev {interface} 2>/dev/null || true",
                f"echo '{sudo_password}' | sudo -S ip addr add {jetson_ip}/24 dev {interface} 2>/dev/null || true",
                f"echo '{sudo_password}' | sudo -S ethtool -s {interface} speed 100 duplex full autoneg off 2>/dev/null || true"
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
    
    def setup_ptz(self):
        """Configurar control PTZ"""
        if not PTZ_AVAILABLE:
            self.logger.warning("[WARN] Controlador PTZ no disponible - continuando sin PTZ")
            return
        
        if not self.config.get("ptz", {}).get("enabled", False):
            self.logger.info("[INFO] Control PTZ deshabilitado en configuración")
            return
        
        try:
            camera_config = self.config["camera"]
            self.ptz_controller = PTZController(
                camera_ip=camera_config["ip"],
                username=camera_config["user"],
                password=camera_config["password"]
            )
            self.logger.info("[OK] Controlador PTZ inicializado")
            
        except Exception as e:
            self.logger.warning(f"[WARN] No se pudo inicializar control PTZ: {e}")
            self.logger.info("[INFO] Continuando sin control PTZ")
            self.ptz_controller = None
    
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
    
    def detect_colombian_plate_colors(self, roi):
        """
        Detectar si el ROI contiene colores de placa colombiana (amarilla/blanca)
        Retorna True si detecta colores típicos de placas colombianas
        """
        if not self.config["processing"].get("color_detection_enabled", True):
            return True  # Si está deshabilitado, aceptar todas
        
        try:
            if roi.size == 0 or len(roi.shape) != 3:
                return True  # Si no hay color, asumir válido
            
            # Convertir a HSV para mejor detección de color
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Rango para amarillo (placas amarillas colombianas)
            # Amarillo en HSV: H=15-30, S>50, V>50
            yellow_lower = np.array([15, 50, 50])
            yellow_upper = np.array([30, 255, 255])
            yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
            
            # Rango para blanco (placas blancas colombianas)
            # Blanco en HSV: S<30, V>200
            white_lower = np.array([0, 0, 200])
            white_upper = np.array([180, 30, 255])
            white_mask = cv2.inRange(hsv, white_lower, white_upper)
            
            # Combinar máscaras
            combined_mask = cv2.bitwise_or(yellow_mask, white_mask)
            
            # Calcular porcentaje de píxeles que coinciden con colores de placa
            total_pixels = roi.shape[0] * roi.shape[1]
            matching_pixels = cv2.countNonZero(combined_mask)
            color_ratio = matching_pixels / total_pixels if total_pixels > 0 else 0
            
            # Si al menos 10% del ROI tiene colores de placa colombiana, es válido (más permisivo)
            return color_ratio >= 0.10
            
        except Exception as e:
            self.logger.debug(f"[COLOR] Error en detección de color: {e}")
            return True  # En caso de error, aceptar
    
    def preprocess_colombian_plate(self, roi):
        """
        Preprocesamiento agresivo específico para placas colombianas (amarillas/blancas)
        Aplica múltiples técnicas para mejorar contraste y legibilidad
        """
        if not self.config["processing"].get("preprocess_aggressive", True):
            return roi
        
        try:
            if roi.size == 0:
                return roi
            
            # Convertir a escala de grises si es necesario
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi.copy()
            
            # TÉCNICA 1: CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # Mejora contraste local, especialmente útil para placas con iluminación desigual
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # TÉCNICA 2: Sharpening para mejorar bordes de caracteres
            kernel_sharpen = np.array([[-1, -1, -1],
                                      [-1,  9, -1],
                                      [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
            
            # TÉCNICA 3: Bilateral filter para reducir ruido manteniendo bordes
            denoised = cv2.bilateralFilter(sharpened, 5, 50, 50)
            
            # TÉCNICA 4: OTSU thresholding para binarización adaptativa
            _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # TÉCNICA 5: Morphological operations para limpiar
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
            
        except Exception as e:
            self.logger.debug(f"[PREPROCESS] Error en preprocesamiento: {e}")
            return roi
    
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
                
                # IA CADA FRAME para detección agresiva de placas colombianas
                ai_every = self.config["realtime_optimization"]["ai_process_every"]
                if self.capture_frame_count % ai_every == 0:
                    # Si motion_activation está deshabilitado, procesar siempre
                    if not self.config["realtime_optimization"]["motion_activation"]:
                        try:
                            if not self.ai_queue.full():
                                self.ai_queue.put(frame.copy(), block=False)
                        except queue.Full:
                            pass
                    elif self.detect_motion(frame):
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
                
                # Limpiar detecciones expiradas (aumentado a 10 segundos para mejor visualización)
                current_time_float = time.time()
                timeout_sec = self.config["processing"].get("detection_display_timeout_sec", 10.0)
                
                # Combinar display_detections y active_detections
                all_detections = list(self.display_detections) + list(self.active_detections)
                
                # Limpiar expiradas
                valid_detections = [
                    d for d in all_detections 
                    if (current_time_float - d.get('display_time', 0)) < timeout_sec
                ]
                
                # Actualizar listas
                self.display_detections = [d for d in valid_detections if d in self.display_detections]
                self.active_detections = [d for d in valid_detections if d in self.active_detections]
                
                # Mostrar todas las detecciones válidas
                for i, detection in enumerate(valid_detections):
                    bbox = detection['bbox']
                    x1, y1, x2, y2 = [int(coord * scale) for coord in bbox]
                    
                    # Calcular tiempo restante para desvanecer
                    elapsed = current_time_float - detection.get('display_time', 0)
                    time_remaining = timeout_sec - elapsed
                    
                    # Sistema de colores según estado:
                    # - Amarillo (0, 255, 255): posible detección (en proceso)
                    # - Verde (0, 255, 0): detección final confirmada y guardada
                    # - Verde claro (0, 200, 0): confirmada pero no guardada (sin BD o error de guardado)
                    # - Rojo (0, 0, 255): detección errónea (formato inválido, OCR muy bajo)
                    status = detection.get('status', 'possible')
                    saved_to_db = detection.get('saved_to_db', False)
                    
                    if status == 'saved' and saved_to_db:
                        # Verde: guardada correctamente en BD
                        color = (0, 255, 0)
                        status_text = "✓ GUARDADA"
                    elif status == 'confirmed':
                        # Verde claro: confirmada pero aún no guardada (sin BD o error de guardado)
                        color = (0, 200, 0)
                        status_text = "CONFIRMADA"
                    elif status == 'error':
                        # Rojo: error de detección (formato inválido, OCR muy bajo)
                        color = (0, 0, 255)
                        status_text = "✗ ERROR"
                    else:
                        # Amarillo: posible detección (en proceso de agrupación)
                        alpha = max(0.5, time_remaining / timeout_sec)
                        color = (0, int(255 * alpha), 255)  # Amarillo
                        status_text = "PROCESANDO"
                    
                    # Dibujar rectángulo con color según estado (línea más gruesa para mejor visibilidad)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Texto de la placa (más grande y visible)
                    plate_text = detection['plate_text']
                    font_scale = 0.6
                    font_thickness = 2
                    (text_width, text_height), baseline = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    
                    # Fondo semitransparente para el texto
                    cv2.rectangle(display_frame, (x1, y1-text_height-5), (x1+text_width+5, y1+5), (0, 0, 0), -1)
                    cv2.putText(display_frame, plate_text, (x1+2, y1-2), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
                    
                    # Mostrar estado debajo de la placa (más pequeño)
                    (status_width, status_height), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cv2.rectangle(display_frame, (x1, y2+2), (x1+status_width+5, y2+status_height+8), (0, 0, 0), -1)
                    cv2.putText(display_frame, status_text, (x1+2, y2+status_height+5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return display_frame
    
    def calculate_bbox_hash(self, x1, y1, x2, y2):
        """Calcular hash de bbox para cooldown por ubicación"""
        # Redondear a múltiplos de 20 para agrupar bboxes similares
        x1_rounded = (x1 // 20) * 20
        y1_rounded = (y1 // 20) * 20
        x2_rounded = (x2 // 20) * 20
        y2_rounded = (y2 // 20) * 20
        return f"{x1_rounded}_{y1_rounded}_{x2_rounded}_{y2_rounded}"
    
    def estimate_distance_from_bbox(self, bbox, frame_width, frame_height):
        """
        Estimar distancia de la placa basándose en el tamaño del bbox
        Retorna distancia estimada en metros
        Optimizado para mayor alcance (hasta 30 metros)
        """
        x1, y1, x2, y2 = bbox
        plate_width = x2 - x1
        plate_height = y2 - y1
        
        # Tamaño promedio de una placa real: ~52cm x 11cm
        # Usar el ancho como referencia principal (más estable)
        real_plate_width_cm = 52.0  # cm
        real_plate_height_cm = 11.0  # cm
        
        # Calibración mejorada para mayor alcance:
        # - Placa de 80px de ancho = 5 metros (cerca)
        # - Placa de 40px de ancho = 10 metros (media distancia)
        # - Placa de 20px de ancho = 20 metros (lejos)
        # - Placa de 10px de ancho = 30+ metros (muy lejos)
        
        reference_width_px = 80
        reference_distance_m = 5.0
        
        if plate_width < 10:  # Muy pequeña, probablemente muy lejos (30+ metros)
            return 35.0  # Retornar distancia muy grande
        elif plate_width < 15:  # Muy pequeña, lejos (25-30 metros)
            return 30.0
        elif plate_width < 20:  # Pequeña, lejos (20-25 metros)
            return 25.0
        elif plate_width < 30:  # Pequeña, media distancia (15-20 metros)
            return 18.0
        
        # Estimar distancia usando relación inversa mejorada
        # d = d_ref * (w_ref / w_actual)
        # Aplicar factor de corrección para distancias mayores
        base_distance = reference_distance_m * (reference_width_px / plate_width)
        
        # Factor de corrección para distancias mayores (más preciso)
        if base_distance > 15:
            # Para distancias mayores, usar modelo más conservador
            estimated_distance = base_distance * 0.9  # Reducir ligeramente
        else:
            estimated_distance = base_distance
        
        # Limitar a máximo 30 metros
        return min(estimated_distance, 30.0)
    
    def is_within_detection_range(self, bbox, frame_width, frame_height):
        """
        Verificar si la placa está dentro del rango de detección permitido
        Retorna (bool, distancia_estimada)
        """
        if not self.config["processing"].get("distance_filter_enabled", True):
            return True, 0.0
        
        # Verificar tamaño mínimo de placa (filtro rápido)
        x1, y1, x2, y2 = bbox
        plate_width = x2 - x1
        plate_height = y2 - y1
        
        min_width = self.config["processing"].get("min_plate_width_px", 80)
        min_height = self.config["processing"].get("min_plate_height_px", 30)
        
        # Si la placa es muy pequeña, está fuera de rango
        if plate_width < min_width or plate_height < min_height:
            estimated_dist = self.estimate_distance_from_bbox(bbox, frame_width, frame_height)
            return False, estimated_dist
        
        # Estimar distancia
        estimated_distance = self.estimate_distance_from_bbox(bbox, frame_width, frame_height)
        
        # Verificar si está dentro del rango máximo
        max_distance = self.config["processing"].get("max_detection_distance_m", 5.0)
        
        is_within_range = estimated_distance <= max_distance
        
        return is_within_range, estimated_distance
    
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
    
    def find_or_create_detection_group(self, text, bbox, timestamp_float):
        """
        Encontrar o crear un grupo de detecciones similares.
        Agrupa detecciones que probablemente son del mismo vehículo.
        """
        with self.group_lock:
            # Normalizar texto para comparación
            normalized_text = re.sub(r'[^A-Z0-9]', '', text.upper())
            
            # Buscar grupo existente similar
            for group_id, group_data in list(self.detection_groups.items()):
                group_texts = group_data.get('texts', [])
                group_bboxes = group_data.get('bboxes', [])
                group_start_time = group_data.get('start_time', 0)
                
                # Verificar si el tiempo del grupo no ha expirado
                if (timestamp_float - group_start_time) > self.group_timeout_sec:
                    continue  # Grupo expirado, no considerar
                
                # Verificar similitud con textos del grupo
                for group_text in group_texts:
                    similarity = self.calculate_text_similarity(text, group_text)
                    if similarity >= 0.6:  # 60% de similitud mínimo
                        # Verificar si el bbox está cerca (mismo vehículo)
                        for group_bbox in group_bboxes:
                            if group_bbox:
                                gx1, gy1, gx2, gy2 = group_bbox
                                x1, y1, x2, y2 = bbox
                                
                                # Calcular distancia entre centros
                                center_x = (x1 + x2) / 2
                                center_y = (y1 + y2) / 2
                                g_center_x = (gx1 + gx2) / 2
                                g_center_y = (gy1 + gy2) / 2
                                
                                distance = ((center_x - g_center_x) ** 2 + (center_y - g_center_y) ** 2) ** 0.5
                                
                                # Si está cerca (mismo vehículo), agregar a este grupo
                                if distance < 150:  # 150 píxeles de tolerancia
                                    return group_id
            
            # No se encontró grupo similar, crear uno nuevo
            new_group_id = f"group_{timestamp_float}_{normalized_text[:3]}"
            self.detection_groups[new_group_id] = {
                'start_time': timestamp_float,
                'texts': [text],
                'bboxes': [bbox],
                'detections': [],
                'last_update': timestamp_float
            }
            return new_group_id
    
    def add_detection_to_group(self, group_id, detection):
        """Agregar detección a un grupo existente"""
        with self.group_lock:
            if group_id in self.detection_groups:
                group = self.detection_groups[group_id]
                group['detections'].append(detection)
                group['texts'].append(detection['plate_text'])
                group['bboxes'].append(detection['bbox'])
                group['last_update'] = time.time()
    
    def select_best_detection_from_group(self, group_id):
        """
        Seleccionar la mejor detección de un grupo basándose en:
        1. Confianza OCR más alta
        2. Formato más válido (longitud correcta, ej: NON491 > NON49)
        3. Distancia más cercana (menor distancia = mejor)
        4. Confianza YOLO más alta
        """
        with self.group_lock:
            if group_id not in self.detection_groups:
                return None
            
            group = self.detection_groups[group_id]
            detections = group.get('detections', [])
            
            if not detections:
                return None
            
            # Función de scoring para cada detección
            def calculate_score(det):
                ocr_conf = det.get('ocr_confidence', 0)
                yolo_conf = det.get('yolo_confidence', 0)
                text = det.get('plate_text', '')
                distance = det.get('estimated_distance_m', 999)
                
                # Score base: confianza OCR (peso 40%)
                score = ocr_conf * 0.4
                
                # Bonus por confianza YOLO (peso 20%)
                score += yolo_conf * 0.2
                
                # Bonus por formato válido (peso 20%)
                # Preferir placas de 6 caracteres (formato estándar colombiano)
                if len(text) == 6:
                    score += 0.2
                elif len(text) == 5:
                    score += 0.1
                elif len(text) == 7:
                    score += 0.15
                else:
                    score += 0.05
                
                # Bonus por distancia cercana (peso 20%)
                # Menor distancia = mejor (invertir)
                if distance > 0:
                    distance_score = max(0, 1.0 - (distance / 10.0))  # Normalizar a 0-1
                    score += distance_score * 0.2
                
                # Penalizar textos muy cortos o muy largos
                if len(text) < 4 or len(text) > 8:
                    score *= 0.5
                
                return score
            
            # Calcular score para cada detección
            scored_detections = [(det, calculate_score(det)) for det in detections]
            
            # Ordenar por score descendente
            scored_detections.sort(key=lambda x: x[1], reverse=True)
            
            # Retornar la mejor detección
            best_detection, best_score = scored_detections[0]
            
            # Asegurar que tiene los campos de estado
            if 'status' not in best_detection:
                best_detection['status'] = 'confirmed'
            if 'saved_to_db' not in best_detection:
                best_detection['saved_to_db'] = False
            
            # No loguear aquí - solo mostrar placa final en finalize_detection
            
            return best_detection
    
    def cleanup_expired_groups(self):
        """Limpiar grupos de detecciones expirados y procesar los que están listos - OPTIMIZADO"""
        current_time = time.time()
        
        # Obtener grupos a procesar sin bloquear mucho tiempo
        groups_to_process = []
        groups_to_remove = []
        
        with self.group_lock:
            for group_id, group_data in list(self.detection_groups.items()):
                # Evitar procesar grupos que ya están siendo procesados
                if group_id in self.processing_groups:
                    continue
                    
                start_time = group_data.get('start_time', 0)
                last_update = group_data.get('last_update', 0)
                
                # Procesar grupos más rápido: 0.3 segundos sin actualizaciones
                if (current_time - last_update) >= self.group_timeout_sec:
                    groups_to_process.append(group_id)
                # Si el grupo es muy antiguo (más de 2 segundos), eliminarlo sin procesar
                elif (current_time - start_time) > 2.0:
                    groups_to_remove.append(group_id)
        
        # Procesar grupos en thread separado para no bloquear
        for group_id in groups_to_process:
            if group_id in self.processing_groups:
                continue
                
            self.processing_groups.add(group_id)
            
            # Procesar en thread separado para no bloquear el flujo principal
            def process_group_async(gid):
                try:
                    self.logger.info(f"[GROUP] Procesando grupo {gid} - seleccionando mejor detección...")
                    best_detection = self.select_best_detection_from_group(gid)
                    if best_detection:
                        plate_text = best_detection.get('plate_text', 'UNKNOWN')
                        self.logger.info(f"[GROUP] ✅ Mejor detección seleccionada: {plate_text} - Finalizando...")
                        try:
                            self.finalize_best_detection(best_detection)
                            self.logger.info(f"[GROUP] ✅ Detección {plate_text} finalizada correctamente")
                        except Exception as e:
                            self.logger.error(f"[GROUP] ❌ Error finalizando detección {plate_text}: {e}")
                            import traceback
                            self.logger.error(traceback.format_exc())
                    else:
                        self.logger.warning(f"[GROUP] ⚠️ No se pudo seleccionar mejor detección del grupo {gid}")
                    
                    # Eliminar grupo después de procesar
                    with self.group_lock:
                        if gid in self.detection_groups:
                            del self.detection_groups[gid]
                    self.processing_groups.discard(gid)
                except Exception as e:
                    self.logger.error(f"[GROUP] Error procesando grupo {gid}: {e}")
                    self.processing_groups.discard(gid)
            
            # Ejecutar en thread daemon para no bloquear
            thread = threading.Thread(target=process_group_async, args=(group_id,), daemon=True)
            thread.start()
        
        # Eliminar grupos expirados inmediatamente
        with self.group_lock:
            for group_id in groups_to_remove:
                if group_id in self.detection_groups and group_id not in self.processing_groups:
                    del self.detection_groups[group_id]
    
    def finalize_best_detection(self, detection):
        """Finalizar y guardar la mejor detección de un grupo - OPTIMIZADO para no bloquear"""
        try:
            # Usar frame guardado en la detección o frame actual como fallback
            frame = detection.get('frame')
            if frame is None:
                frame = self.last_frame.copy() if self.last_frame is not None else None
            
            if frame is None:
                self.logger.warning("[WARN] No hay frame disponible para guardar detección")
                return
            
            # Crear timestamp
            timestamp = datetime.now()
            
            # No loguear aquí - solo en finalize_detection para mostrar placa final
            # Procesar directamente sin bloqueos - simplificado
            self.finalize_detection(detection, frame, timestamp)
                
        except Exception as e:
            self.logger.error(f"[ERROR] Error finalizando mejor detección: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def process_frame_ai_realtime(self, frame, frame_time):
        """Procesamiento IA optimizado para tiempo real con cooldown inteligente"""
        try:
            self.ai_processed_frames += 1
            
            # YOLO más agresivo con umbral bajo para placas colombianas
            results = self.yolo_model(frame, verbose=False, 
                                     conf=self.config["processing"]["confidence_threshold"],
                                     iou=0.45)  # NMS menos agresivo para capturar más detecciones
            
            detections = []
            current_time = datetime.now()
            current_time_float = time.time()
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        # Umbral de confianza aumentado para evitar falsos positivos
                        min_yolo_conf = self.config["processing"]["confidence_threshold"]
                        if confidence < min_yolo_conf:
                            continue
                        
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)
                        
                        if x2 > x1 and y2 > y1:
                            # Filtrar por distancia/rango de detección
                            frame_height, frame_width = frame.shape[:2]
                            is_within_range, estimated_distance = self.is_within_detection_range(
                                [x1, y1, x2, y2], frame_width, frame_height
                            )
                            
                            if not is_within_range:
                                self.logger.debug(f"[FILTER] Placa rechazada por distancia: {estimated_distance:.1f}m "
                                                 f"(máximo: {self.config['processing'].get('max_detection_distance_m', 5.0)}m)")
                                continue  # Fuera del rango permitido
                            
                            # Cooldown por ubicación (bbox)
                            bbox_hash = self.calculate_bbox_hash(x1, y1, x2, y2)
                            bbox_cooldown_sec = self.config["processing"]["bbox_cooldown_sec"]
                            
                            if bbox_hash in self.bbox_cooldown:
                                last_bbox_time = self.bbox_cooldown[bbox_hash]
                                if (current_time_float - last_bbox_time) < bbox_cooldown_sec:
                                    continue  # Esta ubicación fue detectada recientemente
                            
                            roi = frame[y1:y2, x1:x2]
                            
                            # FILTRO DE COLOR: Verificar si es placa colombiana (amarilla/blanca)
                            # Deshabilitado temporalmente para detección más agresiva
                            # if not self.detect_colombian_plate_colors(roi):
                            #     self.logger.debug(f"[FILTER] Placa rechazada por color: no coincide con placas colombianas")
                            #     continue  # No es una placa colombiana típica
                            
                            # OCR con cache
                            plate_texts = self.get_plate_text_cached_realtime(roi)
                            
                            for plate_text in plate_texts:
                                text = plate_text['text']
                                ocr_conf = plate_text['confidence']
                                
                                # Validar formato de placa
                                if not is_valid_license_plate(text):
                                    self.logger.warning(f"[FILTER] Placa rechazada por formato inválido: {text} (OCR: {ocr_conf:.2f}, YOLO: {confidence:.2f})")
                                    # Agregar a display como error (rojo) brevemente
                                    error_detection = {
                                        'plate_text': text,
                                        'bbox': [x1, y1, x2, y2],
                                        'status': 'error',
                                        'saved_to_db': False,
                                        'display_time': current_time_float,
                                        'ocr_confidence': ocr_conf,
                                        'yolo_confidence': confidence
                                    }
                                    self.display_detections.append(error_detection)
                                    if len(self.display_detections) > 5:
                                        self.display_detections = self.display_detections[-5:]
                                    continue
                                
                                # Validación adicional: rechazar placas con confianza OCR muy baja
                                ocr_min_threshold = self.config["processing"]["plate_confidence_min"]
                                if ocr_conf < ocr_min_threshold:
                                    self.logger.debug(f"[FILTER] Placa rechazada por baja confianza OCR: {text} (OCR: {ocr_conf:.2f}, YOLO: {confidence:.2f}, mínimo: {ocr_min_threshold:.2f})")
                                    continue
                                
                                # Validación adicional: rechazar placas muy cortas (probablemente errores OCR)
                                if len(text) < 5:
                                    self.logger.debug(f"[FILTER] Placa rechazada por longitud insuficiente: {text} (longitud: {len(text)}, mínimo: 5)")
                                    continue
                                
                                # Validación adicional: debe tener al menos 2 letras y 2 números
                                letters = len(re.findall(r'[A-Z]', text))
                                numbers = len(re.findall(r'[0-9]', text))
                                if letters < 2 or numbers < 2:
                                    self.logger.debug(f"[FILTER] Placa rechazada por formato inválido: {text} (letras: {letters}, números: {numbers})")
                                    continue
                                
                                # Verificar si es similar a una detección reciente (evitar variaciones OCR)
                                is_similar, better_text = self.is_similar_to_recent_detection(text, [x1, y1, x2, y2])
                                
                                if is_similar and better_text:
                                    # Si hay una detección similar reciente con mayor confianza, usar esa
                                    # Pero solo si la diferencia de confianza es significativa
                                    recent_det = next((d for d in self.recent_detections if d.get('plate_text') == better_text), None)
                                    if recent_det and recent_det.get('ocr_confidence', 0) > ocr_conf + 0.15:
                                        self.logger.debug(f"[FILTER] Usando detección previa más confiable: {better_text}")
                                        continue  # Ignorar esta detección, usar la anterior
                                
                                # Cooldown por texto de placa - REDUCIDO para detección más frecuente
                                cooldown_sec = self.config["processing"]["detection_cooldown_sec"]
                                if text in self.detection_cooldown:
                                    last_time = self.detection_cooldown[text]
                                    # Solo aplicar cooldown si la placa es exactamente igual (evitar duplicados exactos)
                                    if (current_time_float - last_time) < cooldown_sec:
                                        # Permitir si es una variación diferente (ej: LCP909 vs ILCP9091)
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
                                    'estimated_distance_m': round(estimated_distance, 2),
                                    'processing_latency_ms': int(processing_latency * 1000),
                                    'valid': True,
                                    'frame': frame.copy(),  # Guardar frame para análisis mejorado
                                    'status': 'possible',  # Estado: possible, confirmed, error, saved
                                    'saved_to_db': False  # Indica si se guardó en BD
                                }
                                
                                detections.append(detection)
                                
                                # SISTEMA DE AGRUPACIÓN INTELIGENTE
                                # Agrupar detecciones similares para seleccionar la mejor
                                group_id = self.find_or_create_detection_group(
                                    text, [x1, y1, x2, y2], current_time_float
                                )
                                self.add_detection_to_group(group_id, detection)
                                
                                # Agregar a detecciones para mostrar INMEDIATAMENTE (con timestamp)
                                display_detection = detection.copy()
                                display_detection['display_time'] = current_time_float
                                display_detection['status'] = 'possible'  # Estado inicial
                                
                                # Agregar a ambas listas para asegurar visualización
                                self.display_detections.append(display_detection)
                                self.active_detections.append(display_detection)
                                
                                # Mantener hasta 20 detecciones para mostrar múltiples en el stream
                                if len(self.display_detections) > 20:
                                    self.display_detections = self.display_detections[-20:]
                                if len(self.active_detections) > 20:
                                    self.active_detections = self.active_detections[-20:]
                                
                                # Solo loguear en modo debug - no mostrar en terminal todas las detecciones
                                # self.logger.debug(f"[GROUP] Detección agregada al grupo {group_id}: {text} "
                                #                f"(OCR: {ocr_conf:.2f}, Distancia: {estimated_distance:.1f}m)")
                                
                                # Si la confianza es muy alta, procesar inmediatamente sin esperar agrupación
                                # Solo si cumple validación estricta y tiene alta confianza
                                if ocr_conf >= 0.65 and is_valid_license_plate(text) and len(text) >= 5:
                                    # Procesar inmediatamente si la confianza es alta y formato válido
                                    try:
                                        # Actualizar estado en display antes de finalizar
                                        for det in self.display_detections:
                                            if det.get('plate_text') == text and det.get('display_time') == current_time_float:
                                                det['status'] = 'confirmed'
                                        for det in self.active_detections:
                                            if det.get('plate_text') == text and det.get('display_time') == current_time_float:
                                                det['status'] = 'confirmed'
                                        self.finalize_detection(detection, frame, current_time)
                                    except Exception as e:
                                        self.logger.error(f"[ERROR] Error procesando inmediatamente: {e}")
            
            # Limpiar grupos expirados y procesar los listos (más frecuente)
            # Limpiar cada 3 frames para mejor balance
            if self.ai_processed_frames % 3 == 0:
                self.cleanup_expired_groups()
            
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
        """
        OCR optimizado AGRESIVO para placas colombianas (amarillas/blancas)
        Aplica múltiples técnicas de preprocesamiento y OCR para máxima precisión
        """
        try:
            if roi.size == 0:
                return []
            
            # 🔥 OPTIMIZACIÓN 1: Verificar tamaño mínimo antes de procesar OCR
            # Las placas colombianas necesitan tamaño mínimo para reconocimiento preciso
            # Valores reducidos para permitir mayor alcance
            min_width = self.config["processing"].get("min_roi_width_for_ocr", 60)
            min_height = self.config["processing"].get("min_roi_height_for_ocr", 20)
            
            # Si el ROI es muy pequeño, intentar ampliarlo antes de rechazar
            if roi.shape[1] < min_width or roi.shape[0] < min_height:
                # Intentar ampliar si es muy pequeño pero no extremadamente pequeño
                if roi.shape[1] >= 30 and roi.shape[0] >= 12:
                    # Ampliar para alcanzar mínimo
                    scale_w = min_width / roi.shape[1] if roi.shape[1] < min_width else 1.0
                    scale_h = min_height / roi.shape[0] if roi.shape[0] < min_height else 1.0
                    scale = max(scale_w, scale_h)
                    new_w = int(roi.shape[1] * scale)
                    new_h = int(roi.shape[0] * scale)
                    roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                else:
                    # Muy pequeño, rechazar
                    self.logger.debug(f"[OCR] ROI muy pequeño para OCR: {roi.shape[1]}x{roi.shape[0]} (mínimo: {min_width}x{min_height})")
                    return []
            
            # Aumentar tamaño para mejor OCR
            target_height = 100   # Altura aumentada para mejor OCR
            target_width = 300   # Ancho aumentado para mejor OCR
        
            if roi.shape[0] > target_height or roi.shape[1] > target_width:
                scale_h = target_height / roi.shape[0] if roi.shape[0] > target_height else 1.0
                scale_w = target_width / roi.shape[1] if roi.shape[1] > target_width else 1.0
                scale = min(scale_h, scale_w)
            
                new_h = max(min_height, int(roi.shape[0] * scale))
                new_w = max(min_width, int(roi.shape[1] * scale))
                roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            else:
                # Si es pequeño pero cumple mínimo, ampliar para mejor OCR
                if roi.shape[0] < target_height or roi.shape[1] < target_width:
                    scale_factor = max(target_height / roi.shape[0], target_width / roi.shape[1])
                    new_h = int(roi.shape[0] * scale_factor)
                    new_w = int(roi.shape[1] * scale_factor)
                    roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # PREPROCESAMIENTO AGRESIVO para placas colombianas
            if self.config["processing"].get("colombian_plate_optimization", True):
                processed_roi = self.preprocess_colombian_plate(roi)
            else:
                # Preprocesamiento básico
                if len(roi.shape) == 3:
                    processed_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                else:
                    processed_roi = roi.copy()
                processed_roi = cv2.bilateralFilter(processed_roi, 5, 50, 50)
            
            # MÚLTIPLES INTENTOS DE OCR con diferentes preprocesamientos
            all_texts = []
            
            # INTENTO 1: Con preprocesamiento agresivo
            try:
                ocr_results = self.ocr_reader.readtext(processed_roi)
                for (bbox, text, confidence) in ocr_results:
                    if confidence >= self.config["processing"]["plate_confidence_min"] and len(text.strip()) > 2:
                        cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                        if len(cleaned_text) >= 3:
                            all_texts.append({
                                'text': cleaned_text,
                                'confidence': confidence
                            })
            except:
                pass
            
            # INTENTO 2: Con ROI original (por si el preprocesamiento no funciona bien)
            if len(roi.shape) == 3:
                gray_original = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray_original = roi.copy()
            
            try:
                ocr_results2 = self.ocr_reader.readtext(gray_original)
                for (bbox, text, confidence) in ocr_results2:
                    if confidence >= self.config["processing"]["plate_confidence_min"] and len(text.strip()) > 2:
                        cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                        if len(cleaned_text) >= 3:
                            all_texts.append({
                                'text': cleaned_text,
                                'confidence': confidence
                            })
            except:
                pass
            
            # INTENTO 3: Con threshold adaptativo adicional
            try:
                adaptive_thresh = cv2.adaptiveThreshold(
                    processed_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
                ocr_results3 = self.ocr_reader.readtext(adaptive_thresh)
                for (bbox, text, confidence) in ocr_results3:
                    if confidence >= self.config["processing"]["plate_confidence_min"] and len(text.strip()) > 2:
                        cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                        if len(cleaned_text) >= 3:
                            all_texts.append({
                                'text': cleaned_text,
                                'confidence': confidence
                            })
            except:
                pass
            
            # Eliminar duplicados y mantener el de mayor confianza
            unique_texts = {}
            for text_data in all_texts:
                text = text_data['text']
                conf = text_data['confidence']
                if text not in unique_texts or conf > unique_texts[text]['confidence']:
                    unique_texts[text] = text_data
            
            # Ordenar por confianza descendente
            texts = sorted(unique_texts.values(), key=lambda x: x['confidence'], reverse=True)
            
            return texts
            
        except Exception as e:
            self.logger.debug(f"[OCR] Error en lectura de placa: {e}")
            return []
    
    # Funcionalidad de detección mejorada eliminada - mostrar detecciones inmediatamente
    
    def finalize_detection(self, detection, frame, timestamp):
        """Finalizar y guardar detección"""
        try:
            # Actualizar estado a confirmado
            detection['status'] = 'confirmed'
            detection['saved_to_db'] = False
            
            # Agregar a detecciones recientes
            self.recent_detections.append(detection)
            if len(self.recent_detections) > 10:
                self.recent_detections = self.recent_detections[-10:]
            
            # Agregar a display con estado inicial - INMEDIATAMENTE
            display_detection = detection.copy()
            display_detection['display_time'] = time.time()
            # Asegurar que tiene los campos de estado
            if 'status' not in display_detection:
                display_detection['status'] = 'confirmed'
            if 'saved_to_db' not in display_detection:
                display_detection['saved_to_db'] = False
            
            # Agregar a ambas listas para asegurar visualización
            self.display_detections.append(display_detection)
            self.active_detections.append(display_detection)
            
            # Mantener más detecciones para mejor visualización
            if len(self.display_detections) > 10:
                self.display_detections = self.display_detections[-10:]
            if len(self.active_detections) > 10:
                self.active_detections = self.active_detections[-10:]
            
            # Log final - SOLO mostrar en terminal la placa final confirmada
            self.detections_count += 1
            plate_text = detection['plate_text']
            # Solo loguear placas finales confirmadas (no todas las detecciones intermedias)
            self.logger.info(f"[TARGET] ✅ PLACA FINAL: {plate_text} "
                           f"(YOLO: {detection['yolo_confidence']:.2f}, "
                           f"OCR: {detection['ocr_confidence']:.2f}, "
                           f"Distancia: {detection['estimated_distance_m']:.1f}m)")
            
            # Guardar en base de datos y validar
            if self.db_manager:
                try:
                    db_data = {
                        'timestamp': timestamp,
                        'plate_text': detection['plate_text'],
                        'confidence': detection['yolo_confidence'],
                        'plate_score': detection['ocr_confidence'],
                        'vehicle_bbox': None,
                        'plate_bbox': json.dumps(detection['bbox']),
                        'camera_location': 'entrada_principal',
                        'estimated_distance_m': detection.get('estimated_distance_m')
                    }
                    self.logger.info(f"[DB] Intentando guardar placa {plate_text} en BD...")
                    saved = self.db_manager.insert_detection(db_data)
                    
                    if saved:
                        detection['saved_to_db'] = True
                        detection['status'] = 'saved'
                        # Actualizar también en display_detections y active_detections - buscar por placa y tiempo reciente
                        current_time_now = time.time()
                        updated_count = 0
                        for disp_det in self.display_detections + self.active_detections:
                            if disp_det.get('plate_text') == plate_text:
                                # Actualizar si es la misma placa y está dentro de 30 segundos
                                time_diff = abs(disp_det.get('display_time', 0) - current_time_now)
                                if time_diff < 30.0:
                                    disp_det['saved_to_db'] = True
                                    disp_det['status'] = 'saved'
                                    updated_count += 1
                                    self.logger.debug(f"[DB] Display actualizado: {plate_text} -> status=saved, saved_to_db=True")
                        if updated_count > 0:
                            self.logger.info(f"[DB] ✅ Estado actualizado en {updated_count} detección(es) del display: {plate_text} -> GUARDADA (VERDE)")
                        else:
                            # Si no se encontró en display, agregar una nueva entrada
                            new_display_det = detection.copy()
                            new_display_det['display_time'] = current_time_now
                            new_display_det['status'] = 'saved'
                            new_display_det['saved_to_db'] = True
                            self.display_detections.append(new_display_det)
                            self.active_detections.append(new_display_det)
                            if len(self.display_detections) > 10:
                                self.display_detections = self.display_detections[-10:]
                            if len(self.active_detections) > 10:
                                self.active_detections = self.active_detections[-10:]
                            self.logger.info(f"[DB] ✅ Nueva entrada agregada al display: {plate_text} -> GUARDADA (VERDE)")
                        self.logger.info(f"[DB] ✅✅✅ VALIDACIÓN EXITOSA: Placa {plate_text} guardada correctamente en BD - CUADRO VERDE ACTIVADO")
                    else:
                        # Error al guardar - mantener como confirmada pero no guardada
                        detection['status'] = 'confirmed'
                        detection['saved_to_db'] = False
                        # Actualizar también en display_detections
                        current_time_now = time.time()
                        for disp_det in self.display_detections:
                            if disp_det.get('plate_text') == plate_text:
                                time_diff = abs(disp_det.get('display_time', 0) - current_time_now)
                                if time_diff < 15.0:
                                    disp_det['saved_to_db'] = False
                                    disp_det['status'] = 'confirmed'
                        self.logger.error(f"[DB] ❌❌❌ ERROR DE VALIDACIÓN: No se pudo guardar placa {plate_text} en BD - insert_detection retornó False")
                        self.logger.error(f"[DB] Detalles: YOLO={detection['yolo_confidence']:.2f}, OCR={detection['ocr_confidence']:.2f}, Distancia={detection.get('estimated_distance_m', 0):.1f}m")
                        self.logger.error(f"[DB] La placa {plate_text} NO se guardó en la base de datos - CUADRO VERDE NO ACTIVADO")
                except Exception as e:
                    # Error de excepción - mantener como confirmada pero loguear el error
                    detection['status'] = 'confirmed'
                    detection['saved_to_db'] = False
                    self.logger.error(f"[DB] ❌ EXCEPCIÓN al guardar placa {detection['plate_text']} en BD: {e}")
                    self.logger.error(f"[DB] Tipo de error: {type(e).__name__}")
                    import traceback
                    self.logger.error(f"[DB] Traceback: {traceback.format_exc()}")
            else:
                # Sin BD disponible - mantener como confirmada (no es un error, solo no hay BD)
                detection['status'] = 'confirmed'
                detection['saved_to_db'] = False
                self.logger.info(f"[DB] ℹ️ BD no disponible - placa {detection['plate_text']} confirmada pero no guardada")
            
            # Guardar en archivo
            if self.config["output"]["save_results"]:
                self.save_detection(detection)
                
        except Exception as e:
            self.logger.error(f"[ERROR] Error finalizando detección: {e}")
    
    def save_detection(self, detection):
        """Guardar detección - Corregido para evitar error de serialización JSON"""
        try:
            results_file = self.results_dir / f"realtime_detections_{datetime.now().strftime('%Y%m%d')}.jsonl"
            
            # Crear copia de la detección sin objetos no serializables
            detection_copy = detection.copy()
            
            # Eliminar frame (ndarray no es serializable en JSON)
            if 'frame' in detection_copy:
                del detection_copy['frame']
            
            # Convertir bbox a lista si es ndarray
            if 'bbox' in detection_copy and isinstance(detection_copy['bbox'], np.ndarray):
                detection_copy['bbox'] = detection_copy['bbox'].tolist()
            
            # Convertir cualquier otro ndarray a lista
            for key, value in detection_copy.items():
                if isinstance(value, np.ndarray):
                    detection_copy[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    detection_copy[key] = float(value)
            
            with open(results_file, 'a', encoding='utf-8') as f:
                json.dump(detection_copy, f, ensure_ascii=False)
                f.write('\n')
                
        except Exception as e:
            self.logger.error(f"[ERROR] Error guardando: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
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
        self.display_detections.clear()
        self.active_detections.clear()
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
        
        # Detener control PTZ
        if self.ptz_controller:
            try:
                self.ptz_controller.stop()
            except Exception as e:
                self.logger.warning(f"[WARN] Error deteniendo PTZ: {e}")
        
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
    
    print("SISTEMA LPR TIEMPO REAL")
    print("=" * 60)
    print("[TARGET] Enfoque: DETECCIÓN AGRESIVA DE PLACAS COLOMBIANAS")
    print("[COLOMBIA] Optimizado para placas AMARILLAS y BLANCAS")
    print("[VIDEO] IA cada FRAME (máxima frecuencia posible)")
    print("[TIME] Cooldown 1.5 segundos (detección más frecuente)")
    print("[AGGRESSIVE] Umbrales reducidos: YOLO 0.20, OCR 0.25")
    print("[COLOR] Detección de color: Filtra placas amarillas/blancas")
    print("[PREPROCESS] Preprocesamiento agresivo: CLAHE + Sharpening + OTSU")
    print("[OCR] Múltiples intentos OCR con diferentes técnicas")
    print("[FAST] Procesamiento optimizado para velocidad máxima")
    print("[ROBOT] Optimizado para Jetson Orin Nano")
    print("[DATABASE] Conexión MySQL habilitada")
    print("[IMPROVED] Cooldown inteligente por ubicación y texto")
    print("[PTZ] Control automático: Scroll y zoom hacia placas detectadas")
    print("[DISTANCE] Filtro de distancia DESHABILITADO (detección agresiva)")
    print("[ENHANCED] Detección mejorada: Congelar 1.5s -> Zoom -> Análisis mejorado")
    print("=" * 60)
    
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