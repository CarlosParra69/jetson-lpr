#!/usr/bin/env python3
"""
⚡ SISTEMA LPR ULTRA-OPTIMIZADO TIEMPO REAL
==========================================
Versión modificada para detección casi instantánea de placas

Optimizaciones para tiempo real:
- IA cada 2-3 frames máximo
- Cooldown reducido a 0.5 segundos
- Detección de movimiento para activar IA
- Cache agresivo
- Prioridad a detección sobre FPS

Autor: Sistema LPR automatizado - Versión Tiempo Real
Fecha: 2025-06-29
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
    """Sistema LPR optimizado para tiempo real"""
    
    def __init__(self, config_path="config/ptz_config.json"):
        self.config_path = Path(config_path)
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
        self.detection_cooldown = {}
        self.recent_detections = []
        
        # Detección de movimiento para activar IA
        self.motion_detector = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.last_motion_time = 0
        self.motion_threshold = 1000  # Píxeles en movimiento
        
        # Display optimizado
        self.window_created = False
        self.last_frame = None
        
        # Configurar modelos
        self.setup_models()
        
        # Configurar red PTZ
        self.setup_network()
        
        # Directorios
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.logger.info("⚡ Sistema LPR TIEMPO REAL inicializado")
    
    def setup_logging(self):
        """Configurar logging"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"realtime_lpr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - REALTIME-LPR - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        print(f"📝 Logs guardados en: {log_file}")
    
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
                "display_scale": 0.25,         # Display más pequeño para más FPS
                "minimal_rendering": True,     
                "fast_resize": True,           
                "aggressive_cache": True       # Cache más agresivo
            },
            "processing": {
                "confidence_threshold": 0.30,  # Umbral más bajo para no perder detecciones
                "plate_confidence_min": 0.25,  # OCR más permisivo
                "max_detections": 3,
                "ocr_cache_enabled": True,
                "detection_cooldown_sec": 0.5,  # COOLDOWN MUY CORTO (0.5s)
                "motion_cooldown_sec": 2        # Cooldown para detección de movimiento
            },
            "output": {
                "save_results": True,
                "save_images": False,           
                "show_video": True,
                "show_minimal_overlay": True,   
                "window_title": "Tiempo Real LPR ⚡⚡"
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
            
            self.logger.info(f"✅ Red tiempo real: {interface} -> {jetson_ip}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Configuración de red: {e}")
    
    def setup_models(self):
        """Inicializar modelos de IA"""
        self.logger.info("🤖 Cargando modelos de IA...")
        
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
            
            self.logger.info(f"📦 Cargando modelo: {selected_model}")
            self.yolo_model = YOLO(selected_model)
            
            # Warm-up más agresivo
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            for _ in range(3):  # Múltiples warm-ups
                self.yolo_model(dummy_frame, verbose=False)
            
            # EasyOCR
            self.logger.info("📝 Inicializando EasyOCR...")
            self.ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False, detector='dbnet18')
            
            # Warm-up OCR
            dummy_roi = np.zeros((50, 150, 3), dtype=np.uint8)
            for _ in range(2):
                self.ocr_reader.readtext(dummy_roi)
            
            self.logger.info("✅ Modelos listos para TIEMPO REAL")
            
        except Exception as e:
            self.logger.error(f"❌ Error cargando modelos: {e}")
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
        self.logger.info("📹 Iniciando captura TIEMPO REAL...")
        
        rtsp_url = self.config["camera"]["rtsp_url"]
        
        try:
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, self.config["realtime_optimization"]["capture_target_fps"])
            
            self.logger.info("✅ Captura TIEMPO REAL conectada")
            
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
            self.logger.error(f"❌ Error en captura: {e}")
    
    def display_worker(self):
        """Thread de display optimizado"""
        self.logger.info("🖥️ Iniciando display TIEMPO REAL...")
        
        window_title = self.config["output"]["window_title"]
        cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
        self.window_created = True
        
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
            self.logger.error(f"❌ Error en display: {e}")
    
    def ai_worker(self):
        """Thread de IA para tiempo real"""
        self.logger.info("🧠 Iniciando IA TIEMPO REAL...")
        
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
            self.logger.error(f"❌ Error en IA: {e}")
    
    def realtime_overlay(self, frame):
        """Overlay mínimo para tiempo real"""
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
    
    def process_frame_ai_realtime(self, frame, frame_time):
        """Procesamiento IA optimizado para tiempo real"""
        try:
            self.ai_processed_frames += 1
            
            # YOLO más rápido
            results = self.yolo_model(frame, verbose=False, 
                                     conf=self.config["processing"]["confidence_threshold"],
                                     iou=0.5)  # NMS más agresivo
            
            detections = []
            current_time = datetime.now()
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        if confidence < self.config["processing"]["plate_confidence_min"]:
                            continue
                        
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)
                        
                        if x2 > x1 and y2 > y1:
                            roi = frame[y1:y2, x1:x2]
                            
                            # OCR con cache ULTRA-agresivo
                            plate_texts = self.get_plate_text_cached_realtime(roi)
                            
                            for plate_text in plate_texts:
                                text = plate_text['text']
                                
                                if is_valid_license_plate(text):
                                    # Cooldown MUY CORTO
                                    cooldown_sec = self.config["processing"]["detection_cooldown_sec"]
                                    if text in self.detection_cooldown:
                                        last_time = self.detection_cooldown[text]
                                        if (current_time - last_time).total_seconds() < cooldown_sec:
                                            continue
                                    
                                    self.detection_cooldown[text] = current_time
                                    
                                    # Calcular latencia real
                                    processing_latency = time.time() - frame_time
                                    
                                    detection = {
                                        'timestamp': current_time.isoformat(),
                                        'frame_number': self.capture_frame_count,
                                        'ai_frame_number': self.ai_processed_frames,
                                        'plate_text': text,
                                        'yolo_confidence': confidence,
                                        'ocr_confidence': plate_text['confidence'],
                                        'bbox': [x1, y1, x2, y2],
                                        'processing_latency_ms': int(processing_latency * 1000),
                                        'valid': True
                                    }
                                    
                                    detections.append(detection)
                                    self.detections_count += 1
                                    
                                    self.logger.info(f"🎯 PLACA: {text} "
                                                   f"(YOLO: {confidence:.2f}, OCR: {plate_text['confidence']:.2f}, "
                                                   f"Latencia: {int(processing_latency * 1000)}ms)")
                                    
                                    if self.config["output"]["save_results"]:
                                        self.save_detection(detection)
            
            return detections
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error en IA: {e}")
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
                if confidence > 0.2 and len(text.strip()) > 2:  # Umbral más bajo
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
            self.logger.error(f"❌ Error guardando: {e}")
    
    def reset_stats(self):
        """Reset estadísticas"""
        self.capture_frame_count = 0
        self.display_frame_count = 0
        self.ai_processed_frames = 0
        self.detections_count = 0
        self.recent_detections = []
        self.start_time = time.time()
        self.logger.info("🔄 Reset TIEMPO REAL")
    
    def clear_cache(self):
        """Limpiar cache"""
        self.ocr_cache.clear()
        self.detection_cooldown.clear()
        self.logger.info("🧹 Cache limpiado")
    
    def save_screenshot(self, frame):
        """Guardar captura"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        cv2.imwrite(f"realtime_capture_{timestamp}.jpg", frame)
        self.logger.info(f"📸 Captura: realtime_capture_{timestamp}.jpg")
    
    def run(self):
        """Ejecutar sistema tiempo real"""
        self.logger.info("⚡ Iniciando sistema TIEMPO REAL...")
        
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
            
            self.logger.info("✅ Sistema TIEMPO REAL iniciado")
            self.logger.info("🎮 Controles: 'q'=salir, 'r'=reset, 'c'=cache, 's'=captura")
            self.logger.info("⚡ IA CADA 2 FRAMES + Cooldown 0.5s = DETECCIÓN CASI INSTANTÁNEA")
            
            # Esperar threads
            while self.running:
                time.sleep(0.1)
                
                # Verificar threads
                if not self.capture_thread.is_alive():
                    self.logger.error("❌ Thread de captura murió")
                    break
                if not self.display_thread.is_alive():
                    self.logger.error("❌ Thread de display murió")
                    break
                    
        except KeyboardInterrupt:
            self.logger.info("⏹️ Detenido por usuario")
        except Exception as e:
            self.logger.error(f"❌ Error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        finally:
            self.stop()
    
    def stop(self):
        """Detener sistema"""
        self.logger.info("🛑 Deteniendo sistema TIEMPO REAL...")
        
        self.running = False
        
        # Esperar threads
        threads = [self.capture_thread, self.display_thread, self.ai_thread]
        for thread in threads:
            if thread and thread.is_alive():
                thread.join(timeout=2)
        
        # Estadísticas finales
        if self.start_time:
            runtime = time.time() - self.start_time
            capture_fps = self.capture_frame_count / runtime if runtime > 0 else 0
            display_fps = self.display_frame_count / runtime if runtime > 0 else 0
            ai_fps = self.ai_processed_frames / runtime if runtime > 0 else 0
            
            # Calcular latencia promedio
            avg_detections_per_minute = (self.detections_count / runtime) * 60 if runtime > 0 else 0
            
            self.logger.info("📊 ESTADÍSTICAS TIEMPO REAL:")
            self.logger.info(f"   ⏱️  Tiempo: {runtime:.1f}s")
            self.logger.info(f"   📹 FPS Captura: {capture_fps:.1f}")
            self.logger.info(f"   🖥️ FPS Display: {display_fps:.1f}")
            self.logger.info(f"   🧠 FPS IA: {ai_fps:.1f}")
            self.logger.info(f"   🎯 Detecciones: {self.detections_count}")
            self.logger.info(f"   ⚡ Detecciones/min: {avg_detections_per_minute:.1f}")
            self.logger.info(f"   📊 Eficiencia IA: {(ai_fps*100):.1f}% más frecuente")

def main():
    """Función principal tiempo real"""
    parser = argparse.ArgumentParser(description="Sistema LPR Tiempo Real")
    parser.add_argument("--config", default="config/ptz_config.json")
    parser.add_argument("--ai-every", type=int, default=2, help="Procesar IA cada N frames (por defecto: 2)")
    parser.add_argument("--cooldown", type=float, default=0.5, help="Cooldown en segundos (por defecto: 0.5)")
    parser.add_argument("--motion", action="store_true", help="Activar detección de movimiento")
    parser.add_argument("--confidence", type=float, default=0.30, help="Umbral confianza YOLO")
    parser.add_argument("--display-scale", type=float, default=0.25, help="Escala display")
    
    args = parser.parse_args()
    
    print("⚡⚡ SISTEMA LPR TIEMPO REAL ⚡⚡")
    print("=" * 50)
    print("🎯 Enfoque: DETECCIÓN CASI INSTANTÁNEA")
    print("📹 IA cada 2 frames (máxima frecuencia)")
    print("⏱️ Cooldown 0.5 segundos (ultra-corto)")
    print("🔍 Detección de movimiento opcional")
    print("💨 Procesamiento ultra-agresivo")
    print("=" * 50)
    
    try:
        system = RealtimeLPRSystem(args.config)
        
        # Aplicar configuraciones tiempo real
        system.config["realtime_optimization"]["ai_process_every"] = args.ai_every
        system.config["processing"]["detection_cooldown_sec"] = args.cooldown
        system.config["realtime_optimization"]["motion_activation"] = args.motion
        system.config["processing"]["confidence_threshold"] = args.confidence
        system.config["realtime_optimization"]["display_scale"] = args.display_scale
        
        print(f"⚙️ Configuración TIEMPO REAL:")
        print(f"   🧠 IA cada: {args.ai_every} frames")
        print(f"   ⏱️ Cooldown: {args.cooldown}s")
        print(f"   🔍 Detección movimiento: {'Sí' if args.motion else 'No'}")
        print(f"   🎯 Confianza: {args.confidence}")
        print(f"   📏 Escala: {args.display_scale}")
        print()
        print("🚨 ADVERTENCIA: Este modo consume más recursos")
        print("🚨 pero detecta placas CASI INSTANTÁNEAMENTE")
        print()
        
        system.run()
        
    except Exception as e:
        print(f"❌ Error fatal: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
