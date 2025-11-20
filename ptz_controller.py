#!/usr/bin/env python3
"""
🎥 Controlador PTZ para cámaras IP
Soporta control mediante HTTP API y ONVIF
"""

import requests
import time
import threading
from urllib.parse import urlencode
import logging

class PTZController:
    """Controlador PTZ para cámaras IP"""
    
    def __init__(self, camera_ip, username="admin", password="admin", protocol="http"):
        self.camera_ip = camera_ip
        self.username = username
        self.password = password
        self.protocol = protocol
        self.base_url = f"{protocol}://{camera_ip}"
        
        # Estado actual de la cámara
        self.current_pan = 0.0
        self.current_tilt = 0.0
        self.current_zoom = 0.0
        
        # Posición guardada antes de movimiento
        self.saved_pan = 0.0
        self.saved_tilt = 0.0
        self.saved_zoom = 0.0
        
        # Lock para operaciones thread-safe
        self.lock = threading.Lock()
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Verificar conexión
        self.test_connection()
    
    def test_connection(self):
        """Verificar conexión con la cámara"""
        try:
            # Intentar obtener snapshot o información básica
            test_url = f"{self.base_url}/cgi-bin/snapshot.cgi"
            response = requests.get(test_url, auth=(self.username, self.password), timeout=2)
            if response.status_code == 200:
                self.logger.info(f"[PTZ] Conexión exitosa con cámara {self.camera_ip}")
                return True
        except Exception as e:
            self.logger.warning(f"[PTZ] No se pudo verificar conexión: {e}")
            self.logger.info("[PTZ] Continuando con control PTZ (puede fallar si la cámara no soporta)")
        
        return False
    
    def send_ptz_command(self, command, params=None):
        """Enviar comando PTZ genérico - Soporta múltiples formatos de cámaras"""
        try:
            if params is None:
                params = {}
            
            # Intentar diferentes formatos de API según el tipo de cámara
            urls_to_try = [
                f"{self.base_url}/cgi-bin/ptz.cgi",  # Dahua/Hikvision estándar
                f"{self.base_url}/cgi-bin/ptzctrl.cgi",  # Variante
            ]
            
            # Agregar autenticación
            auth = (self.username, self.password)
            
            for url in urls_to_try:
                try:
                    params_with_auth = params.copy()
                    params_with_auth['action'] = command
                    params_with_auth['user'] = self.username
                    params_with_auth['pwd'] = self.password
                    
                    response = requests.get(url, params=params_with_auth, auth=auth, timeout=2)
                    
                    if response.status_code in [200, 204]:
                        return True
                        
                except requests.exceptions.RequestException:
                    continue
            
            self.logger.warning(f"[PTZ] No se pudo ejecutar comando {command} en ninguna URL")
            return False
                
        except Exception as e:
            self.logger.warning(f"[PTZ] Error enviando comando {command}: {e}")
            return False
    
    def move_to_position(self, pan, tilt, speed=0.5):
        """
        Mover cámara a posición específica (pan, tilt)
        pan: -1.0 a 1.0 (izquierda a derecha)
        tilt: -1.0 a 1.0 (abajo a arriba)
        speed: 0.0 a 1.0 (velocidad de movimiento)
        """
        with self.lock:
            try:
                # Convertir coordenadas normalizadas a valores de cámara
                # Muchas cámaras usan valores de -1 a 1 o 0 a 360
                pan_value = int(pan * 100)  # Convertir a porcentaje
                tilt_value = int(tilt * 100)
                
                # Comando para movimiento continuo
                params = {
                    'action': 'start',
                    'code': 'PositionABS',
                    'arg1': pan_value,
                    'arg2': tilt_value,
                    'arg3': int(speed * 100)
                }
                
                result = self.send_ptz_command('ptz', params)
                
                if result:
                    self.current_pan = pan
                    self.current_tilt = tilt
                    self.logger.info(f"[PTZ] Moviendo a posición: pan={pan:.2f}, tilt={tilt:.2f}")
                
                return result
                
            except Exception as e:
                self.logger.error(f"[PTZ] Error moviendo cámara: {e}")
                return False
    
    def move_relative(self, pan_delta, tilt_delta, speed=0.5):
        """
        Mover cámara relativamente desde posición actual
        pan_delta: cambio en pan (-1.0 a 1.0)
        tilt_delta: cambio en tilt (-1.0 a 1.0)
        """
        with self.lock:
            new_pan = max(-1.0, min(1.0, self.current_pan + pan_delta))
            new_tilt = max(-1.0, min(1.0, self.current_tilt + tilt_delta))
            return self.move_to_position(new_pan, new_tilt, speed)
    
    def set_zoom(self, zoom_level):
        """
        Establecer nivel de zoom
        zoom_level: 0.0 (sin zoom) a 1.0 (zoom máximo)
        """
        with self.lock:
            try:
                zoom_value = int(zoom_level * 100)
                
                params = {
                    'action': 'start',
                    'code': 'Zoom',
                    'arg1': zoom_value
                }
                
                result = self.send_ptz_command('ptz', params)
                
                if result:
                    self.current_zoom = zoom_level
                    self.logger.info(f"[PTZ] Zoom establecido a: {zoom_level:.2f}")
                
                return result
                
            except Exception as e:
                self.logger.error(f"[PTZ] Error estableciendo zoom: {e}")
                return False
    
    def zoom_in(self, steps=10):
        """Hacer zoom in"""
        with self.lock:
            new_zoom = min(1.0, self.current_zoom + (steps / 100.0))
            return self.set_zoom(new_zoom)
    
    def zoom_out(self, steps=10):
        """Hacer zoom out"""
        with self.lock:
            new_zoom = max(0.0, self.current_zoom - (steps / 100.0))
            return self.set_zoom(new_zoom)
    
    def save_position(self):
        """Guardar posición actual"""
        with self.lock:
            self.saved_pan = self.current_pan
            self.saved_tilt = self.current_tilt
            self.saved_zoom = self.current_zoom
            self.logger.info(f"[PTZ] Posición guardada: pan={self.saved_pan:.2f}, tilt={self.saved_tilt:.2f}, zoom={self.saved_zoom:.2f}")
    
    def restore_position(self):
        """Restaurar posición guardada"""
        with self.lock:
            self.logger.info(f"[PTZ] Restaurando posición: pan={self.saved_pan:.2f}, tilt={self.saved_tilt:.2f}, zoom={self.saved_zoom:.2f}")
            self.move_to_position(self.saved_pan, self.saved_tilt)
            self.set_zoom(self.saved_zoom)
    
    def stop(self):
        """Detener movimiento PTZ"""
        try:
            params = {
                'action': 'stop',
                'code': 'Stop'
            }
            return self.send_ptz_command('ptz', params)
        except Exception as e:
            self.logger.warning(f"[PTZ] Error deteniendo movimiento: {e}")
            return False
    
    def calculate_pan_tilt_from_bbox(self, bbox, frame_width, frame_height):
        """
        Calcular pan/tilt necesario para centrar la placa en el frame
        bbox: [x1, y1, x2, y2]
        Retorna: (pan_delta, tilt_delta) normalizados
        """
        x1, y1, x2, y2 = bbox
        
        # Calcular centro de la placa
        plate_center_x = (x1 + x2) / 2.0
        plate_center_y = (y1 + y2) / 2.0
        
        # Calcular centro del frame
        frame_center_x = frame_width / 2.0
        frame_center_y = frame_height / 2.0
        
        # Calcular diferencia (normalizada)
        pan_delta = (plate_center_x - frame_center_x) / frame_width
        tilt_delta = (frame_center_y - plate_center_y) / frame_height  # Invertido porque Y crece hacia abajo
        
        # Limitar valores
        pan_delta = max(-1.0, min(1.0, pan_delta))
        tilt_delta = max(-1.0, min(1.0, tilt_delta))
        
        return pan_delta, tilt_delta
    
    def focus_on_plate(self, bbox, frame_width, frame_height, zoom_level=0.7, restore_after=3.0):
        """
        Enfocar automáticamente en una placa detectada
        bbox: [x1, y1, x2, y2] de la placa
        zoom_level: nivel de zoom a aplicar (0.0 a 1.0)
        restore_after: segundos después de los cuales restaurar posición
        """
        def focus_thread():
            try:
                # Guardar posición actual
                self.save_position()
                
                # Calcular movimiento necesario
                pan_delta, tilt_delta = self.calculate_pan_tilt_from_bbox(bbox, frame_width, frame_height)
                
                # Mover cámara hacia la placa
                self.logger.info(f"[PTZ] Enfocando en placa: pan_delta={pan_delta:.2f}, tilt_delta={tilt_delta:.2f}")
                self.move_relative(pan_delta * 0.5, tilt_delta * 0.5, speed=0.6)  # Movimiento suave
                
                # Esperar a que la cámara se mueva
                time.sleep(1.0)
                
                # Aplicar zoom
                self.logger.info(f"[PTZ] Aplicando zoom: {zoom_level:.2f}")
                self.set_zoom(zoom_level)
                
                # Esperar tiempo de detección
                time.sleep(restore_after)
                
                # Restaurar posición
                self.logger.info("[PTZ] Restaurando posición original")
                self.restore_position()
                
            except Exception as e:
                self.logger.error(f"[PTZ] Error en enfoque automático: {e}")
                # Intentar restaurar en caso de error
                try:
                    self.restore_position()
                except:
                    pass
        
        # Ejecutar en thread separado para no bloquear
        thread = threading.Thread(target=focus_thread, daemon=True)
        thread.start()
        
        return thread

