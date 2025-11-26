#!/usr/bin/env python3
"""
💾 Gestor de Base de Datos MySQL para LPR Stream
Solo soporta MySQL (producción)
"""

import json
from datetime import datetime

class DatabaseManager:
    """Gestor de base de datos MySQL para sistema LPR"""
    
    def __init__(self, config):
        self.config = config
        self.connection = None
        self.setup_database()
    
    def setup_database(self):
        """Configurar conexión MySQL"""
        if self.config.get('type', 'mysql') != 'mysql':
            raise ValueError("Solo se soporta MySQL. Configuración debe tener 'type': 'mysql'")
        
        self.setup_mysql()
    
    def setup_mysql(self):
        """Configurar MySQL para producción"""
        try:
            import mysql.connector
            mysql_config = {k: v for k, v in self.config.items() if k != 'type'}
            self.connection = mysql.connector.connect(**mysql_config)
            
            if self.connection.is_connected():
                self.create_mysql_tables()
                print(f"✅ MySQL conectado: {self.config['host']}")
            
        except Exception as e:
            print(f"❌ Error MySQL: {e}")
            raise
    
    def create_mysql_tables(self):
        """Crear tablas MySQL para producción"""
        cursor = self.connection.cursor()
        
        # Tabla de detecciones
        create_detections = '''
        CREATE TABLE IF NOT EXISTS lpr_detections (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            plate_text VARCHAR(10) NOT NULL,
            confidence FLOAT,
            plate_score FLOAT,
            vehicle_bbox TEXT,
            plate_bbox TEXT,
            camera_location VARCHAR(100) DEFAULT 'entrada_principal',
            processed BOOLEAN DEFAULT FALSE,
            entry_type ENUM('entrada', 'salida') DEFAULT 'entrada',
            estimated_distance_m FLOAT,
            
            INDEX idx_timestamp (timestamp),
            INDEX idx_plate (plate_text),
            INDEX idx_location (camera_location),
            INDEX idx_processed (processed),
            INDEX idx_entry_type (entry_type)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        '''
        
        # Tabla de vehículos registrados
        create_vehicles = '''
        CREATE TABLE IF NOT EXISTS registered_vehicles (
            id INT AUTO_INCREMENT PRIMARY KEY,
            plate_number VARCHAR(10) UNIQUE NOT NULL,
            owner_name VARCHAR(100),
            owner_phone VARCHAR(20),
            vehicle_type ENUM('particular', 'moto', 'diplomatico', 'comercial') DEFAULT 'particular',
            vehicle_brand VARCHAR(50),
            vehicle_color VARCHAR(30),
            authorized BOOLEAN DEFAULT TRUE,
            authorization_start DATE,
            authorization_end DATE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            notes TEXT,
            
            INDEX idx_plate (plate_number),
            INDEX idx_authorized (authorized),
            INDEX idx_vehicle_type (vehicle_type),
            INDEX idx_authorization_end (authorization_end)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        '''
        
        # Tabla de log de acceso
        create_access_log = '''
        CREATE TABLE IF NOT EXISTS access_log (
            id INT AUTO_INCREMENT PRIMARY KEY,
            detection_id INT,
            plate_number VARCHAR(10) NOT NULL,
            access_granted BOOLEAN DEFAULT FALSE,
            access_reason VARCHAR(100),
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            camera_location VARCHAR(100),
            
            FOREIGN KEY (detection_id) REFERENCES lpr_detections(id) ON DELETE SET NULL,
            INDEX idx_plate (plate_number),
            INDEX idx_timestamp (timestamp),
            INDEX idx_access (access_granted)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        '''
        
        cursor.execute(create_detections)
        cursor.execute(create_vehicles)
        cursor.execute(create_access_log)
<<<<<<< HEAD
        
        # Agregar columna estimated_distance_m si no existe (migración)
        try:
            cursor.execute("ALTER TABLE lpr_detections ADD COLUMN estimated_distance_m FLOAT")
            self.connection.commit()
            print("📊 Columna estimated_distance_m agregada")
        except Exception as e:
            # La columna ya existe o hay otro error, continuar
            pass
        
=======
>>>>>>> 95fc484 (Fixing_Files_Identical)
        self.connection.commit()
        print("📊 Tablas MySQL creadas")
    
    def insert_detection(self, detection_data):
        """Insertar detección en BD MySQL"""
        try:
            cursor = self.connection.cursor()
            
            # Preparar datos
            plate_bbox_json = detection_data.get('plate_bbox')
            if isinstance(plate_bbox_json, str):
                plate_bbox = plate_bbox_json
            else:
                plate_bbox = json.dumps(plate_bbox_json) if plate_bbox_json else None
            
            vehicle_bbox_json = detection_data.get('vehicle_bbox')
            if isinstance(vehicle_bbox_json, str):
                vehicle_bbox = vehicle_bbox_json
            else:
                vehicle_bbox = json.dumps(vehicle_bbox_json) if vehicle_bbox_json else None
            
<<<<<<< HEAD
            # Verificar si la columna estimated_distance_m existe
            cursor.execute("SHOW COLUMNS FROM lpr_detections LIKE 'estimated_distance_m'")
            has_distance_column = cursor.fetchone() is not None
            
            if has_distance_column:
                cursor.execute('''
                    INSERT INTO lpr_detections 
                    (timestamp, plate_text, confidence, plate_score, vehicle_bbox, plate_bbox, 
                     camera_location, estimated_distance_m)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ''', (
                    detection_data.get('timestamp'),
                    detection_data['plate_text'],
                    detection_data.get('confidence'),
                    detection_data.get('plate_score'),
                    vehicle_bbox,
                    plate_bbox,
                    detection_data.get('camera_location', 'entrada_principal'),
                    detection_data.get('estimated_distance_m')
                ))
            else:
                # Insertar sin estimated_distance_m si la columna no existe
                cursor.execute('''
                    INSERT INTO lpr_detections 
                    (timestamp, plate_text, confidence, plate_score, vehicle_bbox, plate_bbox, 
                     camera_location)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                ''', (
                    detection_data.get('timestamp'),
                    detection_data['plate_text'],
                    detection_data.get('confidence'),
                    detection_data.get('plate_score'),
                    vehicle_bbox,
                    plate_bbox,
                    detection_data.get('camera_location', 'entrada_principal')
                ))
=======
            cursor.execute('''
                INSERT INTO lpr_detections 
                (timestamp, plate_text, confidence, plate_score, vehicle_bbox, plate_bbox, 
                 camera_location, estimated_distance_m)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ''', (
                detection_data.get('timestamp'),
                detection_data['plate_text'],
                detection_data.get('confidence'),
                detection_data.get('plate_score'),
                vehicle_bbox,
                plate_bbox,
                detection_data.get('camera_location', 'entrada_principal'),
                detection_data.get('estimated_distance_m')
            ))
>>>>>>> 95fc484 (Fixing_Files_Identical)
            
            self.connection.commit()
            return True
            
        except Exception as e:
            print(f"❌ Error insertando detección: {e}")
            return False
    
    def check_authorized_vehicle(self, plate_text):
        """Verificar si vehículo está autorizado"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                'SELECT authorized, owner_name FROM registered_vehicles WHERE plate_number = %s',
                (plate_text,)
            )
            
            result = cursor.fetchone()
            
            if result:
                return {
                    'authorized': bool(result[0]),
                    'owner_name': result[1],
                    'registered': True
                }
            else:
                return {
                    'authorized': False,
                    'owner_name': None,
                    'registered': False
                }
                
        except Exception as e:
            print(f"❌ Error verificando autorización: {e}")
            return {'authorized': False, 'owner_name': None, 'registered': False}
    
    def get_recent_detections(self, hours=24):
        """Obtener detecciones recientes"""
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                SELECT plate_text, timestamp, confidence, camera_location, estimated_distance_m
                FROM lpr_detections 
                WHERE timestamp >= DATE_SUB(NOW(), INTERVAL %s HOUR)
                ORDER BY timestamp DESC
            ''', (hours,))
            
            return cursor.fetchall()
            
        except Exception as e:
            print(f"❌ Error obteniendo detecciones: {e}")
            return []
    
    def close(self):
        """Cerrar conexión"""
        if self.connection:
            self.connection.close()
            print("💾 Conexión BD cerrada")
