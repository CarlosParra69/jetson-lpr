#!/usr/bin/env python3
"""
Database Manager for LPR Stream
Supports SQLite (development) and MySQL (production)
"""

import sqlite3
import json
import os
from datetime import datetime

class DatabaseManager:
    def __init__(self, config):
        self.config = config
        self.connection = None
        self.setup_database()
    
    def setup_database(self):
        """Setup connection according to DB type"""
        if self.config['type'] == 'sqlite':
            self.setup_sqlite()
        elif self.config['type'] == 'mysql':
            self.setup_mysql()
        else:
            raise ValueError(f"Unsupported DB type: {self.config['type']}")
    
    def setup_sqlite(self):
        """Setup SQLite for development"""
        try:
            db_path = self.config['database']
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            self.connection = sqlite3.connect(db_path, check_same_thread=False)
            self.connection.execute('PRAGMA foreign_keys = ON')
            self.create_sqlite_tables()
            print(f"[OK] SQLite connected: {db_path}")
            
        except Exception as e:
            print(f"[ERROR] SQLite error: {e}")
            raise
    
    def setup_mysql(self):
        """Setup MySQL for production"""
        try:
            import mysql.connector
            mysql_config = {k: v for k, v in self.config.items() if k != 'type'}
            self.connection = mysql.connector.connect(**mysql_config)
            
            if self.connection.is_connected():
                self.create_mysql_tables()
                print(f"[OK] MySQL connected: {self.config['host']}")
            
        except Exception as e:
            print(f"[ERROR] MySQL error: {e}")
            raise
    
    def create_sqlite_tables(self):
        """Create SQLite tables for development"""
        cursor = self.connection.cursor()
        
        # Detections table
        create_detections = '''
        CREATE TABLE IF NOT EXISTS lpr_detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            plate_text VARCHAR(10) NOT NULL,
            confidence REAL,
            plate_score REAL,
            vehicle_bbox TEXT,
            plate_bbox TEXT,
            camera_location VARCHAR(100) DEFAULT 'development',
            processed BOOLEAN DEFAULT 0
        )
        '''
        
        # Registered vehicles table
        create_vehicles = '''
        CREATE TABLE IF NOT EXISTS registered_vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number VARCHAR(10) UNIQUE NOT NULL,
            owner_name VARCHAR(100),
            vehicle_type VARCHAR(20) DEFAULT 'particular',
            authorized BOOLEAN DEFAULT 1,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        '''
        
        cursor.execute(create_detections)
        cursor.execute(create_vehicles)
        
        # Insert test data
        test_vehicles = [
            ('ABC123', 'Juan Perez', 'particular', 1),
            ('XYZ789', 'Maria Garcia', 'particular', 1),
            ('MOT45A', 'Carlos Lopez', 'moto', 1),
            ('CD1234', 'Embajada Test', 'diplomatico', 1),
            ('DEF456', 'Ana Rodriguez', 'particular', 0)
        ]
        
        cursor.executemany('''
            INSERT OR IGNORE INTO registered_vehicles 
            (plate_number, owner_name, vehicle_type, authorized) 
            VALUES (?, ?, ?, ?)
        ''', test_vehicles)
        
        self.connection.commit()
        print("[OK] SQLite tables created with test data")
    
    def create_mysql_tables(self):
        """Create MySQL tables for production"""
        cursor = self.connection.cursor()
        
        # Detections table
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
            
            INDEX idx_timestamp (timestamp),
            INDEX idx_plate (plate_text),
            INDEX idx_location (camera_location)
        )
        '''
        
        # Registered vehicles table
        create_vehicles = '''
        CREATE TABLE IF NOT EXISTS registered_vehicles (
            id INT AUTO_INCREMENT PRIMARY KEY,
            plate_number VARCHAR(10) UNIQUE NOT NULL,
            owner_name VARCHAR(100),
            vehicle_type ENUM('particular', 'moto', 'diplomatico') DEFAULT 'particular',
            authorized BOOLEAN DEFAULT TRUE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            
            INDEX idx_plate (plate_number)
        )
        '''
        
        cursor.execute(create_detections)
        cursor.execute(create_vehicles)
        self.connection.commit()
        print("[OK] MySQL tables created")
    
    def insert_detection(self, detection_data):
        """Insert detection in DB"""
        try:
            if self.config['type'] == 'sqlite':
                cursor = self.connection.cursor()
                cursor.execute('''
                    INSERT INTO lpr_detections 
                    (timestamp, plate_text, confidence, plate_score, vehicle_bbox, plate_bbox, camera_location)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    detection_data['timestamp'],
                    detection_data['plate_text'],
                    detection_data['confidence'],
                    detection_data['plate_score'],
                    json.dumps(detection_data['vehicle_bbox']),
                    json.dumps(detection_data['plate_bbox']),
                    detection_data.get('camera_location', 'development')
                ))
                
            else:  # MySQL
                cursor = self.connection.cursor()
                cursor.execute('''
                    INSERT INTO lpr_detections 
                    (timestamp, plate_text, confidence, plate_score, vehicle_bbox, plate_bbox, camera_location)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                ''', (
                    detection_data['timestamp'],
                    detection_data['plate_text'],
                    detection_data['confidence'],
                    detection_data['plate_score'],
                    json.dumps(detection_data['vehicle_bbox']),
                    json.dumps(detection_data['plate_bbox']),
                    detection_data.get('camera_location', 'entrada_principal')
                ))
            
            self.connection.commit()
            return True
            
        except Exception as e:
            print(f"[ERROR] Error inserting detection: {e}")
            return False
    
    def check_authorized_vehicle(self, plate_text):
        """Check if vehicle is authorized"""
        try:
            cursor = self.connection.cursor()
            
            if self.config['type'] == 'sqlite':
                cursor.execute(
                    'SELECT authorized, owner_name FROM registered_vehicles WHERE plate_number = ?',
                    (plate_text,)
                )
            else:  # MySQL
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
            print(f"[ERROR] Error checking authorization: {e}")
            return {'authorized': False, 'owner_name': None, 'registered': False}
    
    def get_recent_detections(self, hours=24):
        """Get recent detections"""
        try:
            cursor = self.connection.cursor()
            
            if self.config['type'] == 'sqlite':
                cursor.execute('''
                    SELECT plate_text, timestamp, confidence, camera_location
                    FROM lpr_detections 
                    WHERE datetime(timestamp) >= datetime('now', '-{} hours')
                    ORDER BY timestamp DESC
                '''.format(hours))
            else:  # MySQL
                cursor.execute('''
                    SELECT plate_text, timestamp, confidence, camera_location
                    FROM lpr_detections 
                    WHERE timestamp >= DATE_SUB(NOW(), INTERVAL %s HOUR)
                    ORDER BY timestamp DESC
                ''', (hours,))
            
            return cursor.fetchall()
            
        except Exception as e:
            print(f"[ERROR] Error getting detections: {e}")
            return []
    
    def close(self):
        """Close connection"""
        if self.connection:
            self.connection.close()
            print("[OK] DB connection closed")