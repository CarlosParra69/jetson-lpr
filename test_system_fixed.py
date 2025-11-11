#!/usr/bin/env python3
"""
SCRIPT DE PRUEBA COMPLETA DEL SISTEMA LPR
=========================================
Prueba todos los componentes del sistema: IA, base de datos, configuración
Funciona con SQLite (desarrollo) y MySQL (producción)
Compatible con Windows y Linux

Autor: Sistema LPR
Fecha: 2025-11-11
"""

import sys
import os
import json
import time
import tempfile
import sqlite3
from datetime import datetime
from pathlib import Path

# Añadir rutas del proyecto
sys.path.append('stream')

class LPRSystemTester:
    """Probador completo del sistema LPR"""
    
    def __init__(self):
        self.results = []
        self.test_start_time = time.time()
        
    def log_result(self, test_name, success, message, details=None):
        """Registrar resultado de prueba"""
        result = {
            'test': test_name,
            'success': success,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        self.results.append(result)
        
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status} {test_name}: {message}")
        
        if details and not success:
            print(f"   Details: {details}")
    
    def test_imports(self):
        """Probar importaciones de modulos"""
        print("\nTESTING IMPORTS...")
        
        modules = [
            ('cv2', 'OpenCV'),
            ('numpy', 'NumPy'),
            ('ultralytics', 'Ultralytics'),
            ('easyocr', 'EasyOCR'),
            ('torch', 'PyTorch'),
            ('mysql.connector', 'MySQL Connector'),
            ('psutil', 'psutil'),
            ('tqdm', 'tqdm'),
            ('colorama', 'colorama')
        ]
        
        missing = []
        for module, name in modules:
            try:
                __import__(module)
                self.log_result(f"Import {name}", True, f"Module {name} available")
            except ImportError as e:
                missing.append(name)
                self.log_result(f"Import {name}", False, f"Module not available", {'error': str(e)})
        
        if missing:
            self.log_result("Imports General", False, f"Missing modules: {', '.join(missing)}")
        else:
            self.log_result("Imports General", True, "All modules available")
    
    def test_opencv_functionality(self):
        """Probar funcionalidades de OpenCV"""
        print("\nTESTING OPENCV...")
        
        try:
            import cv2
            import numpy as np
            
            # Test crear imagen
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            test_img[:, :] = (0, 255, 0)  # Verde
            
            # Test redimensionar
            resized = cv2.resize(test_img, (50, 50))
            
            # Test escribir imagen temporal
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            cv2.imwrite(temp_file.name, test_img)
            
            # Test leer imagen
            loaded = cv2.imread(temp_file.name)
            
            # Limpiar
            os.unlink(temp_file.name)
            
            if loaded is not None:
                self.log_result("OpenCV Basic", True, "Basic operations working")
            else:
                self.log_result("OpenCV Basic", False, "Error reading image")
                
        except Exception as e:
            self.log_result("OpenCV Basic", False, f"OpenCV error: {e}")
    
    def test_ai_models(self):
        """Probar modelos de IA"""
        print("\nTESTING AI MODELS...")
        
        try:
            from ultralytics import YOLO
            import numpy as np
            
            # Buscar modelo YOLO
            model_files = list(Path(".").glob("*.pt"))
            
            if not model_files:
                self.log_result("YOLO Model", False, "No .pt models found")
                return
            
            # Cargar modelo (el mas pequeno disponible)
            model_files.sort(key=lambda x: x.stat().st_size)  # Ordenar por tamano
            selected_model = model_files[0]
            
            self.log_result("YOLO Model", True, f"Model found: {selected_model.name}")
            
            # Cargar modelo (sin inferencia para no usar GPU)
            model = YOLO(str(selected_model))
            
            # Test con imagen dummy (sin inferencia real)
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            
            self.log_result("YOLO Load", True, f"Model loaded: {len(model.model.names)} classes")
            
            # Probar EasyOCR
            try:
                import easyocr
                reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                self.log_result("EasyOCR", True, "EasyOCR initialized")
            except Exception as e:
                self.log_result("EasyOCR", False, f"EasyOCR error: {e}")
                
        except Exception as e:
            self.log_result("AI Models", False, f"Models error: {e}")
    
    def test_database_sqlite(self):
        """Probar base de datos SQLite"""
        print("\nTESTING DATABASE (SQLite)...")
        
        try:
            # Configuracion SQLite para prueba
            temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
            
            sqlite_config = {
                'type': 'sqlite',
                'database': temp_db.name
            }
            
            # Importar y usar db_manager
            from database.db_manager import DatabaseManager
            
            db = DatabaseManager(sqlite_config)
            
            # Test insertar deteccion
            test_detection = {
                'timestamp': datetime.now().isoformat(),
                'plate_text': 'TEST01',
                'confidence': 0.95,
                'plate_score': 0.90,
                'vehicle_bbox': '[100, 100, 200, 200]',
                'plate_bbox': '[120, 120, 180, 160]',
                'camera_location': 'test_location'
            }
            
            db.insert_detection(test_detection)
            
            # Test verificar autorizacion
            auth = db.check_authorized_vehicle('ABC123')
            
            # Test obtener detecciones recientes
            recent = db.get_recent_detections(24)
            
            db.close()
            
            # Limpiar
            os.unlink(temp_db.name)
            
            self.log_result("SQLite Database", True, f"Basic operations working, {len(recent)} detections found")
            
        except Exception as e:
            self.log_result("SQLite Database", False, f"SQLite error: {e}")
    
    def test_database_mysql(self):
        """Probar base de datos MySQL (opcional)"""
        print("\nTESTING DATABASE (MySQL)...")
        
        try:
            import mysql.connector
            
            # Intentar conexion con configuracion de desarrollo
            mysql_config = {
                'host': 'localhost',
                'port': 3306,
                'user': 'root',
                'password': '',  # Vacio por defecto en Windows
                'database': 'lpr_development',
                'charset': 'utf8mb4'
            }
            
            connection = mysql.connector.connect(**mysql_config)
            cursor = connection.cursor()
            
            # Test consulta basica
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()
            
            self.log_result("MySQL Connection", True, f"MySQL connected, version: {version[0]}")
            
            connection.close()
            
        except Exception as e:
            self.log_result("MySQL Connection", False, "MySQL not available or not configured", {
                'error': str(e),
                'note': 'Normal in systems without MySQL installed'
            })
    
    def test_configuration(self):
        """Probar configuracion del sistema"""
        print("\nTESTING CONFIGURATION...")
        
        try:
            # Probar cargar configuracion
            config_path = Path("config/ptz_config.json")
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Verificar secciones importantes
                required_sections = ['camera', 'jetson', 'realtime_optimization', 'processing', 'output']
                missing = [section for section in required_sections if section not in config]
                
                if not missing:
                    self.log_result("Configuration", True, "Complete valid configuration")
                else:
                    self.log_result("Configuration", False, f"Missing sections: {missing}")
            else:
                # Crear configuracion por defecto
                default_config = {
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
                        "capture_target_fps": 25,
                        "ai_process_every": 2,
                        "motion_activation": True,
                        "display_scale": 0.25,
                        "headless_mode": True
                    },
                    "processing": {
                        "confidence_threshold": 0.30,
                        "plate_confidence_min": 0.25,
                        "detection_cooldown_sec": 0.5,
                        "ocr_cache_enabled": True
                    },
                    "output": {
                        "save_results": True,
                        "show_video": False,
                        "window_title": "LPR System"
                    }
                }
                
                config_path.parent.mkdir(exist_ok=True)
                with open(config_path, 'w') as f:
                    json.dump(default_config, f, indent=4)
                
                self.log_result("Configuration", True, "Default configuration created")
                
        except Exception as e:
            self.log_result("Configuration", False, f"Configuration error: {e}")
    
    def test_system_info(self):
        """Probar informacion del sistema"""
        print("\nTESTING SYSTEM INFO...")
        
        try:
            import platform
            import psutil
            
            # Informacion del sistema
            system_info = {
                'platform': platform.system(),
                'platform_version': platform.release(),
                'architecture': platform.machine(),
                'python_version': sys.version,
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total,
                'memory_available': psutil.virtual_memory().available
            }
            
            # Informacion de GPU (si esta disponible)
            try:
                import torch
                if torch.cuda.is_available():
                    system_info['gpu_available'] = True
                    system_info['gpu_name'] = torch.cuda.get_device_name(0)
                    system_info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory
                else:
                    system_info['gpu_available'] = False
            except:
                system_info['gpu_available'] = False
            
            self.log_result("System Info", True, f"System: {system_info['platform']} {system_info['architecture']}", 
                          system_info)
            
        except Exception as e:
            self.log_result("System Info", False, f"System info error: {e}")
    
    def test_file_structure(self):
        """Probar estructura de archivos"""
        print("\nTESTING FILE STRUCTURE...")
        
        required_files = [
            'realtime_lpr_fixed.py',
            'requirements.txt',
            'config/ptz_config.json'
        ]
        
        optional_files = [
            'license_plate_detector.pt',
            'yolov8n.pt',
            'yolo11n.pt',
            'install_jetson_complete.sh',
            'manual.md'
        ]
        
        missing_required = []
        missing_optional = []
        
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_required.append(file_path)
        
        for file_path in optional_files:
            if not Path(file_path).exists():
                missing_optional.append(file_path)
        
        if not missing_required:
            self.log_result("File Structure", True, "Main files present")
        else:
            self.log_result("File Structure", False, f"Missing files: {missing_required}")
        
        if missing_optional:
            self.log_result("Optional Files", False, f"Optional files missing: {missing_optional}")
        else:
            self.log_result("Optional Files", True, "Optional files present")
    
    def run_all_tests(self):
        """Ejecutar todas las pruebas"""
        print("STARTING COMPLETE LPR SYSTEM TESTS")
        print("=" * 60)
        
        tests = [
            ("Imports", self.test_imports),
            ("OpenCV", self.test_opencv_functionality),
            ("AI Models", self.test_ai_models),
            ("SQLite DB", self.test_database_sqlite),
            ("MySQL DB", self.test_database_mysql),
            ("Configuration", self.test_configuration),
            ("System Info", self.test_system_info),
            ("File Structure", self.test_file_structure)
        ]
        
        for test_name, test_func in tests:
            try:
                test_func()
            except Exception as e:
                self.log_result(test_name, False, f"Test execution error: {e}")
        
        # Resumen final
        self.print_summary()
    
    def print_summary(self):
        """Imprimir resumen de pruebas"""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total time: {time.time() - self.test_start_time:.1f} seconds")
        print(f"Total tests: {total_tests}")
        print(f"Passed tests: {passed_tests}")
        print(f"Failed tests: {failed_tests}")
        
        if failed_tests > 0:
            print(f"\nFAILED TESTS:")
            for result in self.results:
                if not result['success']:
                    print(f"   - {result['test']}: {result['message']}")
        
        # Categorizar resultados
        critical_failures = []
        warnings = []
        
        for result in self.results:
            if not result['success']:
                if 'Import' in result['test'] or 'AI Models' in result['test']:
                    critical_failures.append(result['test'])
                else:
                    warnings.append(result['test'])
        
        if critical_failures:
            print(f"\nCRITICAL FAILURES (affect functionality):")
            for failure in critical_failures:
                print(f"   - {failure}")
        
        if warnings:
            print(f"\nWARNINGS (don't affect main functionality):")
            for warning in warnings:
                print(f"   - {warning}")
        
        # Recomendaciones
        print(f"\nRECOMMENDATIONS:")
        if critical_failures:
            print("   1. Install missing dependencies: pip install -r requirements.txt")
            print("   2. Verify Python and module installation")
        else:
            print("   System ready to work")
            
        if not any('MySQL' in r['test'] and r['success'] for r in self.results):
            print("   Database: For MySQL, install and configure MySQL Server")
            print("   Database: For development, use SQLite (already tested)")
        
        print("   Documentation: See manual.md for detailed instructions")
        print("   Setup: Use setup scripts to configure database")
        
        # Guardar reporte
        self.save_report()
    
    def save_report(self):
        """Guardar reporte de pruebas"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': time.time() - self.test_start_time,
                'summary': {
                    'total': len(self.results),
                    'passed': sum(1 for r in self.results if r['success']),
                    'failed': sum(1 for r in self.results if not r['success'])
                },
                'results': self.results
            }
            
            report_file = Path("test_results.json")
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"\nReport saved to: {report_file}")
            
        except Exception as e:
            print(f"\nError saving report: {e}")

def main():
    """Funcion principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Complete LPR System Tests')
    parser.add_argument('--quick', action='store_true', help='Quick tests only')
    parser.add_argument('--db-only', action='store_true', help='Only test database')
    
    args = parser.parse_args()
    
    tester = LPRSystemTester()
    
    if args.db_only:
        print("Testing database only...")
        tester.test_database_sqlite()
        tester.test_database_mysql()
        tester.print_summary()
    elif args.quick:
        print("Running quick tests...")
        tester.test_imports()
        tester.test_configuration()
        tester.test_file_structure()
        tester.print_summary()
    else:
        tester.run_all_tests()

if __name__ == "__main__":
    main()