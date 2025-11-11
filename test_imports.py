#!/usr/bin/env python3
"""
Script para verificar los imports necesarios para realtime_lpr.py
"""

def test_imports():
    """Probar todos los imports necesarios"""
    errors = []
    
    try:
        import cv2
        print("OK - cv2 (OpenCV)")
    except ImportError as e:
        errors.append(f"ERROR - cv2 (OpenCV): {e}")
    
    try:
        import numpy as np
        print("OK - numpy")
    except ImportError as e:
        errors.append(f"ERROR - numpy: {e}")
    
    try:
        from ultralytics import YOLO
        print("OK - ultralytics (YOLO)")
    except ImportError as e:
        errors.append(f"ERROR - ultralytics (YOLO): {e}")
    
    try:
        import easyocr
        print("OK - easyocr")
    except ImportError as e:
        errors.append(f"ERROR - easyocr: {e}")
    
    print("\n" + "="*50)
    if errors:
        print("ERRORES ENCONTRADOS:")
        for error in errors:
            print(error)
        return False
    else:
        print("TODOS LOS IMPORTS FUNCIONAN CORRECTAMENTE")
        return True

if __name__ == "__main__":
    test_imports()