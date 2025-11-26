# 📁 Estructura del Proyecto LPR

## ✅ Archivos Principales (Mantenidos)

### 🚀 Scripts de Ejecución
- **`realtime_lpr_fixed.py`** ⭐ - Script principal del sistema LPR
- **`iniciar_lpr.sh`** - Script ejecutable para iniciar el sistema
- **`INICIAR_LPR.desktop`** - Acceso directo clickeable para Ubuntu
- **`INSTALAR_DESKTOP.sh`** - Script para instalar acceso directo en Desktop

### 🔧 Módulos del Sistema
- **`ptz_controller.py`** - Controlador PTZ para cámaras
- **`plate_validator.py`** - Validador de formatos de placas
- **`util.py`** - Funciones utilitarias
- **`visualize.py`** - Visualización de resultados

### 📦 Modelos de IA
- **`license_plate_detector.pt`** - Modelo YOLO para detección de placas
- **`yolo11n.pt`** - Modelo YOLO11 nano
- **`yolov8n.pt`** - Modelo YOLOv8 nano

### ⚙️ Configuración
- **`config/`** - Carpeta de configuración
  - `ptz_config.json` - Configuración de cámara y sistema
- **`requirements.txt`** - Dependencias de Python

### 📚 Documentación
- **`README_INSTALACION.md`** - Guía completa de instalación
- **`RESUMEN_INSTALACION.txt`** - Guía rápida
- **`manual.md`** - Manual del sistema
- **`SOLUCIONES_LPR.md`** - Soluciones a problemas comunes
- **`RESUMEN`** - Resumen del proyecto

### 🗂️ Carpetas del Sistema
- **`stream/`** - Módulos de streaming y base de datos
- **`logs/`** - Logs del sistema (se crea automáticamente)
- **`results/`** - Resultados de detecciones (se crea automáticamente)

### 🛠️ Scripts de Instalación
- **`install_jetson_complete.sh`** - Instalación completa para Jetson
- **`jetson_lpr_start.sh`** - Script de inicio para Jetson
- **`ptz_startup.sh`** - Script de inicio para PTZ

### 🧹 Utilidades
- **`LIMPIAR_SCRIPTS.sh`** - Script para limpiar archivos innecesarios

## ❌ Archivos Eliminados (Limpieza)

Los siguientes archivos fueron eliminados por ser innecesarios:

- ~~`test_imports.py`~~ - Script de prueba de imports
- ~~`test_system_fixed.py`~~ - Script de prueba del sistema
- ~~`test_results.json`~~ - Resultados de pruebas
- ~~`parqueadero_simple.py`~~ - Sistema diferente con GUI (tkinter)
- ~~`add_missing_data.py`~~ - Utilidad de datos

## 📋 Estructura Final

```
jetson-lpr/
├── 🚀 EJECUCIÓN
│   ├── realtime_lpr_fixed.py      ⭐ Principal
│   ├── iniciar_lpr.sh             🖱️ Ejecutable
│   ├── INICIAR_LPR.desktop        🖱️ Acceso directo
│   └── INSTALAR_DESKTOP.sh        📦 Instalador
│
├── 🔧 MÓDULOS
│   ├── ptz_controller.py
│   ├── plate_validator.py
│   ├── util.py
│   └── visualize.py
│
├── 📦 MODELOS
│   ├── license_plate_detector.pt
│   ├── yolo11n.pt
│   └── yolov8n.pt
│
├── ⚙️ CONFIGURACIÓN
│   ├── config/
│   │   └── ptz_config.json
│   └── requirements.txt
│
├── 📚 DOCUMENTACIÓN
│   ├── README_INSTALACION.md
│   ├── RESUMEN_INSTALACION.txt
│   ├── manual.md
│   ├── SOLUCIONES_LPR.md
│   ├── RESUMEN
│   └── ESTRUCTURA_PROYECTO.md
│
├── 🗂️ CARPETAS
│   ├── stream/          (módulos de BD y streaming)
│   ├── logs/            (generado automáticamente)
│   └── results/         (generado automáticamente)
│
└── 🛠️ UTILIDADES
    ├── install_jetson_complete.sh
    ├── jetson_lpr_start.sh
    ├── ptz_startup.sh
    └── LIMPIAR_SCRIPTS.sh
```

## 🎯 Uso Rápido

1. **Instalar acceso directo:**
   ```bash
   ./INSTALAR_DESKTOP.sh
   ```

2. **Ejecutar sistema:**
   - Doble clic en `INICIAR_LPR.desktop` en Desktop
   - O ejecutar: `./iniciar_lpr.sh`

3. **Limpiar (si es necesario):**
   ```bash
   ./LIMPIAR_SCRIPTS.sh
   ```

