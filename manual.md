# 🚗 Sistema LPR (License Plate Recognition) - Manual Completo

## 📋 Índice

1. [Descripción del Proyecto](#descripción-del-proyecto)
2. [Características Principales](#características-principales)
3. [Requisitos del Sistema](#requisitos-del-sistema)
4. [Instalación](#instalación)
5. [Configuración](#configuración)
6. [Base de Datos](#base-de-datos)
7. [Uso del Sistema](#uso-del-sistema)
8. [Scripts y Herramientas](#scripts-y-herramientas)
9. [Solución de Problemas](#solución-de-problemas)
10. [Monitoreo y Mantenimiento](#monitoreo-y-mantenimiento)

---

## 📖 Descripción del Proyecto

El **Sistema LPR** es una solución completa de reconocimiento automático de
placas vehiculares en tiempo real, optimizada para funcionar en **Jetson Orin
Nano** y entornos de producción.

### 🎯 Propósito

- Detección automática de placas de vehículos en tiempo real
- Reconocimiento óptico de caracteres (OCR) para extraer texto
- Validación y verificación de placas autorizadas
- Integración con base de datos MySQL para gestión de vehículos
- Optimizado para cámaras PTZ y sistemas de estacionamiento

### 🚀 Versiones

- **Desarrollo (Windows/Linux)**: `realtime_lpr_fixed.py`
- **Producción (Jetson Orin Nano)**: Optimizado con modo headless
- **Base de datos**: SQLite (desarrollo) y MySQL (producción)

---

## ⚡ Características Principales

### 🤖 Inteligencia Artificial

- **YOLOv8/YOLOv11**: Detección de objetos optimizada
- **EasyOCR**: Reconocimiento óptico de caracteres avanzado
- **Procesamiento en tiempo real**: IA cada 2 frames
- **Cache inteligente**: Optimización de rendimiento
- **Detección de movimiento**: Activación inteligente de IA

### 🔧 Optimizaciones Técnicas

- **Modo headless**: Sin dependencias de GUI
- **Logging UTF-8**: Compatible con múltiples idiomas
- **Threading optimizado**: Procesamiento paralelo
- **Configuración flexible**: Parámetros adaptables
- **Variables de entorno**: CUDA y rendimiento optimizados

### 📊 Base de Datos

- **Soporte dual**: SQLite (desarrollo) y MySQL (producción)
- **Gestión de vehículos**: Autorizaciones y perfiles
- **Log de detecciones**: Historial completo
- **Índices optimizados**: Consultas rápidas
- **Backup automático**: Respaldo de datos

### 🌐 Conectividad

- **Cámaras IP/RTSP**: Múltiples formatos soportados
- **Red Jetson**: Configuración automática
- **Protocolos**: RTSP, HTTP, MySQL
- **Monitoreo**: Logs detallados y estadísticas

---

## 💻 Requisitos del Sistema

### 🖥️ Requisitos Mínimos

- **OS**: Ubuntu 20.04+ / Windows 10+
- **Python**: 3.8+
- **RAM**: 4GB mínimo, 8GB recomendado
- **Almacenamiento**: 10GB libres
- **CPU**: x86_64 o ARM64 (Jetson)

### 🤖 Jetson Orin Nano (Producción)

- **GPU**: NVIDIA Jetson Orin Nano
- **RAM**: 8GB (configuración mínima recomendada)
- **Almacenamiento**: 32GB+ microSD o SSD
- **Conectividad**: Gigabit Ethernet
- **Cámara**: IP PTZ con soporte RTSP

### 🛠️ Dependencias del Sistema

#### Linux (Ubuntu/Jetson)

```bash
sudo apt update
sudo apt install -y \
    python3-dev python3-pip python3-setuptools \
    build-essential cmake git wget unzip curl \
    mysql-server mysql-client libmysqlclient-dev \
    libopencv-dev python3-opencv \
    libgtk2.0-dev libavcodec-dev libavformat-dev \
    libswscale-dev libv4l-dev libxvidcore-dev \
    libx264-dev libjpeg-dev libpng-dev libtiff-dev
```

#### Windows

- Visual Studio Build Tools 2019+
- Git for Windows
- MySQL Server 8.0+
- Python 3.8+ con pip

### 📦 Dependencias Python

Todas las dependencias están listadas en `requirements.txt`:

```bash
# Principales
opencv-python>=4.8.0
numpy>=1.24.0
ultralytics>=8.2.0
easyocr>=1.7.0
torch>=2.0.0
torchvision>=0.15.0
mysql-connector-python

# Soporte
psutil>=5.9.0
tqdm>=4.65.0
colorama>=0.4.6
```

---

## 🚀 Instalación

### 🔧 Instalación Automática (Jetson Orin Nano)

**Script completo de instalación:**

```bash
# Descargar e instalar todo automáticamente
wget -O install_jetson.sh https://raw.githubusercontent.com/tu-repo/install_jetson_complete.sh
chmod +x install_jetson_complete.sh
./install_jetson_complete.sh
```

**Lo que hace el script:**

1. Actualiza el sistema Jetson
2. Instala todas las dependencias necesarias
3. Configura MySQL
4. Optimiza variables de entorno
5. Configura red Jetson
6. Crea scripts de utilidad
7. Verifica la instalación

### 📱 Instalación Manual

#### 1. Clonar Repositorio

```bash
git clone https://github.com/tu-repo/jetson-lpr.git
cd jetson-lpr
```

#### 2. Instalar Dependencias Python

```bash
# Desarrollo (Windows/Linux)
pip install -r requirements.txt

# Jetson Orin Nano (optimizado)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt
```

#### 3. Hacer Ejecutables

```bash
chmod +x *.py *.sh
```

#### 4. Verificar Instalación

```bash
python test_imports.py
```

---

## ⚙️ Configuración

### 📋 Configuración Principal

Archivo: `config/ptz_config.json`

```json
{
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
        "motion_activation": true,
        "display_scale": 0.25,
        "headless_mode": true
    },
    "processing": {
        "confidence_threshold": 0.30,
        "plate_confidence_min": 0.25,
        "detection_cooldown_sec": 0.5,
        "ocr_cache_enabled": true
    },
    "output": {
        "save_results": true,
        "show_video": false,
        "window_title": "LPR Sistema"
    }
}
```

### 🌐 Configuración de Red Jetson

#### Configuración Automática

```bash
# El sistema configura automáticamente al inicio
sudo ip addr add 192.168.1.100/24 dev enP8p1s0
sudo ethtool -s enP8p1s0 speed 100 duplex full autoneg off
```

#### Configuración Manual

```bash
# Configurar interfaz de red
sudo ip addr flush dev enP8p1s0
sudo ip addr add 192.168.1.100/24 dev enP8p1s0
sudo ethtool -s enP8p1s0 speed 100 duplex full autoneg off

# Verificar configuración
ip addr show enP8p1s0
```

### 🔧 Variables de Entorno

Agregar a `~/.bashrc`:

```bash
# ===== CONFIGURACIÓN SISTEMA LPR =====
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"
export PYTHONUNBUFFERED=1

# Rutas del proyecto
export LPR_HOME="$HOME/jetson-lpr"
export PYTHONPATH="$LPR_HOME:$PYTHONPATH"
```

---

## 💾 Base de Datos

### 🗄️ MySQL para Desarrollo

#### Instalación MySQL

```bash
# Ubuntu/Debian
sudo apt install mysql-server mysql-client libmysqlclient-dev

# Windows
# Descargar MySQL Installer desde https://dev.mysql.com/downloads/installer/
```

#### Configuración Desarrollo

```bash
cd jetson-lpr
python stream/database/setup_mysql_dev.py
```

**Configuración creada:**

- **Base de datos**: `lpr_development`
- **Usuario**: `lpr_dev_user`
- **Password**: `lpr_dev_pass`
- **Puerto**: `3306`

### 🤖 MySQL para Jetson (Producción)

#### Configuración Producción

```bash
cd jetson-lpr
python stream/database/setup_mysql_jetson.py --optimize
```

**Configuración creada:**

- **Base de datos**: `parqueadero_jetson`
- **Usuario**: `lpr_user`
- **Password**: `lpr_password`
- **Puerto**: `3306`

### 📊 Estructura de Tablas

#### Tabla: `lpr_detections`

```sql
CREATE TABLE lpr_detections (
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
    
    INDEX idx_timestamp (timestamp),
    INDEX idx_plate (plate_text),
    INDEX idx_location (camera_location)
);
```

#### Tabla: `registered_vehicles`

```sql
CREATE TABLE registered_vehicles (
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
    INDEX idx_authorized (authorized)
);
```

#### Tabla: `access_log`

```sql
CREATE TABLE access_log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    detection_id INT,
    plate_number VARCHAR(10) NOT NULL,
    access_granted BOOLEAN DEFAULT FALSE,
    access_reason VARCHAR(100),
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    camera_location VARCHAR(100),
    
    FOREIGN KEY (detection_id) REFERENCES lpr_detections(id),
    INDEX idx_plate (plate_number),
    INDEX idx_timestamp (timestamp)
);
```

### 🧪 Testing Base de Datos

#### Test de Conexión

```bash
# Desarrollo
python stream/database/setup_mysql_dev.py --test

# Producción (Jetson)
python stream/database/setup_mysql_jetson.py --test
```

#### Script de Prueba

```python
#!/usr/bin/env python3
"""Test completo de base de datos"""

import sys
sys.path.append('stream')

from database.db_manager import DatabaseManager

# Configuración de prueba
mysql_config = {
    'host': 'localhost',
    'port': 3306,
    'database': 'lpr_development',  # o 'parqueadero_jetson'
    'user': 'lpr_dev_user',        # o 'lpr_user'
    'password': 'lpr_dev_pass',    # o 'lpr_password'
    'charset': 'utf8mb4'
}

def test_database():
    try:
        db = DatabaseManager(mysql_config)
        
        # Test insertar detección
        test_detection = {
            'timestamp': '2025-11-11 20:00:00',
            'plate_text': 'TEST01',
            'confidence': 0.95,
            'plate_score': 0.90,
            'vehicle_bbox': '[100, 100, 200, 200]',
            'plate_bbox': '[120, 120, 180, 160]',
            'camera_location': 'test_location'
        }
        
        db.insert_detection(test_detection)
        
        # Test verificar autorización
        auth = db.check_authorized_vehicle('TEST01')
        print(f"Autorización TEST01: {auth}")
        
        # Test obtener detecciones recientes
        recent = db.get_recent_detections(24)
        print(f"Detecciones recientes: {len(recent)}")
        
        db.close()
        print("✅ Test de base de datos exitoso")
        
    except Exception as e:
        print(f"❌ Error en test: {e}")

if __name__ == "__main__":
    test_database()
```

---

## 🎮 Uso del Sistema

### 🚀 Inicio Rápido

#### Jetson Orin Nano

```bash
# Inicio automático completo
./quick_start.sh

# Inicio manual con opciones
python realtime_lpr_fixed.py \
    --headless \
    --ai-every 2 \
    --cooldown 0.5 \
    --confidence 0.30 \
    --motion \
    --display-scale 0.50
```

#### Windows/Linux (Desarrollo)

```bash
# Modo con GUI (si está disponible)
python realtime_lpr_fixed.py

# Modo sin GUI (recomendado)
python realtime_lpr_fixed.py --headless
```

### 📋 Parámetros de Línea de Comandos

```bash
python realtime_lpr_fixed.py [OPTIONS]

OPCIONES:
  -h, --help            show help message and exit
  --config CONFIG       archivo de configuración (default: config/ptz_config.json)
  --ai-every AI_EVERY   procesar IA cada N frames (default: 2)
  --cooldown COOLDOWN   cooldown en segundos (default: 0.5)
  --motion              activar detección de movimiento
  --confidence CONFIDENCE umbral confianza YOLO (default: 0.30)
  --display-scale SCALE escala display (default: 0.25)
  --headless           activar modo sin GUI (recomendado para Jetson)
```

### 🎛️ Controles Durante la Ejecución

Cuando el sistema está ejecutándose con GUI:

- `q` o `ESC`: Salir del programa
- `r`: Reset de estadísticas
- `c`: Limpiar cache
- `s`: Guardar captura de pantalla

### 📊 Monitoreo en Tiempo Real

#### Ver Logs

```bash
# Ver logs en tiempo real
tail -f logs/realtime_lpr_*.log

# Ver solo errores
tail -f logs/realtime_lpr_*.log | grep ERROR

# Filtrar por tipo de evento
grep "PLACA" logs/realtime_lpr_*.log
```

#### Verificar Estado

```bash
# Verificar procesos
ps aux | grep realtime_lpr_fixed

# Verificar memoria y CPU
htop

# Verificar red Jetson
ip addr show enP8p1s0

# Verificar MySQL
sudo systemctl status mysql
```

### 📈 Análisis de Resultados

#### Logs de Detección

Ubicación: `logs/realtime_lpr_YYYYMMDD_HHMMSS.log`

Ejemplo de entrada:

```
2025-11-11 15:20:31,830 - REALTIME-LPR - INFO - [TARGET] PLACA: ABC123 (YOLO: 0.95, OCR: 0.88, Latencia: 45ms)
```

#### Resultados Estructurados

Ubicación: `results/realtime_detections_YYYYMMDD.jsonl`

Ejemplo de entrada:

```json
{
    "timestamp": "2025-11-11T15:20:31.830",
    "frame_number": 1250,
    "ai_frame_number": 625,
    "plate_text": "ABC123",
    "yolo_confidence": 0.95,
    "ocr_confidence": 0.88,
    "bbox": [100, 100, 200, 150],
    "processing_latency_ms": 45,
    "valid": true
}
```

---

## 🛠️ Scripts y Herramientas

### 📁 Estructura del Proyecto

```
jetson-lpr/
├── realtime_lpr_fixed.py      # Programa principal
├── install_jetson_complete.sh # Script instalación completa
├── jetson_lpr_start.sh        # Script inicio Jetson
├── ptz_startup.sh             # Script configuración PTZ
├── test_imports.py            # Test dependencias
├── test_system_fixed.py       # Pruebas del sistema
├── requirements.txt           # Dependencias Python
├── SOLUCIONES_LPR.md         # Documentación de soluciones
├── manual.md                  # Este manual
├── config/
│   └── ptz_config.json       # Configuración principal
├── logs/                     # Logs del sistema
├── results/                  # Resultados de detecciones
└── stream/
    ├── database/
    │   ├── db_manager.py     # Gestor de base de datos MySQL
    │   ├── setup_mysql_dev.py # Configuración MySQL desarrollo
    │   ├── setup_mysql_jetson.py # Configuración MySQL producción
    │   └── __init__.py
    ├── config/
    └── utils/
```

### 🔧 Scripts de Utilidad

#### `verify_installation.py`

Verifica que todas las dependencias estén correctamente instaladas:

```bash
python verify_installation.py
```

#### `test_imports.py`

Test simple de importaciones:

```bash
python test_imports.py
```

#### `jetson_lpr_start.sh`

Script completo de inicio para Jetson:

```bash
./jetson_lpr_start.sh
```

#### `quick_start.sh`

Inicio rápido con verificaciones:

```bash
./quick_start.sh
```

### 🗄️ Scripts de Base de Datos

#### `setup_mysql_dev.py`

Configura MySQL para desarrollo:

```bash
python stream/database/setup_mysql_dev.py
python stream/database/setup_mysql_dev.py --test
```

#### `setup_mysql_jetson.py`

Configura MySQL para producción (Jetson):

```bash
python stream/database/setup_mysql_jetson.py
python stream/database/setup_mysql_jetson.py --optimize
python stream/database/setup_mysql_jetson.py --test
```

---

## 🐛 Solución de Problemas

### ❌ Problemas Comunes

#### 1. Error: `ModuleNotFoundError: No module named 'cv2'`

**Síntoma**: Error de importación de OpenCV

**Solución**:

```bash
# Reinstalar OpenCV
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python opencv-contrib-python

# Verificar instalación
python -c "import cv2; print(cv2.__version__)"
```

#### 2. Error: `UnicodeEncodeError: 'charmap' codec can't encode character`

**Síntoma**: Errores de codificación en Windows

**Solución**: Usar la versión corregida `realtime_lpr_fixed.py` que maneja UTF-8
correctamente.

#### 3. Error: `cv2.error: OpenCV ... The function is not implemented`

**Síntoma**: OpenCV no soporta GUI en el entorno actual

**Solución**:

```bash
# Usar modo headless
python realtime_lpr_fixed.py --headless

# O reinstalar OpenCV con soporte GUI
pip uninstall opencv-python
pip install opencv-contrib-python
```

#### 4. Error: `MySQL connection failed`

**Síntoma**: No se puede conectar a MySQL

**Solución**:

```bash
# Verificar que MySQL esté corriendo
sudo systemctl status mysql

# Iniciar MySQL si está parado
sudo systemctl start mysql

# Configurar MySQL si es la primera vez
python stream/database/setup_mysql_dev.py  # Desarrollo
python stream/database/setup_mysql_jetson.py  # Producción
```

#### 5. Error: `CUDA out of memory`

**Síntoma**: Error de memoria en GPU

**Solución**:

```bash
# Reducir resolución de procesamiento
python realtime_lpr_fixed.py --ai-every 3 --display-scale 0.2

# Limpiar cache de GPU
python -c "import torch; torch.cuda.empty_cache()"

# Verificar memoria disponible
nvidia-smi
```

#### 6. Error: `RTSP connection failed`

**Síntoma**: No se puede conectar a la cámara

**Solución**:

```bash
# Verificar configuración de red
ping 192.168.1.101

# Verificar URL RTSP
ffmpeg -i rtsp://admin:admin@192.168.1.101/cam/realmonitor?channel=1&subtype=1 -t 5 -f null -

# Verificar credenciales de cámara
# Usuario: admin, Password: admin (por defecto)
```

### 🔍 Comandos de Diagnóstico

#### Verificar Sistema

```bash
# Información general
python verify_installation.py

# Verificar GPU
nvidia-smi

# Verificar red
ip addr show
ping -c 3 192.168.1.101

# Verificar MySQL
sudo systemctl status mysql
mysql -u root -p -e "SHOW DATABASES;"
```

#### Verificar Logs

```bash
# Ver errores recientes
tail -50 logs/realtime_lpr_*.log | grep ERROR

# Ver estadísticas
tail -50 logs/realtime_lpr_*.log | grep "ESTADÍSTICAS"

# Contar detecciones
grep "PLACA" logs/realtime_lpr_*.log | wc -l
```

#### Verificar Base de Datos

```bash
# Test de conexión
python stream/database/setup_mysql_dev.py --test

# Verificar tablas
mysql -u lpr_dev_user -p lpr_development -e "SHOW TABLES; SELECT COUNT(*) FROM lpr_detections;"

# Ver vehículos autorizados
mysql -u lpr_user -p parqueadero_jetson -e "SELECT plate_number, owner_name FROM registered_vehicles WHERE authorized = TRUE;"
```

### 🛠️ Optimización de Rendimiento

#### Para Jetson Orin Nano

```bash
# Configurar gobernor de CPU
sudo cpupower frequency-set -g performance

# Configurar swappiness
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf

# Optimizar memoria GPU
sudo nvpmodel -m 0  # Modo máximo rendimiento
```

#### Para Base de Datos

```bash
# Optimizar consultas
mysql -u root -p -e "
    OPTIMIZE TABLE lpr_detections;
    OPTIMIZE TABLE registered_vehicles;
    ANALYZE TABLE lpr_detections;
"

# Verificar índices
mysql -u root -p -e "
    SHOW INDEX FROM lpr_detections;
    SHOW INDEX FROM registered_vehicles;
"
```

---

## 📊 Monitoreo y Mantenimiento

### 📈 Métricas de Rendimiento

#### KPIs del Sistema

- **FPS de captura**: Frames por segundo de la cámara
- **FPS de IA**: Frames procesados por segundo con IA
- **Latencia**: Tiempo de procesamiento por frame
- **Precisión**: Porcentaje de detecciones correctas
- **Uptime**: Tiempo de funcionamiento continuo

#### Ubicación de Métricas

- **Logs**: `logs/realtime_lpr_*.log`
- **Estadísticas en tiempo real**: Logs con `ESTADÍSTICAS TIEMPO REAL`
- **Base de datos**: Tabla `lpr_detections`

### 🔧 Tareas de Mantenimiento

#### Diario

- Verificar logs de errores
- Revisar detecciones de las últimas 24 horas
- Verificar estado de MySQL
- Monitorear uso de CPU/GPU

#### Semanal

- Analizar estadísticas de rendimiento
- Limpiar logs antiguos (>30 días)
- Verificar integridad de la base de datos
- Backup de la base de datos

#### Mensual

- Optimizar tablas de la base de datos
- Actualizar listas de vehículos autorizados
- Revisar configuración de red
- Actualizar dependencias si es necesario

### 📋 Scripts de Mantenimiento

#### Backup de Base de Datos

```bash
#!/bin/bash
# backup_db.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$HOME/backups/lpr"

mkdir -p $BACKUP_DIR

# Backup MySQL
mysqldump -u lpr_user -p'lpr_password' parqueadero_jetson > $BACKUP_DIR/lpr_db_$DATE.sql

# Comprimir
gzip $BACKUP_DIR/lpr_db_$DATE.sql

# Mantener solo los últimos 30 backups
find $BACKUP_DIR -name "lpr_db_*.sql.gz" -mtime +30 -delete

echo "Backup completado: lpr_db_$DATE.sql.gz"
```

#### Limpieza de Logs

```bash
#!/bin/bash
# cleanup_logs.sh

LOG_DIR="$HOME/jetson-lpr/logs"
RESULTS_DIR="$HOME/jetson-lpr/results"

# Eliminar logs antiguos (>30 días)
find $LOG_DIR -name "*.log" -mtime +30 -delete

# Eliminar capturas antiguas (>7 días)
find . -name "realtime_capture_*.jpg" -mtime +7 -delete

# Comprimir logs grandes
find $LOG_DIR -name "*.log" -size +100M -exec gzip {} \;

echo "Limpieza completada"
```

#### Monitoreo Automático

```bash
#!/bin/bash
# monitor_system.sh

LOG_FILE="$HOME/jetson-lpr/logs/monitor.log"

# Verificar estado del proceso LPR
if ! pgrep -f "realtime_lpr_fixed.py" > /dev/null; then
    echo "$(date): ERROR - Proceso LPR no está corriendo" >> $LOG_FILE
    # Reiniciar sistema
    cd ~/jetson-lpr && ./quick_start.sh &
fi

# Verificar MySQL
if ! systemctl is-active --quiet mysql; then
    echo "$(date): ERROR - MySQL no está activo" >> $LOG_FILE
    sudo systemctl start mysql
fi

# Verificar uso de memoria
MEM_USAGE=$(free | grep Mem | awk '{printf("%.1f"), $3/$2 * 100.0}')
if (( $(echo "$MEM_USAGE > 90" | bc -l) )); then
    echo "$(date): WARNING - Uso de memoria alto: $MEM_USAGE%" >> $LOG_FILE
fi

# Verificar espacio en disco
DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 90 ]; then
    echo "$(date): WARNING - Espacio en disco bajo: $DISK_USAGE%" >> $LOG_FILE
fi
```

#### Programar Tareas Automáticas

```bash
# Añadir a crontab
crontab -e

# Tareas programadas:
# Limpieza de logs - diario a las 2 AM
0 2 * * * /home/user/jetson-lpr/cleanup_logs.sh

# Backup de base de datos - diario a las 3 AM
0 3 * * * /home/user/jetson-lpr/backup_db.sh

# Monitoreo del sistema - cada 5 minutos
*/5 * * * * /home/user/jetson-lpr/monitor_system.sh

# Optimización de BD - semanal domingo a las 4 AM
0 4 * * 0 /usr/bin/mysql -u root -p'your_password' -e "OPTIMIZE TABLE lpr_detections, registered_vehicles; ANALYZE TABLE lpr_detections, registered_vehicles;"
```

### 📧 Alertas y Notificaciones

#### Script de Alertas

```python
#!/usr/bin/env python3
"""Sistema de alertas para LPR"""

import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import subprocess
import json

def send_alert(subject, message):
    """Enviar alerta por email"""
    try:
        # Configuración SMTP (personalizar según necesidades)
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        sender_email = "alerts@tusistema.com"
        sender_password = "tu_password"
        recipient_email = "admin@tusistema.com"
        
        msg = MimeMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = f"LPR Alert: {subject}"
        
        msg.attach(MimeText(message, 'plain'))
        
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        
        print(f"Alerta enviada: {subject}")
        
    except Exception as e:
        print(f"Error enviando alerta: {e}")

def check_system_health():
    """Verificar salud del sistema"""
    alerts = []
    
    # Verificar proceso LPR
    result = subprocess.run(['pgrep', '-f', 'realtime_lpr_fixed.py'], 
                          capture_output=True)
    if result.returncode != 0:
        alerts.append("Proceso LPR no está ejecutándose")
    
    # Verificar MySQL
    result = subprocess.run(['systemctl', 'is-active', 'mysql'], 
                          capture_output=True)
    if result.stdout.decode().strip() != 'active':
        alerts.append("MySQL no está activo")
    
    # Verificar uso de memoria
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if 'MemAvailable' in line:
                    available_kb = int(line.split()[1])
                    available_gb = available_kb / 1024 / 1024
                    if available_gb < 1:  # Menos de 1GB disponible
                        alerts.append(f"Memoria baja: {available_gb:.1f}GB disponible")
                    break
    except:
        pass
    
    # Enviar alertas si las hay
    if alerts:
        message = "\\n".join(alerts)
        send_alert("System Health Issues", message)

if __name__ == "__main__":
    check_system_health()
```

---

## 🎯 Conclusión

Este manual cubre todos los aspectos del Sistema LPR, desde la instalación
básica hasta la optimización avanzada y el mantenimiento en producción.

### 📞 Soporte

Para soporte adicional:

- **Documentación**: Ver `SOLUCIONES_LPR.md`
- **Logs**: Revisar `logs/` para errores específicos
- **Scripts**: Usar herramientas de diagnóstico incluidas

### 🚀 Próximos Pasos

1. **Instalación**: Ejecutar `install_jetson_complete.sh`
2. **Configuración**: Configurar base de datos con `setup_mysql_jetson.py`
3. **Verificación**: Ejecutar `verify_installation.py`
4. **Inicio**: Usar `quick_start.sh`
5. **Monitoreo**: Configurar alertas y tareas programadas

**¡El sistema está listo para detectar placas vehiculares en tiempo real!**

---

_Última actualización: 2025-11-11_
