#!/bin/bash
# =========================================================
# SCRIPT DE INSTALACIÓN COMPLETA - JETSON ORIN NANO
# Sistema LPR (License Plate Recognition)
# Fecha: 2025-11-11
# =========================================================

set -e  # Salir en caso de error

echo "🚀 INSTALACIÓN COMPLETA SISTEMA LPR - JETSON ORIN NANO"
echo "=========================================================="

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para imprimir con color
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Verificar que estamos en Jetson
if ! grep -q "NVIDIA" /proc/cpuinfo; then
    print_warning "Este script está optimizado para Jetson Orin Nano"
    read -p "¿Continuar de todos modos? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Actualizar sistema
print_info "Actualizando sistema Jetson..."
sudo apt update && sudo apt upgrade -y

# Instalar dependencias del sistema
print_info "Instalando dependencias del sistema..."
sudo apt install -y \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    curl \
    vim \
    htop \
    tree \
    mysql-server \
    mysql-client \
    libmysqlclient-dev \
    pkg-config \
    libgtk2.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libatlas-base-dev \
    gfortran \
    libtbb2 \
    libtbb-dev \
    libdc1394-22-dev

# Configurar MySQL
print_info "Configurando MySQL..."
sudo systemctl enable mysql
sudo systemctl start mysql
sudo mysql_secure_installation -y

# Instalar dependencias Python optimizadas para Jetson
print_info "Instalando dependencias Python optimizadas..."

# PyTorch con soporte CUDA para Jetson
print_info "Instalando PyTorch con soporte CUDA..."
pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Dependencias principales para LPR
print_info "Instalando dependencias principales..."
pip3 install --no-cache-dir \
    ultralytics \
    opencv-python \
    opencv-contrib-python \
    numpy \
    scipy \
    pandas \
    Pillow \
    easyocr \
    mysql-connector-python \
    psutil \
    tqdm \
    colorama \
    requests

# Dependencias adicionales para rendimiento
print_info "Instalando dependencias adicionales..."
pip3 install --no-cache-dir \
    filterpy \
    matplotlib \
    seaborn \
    imageio \
    scikit-image \
    pytesseract

# Configurar variables de entorno
print_info "Configurando variables de entorno..."
cat << 'EOF' >> ~/.bashrc

# ===== CONFIGURACIÓN SISTEMA LPR =====
# Variables de entorno optimizadas para Jetson
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"
export PYTHONUNBUFFERED=1

# Rutas del proyecto
export LPR_HOME="$HOME/jetson-lpr"
export PYTHONPATH="$LPR_HOME:$PYTHONPATH"

# Aliases útiles
alias lpr-start="cd $LPR_HOME && python3 realtime_lpr_fixed.py --headless"
alias lpr-mysql="cd $LPR_HOME && python3 stream/database/setup_mysql_jetson.py"
alias lpr-logs="tail -f $LPR_HOME/logs/realtime_lpr_*.log"
alias lpr-status="ps aux | grep realtime_lpr_fixed"

# Configuración MySQL
export MYSQL_HOST=localhost
export MYSQL_DATABASE=parqueadero_jetson
export MYSQL_USER=lpr_user
export MYSQL_PASSWORD=lpr_password
EOF

source ~/.bashrc

# Crear directorios del proyecto
print_info "Creando estructura de directorios..."
mkdir -p ~/jetson-lpr
mkdir -p ~/jetson-lpr/{logs,results,config,models}
mkdir -p ~/jetson-lpr/stream/{database,config,utils}

# Copiar archivos del proyecto (si existen)
print_info "Verificando archivos del proyecto..."
cd ~/jetson-lpr

# Descargar modelos YOLO si no existen
if [ ! -f "license_plate_detector.pt" ]; then
    print_info "Descargando modelo de detección de placas..."
    wget -q https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O yolov8n.pt
    print_status "Modelo YOLOv8n descargado"
fi

# Configurar permisos
print_info "Configurando permisos..."
chmod +x *.py
chmod +x *.sh

# Configurar red para Jetson
print_info "Configurando red Jetson..."
sudo ip addr flush dev enP8p1s0 2>/dev/null || true
sudo ip addr add 192.168.1.100/24 dev enP8p1s0 2>/dev/null || true

# Optimizar configuración del sistema
print_info "Aplicando optimizaciones del sistema..."

# Configurar swappiness
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf

# Configurar rendimiento de la GPU
echo 'for f in /sys/bus/pci/devices/0000:00:10.0/power/control; do echo on > $f; done 2>/dev/null || true' | sudo tee -a /etc/rc.local

# Crear script de verificación
print_info "Creando script de verificación..."
cat > ~/jetson-lpr/verify_installation.py << 'EOF'
#!/usr/bin/env python3
"""
Script de verificación de instalación del sistema LPR
"""

import sys
import platform
import subprocess

def check_python():
    """Verificar versión de Python"""
    version = sys.version_info
    print(f"🐍 Python: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ requerido")
        return False
    
    print("✅ Versión de Python compatible")
    return True

def check_cuda():
    """Verificar soporte CUDA"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🖥️ GPU: {gpu_name}")
            print(f"💾 VRAM: {gpu_memory:.1f} GB")
            print("✅ CUDA disponible")
            return True
        else:
            print("⚠️  CUDA no disponible")
            return False
    except:
        print("❌ PyTorch no disponible")
        return False

def check_dependencies():
    """Verificar dependencias principales"""
    deps = {
        'cv2': 'OpenCV',
        'numpy': 'NumPy', 
        'ultralytics': 'Ultralytics',
        'easyocr': 'EasyOCR',
        'mysql': 'MySQL Connector',
        'torch': 'PyTorch'
    }
    
    missing = []
    
    for module, name in deps.items():
        try:
            if module == 'cv2':
                import cv2
                print(f"✅ {name}: {cv2.__version__}")
            elif module == 'mysql':
                import mysql.connector
                print(f"✅ {name}: Disponible")
            else:
                __import__(module)
                print(f"✅ {name}: OK")
        except ImportError:
            print(f"❌ {name}: NO DISPONIBLE")
            missing.append(name)
    
    return len(missing) == 0

def check_system():
    """Verificar información del sistema"""
    print(f"💻 Sistema: {platform.system()} {platform.release()}")
    print(f"🔧 Arquitectura: {platform.machine()}")
    print(f"🖥️ Procesador: {platform.processor()}")
    
    # Información de memoria
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if 'MemTotal' in line:
                    mem_kb = int(line.split()[1])
                    mem_gb = mem_kb / 1024 / 1024
                    print(f"💾 RAM: {mem_gb:.1f} GB")
                    break
    except:
        pass
    
    # Información de almacenamiento
    try:
        result = subprocess.run(['df', '-h', '/'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            fields = lines[1].split()
            if len(fields) >= 4:
                total = fields[1]
                used = fields[2]
                print(f"💿 Espacio: {used} de {total} usado")
    except:
        pass

def main():
    """Función principal"""
    print("🔍 VERIFICACIÓN DE INSTALACIÓN SISTEMA LPR")
    print("=" * 50)
    
    checks = [
        ("Sistema", check_system),
        ("Python", check_python),
        ("CUDA/GPU", check_cuda),
        ("Dependencias", check_dependencies)
    ]
    
    results = []
    
    for name, check_func in checks:
        print(f"\n📋 Verificando {name}...")
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Error verificando {name}: {e}")
            results.append((name, False))
    
    # Resumen final
    print("\n" + "=" * 50)
    print("📊 RESUMEN DE VERIFICACIÓN:")
    
    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {name}: {status}")
        if result:
            passed += 1
    
    total = len(results)
    print(f"\n🎯 Resultado: {passed}/{total} verificaciones pasaron")
    
    if passed == total:
        print("🎉 ¡INSTALACIÓN COMPLETA Y EXITOSA!")
        print("💡 Para iniciar el sistema: lpr-start")
    else:
        print("⚠️  INSTALACIÓN INCOMPLETA")
        print("💡 Revisar dependencias faltantes")

if __name__ == "__main__":
    main()
EOF

chmod +x ~/jetson-lpr/verify_installation.py

# Crear script de inicio rápido
print_info "Creando script de inicio rápido..."
cat > ~/jetson-lpr/quick_start.sh << 'EOF'
#!/bin/bash
# Inicio rápido del sistema LPR

cd ~/jetson-lpr

echo "🚀 INICIANDO SISTEMA LPR..."
echo "Verificando dependencias..."

python3 verify_installation.py

echo ""
echo "📊 Estado de servicios:"

# Verificar MySQL
if systemctl is-active --quiet mysql; then
    echo "✅ MySQL: Activo"
else
    echo "⚠️  MySQL: Inactivo"
    echo "   💡 Iniciar con: sudo systemctl start mysql"
fi

# Verificar red Jetson
if ip addr show enP8p1s0 | grep -q "192.168.1.100"; then
    echo "✅ Red Jetson: Configurada"
else
    echo "⚠️  Red Jetson: No configurada"
    echo "   💡 Configurar con: sudo ip addr add 192.168.1.100/24 dev enP8p1s0"
fi

echo ""
echo "🎯 Iniciando sistema LPR en modo headless..."
python3 realtime_lpr_fixed.py \
    --headless \
    --ai-every 2 \
    --cooldown 0.5 \
    --confidence 0.30 \
    --motion \
    --display-scale 0.50

echo "🛑 Sistema LPR detenido"
EOF

chmod +x ~/jetson-lpr/quick_start.sh

# Ejecutar verificación inicial
print_info "Ejecutando verificación inicial..."
python3 ~/jetson-lpr/verify_installation.py

# Configurar base de datos MySQL (opcional)
print_info "¿Configurar base de datos MySQL para Jetson? (recomendado)"
read -p "Esto creará la base de datos parqueadero_jetson (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Configurando MySQL..."
    cd ~/jetson-lpr
    python3 stream/database/setup_mysql_jetson.py
fi

# Resumen final
echo ""
echo "🎉 ¡INSTALACIÓN COMPLETADA!"
echo "=========================================================="
print_status "Jetson Orin Nano configurado para Sistema LPR"
echo ""
print_info "Archivos importantes:"
echo "   📁 ~/jetson-lpr/ - Directorio principal"
echo "   🚀 ~/jetson-lpr/quick_start.sh - Inicio rápido"
echo "   🔍 ~/jetson-lpr/verify_installation.py - Verificación"
echo "   📊 ~/jetson-lpr/logs/ - Logs del sistema"
echo "   💾 ~/jetson-lpr/results/ - Resultados de detecciones"
echo ""
print_info "Comandos útiles:"
echo "   lpr-start - Iniciar sistema LPR"
echo "   lpr-mysql - Configurar MySQL"
echo "   lpr-logs - Ver logs en tiempo real"
echo "   lpr-status - Verificar estado"
echo ""
print_warning "⚠️  IMPORTANTE:"
echo "   - Reiniciar la terminal para cargar las variables de entorno"
echo "   - Ejecutar 'lpr-mysql' si no se configuró MySQL durante la instalación"
echo "   - Verificar que la cámara esté conectada antes de iniciar"
echo ""
print_info "Para empezar: cd ~/jetson-lpr && ./quick_start.sh"
echo "=========================================================="