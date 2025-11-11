#!/bin/bash
# =========================================================
# SCRIPT DE INICIO PARA JETSON ORIN NANO - SISTEMA LPR
# =========================================================
# Versión optimizada para jetson-lpr
# Fecha: 2025-11-11

echo "=========================================================="
echo "🚗 SISTEMA LPR - JETSON ORIN NANO"
echo "=========================================================="

# Configurar entorno
export PYTHONPATH=/usr/local/lib/python3.10/site-packages
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Verificar GPU NVIDIA
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU NVIDIA detectada:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "⚠️  GPU NVIDIA no detectada - usando CPU"
fi

# Verificar Python y dependencias
echo "🐍 Verificando Python..."
python3 --version

echo "📦 Verificando dependencias críticas..."
python3 -c "
import sys
try:
    import cv2; print('✅ OpenCV:', cv2.__version__)
except: print('❌ OpenCV no disponible')
try:
    import torch; print('✅ PyTorch:', torch.__version__, '- CUDA:', torch.cuda.is_available())
except: print('❌ PyTorch no disponible')
try:
    import ultralytics; print('✅ Ultralytics disponible')
except: print('❌ Ultralytics no disponible')
try:
    import easyocr; print('✅ EasyOCR disponible')
except: print('❌ EasyOCR no disponible')
"

# Verificar modelos YOLO
echo "🤖 Verificando modelos..."
if [ -f "license_plate_detector.pt" ]; then
    echo "✅ license_plate_detector.pt encontrado"
elif [ -f "yolo11n.pt" ]; then
    echo "✅ yolo11n.pt encontrado"
elif [ -f "yolov8n.pt" ]; then
    echo "✅ yolov8n.pt encontrado"
else
    echo "❌ No se encontraron modelos YOLO (.pt)"
fi

# Verificar configuración
echo "⚙️ Verificando configuración..."
if [ -f "config/ptz_config.json" ]; then
    echo "✅ Configuración encontrada"
else
    echo "⚠️  Usando configuración por defecto"
fi

# Crear directorios necesarios
echo "📁 Creando directorios..."
mkdir -p logs
mkdir -p results
mkdir -p config

# Configurar interfaz de red
echo "🌐 Configurando red..."
INTERFACE="enP8p1s0"
JETSON_IP="192.168.1.100"

if ip link show $INTERFACE &> /dev/null; then
    echo "✅ Interfaz $INTERFACE detectada"
    sudo ip addr flush dev $INTERFACE 2>/dev/null || true
    sudo ip addr add $JETSON_IP/24 dev $INTERFACE 2>/dev/null || true
    sudo ethtool -s $INTERFACE speed 100 duplex full autoneg off 2>/dev/null || true
    echo "✅ Red configurada: $INTERFACE -> $JETSON_IP"
else
    echo "⚠️  Interfaz $INTERFACE no detectada"
fi

# Configurar permisos
echo "🔐 Configurando permisos..."
chmod +x realtime_lpr_fixed.py

echo "=========================================================="
echo "🚀 INICIANDO SISTEMA LPR EN MODO HEADLESS"
echo "=========================================================="

# Ejecutar con parámetros optimizados para Jetson
python3 realtime_lpr_fixed.py \
    --headless \
    --ai-every 2 \
    --cooldown 0.5 \
    --confidence 0.30 \
    --motion \
    --display-scale 0.50

echo "=========================================================="
echo "🛑 Sistema LPR detenido"
echo "=========================================================="