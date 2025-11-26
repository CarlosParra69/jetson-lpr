#!/bin/bash
# =========================================================
# SCRIPT AUTOEJECUTABLE - SISTEMA LPR
# =========================================================
# Este script abre la carpeta jetson-lpr y ejecuta realtime_lpr_fixed.py
# Fecha: 2025-11-26

# Colores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=========================================================="
echo "🚗 SISTEMA LPR - INICIANDO"
echo "==========================================================${NC}"

# Obtener la ruta del script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${YELLOW}📁 Directorio: $SCRIPT_DIR${NC}"

# Verificar que existe el archivo principal
if [ ! -f "realtime_lpr_fixed.py" ]; then
    echo -e "${RED}❌ Error: No se encontró realtime_lpr_fixed.py${NC}"
    echo -e "${YELLOW}Presiona Enter para salir...${NC}"
    read
    exit 1
fi

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Error: Python3 no está instalado${NC}"
    echo -e "${YELLOW}Presiona Enter para salir...${NC}"
    read
    exit 1
fi

# Verificar dependencias básicas
echo -e "${YELLOW}🔍 Verificando dependencias...${NC}"
python3 -c "import cv2, ultralytics, easyocr" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}⚠️  Advertencia: Algunas dependencias pueden no estar instaladas${NC}"
    echo -e "${YELLOW}Continuando de todas formas...${NC}"
fi

# Crear directorios necesarios
mkdir -p logs
mkdir -p results
mkdir -p config

# Configurar sudoers para no pedir contraseña (opcional, solo si es necesario)
# Esto permite usar sudo sin contraseña para comandos de red específicos
SUDOERS_FILE="/etc/sudoers.d/jetson-lpr"
if [ ! -f "$SUDOERS_FILE" ]; then
    echo -e "${YELLOW}🔐 Configurando permisos sudo (puede pedir contraseña una vez)...${NC}"
    echo "proyecto" | sudo -S bash -c "echo '$USER ALL=(ALL) NOPASSWD: /sbin/ip, /sbin/ethtool' > $SUDOERS_FILE 2>/dev/null" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Permisos sudo configurados${NC}"
    else
        echo -e "${YELLOW}⚠️  No se pudieron configurar permisos sudo (se usará contraseña automática)${NC}"
    fi
fi

echo -e "${GREEN}✅ Todo listo${NC}"
echo -e "${GREEN}🚀 Iniciando sistema LPR...${NC}"
echo ""

# Ejecutar el sistema LPR
python3 realtime_lpr_fixed.py

# Si el script termina, mostrar mensaje
echo ""
echo -e "${YELLOW}=========================================================="
echo "🛑 Sistema LPR detenido"
echo "==========================================================${NC}"
echo -e "${YELLOW}Presiona Enter para cerrar...${NC}"
read

