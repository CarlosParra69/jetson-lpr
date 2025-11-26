#!/bin/bash
# =========================================================
# SCRIPT DE INSTALACIÓN - CREAR ACCESO DIRECTO EN DESKTOP
# =========================================================
# Este script copia el acceso directo al Desktop y lo hace ejecutable

DESKTOP_DIR="$HOME/Desktop"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=========================================================="
echo "📋 Instalando acceso directo en Desktop..."
echo "=========================================================="

# Copiar script al Desktop
if [ -f "$SCRIPT_DIR/iniciar_lpr.sh" ]; then
    cp "$SCRIPT_DIR/iniciar_lpr.sh" "$DESKTOP_DIR/iniciar_lpr.sh"
    chmod +x "$DESKTOP_DIR/iniciar_lpr.sh"
    echo "✅ Script copiado a Desktop: $DESKTOP_DIR/iniciar_lpr.sh"
else
    echo "❌ Error: No se encontró iniciar_lpr.sh"
    exit 1
fi

# Copiar archivo .desktop al Desktop
if [ -f "$SCRIPT_DIR/INICIAR_LPR.desktop" ]; then
    cp "$SCRIPT_DIR/INICIAR_LPR.desktop" "$DESKTOP_DIR/INICIAR_LPR.desktop"
    chmod +x "$DESKTOP_DIR/INICIAR_LPR.desktop"
    echo "✅ Acceso directo copiado a Desktop: $DESKTOP_DIR/INICIAR_LPR.desktop"
else
    echo "❌ Error: No se encontró INICIAR_LPR.desktop"
    exit 1
fi

# Configurar sudo (opcional)
echo ""
read -p "¿Configurar sudo para no pedir contraseña? (s/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Ss]$ ]]; then
    if [ -f "$SCRIPT_DIR/configurar_sudo.sh" ]; then
        chmod +x "$SCRIPT_DIR/configurar_sudo.sh"
        "$SCRIPT_DIR/configurar_sudo.sh"
    else
        echo "⚠️  Script configurar_sudo.sh no encontrado"
    fi
fi

# Actualizar base de datos de aplicaciones
if command -v update-desktop-database &> /dev/null; then
    update-desktop-database "$HOME/.local/share/applications" 2>/dev/null
    echo "✅ Base de datos de aplicaciones actualizada"
fi

echo ""
echo "=========================================================="
echo "✅ INSTALACIÓN COMPLETADA"
echo "=========================================================="
echo ""
echo "Ahora puedes:"
echo "1. Hacer doble clic en 'INICIAR_LPR.desktop' en el Desktop"
echo "2. O ejecutar: ~/Desktop/iniciar_lpr.sh"
echo ""
echo "Si el acceso directo no funciona, ejecuta:"
echo "  chmod +x ~/Desktop/INICIAR_LPR.desktop"
echo "  chmod +x ~/Desktop/iniciar_lpr.sh"
echo ""

