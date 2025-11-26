#!/bin/bash
# =========================================================
# CONFIGURAR SUDO SIN CONTRASEÑA PARA JETSON-LPR
# =========================================================
# Este script configura sudoers para que no pida contraseña
# para comandos de red necesarios para el sistema LPR

echo "=========================================================="
echo "🔐 CONFIGURACIÓN DE PERMISOS SUDO"
echo "=========================================================="
echo ""
echo "Este script configurará sudo para que NO pida contraseña"
echo "para los comandos de red necesarios (ip, ethtool)"
echo ""
echo "Contraseña del sistema: proyecto"
echo ""

# Contraseña
PASSWORD="proyecto"
USERNAME=$(whoami)

# Crear archivo sudoers
SUDOERS_FILE="/etc/sudoers.d/jetson-lpr"

echo "Configurando permisos sudo..."
echo "$PASSWORD" | sudo -S bash -c "cat > $SUDOERS_FILE << 'EOF'
# Permisos para sistema LPR - No pedir contraseña para comandos de red
$USERNAME ALL=(ALL) NOPASSWD: /sbin/ip
$USERNAME ALL=(ALL) NOPASSWD: /sbin/ethtool
EOF
" 2>/dev/null

if [ $? -eq 0 ]; then
    # Verificar que el archivo se creó correctamente
    if [ -f "$SUDOERS_FILE" ]; then
        echo "$PASSWORD" | sudo -S chmod 0440 "$SUDOERS_FILE" 2>/dev/null
        echo "✅ Permisos sudo configurados correctamente"
        echo ""
        echo "Ahora puedes ejecutar el sistema LPR sin que pida contraseña"
    else
        echo "❌ Error: No se pudo crear el archivo sudoers"
        exit 1
    fi
else
    echo "❌ Error: No se pudo configurar sudo"
    echo "Asegúrate de que la contraseña 'proyecto' sea correcta"
    exit 1
fi

echo ""
echo "=========================================================="
echo "✅ CONFIGURACIÓN COMPLETADA"
echo "=========================================================="
echo ""

