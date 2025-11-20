#!/bin/bash
# Script para configurar base de datos MySQL para Sistema LPR
# Crea la base de datos, usuario y tablas necesarias

set -e  # Salir si hay error

echo "💾 Configurando base de datos MySQL para Sistema LPR"
echo "====================================================="
echo ""

# Configuración por defecto
DB_NAME="parqueadero_jetson"
DB_USER="lpr_user"
DB_PASSWORD="lpr_password"
DB_HOST="localhost"
DB_PORT="3306"

# Leer configuración desde archivo si existe
if [ -f "../config/default_config.json" ]; then
    echo "📄 Leyendo configuración desde config/default_config.json..."
    
    # Usar Python para parsear JSON (más simple que jq que puede no estar instalado)
    DB_NAME=$(python3 -c "
import json
with open('../config/default_config.json') as f:
    config = json.load(f)
    print(config['database'].get('database', 'parqueadero_jetson'))
" 2>/dev/null || echo "parqueadero_jetson")
    
    DB_USER=$(python3 -c "
import json
with open('../config/default_config.json') as f:
    config = json.load(f)
    print(config['database'].get('user', 'lpr_user'))
" 2>/dev/null || echo "lpr_user")
    
    DB_PASSWORD=$(python3 -c "
import json
with open('../config/default_config.json') as f:
    config = json.load(f)
    print(config['database'].get('password', 'lpr_password'))
" 2>/dev/null || echo "lpr_password")
    
    echo "✅ Configuración leída"
fi

echo "Configuración de base de datos:"
echo "  Base de datos: $DB_NAME"
echo "  Usuario: $DB_USER"
echo "  Host: $DB_HOST"
echo "  Puerto: $DB_PORT"
echo ""

# Verificar que MySQL está corriendo
if ! systemctl is-active --quiet mysql; then
    echo "🔄 Iniciando MySQL..."
    sudo systemctl start mysql
    sleep 2
fi

if ! systemctl is-active --quiet mysql; then
    echo "❌ Error: MySQL no está corriendo"
    echo "Inicia MySQL manualmente con: sudo systemctl start mysql"
    exit 1
fi

echo "✅ MySQL está corriendo"
echo ""

# Pedir contraseña de root de MySQL
echo "🔐 Necesitamos la contraseña de root de MySQL"
echo "   (Si no tienes contraseña, presiona Enter)"
read -sp "Contraseña de root: " ROOT_PASSWORD
echo ""

# Intentar conectar sin contraseña primero
if [ -z "$ROOT_PASSWORD" ]; then
    MYSQL_CMD="sudo mysql"
else
    MYSQL_CMD="mysql -u root -p'$ROOT_PASSWORD'"
fi

echo "🔧 Creando base de datos y usuario..."
echo ""

# Crear base de datos
$MYSQL_CMD <<EOF
CREATE DATABASE IF NOT EXISTS $DB_NAME CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE $DB_NAME;

-- Crear usuario
DROP USER IF EXISTS '$DB_USER'@'localhost';
CREATE USER '$DB_USER'@'localhost' IDENTIFIED BY '$DB_PASSWORD';
GRANT ALL PRIVILEGES ON $DB_NAME.* TO '$DB_USER'@'localhost';
FLUSH PRIVILEGES;

SELECT 'Base de datos y usuario creados correctamente' AS status;
EOF

if [ $? -ne 0 ]; then
    echo "❌ Error creando base de datos o usuario"
    echo "Intenta ejecutar manualmente los comandos SQL"
    exit 1
fi

echo ""
echo "✅ Base de datos y usuario creados"
echo ""

# Conectar con el nuevo usuario para crear tablas
echo "🔧 Creando tablas..."
echo ""

mysql -u "$DB_USER" -p"$DB_PASSWORD" -h "$DB_HOST" -P "$DB_PORT" "$DB_NAME" <<EOF
-- Tabla de detecciones
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
    entry_type ENUM('entrada', 'salida') DEFAULT 'entrada',
    
    INDEX idx_timestamp (timestamp),
    INDEX idx_plate (plate_text),
    INDEX idx_location (camera_location),
    INDEX idx_processed (processed),
    INDEX idx_entry_type (entry_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Tabla de vehículos registrados
CREATE TABLE IF NOT EXISTS registered_vehicles (
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Tabla de log de acceso
CREATE TABLE IF NOT EXISTS access_log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    detection_id INT,
    plate_number VARCHAR(10) NOT NULL,
    access_granted BOOLEAN DEFAULT FALSE,
    access_reason VARCHAR(100),
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    camera_location VARCHAR(100),
    
    FOREIGN KEY (detection_id) REFERENCES lpr_detections(id) ON DELETE SET NULL,
    INDEX idx_plate (plate_number),
    INDEX idx_timestamp (timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Insertar algunos vehículos de ejemplo
INSERT IGNORE INTO registered_vehicles (plate_number, owner_name, authorized) VALUES
('ABC123', 'Ejemplo Vehículo 1', TRUE),
('XYZ789', 'Ejemplo Vehículo 2', TRUE),
('CD1234', 'Ejemplo Diplomático', TRUE);

SELECT 'Tablas creadas correctamente' AS status;
SELECT COUNT(*) AS total_vehicles FROM registered_vehicles;
EOF

if [ $? -ne 0 ]; then
    echo "❌ Error creando tablas"
    exit 1
fi

echo ""
echo "✅ Tablas creadas correctamente"
echo ""

# Verificar conexión
echo "🔍 Verificando conexión..."
mysql -u "$DB_USER" -p"$DB_PASSWORD" -h "$DB_HOST" -P "$DB_PORT" "$DB_NAME" -e "SELECT 'Conexión exitosa' AS status;" 2>/dev/null

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Configuración de base de datos completada!"
    echo ""
    echo "📊 Resumen:"
    echo "  Base de datos: $DB_NAME"
    echo "  Usuario: $DB_USER"
    echo "  Contraseña: $DB_PASSWORD"
    echo "  Host: $DB_HOST:$DB_PORT"
    echo ""
    echo "🧪 Para probar la conexión:"
    echo "  mysql -u $DB_USER -p'$DB_PASSWORD' $DB_NAME"
    echo ""
else
    echo "⚠️  No se pudo verificar la conexión, pero las tablas deberían estar creadas"
fi

