# 🚗 Instalación y Uso del Sistema LPR

## 📋 Instrucciones de Instalación

### 1. Crear Acceso Directo en el Desktop

Ejecuta el script de instalación:

```bash
cd ~/Desktop/Jetson/jetson-lpr
chmod +x INSTALAR_DESKTOP.sh
./INSTALAR_DESKTOP.sh
```

Esto creará dos archivos en tu Desktop:
- **INICIAR_LPR.desktop** - Acceso directo clickeable
- **iniciar_lpr.sh** - Script ejecutable

### 2. Hacer el Acceso Directo Ejecutable

Si el acceso directo no funciona al hacer doble clic, ejecuta:

```bash
chmod +x ~/Desktop/INICIAR_LPR.desktop
chmod +x ~/Desktop/iniciar_lpr.sh
```

### 3. Usar el Sistema

**Opción A: Doble clic en el Desktop**
- Haz doble clic en `INICIAR_LPR.desktop`
- Se abrirá una terminal y ejecutará el sistema

**Opción B: Desde terminal**
```bash
~/Desktop/iniciar_lpr.sh
```

**Opción C: Desde la carpeta del proyecto**
```bash
cd ~/Desktop/Jetson/jetson-lpr
python3 realtime_lpr_fixed.py
```

## 🧹 Limpiar Scripts Innecesarios

Para eliminar scripts de prueba y archivos innecesarios:

```bash
cd ~/Desktop/Jetson/jetson-lpr
chmod +x LIMPIAR_SCRIPTS.sh
./LIMPIAR_SCRIPTS.sh
```

Esto eliminará:
- `test_imports.py` - Script de prueba de imports
- `test_system_fixed.py` - Script de prueba del sistema
- `test_results.json` - Resultados de pruebas
- `parqueadero_simple.py` - Sistema diferente con GUI
- `add_missing_data.py` - Utilidad de datos

## 📁 Estructura del Proyecto

```
jetson-lpr/
├── realtime_lpr_fixed.py    # ⭐ Script principal del sistema
├── ptz_controller.py         # Controlador PTZ
├── plate_validator.py        # Validador de placas
├── util.py                  # Utilidades
├── visualize.py             # Visualización
├── config/                  # Configuración
├── stream/                  # Módulos de streaming
├── logs/                    # Logs del sistema
├── results/                 # Resultados
├── *.pt                     # Modelos YOLO
├── iniciar_lpr.sh          # Script ejecutable
├── INICIAR_LPR.desktop     # Acceso directo
└── requirements.txt         # Dependencias
```

## ⚙️ Configuración

El sistema usa el archivo de configuración:
- `config/ptz_config.json`

Puedes editar este archivo para ajustar:
- IP de la cámara
- Configuración de red
- Parámetros de detección
- Base de datos

## 🛑 Detener el Sistema

Para detener el sistema:
- Presiona `Ctrl + C` en la terminal
- O cierra la ventana de terminal

## 📝 Notas

- El sistema requiere Python 3.8+
- Asegúrate de tener todas las dependencias instaladas (`requirements.txt`)
- Los modelos YOLO deben estar en la carpeta principal
- El sistema crea automáticamente las carpetas `logs/` y `results/`

