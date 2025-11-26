#!/bin/bash
# =========================================================
# SCRIPT DE LIMPIEZA - ELIMINAR SCRIPTS INNECESARIOS
# =========================================================
# Este script elimina scripts de prueba y archivos innecesarios

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo "🧹 LIMPIEZA DE SCRIPTS INNECESARIOS"
echo "=========================================================="
echo ""
echo "Los siguientes archivos serán ELIMINADOS:"
echo "  - test_imports.py (script de prueba)"
echo "  - test_system_fixed.py (script de prueba)"
echo "  - test_results.json (resultados de pruebas)"
echo "  - parqueadero_simple.py (sistema diferente con GUI)"
echo "  - add_missing_data.py (utilidad de datos)"
echo ""
read -p "¿Continuar con la limpieza? (s/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Ss]$ ]]; then
    echo "❌ Limpieza cancelada"
    exit 0
fi

# Archivos a eliminar
FILES_TO_REMOVE=(
    "test_imports.py"
    "test_system_fixed.py"
    "test_results.json"
    "parqueadero_simple.py"
    "add_missing_data.py"
)

REMOVED_COUNT=0
for file in "${FILES_TO_REMOVE[@]}"; do
    if [ -f "$file" ]; then
        rm -f "$file"
        echo "✅ Eliminado: $file"
        ((REMOVED_COUNT++))
    else
        echo "⚠️  No encontrado: $file"
    fi
done

echo ""
echo "=========================================================="
echo "✅ LIMPIEZA COMPLETADA"
echo "=========================================================="
echo "Archivos eliminados: $REMOVED_COUNT"
echo ""

