import os

def crear_archivos_prueba():
    """Crear 52 archivos .txt numerados del 1 al 52 en la carpeta sentencias"""
    
    # Definir la ruta de la carpeta
    carpeta = "sentencias"
    
    # Verificar que la carpeta existe
    if not os.path.exists(carpeta):
        print(f"❌ Error: La carpeta {carpeta} no existe")
        print(f"💡 Por favor, asegúrate de que la carpeta existe antes de ejecutar el script")
        print(f"📁 Ruta esperada: {os.path.abspath(carpeta)}")
        return False
    else:
        print(f"✅ Carpeta encontrada: {carpeta}")
    
    # Contenido vacío - el usuario pegará el contenido manualmente
    contenido_base = ""
    
    # Crear los 52 archivos
    archivos_creados = 0
    archivos_existentes = 0
    
    for numero in range(1, 53):  # Del 1 al 52
        nombre_archivo = f"{numero}.txt"
        ruta_completa = os.path.join(carpeta, nombre_archivo)
        
        # Verificar si el archivo ya existe
        if os.path.exists(ruta_completa):
            archivos_existentes += 1
            print(f"⚠️  Archivo ya existe: {nombre_archivo}")
        else:
            # Crear el archivo vacío
            try:
                with open(ruta_completa, 'w', encoding='utf-8') as archivo:
                    archivo.write(contenido_base)
                archivos_creados += 1
                print(f"✅ Creado: {nombre_archivo} (vacío - listo para contenido manual)")
            except Exception as e:
                print(f"❌ Error creando {nombre_archivo}: {e}")
    
    # Resumen final
    print(f"\n{'='*60}")
    print(f"📊 RESUMEN DE CREACIÓN DE ARCHIVOS")
    print(f"{'='*60}")
    print(f"✅ Archivos creados: {archivos_creados}")
    print(f"⚠️  Archivos que ya existían: {archivos_existentes}")
    print(f"🗂️  Total de archivos en carpeta: {archivos_creados + archivos_existentes}")
    print(f"📁 Ubicación: {os.path.abspath(carpeta)}")
    
    # Verificar contenido de la carpeta
    archivos_en_carpeta = [f for f in os.listdir(carpeta) if f.endswith('.txt')]
    print(f"📋 Archivos .txt encontrados: {len(archivos_en_carpeta)}")
    
    if len(archivos_en_carpeta) == 52:
        print("🎉 ¡Todos los 52 archivos están listos!")
        return True
    else:
        print(f"⚠️  Se esperaban 52 archivos, pero se encontraron {len(archivos_en_carpeta)}")
        return True

def limpiar_archivos():
    """Función opcional para eliminar todos los archivos de prueba"""
    carpeta = "sentencias"
    
    if not os.path.exists(carpeta):
        print(f"❌ La carpeta {carpeta} no existe")
        print(f"📁 Ruta esperada: {os.path.abspath(carpeta)}")
        return
    
    archivos = [f for f in os.listdir(carpeta) if f.endswith('.txt')]
    
    if not archivos:
        print("📭 No hay archivos .txt para eliminar")
        return
    
    respuesta = input(f"¿Estás seguro de que quieres eliminar {len(archivos)} archivos? (y/N): ")
    
    if respuesta.lower() in ['y', 'yes', 'sí', 'si']:
        eliminados = 0
        for archivo in archivos:
            try:
                os.remove(os.path.join(carpeta, archivo))
                eliminados += 1
                print(f"🗑️  Eliminado: {archivo}")
            except Exception as e:
                print(f"❌ Error eliminando {archivo}: {e}")
        
        print(f"\n✅ {eliminados} archivos eliminados exitosamente")
    else:
        print("❌ Operación cancelada")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--clean":
        limpiar_archivos()
    else:
        print("🚀 Creando 52 archivos vacíos para sentencias...")
        print("📁 Carpeta destino: Vector_Juli/sentencias")
        print("📋 Archivos: 1.txt hasta 52.txt (vacíos)")
        print("✋ Cada archivo estará listo para contenido manual")
        print()
        
        resultado = crear_archivos_prueba()
        
        if resultado:
            print("\n💡 Comandos útiles:")
            print("   python crear_archivos.py          # Crear archivos")
            print("   python crear_archivos.py --clean  # Eliminar archivos")
            print("   python sentencias_processor.py    # Procesar archivos")
        else:
            print("\n❌ No se pudieron crear los archivos. Verifica que la carpeta Vector_Juli/sentencias exista.")