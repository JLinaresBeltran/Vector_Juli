import os
import re
import json
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from supabase import create_client

# Cargar variables de entorno
load_dotenv()

# Configuración de Supabase
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")

if not supabase_url or not supabase_key:
    raise ValueError("❌ SUPABASE_URL y SUPABASE_SERVICE_KEY deben estar configurados en el archivo .env")

supabase = create_client(supabase_url, supabase_key)

# Configuración de OpenAI para embeddings
openai_api_key = os.environ.get("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("❌ OPENAI_API_KEY debe estar configurado en el archivo .env")

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

class RAC3Processor:
    def __init__(self, text: str, file_name: str = "rac3.txt", creator: str = "System"):
        self.raw_text = text
        self.processed_chunks = []
        self.file_name = file_name
        self.creator = creator
        
    def split_into_sections(self) -> List[Dict[str, Any]]:
        """Dividir el documento RAC 3 en secciones basadas en la numeración jerárquica."""
        sections = []
        
        # Patrón para identificar secciones numeradas del RAC 3
        section_pattern = r'^(3\.\d+(?:\.\d+)*)\s*\.?\s*([^.\n]+?)(?:\.|(?=\s+[A-Z])|\n|$)'
        
        # Identificar todas las secciones con sus posiciones
        matches = list(re.finditer(section_pattern, self.raw_text, re.MULTILINE))
        
        for i, match in enumerate(matches):
            section_num = match.group(1)
            raw_title = match.group(2).strip()
            
            # Limpiar el título
            section_title = self.clean_section_title(raw_title)
            
            # Si no encontramos un título limpio, usar el mapeo
            if not section_title or len(section_title) > 100:
                mapped_title = self.get_section_title(section_num)
                if mapped_title:
                    section_title = mapped_title
                else:
                    section_title = raw_title[:80] + "..." if len(raw_title) > 80 else raw_title
            
            section_start = match.start()
            
            # El contenido de la sección va hasta el inicio de la siguiente sección
            if i < len(matches) - 1:
                section_end = matches[i + 1].start()
            else:
                section_end = len(self.raw_text)
            
            section_text = self.raw_text[section_start:section_end].strip()
            
            # Determinar jerarquía para capítulos
            capitulo_info = self.get_chapter_info(section_num)
            
            # Extraer elementos estructurales del contenido
            structural_elements = self.extract_structural_elements(section_text)
            
            section = {
                "numero": section_num,
                "titulo": section_title,
                "texto": section_text,
                "capitulo_info": capitulo_info,
                "structural_elements": structural_elements,
                "start": section_start,
                "end": section_end,
                "start_line": self.get_line_number(section_start),
                "end_line": self.get_line_number(section_end)
            }
            
            sections.append(section)
        
        return sections
    
    def extract_structural_elements(self, text: str) -> Dict[str, Any]:
        """Extraer elementos estructurales del texto (literales, numerales, parágrafos)."""
        elements = {
            "literales": [],
            "numerales": [],
            "paragrafos": []
        }
        
        # Extraer literales (a), (b), (c), etc.
        literal_matches = re.findall(r'\(([a-z])\)', text)
        elements["literales"] = list(set(literal_matches))  # Eliminar duplicados
        
        # Extraer numerales 1., 2., 3., etc.
        numeral_matches = re.findall(r'^(\d+)\.\s', text, re.MULTILINE)
        elements["numerales"] = list(set(numeral_matches))
        
        # Extraer parágrafos
        paragrafo_matches = re.findall(r'PARÁGRAFO\.?\s*(\d*)', text, re.IGNORECASE)
        elements["paragrafos"] = [p for p in paragrafo_matches if p]
        
        return elements
    
    def clean_section_title(self, title: str) -> str:
        """Limpiar y acortar títulos de sección."""
        title = ' '.join(title.split())
        
        # Patrones para limpiar títulos
        title_end_patterns = [
            r'^([^.]+?)(?:\.\s*[A-Z][a-z])',
            r'^([A-ZÁÉÍÓÚÑ\s]+)(?=\s+[a-z])',
            r'^([^:]+?)(?=:\s*\([a-z]\))',
        ]
        
        for pattern in title_end_patterns:
            match = re.match(pattern, title)
            if match:
                extracted = match.group(1).strip()
                if len(extracted) >= 10:
                    title = extracted
                    break
        
        # Si es muy largo, cortar apropiadamente
        if len(title) > 100:
            for delimiter in ['. ', ', ', ' - ']:
                pos = title.find(delimiter)
                if 10 < pos < 100:
                    title = title[:pos].strip()
                    break
            
            if len(title) > 100:
                title = title[:80].strip() + "..."
        
        return title
    
    def get_chapter_info(self, section_num: str) -> Dict[str, str]:
        """Obtener información del capítulo basado en el número de sección."""
        section_parts = section_num.split('.')
        
        if len(section_parts) >= 2:
            main_section = f"{section_parts[0]}.{section_parts[1]}"
            
            chapter_mapping = {
                "3.10": {"numero": "3.10", "nombre": "TRANSPORTE AÉREO REGULAR DE PASAJEROS DERECHOS Y DEBERES DE LOS USUARIOS"},
                "3.11": {"numero": "3.11", "nombre": "REGISTRO, APROBACIÓN Y MODIFICACIÓN DE ITINERARIOS"},
                "3.12": {"numero": "3.12", "nombre": "VUELOS ADICIONALES"}
            }
            
            return chapter_mapping.get(main_section, {"numero": "3", "nombre": "ACTIVIDADES AÉREAS CIVILES"})
        
        return {"numero": "3", "nombre": "ACTIVIDADES AÉREAS CIVILES"}
    
    def get_section_title(self, section_num: str) -> str:
        """Obtener el título de una sección por su número."""
        section_titles = {
            "3.10": "TRANSPORTE AÉREO REGULAR DE PASAJEROS DERECHOS Y DEBERES DE LOS USUARIOS",
            "3.10.1": "RESERVAS Y TIQUETES",
            "3.10.2": "EJECUCIÓN DEL TRANSPORTE",
            "3.10.3": "Equipajes",
            "3.10.4": "SISTEMA DE ATENCIÓN AL USUARIO",
            "3.10.5": "INCUMPLIMIENTO",
            "3.10.6": "Disposiciones finales",
            "3.11": "REGISTRO, APROBACIÓN Y MODIFICACIÓN DE ITINERARIOS",
            "3.11.1": "Generalidades",
            "3.11.2": "Procedimiento para el registro, aprobación y modificación de Itinerarios",
            "3.11.2.1": "Registro y aprobación",
            "3.11.2.2": "Modificaciones",
            "3.11.3": "Publicación de Itinerarios",
            "3.12": "Vuelos adicionales"
        }
        
        return section_titles.get(section_num, "")
    
    def get_line_number(self, char_position: int) -> int:
        """Calcular el número de línea basado en la posición del carácter."""
        return self.raw_text[:char_position].count('\n') + 1
    
    def determine_subsection_elements(self, section: Dict[str, Any], chunk_content: str) -> Dict[str, Any]:
        """Determinar qué elementos específicos están en este chunk."""
        elements = {"literal_letra": None, "numeral_numero": None, "paragrafo_numero": None}
        
        # Buscar literal específico en el chunk
        literal_match = re.search(r'\(([a-z])\)', chunk_content)
        if literal_match:
            elements["literal_letra"] = literal_match.group(1)
        
        # Buscar numeral específico en el chunk
        numeral_match = re.search(r'^(\d+)\.\s', chunk_content, re.MULTILINE)
        if numeral_match:
            elements["numeral_numero"] = numeral_match.group(1)
        
        # Buscar parágrafo específico en el chunk
        paragrafo_match = re.search(r'PARÁGRAFO\.?\s*(\d*)', chunk_content, re.IGNORECASE)
        if paragrafo_match and paragrafo_match.group(1):
            elements["paragrafo_numero"] = paragrafo_match.group(1)
        elif paragrafo_match:
            elements["paragrafo_numero"] = "1"  # Parágrafo sin número
        
        return elements
    
    def clean_metadata_for_json(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Limpiar metadata para evitar errores de serialización JSON."""
        def clean_value(value):
            if value is None:
                return None
            if isinstance(value, str):
                return value.strip() if value.strip() else None
            if isinstance(value, dict):
                return {k: clean_value(v) for k, v in value.items()}
            return value
        
        cleaned = {}
        for key, value in metadata.items():
            cleaned_value = clean_value(value)
            if cleaned_value is not None:  # Solo incluir valores no None
                cleaned[key] = cleaned_value
        
        return cleaned
    
    def create_standardized_metadata(self, section: Dict[str, Any], chunk_content: str = None,
                                   chunk_start_line: int = None, chunk_end_line: int = None) -> Dict[str, Any]:
        """Crear metadata estandarizada compatible con LangChain."""
        
        now = datetime.now().isoformat()
        
        # Determinar elementos específicos del chunk si se proporciona
        specific_elements = {"literal_letra": None, "numeral_numero": None, "paragrafo_numero": None}
        if chunk_content:
            specific_elements = self.determine_subsection_elements(section, chunk_content)
        
        # Para subsecciones (ej: 3.11.2.1), usar como numeral_numero
        section_parts = section["numero"].split('.')
        subsection_number = None
        if len(section_parts) > 3:  # Es una subsección como 3.11.2.1
            subsection_number = '.'.join(section_parts[3:])
        
        # Asegurar que file_extension no sea vacío
        file_ext = os.path.splitext(self.file_name)[1]
        if not file_ext:
            file_ext = ".txt"
        
        metadata = {
            "loc": {
                "lines": {
                    "from": chunk_start_line or section["start_line"],
                    "to": chunk_end_line or section["end_line"]
                }
            },
            "line": chunk_start_line or section["start_line"],
            "source": "file",
            "creator": self.creator or "System",
            "version": "v1",
            "blobType": "text/plain",
            "id_legal": "RAC3",
            "file_name": self.file_name or "rac3.txt",
            "created_at": now,
            "last_modified": now,
            "file_extension": file_ext,
            "tipo_documento": "Reglamentos Aeronauticos",
            "articulo_numero": section["numero"],  # ← SECCIÓN del RAC (ej: "3.10.1")
            "capitulo_nombre": section["capitulo_info"]["nombre"] or "ACTIVIDADES AÉREAS CIVILES",
            "capitulo_numero": section["capitulo_info"]["numero"] or "3"
        }
        
        # Agregar campos opcionales solo si tienen valor
        if specific_elements["literal_letra"]:
            metadata["literal_letra"] = specific_elements["literal_letra"]
        
        if specific_elements["numeral_numero"] or subsection_number:
            metadata["numeral_numero"] = specific_elements["numeral_numero"] or subsection_number
        
        if specific_elements["paragrafo_numero"]:
            metadata["paragrafo_numero"] = specific_elements["paragrafo_numero"]
        
        # Limpiar metadata antes de retornar
        return self.clean_metadata_for_json(metadata)
    
    def create_chunks_from_sections(self, sections: List[Dict[str, Any]], 
                                  chunk_size: int = 800, 
                                  chunk_overlap: int = 150) -> List[Dict[str, Any]]:
        """Crear chunks a partir de las secciones identificadas."""
        
        for section in sections:
            section_text = section["texto"]
            
            # Decidir si dividir la sección
            if len(section_text) > chunk_size * 1.5:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", ". ", " "],
                    keep_separator=True
                )
                chunks = text_splitter.split_text(section_text)
                
                for i, chunk in enumerate(chunks):
                    # Calcular líneas aproximadas para cada chunk
                    lines_per_chunk = max(1, len(section_text.split('\n')) // len(chunks))
                    chunk_start_line = section["start_line"] + (i * lines_per_chunk)
                    chunk_end_line = chunk_start_line + chunk.count('\n') + 1
                    
                    metadata = self.create_standardized_metadata(
                        section, 
                        chunk_content=chunk,
                        chunk_start_line=chunk_start_line,
                        chunk_end_line=chunk_end_line
                    )
                    
                    self.processed_chunks.append({
                        "content": chunk,
                        "metadata": metadata
                    })
            else:
                # Sección completa como un solo chunk
                metadata = self.create_standardized_metadata(section, section_text)
                
                self.processed_chunks.append({
                    "content": section_text,
                    "metadata": metadata
                })
        
        return self.processed_chunks
    
    def process_document(self, chunk_size: int = 800, chunk_overlap: int = 150) -> List[Dict[str, Any]]:
        """Procesar el documento completo."""
        # Dividir en secciones
        sections = self.split_into_sections()
        print(f"Secciones RAC 3 identificadas: {len(sections)}")
        
        # Mostrar muestra de secciones encontradas
        if sections:
            print("\nMuestra de secciones encontradas:")
            for s in sections[:5]:
                print(f"  {s['numero']}: {s['titulo'][:60]}{'...' if len(s['titulo']) > 60 else ''}")
        
        # Crear chunks
        self.create_chunks_from_sections(sections, chunk_size, chunk_overlap)
        print(f"Chunks generados: {len(self.processed_chunks)}")
        
        return self.processed_chunks
    
    def vectorize_and_store(self, table_name: str = "transporte_aereo"):
        """Generar embeddings y almacenar en Supabase usando estructura compatible con LangChain."""
        total_chunks = len(self.processed_chunks)
        
        if total_chunks == 0:
            print("❌ No hay chunks para procesar")
            return
        
        print(f"📊 Iniciando vectorización de {total_chunks} chunks...")
        
        # Verificar conexión a Supabase
        try:
            # Test de conexión básica
            test_result = supabase.table(table_name).select("count", count="exact").limit(1).execute()
            print(f"✅ Conexión a tabla '{table_name}' exitosa")
        except Exception as e:
            print(f"❌ Error de conexión a Supabase: {e}")
            return
        
        successful_inserts = 0
        failed_inserts = 0
        
        for i, chunk in enumerate(self.processed_chunks, 1):
            try:
                # Validar contenido del chunk
                if not chunk.get("content") or not chunk.get("metadata"):
                    print(f"⚠️  Chunk {i} inválido: contenido o metadata faltante")
                    failed_inserts += 1
                    continue
                
                # Generar el embedding
                try:
                    embedding_vector = embeddings.embed_query(chunk["content"])
                except Exception as embed_error:
                    print(f"❌ Error generando embedding para chunk {i}: {embed_error}")
                    failed_inserts += 1
                    continue
                
                # Validar embedding
                if not embedding_vector or len(embedding_vector) == 0:
                    print(f"⚠️  Embedding vacío para chunk {i}")
                    failed_inserts += 1
                    continue
                
                # Preparar datos para insertar (estructura compatible con LangChain)
                data = {
                    "content": str(chunk["content"]),  # Asegurar que sea string
                    "metadata": chunk["metadata"],     # Ya limpiado por clean_metadata_for_json
                    "embedding": embedding_vector      # Vector de embeddings
                }
                
                # Almacenar en Supabase
                result = supabase.table(table_name).insert(data).execute()
                
                # Verificar que la inserción fue exitosa
                if result.data:
                    seccion_num = chunk['metadata'].get('articulo_numero', 'N/A')
                    successful_inserts += 1
                    if i % 10 == 0 or i == total_chunks:  # Mostrar progreso cada 10 chunks
                        print(f"📝 [{i}/{total_chunks}] RAC 3 - Sección {seccion_num} ✅")
                else:
                    print(f"⚠️  Chunk {i}: inserción sin datos de retorno")
                    failed_inserts += 1
                
            except Exception as e:
                print(f"❌ Error al almacenar chunk {i}/{total_chunks}: {str(e)}")
                # Mostrar detalles del chunk problemático para debug
                if i <= 3:  # Solo mostrar detalles de los primeros errores
                    print(f"   📋 Metadata del chunk problemático: {chunk.get('metadata', {}).keys()}")
                    print(f"   📄 Longitud del contenido: {len(chunk.get('content', ''))}")
                failed_inserts += 1
                continue
        
        # Resumen final
        print(f"\n🎯 Resumen del procesamiento RAC 3:")
        print(f"   ✅ Chunks exitosos: {successful_inserts}")
        print(f"   ❌ Chunks fallidos: {failed_inserts}")
        print(f"   📊 Total procesados: {total_chunks}")
        
        if successful_inserts > 0:
            print(f"✅ RAC 3 procesado. {successful_inserts} chunks almacenados en '{table_name}'.")
        else:
            print(f"❌ No se pudo almacenar ningún chunk. Revisar configuración de Supabase.")
        
        return {"successful": successful_inserts, "failed": failed_inserts, "total": total_chunks}

def verify_or_create_table(table_name: str = "transporte_aereo"):
    """Verificar que la tabla existe, si no, mostrar instrucciones para crearla."""
    try:
        # Intentar hacer una consulta simple para verificar que la tabla existe
        result = supabase.table(table_name).select("count", count="exact").limit(1).execute()
        print(f"✅ Tabla '{table_name}' verificada correctamente")
        return True
    except Exception as e:
        print(f"❌ Error accediendo a la tabla '{table_name}': {e}")
        print(f"\n🔧 Para crear la tabla '{table_name}', ejecuta este SQL en Supabase:")
        print(f"""
-- Habilitar la extensión vector
CREATE EXTENSION IF NOT EXISTS vector;

-- Crear tabla para LangChain
CREATE TABLE IF NOT EXISTS {table_name} (
  id bigserial primary key,
  content text,
  metadata jsonb,
  embedding vector(1536)  -- 1536 para OpenAI embeddings
);

-- Crear función de búsqueda para LangChain
CREATE OR REPLACE FUNCTION match_documents_{table_name} (
  query_embedding vector(1536),
  match_count int DEFAULT NULL,
  filter jsonb DEFAULT '{{}}'
) RETURNS TABLE (
  id bigint,
  content text,
  metadata jsonb,
  similarity float
)
LANGUAGE plpgsql
AS $
#variable_conflict use_column
BEGIN
  RETURN query
  SELECT
    id,
    content,
    metadata,
    1 - ({table_name}.embedding <=> query_embedding) as similarity
  FROM {table_name}
  WHERE metadata @> filter
  ORDER BY {table_name}.embedding <=> query_embedding
  LIMIT match_count;
END;
$;

-- Crear índice para mejorar performance
CREATE INDEX IF NOT EXISTS {table_name}_embedding_idx ON {table_name} 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Crear índice para metadata
CREATE INDEX IF NOT EXISTS {table_name}_metadata_idx ON {table_name} USING GIN (metadata);
        """)
        return False

def process_file(file_path: str, table_name: str = "transporte_aereo", creator: str = "System"):
    """Procesa un archivo de texto del RAC 3 y lo vectoriza en Supabase."""
    print(f"📄 Procesando RAC 3: {file_path}")
    
    # Verificar que la tabla existe
    if not verify_or_create_table(table_name):
        print("❌ No se puede continuar sin la tabla. Crear la tabla primero.")
        return
    
    # Verificar que el archivo existe
    if not os.path.exists(file_path):
        print(f"❌ Archivo no encontrado: {file_path}")
        return
    
    try:
        # Leer el archivo
        with open(file_path, "r", encoding="utf-8") as f:
            document_text = f.read()
        
        if not document_text.strip():
            print(f"❌ Archivo vacío: {file_path}")
            return
            
        print(f"📊 Archivo leído: {len(document_text)} caracteres")
        
    except Exception as e:
        print(f"❌ Error leyendo archivo {file_path}: {e}")
        return
    
    # Obtener nombre del archivo
    file_name = os.path.basename(file_path)
    
    # Crear el procesador
    processor = RAC3Processor(document_text, file_name, creator)
    
    # Procesar el documento
    chunks = processor.process_document(chunk_size=800, chunk_overlap=150)
    
    if not chunks:
        print("❌ No se generaron chunks del documento")
        return
    
    # Vectorizar y almacenar
    result = processor.vectorize_and_store(table_name)
    
    return result

def test_configuration():
    """Probar la configuración antes de procesar documentos."""
    print("🔧 Verificando configuración...")
    
    try:
        # Test OpenAI
        test_embedding = embeddings.embed_query("test")
        print(f"✅ OpenAI API: OK (dimensión: {len(test_embedding)})")
    except Exception as e:
        print(f"❌ OpenAI API: Error - {e}")
        return False
    
    try:
        # Test Supabase - intentar conectar
        result = supabase.auth.get_session()
        print("✅ Supabase conexión: OK")
    except Exception as e:
        print(f"⚠️  Supabase auth test: {e} (normal si usas service key)")
    
    return True

if __name__ == "__main__":
    import sys
    
    # Verificar configuración antes de procesar
    if not test_configuration():
        print("❌ Error en la configuración. Verificar variables de entorno.")
        sys.exit(1)
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        table_name = sys.argv[2] if len(sys.argv) > 2 else "transporte_aereo"
        creator = sys.argv[3] if len(sys.argv) > 3 else "System"
        process_file(file_path, table_name, creator)
    else:
        print("📋 Uso: python vectorizador_rac3.py <archivo> [tabla] [creator]")
        print("📋 Ejemplo: python vectorizador_rac3.py rac3.txt transporte_aereo Jhonathan")
        
        # Si no se proporciona un archivo, usar el archivo predeterminado si existe
        default_file = "rac3.txt"
        if os.path.exists(default_file):
            print(f"🔄 Usando archivo por defecto: {default_file}")
            process_file(default_file, "transporte_aereo", "System")
        else:
            print(f"❌ Archivo por defecto '{default_file}' no encontrado.")