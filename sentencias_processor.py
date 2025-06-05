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

class SentenciaSICProcessor:
    def __init__(self, text: str, file_name: str = "sentencia", creator: str = "System"):
        self.raw_text = text
        self.processed_chunks = []
        self.file_name = file_name
        self.creator = creator
        self.sentencia_data = {}
        
    def extract_metadata(self):
        """Extraer metadata básica de la sentencia."""
        # Patrones para extraer información
        patterns = {
            'radicado': r'(?:radicado|No\.|Número de radicado):\s*(\d+-\d+)',
            'fecha': r'Fecha.*?:\s*(\d{1,2}/\d{1,2}/\d{4})',
            'juez': r'Juez:\s*([^\n]+)',
            'demandante': r'Demandante:\s*([^\n]+)',
            'demandado': r'Demandado[ao]?:\s*([^\n]+)',
            'tipo_accion': r'Tipo de acción:\s*([^\n]+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, self.raw_text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                # Convertir fecha al formato ISO
                if key == 'fecha':
                    try:
                        # Intentar parsear fecha en formato DD/MM/YYYY
                        date_obj = datetime.strptime(value, "%d/%m/%Y")
                        value = date_obj.strftime("%Y-%m-%d")
                    except:
                        # Si falla, intentar otros formatos comunes
                        for fmt in ["%Y/%m/%d", "%d-%m-%Y", "%Y-%m-%d"]:
                            try:
                                date_obj = datetime.strptime(value, fmt)
                                value = date_obj.strftime("%Y-%m-%d")
                                break
                            except:
                                continue
                self.sentencia_data[key] = value
    
    def split_into_sections(self) -> List[Dict[str, Any]]:
        """Dividir sentencia en secciones temáticas."""
        sections = []
        
        # Definir secciones y sus patrones
        section_patterns = {
            'tema_juridico': r'Tema Jurídico(.*?)(?=Subtemas|Ratio|Datos|Hechos|$)',
            'subtemas': r'Subtemas[:\s]+(.*?)(?=Ratio|Datos|Hechos|$)',
            'ratio_decidendi': r'Ratio Decidendi(.*?)(?=Datos|Hechos|Fundamentos|$)',
            'datos_basicos': r'Datos Básicos(.*?)(?=Hechos|Fundamentos|$)',
            'hechos': r'Hechos Clave(.*?)(?=Fundamentos|Consideraciones|$)',
            'fundamentos': r'Fundamentos Jurídicos(.*?)(?=Consideraciones|Decisión|$)',
            'consideraciones': r'Consideraciones Principales(.*?)(?=Decisión|Precedente|$)',
            'decision': r'Decisión(.*?)(?=Precedente|$)',
            'precedente': r'Precedente Establecido(.*?)$'
        }
        
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, self.raw_text, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                if content and len(content) > 50:  # Ignorar secciones muy cortas
                    
                    # Calcular líneas
                    section_start = match.start()
                    section_end = match.end()
                    start_line = self.get_line_number(section_start)
                    end_line = self.get_line_number(section_end)
                    
                    sections.append({
                        'tipo': section_name,
                        'contenido': content,
                        'es_importante': section_name in ['ratio_decidendi', 'decision', 'precedente'],
                        'start': section_start,
                        'end': section_end,
                        'start_line': start_line,
                        'end_line': end_line
                    })
        
        # Si no se encontraron secciones, tratar el documento completo como una sección
        if not sections:
            sections.append({
                'tipo': 'documento_completo',
                'contenido': self.raw_text,
                'es_importante': True,
                'start': 0,
                'end': len(self.raw_text),
                'start_line': 1,
                'end_line': self.get_line_number(len(self.raw_text))
            })
        
        return sections
    
    def get_line_number(self, char_position: int) -> int:
        """Calcular el número de línea basado en la posición del carácter."""
        return self.raw_text[:char_position].count('\n') + 1
    
    def extract_legal_references(self, text: str) -> List[str]:
        """Extraer referencias legales citadas."""
        references = []
        
        # Patrones para diferentes tipos de referencias
        patterns = [
            r'Ley\s+\d+\s+de\s+\d{4}',
            r'Decreto\s+(?:Legislativo\s+)?\d+\s+de\s+\d{4}',
            r'[Aa]rtículo\s+\d+(?:\.\d+)*',
            r'Sentencia\s+[A-Z]-\d+-\d+',
            r'RAC\s+\d+(?:\.\d+)*',
            r'Estatuto del Consumidor',
            r'Código General del Proceso',
            r'Código Civil',
            r'Constitución Política'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            # Normalizar y limpiar las referencias
            for match in matches:
                normalized = re.sub(r'\s+', ' ', match.strip())
                if normalized not in references:
                    references.append(normalized)
        
        return references[:10]  # Máximo 10 referencias
    
    def extract_main_theme(self, content: str) -> str:
        """Extraer tema principal del contenido."""
        content_lower = content.lower()
        
        # Definir temas y sus palabras clave asociadas
        themes = {
            "Responsabilidad en transporte aéreo": ["transporte aéreo", "aerolínea", "vuelo", "pasajero", "tiquete"],
            "Publicidad engañosa": ["publicidad engañosa", "información falsa", "propaganda", "anuncio"],
            "Garantía legal": ["garantía", "efectividad de la garantía", "defecto", "calidad"],
            "Derecho de retracto": ["retracto", "desistimiento", "devolución", "cinco días"],
            "Deber de información": ["información", "deber de informar", "información clara", "información veraz"],
            "Servicios turísticos": ["turismo", "agencia de viajes", "paquete turístico", "hotel"],
            "Emergencia sanitaria COVID-19": ["covid", "pandemia", "emergencia sanitaria", "decreto 482"],
            "Comercio electrónico": ["venta a distancia", "online", "internet", "plataforma digital"],
            "Servicios públicos": ["servicios públicos", "factura", "cobro", "tarifa"],
            "Indemnización de perjuicios": ["perjuicios", "indemnización", "daños", "compensación"]
        }
        
        # Contar coincidencias para cada tema
        theme_scores = {}
        for theme, keywords in themes.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > 0:
                theme_scores[theme] = score
        
        # Retornar el tema con mayor puntuación
        if theme_scores:
            return max(theme_scores, key=theme_scores.get)
        else:
            return "Protección al consumidor"
    
    def extract_tags_from_content(self, content: str) -> List[str]:
        """Extraer tags relevantes del contenido."""
        # Lista expandida de términos jurídicos relevantes
        keywords = [
            # Derechos del consumidor
            'consumidor', 'usuario', 'cliente', 'comprador',
            'proveedor', 'productor', 'comerciante', 'vendedor',
            'garantía', 'garantía legal', 'efectividad de la garantía',
            
            # Acciones y remedios
            'reembolso', 'devolución', 'indemnización', 'perjuicios',
            'reparación', 'compensación', 'resarcimiento',
            
            # Derechos específicos
            'información', 'información veraz', 'información clara',
            'publicidad', 'publicidad engañosa', 'propaganda',
            'retracto', 'desistimiento', 'revocación',
            
            # Sectores específicos
            'aerolínea', 'transporte aéreo', 'aviación', 'vuelo',
            'turismo', 'agencia de viajes', 'hotel',
            'servicios públicos', 'telecomunicaciones',
            
            # Contextos especiales
            'covid', 'covid-19', 'pandemia', 'emergencia sanitaria',
            'voucher', 'bono', 'crédito',
            
            # Conceptos jurídicos
            'vulneración', 'infracción', 'incumplimiento',
            'derechos', 'obligaciones', 'responsabilidad',
            'contrato', 'servicio', 'producto', 'bien',
            'solidaridad', 'responsabilidad solidaria',
            'abuso de posición dominante', 'cláusula abusiva'
        ]
        
        tags = set()
        content_lower = content.lower()
        
        for keyword in keywords:
            if keyword in content_lower:
                # Convertir a formato de tag (sin espacios, guiones bajos)
                tag = keyword.replace(' ', '_')
                tags.add(tag)
        
        # Ordenar por relevancia (frecuencia en el texto)
        tag_list = list(tags)
        tag_list.sort(key=lambda x: content_lower.count(x.replace('_', ' ')), reverse=True)
        
        return tag_list[:7]  # Máximo 7 tags
    
    def clean_metadata_for_json(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Limpiar metadata para evitar errores de serialización JSON."""
        def clean_value(value):
            if value is None:
                return None
            if isinstance(value, str):
                return value.strip() if value.strip() else None
            if isinstance(value, dict):
                return {k: clean_value(v) for k, v in value.items()}
            if isinstance(value, list):
                return [clean_value(v) for v in value if clean_value(v) is not None]
            return value
        
        cleaned = {}
        for key, value in metadata.items():
            cleaned_value = clean_value(value)
            if cleaned_value is not None:
                cleaned[key] = cleaned_value
        
        return cleaned
    
    def create_standardized_metadata(self, section: Dict[str, Any], chunk_content: str = None,
                                   chunk_start_line: int = None, chunk_end_line: int = None) -> Dict[str, Any]:
        """Crear metadata estandarizada compatible con LangChain para sentencias."""
        
        now = datetime.now().isoformat()
        
        # Generar ID legal basado en radicado
        radicado = self.sentencia_data.get('radicado', 'SIN_RADICADO')
        id_legal = f"SIC_{radicado.replace('-', '_')}" if radicado != 'SIN_RADICADO' else "SIC_UNKNOWN"
        
        # Determinar el contenido a analizar (chunk o sección completa)
        content_to_analyze = chunk_content or section['contenido']
        
        # Extraer referencias legales y tags
        referencias = self.extract_legal_references(content_to_analyze)
        tags = self.extract_tags_from_content(content_to_analyze)
        tema_principal = self.extract_main_theme(content_to_analyze)
        
        # Asegurar que file_extension no sea vacío
        file_ext = os.path.splitext(self.file_name)[1]
        if not file_ext:
            file_ext = ".txt"
        
        # Crear metadata estandarizada - ADAPTADA PARA SENTENCIAS
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
            "id_legal": id_legal,
            "file_name": self.file_name or "sentencias",
            "created_at": now,
            "last_modified": now,
            "file_extension": file_ext,
            "tipo_documento": "Sentencia SIC",
            "articulo_numero": section["tipo"],  # ← SECCIÓN TEMÁTICA (ej: "ratio_decidendi", "hechos")
            "capitulo_nombre": None,  # Las sentencias no tienen capítulos
            "capitulo_numero": None   # Las sentencias no tienen capítulos
        }
        
        # Agregar campos específicos de sentencias (opcionales)
        sentencia_metadata = {}
        
        if self.sentencia_data.get('radicado'):
            sentencia_metadata['radicado'] = self.sentencia_data['radicado']
        
        if self.sentencia_data.get('fecha'):
            sentencia_metadata['fecha_sentencia'] = self.sentencia_data['fecha']
        
        if self.sentencia_data.get('demandante'):
            sentencia_metadata['demandante'] = self.sentencia_data['demandante']
        
        if self.sentencia_data.get('demandado'):
            sentencia_metadata['demandado'] = self.sentencia_data['demandado']
        
        if tema_principal:
            sentencia_metadata['tema_principal'] = tema_principal
        
        if tags:
            sentencia_metadata['tags_juridicos'] = tags
        
        if referencias:
            sentencia_metadata['referencias_legales'] = referencias
        
        # Agregar relevancia de la sección
        sentencia_metadata['es_seccion_importante'] = section.get('es_importante', False)
        
        # Combinar metadata base con metadata específica de sentencias
        if sentencia_metadata:
            metadata.update(sentencia_metadata)
        
        # Limpiar metadata antes de retornar
        return self.clean_metadata_for_json(metadata)
    
    def create_chunks_from_sections(self, sections: List[Dict[str, Any]], 
                                  chunk_size: int = 1000, 
                                  chunk_overlap: int = 200):
        """Crear chunks a partir de las secciones identificadas."""
        
        for section in sections:
            section_text = section['contenido']
            
            # Decidir si dividir la sección
            if len(section_text) > chunk_size * 1.5:
                # Dividir secciones largas preservando párrafos
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", ". ", "; ", ", ", " "],
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
                # Mantener sección completa
                metadata = self.create_standardized_metadata(section, section_text)
                
                self.processed_chunks.append({
                    "content": section_text,
                    "metadata": metadata
                })
        
        return self.processed_chunks
    
    def process_document(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """Procesar la sentencia completa."""
        # Extraer metadata básica
        self.extract_metadata()
        print(f"📋 Metadata extraída: Radicado {self.sentencia_data.get('radicado', 'N/A')}")
        
        # Dividir en secciones
        sections = self.split_into_sections()
        print(f"Secciones identificadas: {len(sections)}")
        
        # Mostrar secciones encontradas
        if sections:
            print("\nSecciones encontradas:")
            for section in sections:
                importance = "⭐" if section['es_importante'] else "📄"
                print(f"  {importance} {section['tipo']}: {len(section['contenido'])} caracteres")
        
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
        
        print(f"📊 Iniciando vectorización de {total_chunks} chunks de sentencia...")
        
        # Verificar conexión a Supabase
        try:
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
                
                # Preparar datos para insertar
                data = {
                    "content": str(chunk["content"]),
                    "metadata": chunk["metadata"],
                    "embedding": embedding_vector
                }
                
                # Almacenar en Supabase
                result = supabase.table(table_name).insert(data).execute()
                
                # Verificar que la inserción fue exitosa
                if result.data:
                    seccion_tipo = chunk['metadata'].get('articulo_numero', 'N/A')
                    radicado = chunk['metadata'].get('radicado', 'N/A')
                    successful_inserts += 1
                    if i % 5 == 0 or i == total_chunks:
                        print(f"📝 [{i}/{total_chunks}] Sentencia {radicado} - {seccion_tipo} ✅")
                else:
                    print(f"⚠️  Chunk {i}: inserción sin datos de retorno")
                    failed_inserts += 1
                
            except Exception as e:
                print(f"❌ Error al almacenar chunk {i}/{total_chunks}: {str(e)}")
                if i <= 3:
                    print(f"   📋 Metadata del chunk problemático: {chunk.get('metadata', {}).keys()}")
                    print(f"   📄 Longitud del contenido: {len(chunk.get('content', ''))}")
                failed_inserts += 1
                continue
        
        # Resumen final
        print(f"\n🎯 Resumen del procesamiento de sentencia:")
        print(f"   ✅ Chunks exitosos: {successful_inserts}")
        print(f"   ❌ Chunks fallidos: {failed_inserts}")
        print(f"   📊 Total procesados: {total_chunks}")
        
        if successful_inserts > 0:
            radicado = self.sentencia_data.get('radicado', 'N/A')
            print(f"✅ Sentencia {radicado} procesada. {successful_inserts} chunks almacenados en '{table_name}'.")
        else:
            print(f"❌ No se pudo almacenar ningún chunk. Revisar configuración de Supabase.")
        
        return {"successful": successful_inserts, "failed": failed_inserts, "total": total_chunks}

def verify_or_create_table(table_name: str = "transporte_aereo"):
    """Verificar que la tabla existe, si no, mostrar instrucciones para crearla."""
    try:
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
AS $$
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
$$;

-- Crear índice para mejorar performance
CREATE INDEX IF NOT EXISTS {table_name}_embedding_idx ON {table_name} 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Crear índice para metadata
CREATE INDEX IF NOT EXISTS {table_name}_metadata_idx ON {table_name} USING GIN (metadata);
        """)
        return False

def process_sentencia_file(file_path: str, table_name: str = "transporte_aereo", creator: str = "System"):
    """Procesar archivo de sentencia y vectorizar en Supabase."""
    print(f"📄 Procesando Sentencia SIC: {file_path}")
    
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
    
    # Crear el procesador específico para sentencias
    processor = SentenciaSICProcessor(document_text, file_name, creator)
    
    # Procesar el documento
    chunks = processor.process_document(chunk_size=1000, chunk_overlap=200)
    
    if not chunks:
        print("❌ No se generaron chunks del documento")
        return
    
    # Vectorizar y almacenar
    result = processor.vectorize_and_store(table_name)
    
    return result

def process_multiple_sentencias(directory_path: str, table_name: str = "transporte_aereo", creator: str = "System"):
    """Procesar múltiples archivos de sentencias desde un directorio."""
    if not os.path.exists(directory_path):
        print(f"❌ Directorio no encontrado: {directory_path}")
        return
    
    # Obtener todos los archivos .txt del directorio
    files = [f for f in os.listdir(directory_path) 
             if f.endswith('.txt') or f.endswith('.pdf')]
    
    if not files:
        print(f"❌ No se encontraron archivos .txt o .pdf en {directory_path}")
        return
    
    print(f"📁 Encontrados {len(files)} archivos para procesar")
    
    total_successful = 0
    total_failed = 0
    
    for i, file_name in enumerate(files, 1):
        print(f"\n{'='*60}")
        print(f"📄 Procesando archivo {i}/{len(files)}: {file_name}")
        print(f"{'='*60}")
        
        file_path = os.path.join(directory_path, file_name)
        result = process_sentencia_file(file_path, table_name, creator)
        
        if result:
            total_successful += result.get('successful', 0)
            total_failed += result.get('failed', 0)
    
    print(f"\n🎯 RESUMEN GENERAL:")
    print(f"   📁 Archivos procesados: {len(files)}")
    print(f"   ✅ Chunks exitosos: {total_successful}")
    print(f"   ❌ Chunks fallidos: {total_failed}")
    print(f"   📊 Total chunks: {total_successful + total_failed}")

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
        path = sys.argv[1]
        table_name = sys.argv[2] if len(sys.argv) > 2 else "transporte_aereo"
        creator = sys.argv[3] if len(sys.argv) > 3 else "System"
        
        if os.path.isfile(path):
            # Procesar un archivo único
            process_sentencia_file(path, table_name, creator)
        elif os.path.isdir(path):
            # Procesar un directorio completo
            process_multiple_sentencias(path, table_name, creator)
        else:
            print(f"❌ {path} no es un archivo ni directorio válido")
    else:
        print("📋 Uso:")
        print("  python vectorizador_sentencias.py <archivo> [tabla] [creator]")
        print("  python vectorizador_sentencias.py <directorio/> [tabla] [creator]")
        print("📋 Ejemplos:")
        print("  python vectorizador_sentencias.py sentencias transporte_aereo Jhonathan")
        print("  python vectorizador_sentencias.py sentencia_001.txt transporte_aereo Jhonathan")
        print("  python vectorizador_sentencias.py ./sentencias/ transporte_aereo Jhonathan")
        
        # Buscar archivos por defecto (incluyendo sin extensión)
        default_files = ["sentencias", "sentencia.txt", "sentencia_sic.txt", "sentencias.txt"]
        for default_file in default_files:
            if os.path.exists(default_file):
                print(f"🔄 Archivo encontrado: {default_file}")
                print(f"💡 Para procesarlo, ejecuta:")
                print(f"   python vectorizador_sentencias.py {default_file} transporte_aereo Jhonathan")
                break
        else:
            print(f"❌ No se encontraron archivos por defecto: {default_files}")