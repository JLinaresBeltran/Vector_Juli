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

class EstatutoTransporteProcessor:
    def __init__(self, text: str, file_name: str = "estatuto_transporte.txt", creator: str = "System"):
        self.raw_text = text
        self.processed_chunks = []
        self.file_name = file_name
        self.creator = creator
        self.current_titulo = None
        self.current_capitulo = None
        
    def split_into_articles(self) -> List[Dict[str, Any]]:
        """Dividir el documento en artículos y extraer metadatos contextuales."""
        # Patrón para artículos del Estatuto de Transporte (incluye numeración especial como 16-1, 16-2)
        article_pattern = r'(ART[IÍ]CULO\s+\d+(?:-\d+)?\.?.*?)(?=ART[IÍ]CULO\s+\d+(?:-\d+)?\.?|TÍTULO\s+[A-ZÁÉÍÓÚÑ]+|CAPÍTULO\s+[A-ZÁÉÍÓÚÑ]+|$)'
        articles = []
        
        for match in re.finditer(article_pattern, self.raw_text, re.DOTALL):
            article_text = match.group(1).strip()
            article_start = match.start()
            
            # Extraer número de artículo (incluyendo variantes como "16-1", "16-2")
            article_num_match = re.search(r'ART[IÍ]CULO\s+(\d+(?:-\d+)?)', article_text)
            article_num = article_num_match.group(1) if article_num_match else "Unknown"
            
            # Extraer título del artículo si existe
            article_title = ""
            # Buscar títulos después del número del artículo
            title_patterns = [
                r'ART[IÍ]CULO\s+\d+(?:-\d+)?\.?\s*([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s\-\,\.]+?)\.?\s*(?=\n|$)',
                r'ART[IÍ]CULO\s+\d+(?:-\d+)?\.?\s*([A-ZÁÉÍÓÚÑ][^\.]+)\.'
            ]
            
            for pattern in title_patterns:
                title_match = re.search(pattern, article_text)
                if title_match:
                    article_title = title_match.group(1).strip()
                    if article_title.endswith('.'):
                        article_title = article_title[:-1]
                    break
            
            # Determinar la estructura jerárquica actual
            hierarchy = self.determine_hierarchy(article_start)
            
            # Extraer elementos estructurales del artículo
            structural_elements = self.extract_structural_elements(article_text)
            
            article = {
                "numero": article_num,
                "titulo": article_title,
                "texto": article_text,
                "jerarquia": hierarchy,
                "structural_elements": structural_elements,
                "start": article_start,
                "end": match.end(),
                "start_line": self.get_line_number(article_start),
                "end_line": self.get_line_number(match.end())
            }
            
            articles.append(article)
        
        return articles
    
    def determine_hierarchy(self, article_start: int) -> Dict[str, Any]:
        """Determinar la jerarquía del artículo basada en su posición en el texto."""
        # Determinar el título actual
        titulo_matches = list(re.finditer(r'TÍTULO\s+([A-ZÁÉÍÓÚÑ]+)\s*([A-ZÁÉÍÓÚÑ\s\-\,\.]*)', self.raw_text[:article_start]))
        if titulo_matches:
            latest_match = titulo_matches[-1]
            num_titulo = latest_match.group(1)
            nombre_titulo = latest_match.group(2).strip() if latest_match.group(2) else ""
            if nombre_titulo.endswith('.'):
                nombre_titulo = nombre_titulo[:-1]
            self.current_titulo = {
                "numero": num_titulo,
                "nombre": nombre_titulo
            }
        
        # Determinar el capítulo actual
        capitulo_matches = list(re.finditer(r'CAP[IÍ]TULO\s+([A-ZÁÉÍÓÚÑ]+)\s*([A-ZÁÉÍÓÚÑ\s\-\,\.]*)', self.raw_text[:article_start]))
        if capitulo_matches:
            latest_match = capitulo_matches[-1]
            num_capitulo = latest_match.group(1)
            nombre_capitulo = latest_match.group(2).strip() if latest_match.group(2) else ""
            if nombre_capitulo.endswith('.'):
                nombre_capitulo = nombre_capitulo[:-1]
            self.current_capitulo = {
                "numero": num_capitulo,
                "nombre": nombre_capitulo
            }
        
        return {
            "titulo": self.current_titulo,
            "capitulo": self.current_capitulo
        }
    
    def extract_structural_elements(self, text: str) -> Dict[str, Any]:
        """Extraer elementos estructurales del texto (literales, numerales, parágrafos)."""
        elements = {
            "literales": [],
            "numerales": [],
            "paragrafos": [],
            "tiene_subnumeracion": False
        }
        
        # Extraer literales (a), (b), (c), etc.
        literal_matches = re.findall(r'\(([a-z])\)', text)
        elements["literales"] = list(set(literal_matches))
        
        # Extraer numerales 1., 2., 3., etc.
        numeral_matches = re.findall(r'(\d+)\.\s+([^\.]+)', text)
        elements["numerales"] = [match[0] for match in numeral_matches]
        
        # Extraer parágrafos
        paragrafo_matches = re.findall(r'PAR[ÁA]GRAFO\.?\s*(?:\d+[.-]?)?\s*', text, re.IGNORECASE)
        elements["paragrafos"] = [str(i+1) for i in range(len(paragrafo_matches))]
        
        # Detectar si tiene subnumeración (como 16-1, 16-2)
        elements["tiene_subnumeracion"] = bool(re.search(r'\d+-\d+', text))
        
        return elements
    
    def get_line_number(self, char_position: int) -> int:
        """Calcular el número de línea basado en la posición del carácter."""
        return self.raw_text[:char_position].count('\n') + 1
    
    def extract_tags_from_article(self, article_text: str, article_title: str) -> List[str]:
        """Extraer tags relevantes del artículo para búsquedas."""
        tags = []
        
        # Tags basados en el título del artículo
        if article_title:
            # Convertir a minúsculas y dividir en palabras
            title_words = re.findall(r'\b[a-záéíóúñ]+\b', article_title.lower())
            tags.extend([word for word in title_words if len(word) > 3])
        
        # Tags basados en palabras clave específicas del transporte
        keywords = [
            'transporte', 'servicio público', 'empresa', 'operador', 'habilitación',
            'permiso', 'concesión', 'licencia', 'registro', 'matrícula',
            'seguridad', 'equipos', 'vehículos', 'infraestructura', 'usuarios',
            'tarifa', 'conductor', 'operación', 'prestación', 'autoridad',
            'sanción', 'multa', 'suspensión', 'cancelación', 'inmovilización',
            'terrestre', 'aéreo', 'marítimo', 'fluvial', 'ferroviario', 'masivo',
            'automotor', 'aeronáutico', 'portuario', 'navegación',
            'terminal', 'aeropuerto', 'puerto', 'estación', 'nodo',
            'pasajeros', 'carga', 'mixto', 'público', 'privado',
            'rutas', 'horarios', 'frecuencias', 'itinerarios'
        ]
        
        text_lower = article_text.lower()
        for keyword in keywords:
            if keyword in text_lower:
                tags.append(keyword.replace(' ', '_'))
        
        # Tags específicos por modo de transporte
        transport_modes = {
            'terrestre': r'terrestre|automotor|vehículo|carretera',
            'aéreo': r'aéreo|aeronáutico|avión|aeropuerto',
            'marítimo': r'marítimo|naval|barco|puerto',
            'fluvial': r'fluvial|río|embarcación|navegación',
            'ferroviario': r'ferroviario|tren|férrea|ferrocarril',
            'masivo': r'masivo|metropolitano|urbano'
        }
        
        for mode, pattern in transport_modes.items():
            if re.search(pattern, text_lower):
                tags.append(f'modo_{mode}')
        
        # Remover duplicados y retornar máximo 10 tags
        return list(set(tags))[:10]
    
    def extract_subject_matters(self, text: str) -> List[str]:
        """Extraer materias específicas del derecho del transporte."""
        materias = []
        
        subject_patterns = {
            "habilitacion_empresas": r"habilitación|empresa.*transporte|operador.*transporte",
            "permisos_operacion": r"permiso.*operación|autorización.*servicio",
            "concesiones": r"concesión|licitación.*pública|contrato.*concesión",
            "seguridad_transporte": r"seguridad.*transporte|condiciones.*seguridad",
            "equipos_vehiculos": r"equipos.*transporte|vehículos|homologación",
            "tarifas": r"tarifa|precio.*servicio|costo.*transporte",
            "sanciones": r"sanción|multa|suspensión|cancelación|inmovilización",
            "servicios_conexos": r"terminal|puerto|aeropuerto|estación",
            "transporte_internacional": r"internacional|fronterizo|tratado",
            "infraestructura": r"infraestructura|vía|red.*transporte",
            "procedimientos_administrativos": r"procedimiento.*administrativo|investigación",
            "usuarios_transporte": r"usuario|pasajero|protección.*usuario"
        }
        
        text_lower = text.lower()
        for materia, pattern in subject_patterns.items():
            if re.search(pattern, text_lower):
                materias.append(materia)
        
        return materias
    
    def extract_transport_modes(self, text: str) -> List[str]:
        """Extraer modos de transporte mencionados en el texto."""
        modos = []
        
        mode_patterns = {
            "terrestre": r"terrestre|automotor|carretera",
            "aéreo": r"aéreo|aeronáutico|aviación",
            "marítimo": r"marítimo|naval|marino",
            "fluvial": r"fluvial|río|navegación.*interior",
            "ferroviario": r"ferroviario|férrea|tren",
            "masivo": r"masivo|metropolitano|urbano"
        }
        
        text_lower = text.lower()
        for modo, pattern in mode_patterns.items():
            if re.search(pattern, text_lower):
                modos.append(modo)
        
        return modos
    
    def determine_article_elements(self, article: Dict[str, Any], chunk_content: str) -> Dict[str, Any]:
        """Determinar qué elementos específicos están en este chunk."""
        elements = {"literal_letra": None, "numeral_numero": None, "paragrafo_numero": None}
        
        # Buscar literal específico en el chunk
        literal_match = re.search(r'\(([a-z])\)', chunk_content)
        if literal_match:
            elements["literal_letra"] = literal_match.group(1)
        
        # Buscar numeral específico en el chunk
        numeral_match = re.search(r'(\d+)\.\s+', chunk_content)
        if numeral_match:
            elements["numeral_numero"] = numeral_match.group(1)
        
        # Buscar parágrafo específico en el chunk
        paragrafo_match = re.search(r'PAR[ÁA]GRAFO\.?\s*(\d*)', chunk_content, re.IGNORECASE)
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
            if isinstance(value, list):
                return [clean_value(v) for v in value if clean_value(v) is not None]
            return value
        
        cleaned = {}
        for key, value in metadata.items():
            cleaned_value = clean_value(value)
            if cleaned_value is not None:
                cleaned[key] = cleaned_value
        
        return cleaned
    
    def create_standardized_metadata(self, article: Dict[str, Any], chunk_content: str = None,
                                   chunk_start_line: int = None, chunk_end_line: int = None) -> Dict[str, Any]:
        """Crear metadata estandarizada compatible con LangChain."""
        
        now = datetime.now().isoformat()
        
        # Determinar elementos específicos del chunk si se proporciona
        specific_elements = {"literal_letra": None, "numeral_numero": None, "paragrafo_numero": None}
        if chunk_content:
            specific_elements = self.determine_article_elements(article, chunk_content)
        
        # Asegurar que file_extension no sea vacío
        file_ext = os.path.splitext(self.file_name)[1]
        if not file_ext:
            file_ext = ".txt"
        
        # Construir nombre del capítulo
        capitulo_nombre = "Estatuto Nacional de Transporte"  # Valor por defecto
        capitulo_numero = "I"  # Valor por defecto
        
        if article["jerarquia"].get("capitulo"):
            cap_info = article["jerarquia"]["capitulo"]
            if cap_info.get("nombre"):
                capitulo_nombre = f"Capítulo {cap_info.get('numero', 'I')} - {cap_info['nombre']}"
            capitulo_numero = cap_info.get("numero", "I")
        elif article["jerarquia"].get("titulo"):
            # Si no hay capítulo, usar título
            titulo_info = article["jerarquia"]["titulo"]
            if titulo_info.get("nombre"):
                capitulo_nombre = f"Título {titulo_info.get('numero', 'I')} - {titulo_info['nombre']}"
            capitulo_numero = titulo_info.get("numero", "I")
        
        # Crear metadata base estandarizada
        metadata = {
            "loc": {
                "lines": {
                    "from": chunk_start_line or article["start_line"],
                    "to": chunk_end_line or article["end_line"]
                }
            },
            "line": chunk_start_line or article["start_line"],
            "source": "file",
            "creator": self.creator or "System",
            "version": "v1",
            "blobType": "text/plain",
            "id_legal": "ETRANS",
            "file_name": self.file_name or "estatuto_transporte.txt",
            "created_at": now,
            "last_modified": now,
            "file_extension": file_ext,
            "tipo_documento": "Estatuto Nacional de Transporte",
            "articulo_numero": article["numero"],  # ← ARTÍCULO del Estatuto (ej: "16-1", "25")
            "capitulo_nombre": capitulo_nombre,
            "capitulo_numero": capitulo_numero
        }
        
        # Agregar campos opcionales específicos del transporte
        if specific_elements["literal_letra"]:
            metadata["literal_letra"] = specific_elements["literal_letra"]
        
        if specific_elements["numeral_numero"]:
            metadata["numeral_numero"] = specific_elements["numeral_numero"]
        
        if specific_elements["paragrafo_numero"]:
            metadata["paragrafo_numero"] = specific_elements["paragrafo_numero"]
        
        # Campos específicos del Estatuto de Transporte (información adicional)
        if chunk_content:
            # Extraer información específica del transporte
            tags = self.extract_tags_from_article(chunk_content, article["titulo"])
            materias = self.extract_subject_matters(chunk_content)
            modos_transporte = self.extract_transport_modes(chunk_content)
            
            if tags:
                metadata["tags_transporte"] = tags
            if materias:
                metadata["materias_transporte"] = materias
            if modos_transporte:
                metadata["modos_transporte"] = modos_transporte
            
            # Ubicación legible para el Estatuto de Transporte
            ubicacion_parts = ["Ley 336 de 1996 - Estatuto Nacional de Transporte"]
            if article["jerarquia"].get("titulo"):
                titulo_info = article["jerarquia"]["titulo"]
                titulo_str = f"Título {titulo_info.get('numero', 'I')}"
                if titulo_info.get("nombre"):
                    titulo_str += f" ({titulo_info['nombre']})"
                ubicacion_parts.append(titulo_str)
            
            if article["jerarquia"].get("capitulo"):
                cap_info = article["jerarquia"]["capitulo"]
                cap_str = f"Capítulo {cap_info.get('numero', 'I')}"
                if cap_info.get("nombre"):
                    cap_str += f" ({cap_info['nombre']})"
                ubicacion_parts.append(cap_str)
            
            ubicacion_parts.append(f"Artículo {article['numero']}")
            metadata["ubicacion_legible"] = " - ".join(ubicacion_parts)
        
        # Información sobre elementos estructurales
        if article["structural_elements"].get("tiene_subnumeracion"):
            metadata["tiene_subnumeracion"] = True
        
        # Limpiar metadata antes de retornar
        return self.clean_metadata_for_json(metadata)
    
    def create_chunks_from_articles(self, articles: List[Dict[str, Any]], 
                                  chunk_size: int = 1200, 
                                  chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """Crear chunks a partir de los artículos identificados."""
        
        for article in articles:
            article_text = article["texto"]
            
            # Decidir si dividir el artículo
            if len(article_text) > chunk_size * 1.5:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", "PARÁGRAFO", ".", " "],
                    keep_separator=True
                )
                chunks = text_splitter.split_text(article_text)
                
                for i, chunk in enumerate(chunks):
                    # Calcular líneas aproximadas para cada chunk
                    lines_per_chunk = max(1, len(article_text.split('\n')) // len(chunks))
                    chunk_start_line = article["start_line"] + (i * lines_per_chunk)
                    chunk_end_line = chunk_start_line + chunk.count('\n') + 1
                    
                    metadata = self.create_standardized_metadata(
                        article, 
                        chunk_content=chunk,
                        chunk_start_line=chunk_start_line,
                        chunk_end_line=chunk_end_line
                    )
                    
                    self.processed_chunks.append({
                        "content": chunk,
                        "metadata": metadata
                    })
            else:
                # Artículo completo como un solo chunk
                metadata = self.create_standardized_metadata(article, article_text)
                
                self.processed_chunks.append({
                    "content": article_text,
                    "metadata": metadata
                })
        
        return self.processed_chunks
    
    def process_document(self, chunk_size: int = 1200, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """Procesar el documento completo."""
        # Dividir en artículos
        articles = self.split_into_articles()
        print(f"Artículos del Estatuto de Transporte identificados: {len(articles)}")
        
        # Mostrar muestra de artículos encontrados
        if articles:
            print("\nMuestra de artículos encontrados:")
            for a in articles[:5]:
                titulo_info = a.get('titulo', '')[:40] + '...' if len(a.get('titulo', '')) > 40 else a.get('titulo', '')
                print(f"  Art. {a['numero']}: {titulo_info}")
        
        # Crear chunks
        self.create_chunks_from_articles(articles, chunk_size, chunk_overlap)
        print(f"Chunks generados: {len(self.processed_chunks)}")
        
        return self.processed_chunks
    
    def vectorize_and_store(self, table_name: str = "transporte_aereo"):
        """Generar embeddings y almacenar en Supabase usando estructura compatible con LangChain."""
        total_chunks = len(self.processed_chunks)
        
        if total_chunks == 0:
            print("❌ No hay chunks para procesar")
            return
        
        print(f"📊 Iniciando vectorización de {total_chunks} chunks del Estatuto de Transporte...")
        
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
                    articulo_num = chunk['metadata'].get('articulo_numero', 'N/A')
                    successful_inserts += 1
                    if i % 10 == 0 or i == total_chunks:
                        print(f"📝 [{i}/{total_chunks}] Estatuto Transporte - Art. {articulo_num} ✅")
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
        print(f"\n🎯 Resumen del procesamiento Estatuto de Transporte:")
        print(f"   ✅ Chunks exitosos: {successful_inserts}")
        print(f"   ❌ Chunks fallidos: {failed_inserts}")
        print(f"   📊 Total procesados: {total_chunks}")
        
        if successful_inserts > 0:
            print(f"✅ Estatuto de Transporte procesado. {successful_inserts} chunks almacenados en '{table_name}'.")
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

def process_file(file_path: str, table_name: str = "transporte_aereo", creator: str = "System"):
    """Procesar archivo del Estatuto de Transporte y vectorizar en Supabase."""
    print(f"📄 Procesando Estatuto de Transporte: {file_path}")
    
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
    processor = EstatutoTransporteProcessor(document_text, file_name, creator)
    
    # Procesar el documento
    chunks = processor.process_document(chunk_size=1200, chunk_overlap=200)
    
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
        print("📋 Uso: python vectorizador_estatuto_transporte.py <archivo> [tabla] [creator]")
        print("📋 Ejemplo: python vectorizador_estatuto_transporte.py estatuto_transporte.txt transporte_aereo Jhonathan")
        
        # Si no se proporciona un archivo, usar el archivo predeterminado si existe
        default_file = "estatuto_transporte.txt"
        if os.path.exists(default_file):
            print(f"🔄 Usando archivo por defecto: {default_file}")
            process_file(default_file, "transporte_aereo", "System")
        else:
            print(f"❌ Archivo por defecto '{default_file}' no encontrado.")