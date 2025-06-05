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

class CodigoComercioProcessor:
    def __init__(self, text: str, file_name: str = "codigo_comercio.txt", creator: str = "System"):
        self.raw_text = text
        self.processed_chunks = []
        self.file_name = file_name
        self.creator = creator
        self.current_parte = None
        self.current_titulo = None
        self.current_capitulo = None
        self.current_seccion = None
        
    def split_into_articles(self) -> List[Dict[str, Any]]:
        """Dividir el documento en artículos y extraer metadatos contextuales."""
        articles = []
        
        # Patrón para artículos del Código de Comercio
        article_pattern = r'(ART[IÍ]CULO\s+\d+\..*?)(?=ART[IÍ]CULO\s+\d+\.|$)'
        
        for match in re.finditer(article_pattern, self.raw_text, re.DOTALL):
            article_text = match.group(1).strip()
            article_start = match.start()
            
            # Extraer número de artículo
            article_num_match = re.search(r'ART[IÍ]CULO\s+(\d+)', article_text)
            article_num = article_num_match.group(1) if article_num_match else "Unknown"
            
            # Extraer título del artículo
            article_title_match = re.search(r'ART[IÍ]CULO\s+\d+\.?\s*([A-ZÁÉÍÓÚÑ\s\-]+)\.', article_text)
            article_title = article_title_match.group(1).strip() if article_title_match else ""
            
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
        # Buscar la parte actual
        parte_matches = list(re.finditer(r'PARTE\s+([A-ZÁÉÍÓÚÑ]+)\.?\s*([A-ZÁÉÍÓÚÑ\s\-]*)', 
                                       self.raw_text[:article_start]))
        if parte_matches:
            latest_match = parte_matches[-1]
            self.current_parte = {
                "numero": latest_match.group(1),
                "nombre": latest_match.group(2).strip() if latest_match.group(2) else ""
            }
        
        # Buscar el título actual
        titulo_matches = list(re.finditer(r'TÍ[TD]ULO\s+([A-ZÁÉÍÓÚÑ\s]+)\.?\s*([A-ZÁÉÍÓÚÑ\s\-]*)', 
                                        self.raw_text[:article_start]))
        if titulo_matches:
            latest_match = titulo_matches[-1]
            self.current_titulo = {
                "numero": latest_match.group(1).strip(),
                "nombre": latest_match.group(2).strip() if latest_match.group(2) else ""
            }
        
        # Buscar el capítulo actual
        capitulo_matches = list(re.finditer(r'CAP[IÍ]TULO\s+([IVX]+)\.?\s*([A-ZÁÉÍÓÚÑ\s\-]*)', 
                                          self.raw_text[:article_start]))
        if capitulo_matches:
            latest_match = capitulo_matches[-1]
            self.current_capitulo = {
                "numero": latest_match.group(1),
                "nombre": latest_match.group(2).strip() if latest_match.group(2) else ""
            }
        
        # Buscar la sección actual
        seccion_matches = list(re.finditer(r'SECCI[ÓO]N\s+([IVX]+)\.?\s*([A-ZÁÉÍÓÚÑ\s\-]*)', 
                                         self.raw_text[:article_start]))
        if seccion_matches:
            latest_match = seccion_matches[-1]
            self.current_seccion = {
                "numero": latest_match.group(1),
                "nombre": latest_match.group(2).strip() if latest_match.group(2) else ""
            }
        
        return {
            "parte": self.current_parte,
            "titulo": self.current_titulo,
            "capitulo": self.current_capitulo,
            "seccion": self.current_seccion
        }
    
    def extract_structural_elements(self, text: str) -> Dict[str, Any]:
        """Extraer elementos estructurales del texto (literales, numerales, parágrafos)."""
        elements = {
            "literales": [],
            "numerales": [],
            "paragrafos": []
        }
        
        # Extraer literales (a), (b), (c), etc.
        literal_matches = re.findall(r'\(([a-z])\)', text)
        elements["literales"] = list(set(literal_matches))
        
        # Extraer numerales 1), 2), 3), etc.
        numeral_matches = re.findall(r'(\d+)\)\s+([^;\.]+)', text)
        elements["numerales"] = [match[0] for match in numeral_matches]
        
        # Extraer parágrafos
        paragrafo_matches = re.findall(r'PAR[ÁA]GRAFO\.?\s*(\d*)', text, re.IGNORECASE)
        elements["paragrafos"] = [p for p in paragrafo_matches if p]
        
        return elements
    
    def get_line_number(self, char_position: int) -> int:
        """Calcular el número de línea basado en la posición del carácter."""
        return self.raw_text[:char_position].count('\n') + 1
    
    def determine_article_elements(self, article: Dict[str, Any], chunk_content: str) -> Dict[str, Any]:
        """Determinar qué elementos específicos están en este chunk."""
        elements = {"literal_letra": None, "numeral_numero": None, "paragrafo_numero": None}
        
        # Buscar literal específico en el chunk
        literal_match = re.search(r'\(([a-z])\)', chunk_content)
        if literal_match:
            elements["literal_letra"] = literal_match.group(1)
        
        # Buscar numeral específico en el chunk
        numeral_match = re.search(r'(\d+)\)\s+', chunk_content)
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
        """Limpiar metadata para evitar errores de serialización JSON de forma más agresiva."""
        def clean_value(value):
            if value is None:
                return None
            if isinstance(value, str):
                cleaned = value.strip()
                # Eliminar strings vacíos, solo espacios, o con caracteres problemáticos
                if not cleaned or cleaned in ['', 'None', 'null']:
                    return None
                return cleaned
            if isinstance(value, dict):
                cleaned_dict = {}
                for k, v in value.items():
                    cleaned_v = clean_value(v)
                    if cleaned_v is not None:
                        cleaned_dict[k] = cleaned_v
                # Solo retornar el diccionario si tiene contenido
                return cleaned_dict if cleaned_dict else None
            if isinstance(value, (int, float)):
                return value
            if isinstance(value, bool):
                return value
            # Para otros tipos, convertir a string si es posible
            try:
                return str(value) if value is not None else None
            except:
                return None
        
        # Limpiar toda la metadata
        cleaned = {}
        for key, value in metadata.items():
            cleaned_value = clean_value(value)
            # Solo incluir valores que no sean None y no sean diccionarios vacíos
            if cleaned_value is not None:
                if isinstance(cleaned_value, dict) and not cleaned_value:
                    continue  # Saltar diccionarios vacíos
                cleaned[key] = cleaned_value
        
        # Asegurar que tenemos los campos mínimos requeridos
        required_fields = ['source', 'creator', 'version', 'blobType', 'id_legal', 'file_name', 
                          'tipo_documento', 'articulo_numero']
        
        for field in required_fields:
            if field not in cleaned or cleaned[field] is None:
                # Proporcionar valores por defecto seguros
                defaults = {
                    'source': 'file',
                    'creator': 'System',
                    'version': 'v1',
                    'blobType': 'text/plain',
                    'id_legal': 'CCOM',
                    'file_name': 'codigo_comercio.txt',
                    'tipo_documento': 'Codigo de Comercio',
                    'articulo_numero': 'unknown'
                }
                cleaned[field] = defaults.get(field, 'unknown')
        
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
        
        # Construir nombre del capítulo de forma más robusta
        capitulo_nombre = "Código de Comercio"  # Valor por defecto
        capitulo_numero = "I"  # Valor por defecto
        
        try:
            if article.get("jerarquia") and isinstance(article["jerarquia"], dict):
                jerarquia = article["jerarquia"]
                
                # Intentar obtener información del capítulo
                if jerarquia.get("capitulo") and isinstance(jerarquia["capitulo"], dict):
                    cap_info = jerarquia["capitulo"]
                    if cap_info.get("nombre"):
                        capitulo_nombre = f"Capítulo {cap_info.get('numero', 'I')} - {cap_info['nombre']}"
                    capitulo_numero = cap_info.get("numero", "I")
                
                # Si no hay capítulo, usar título o parte
                elif jerarquia.get("titulo") and isinstance(jerarquia["titulo"], dict):
                    titulo_info = jerarquia["titulo"]
                    if titulo_info.get("nombre"):
                        capitulo_nombre = f"Título {titulo_info.get('numero', 'I')} - {titulo_info['nombre']}"
                    capitulo_numero = titulo_info.get("numero", "I")
                
                elif jerarquia.get("parte") and isinstance(jerarquia["parte"], dict):
                    parte_info = jerarquia["parte"]
                    if parte_info.get("nombre"):
                        capitulo_nombre = f"Parte {parte_info.get('numero', 'I')} - {parte_info['nombre']}"
                    capitulo_numero = parte_info.get("numero", "I")
        except:
            # Si hay cualquier error, usar valores por defecto
            pass
        
        # Crear metadata base con valores seguros
        metadata = {
            "loc": {
                "lines": {
                    "from": chunk_start_line or article.get("start_line", 1),
                    "to": chunk_end_line or article.get("end_line", 1)
                }
            },
            "line": chunk_start_line or article.get("start_line", 1),
            "source": "file",
            "creator": str(self.creator or "System"),
            "version": "v1",
            "blobType": "text/plain",
            "id_legal": "CCOM",
            "file_name": str(self.file_name or "codigo_comercio.txt"),
            "created_at": now,
            "last_modified": now,
            "file_extension": str(file_ext),
            "tipo_documento": "Codigo de Comercio",
            "articulo_numero": str(article.get("numero", "unknown")),
            "capitulo_nombre": str(capitulo_nombre),
            "capitulo_numero": str(capitulo_numero)
        }
        
        # Agregar campos opcionales solo si tienen valor válido
        if specific_elements.get("literal_letra"):
            metadata["literal_letra"] = str(specific_elements["literal_letra"])
        
        if specific_elements.get("numeral_numero"):
            metadata["numeral_numero"] = str(specific_elements["numeral_numero"])
        
        if specific_elements.get("paragrafo_numero"):
            metadata["paragrafo_numero"] = str(specific_elements["paragrafo_numero"])
        
        # Limpiar metadata antes de retornar
        return self.clean_metadata_for_json(metadata)
    
    def create_chunks_from_articles(self, articles: List[Dict[str, Any]], 
                                  chunk_size: int = 800, 
                                  chunk_overlap: int = 150) -> List[Dict[str, Any]]:
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
    
    def process_document(self, chunk_size: int = 800, chunk_overlap: int = 150) -> List[Dict[str, Any]]:
        """Procesar el documento completo."""
        # Dividir en artículos
        articles = self.split_into_articles()
        print(f"Artículos del Código de Comercio identificados: {len(articles)}")
        
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
    
    def debug_metadata_serialization(self, metadata: Dict[str, Any]) -> bool:
        """Verificar que la metadata se puede serializar a JSON."""
        try:
            import json
            json.dumps(metadata)
            return True
        except Exception as e:
            print(f"❌ Error de serialización JSON en metadata: {e}")
            print(f"📋 Metadata problemática: {metadata}")
            return False
    
    def vectorize_and_store(self, table_name: str = "transporte_aereo"):
        """Generar embeddings y almacenar en Supabase usando estructura compatible con LangChain."""
        total_chunks = len(self.processed_chunks)
        
        if total_chunks == 0:
            print("❌ No hay chunks para procesar")
            return
        
        print(f"📊 Iniciando vectorización de {total_chunks} chunks del Código de Comercio...")
        
        successful_inserts = 0
        failed_inserts = 0
        
        for i, chunk in enumerate(self.processed_chunks, 1):
            try:
                # Validar contenido del chunk
                if not chunk.get("content") or not chunk.get("metadata"):
                    print(f"⚠️  Chunk {i} inválido: contenido o metadata faltante")
                    failed_inserts += 1
                    continue
                
                # Verificar que la metadata se puede serializar
                if not self.debug_metadata_serialization(chunk["metadata"]):
                    print(f"⚠️  Chunk {i}: metadata no serializable")
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
                try:
                    # Crear una copia limpia de la metadata
                    clean_metadata = self.clean_metadata_for_json(chunk["metadata"])
                    
                    # Verificar nuevamente que esté limpia
                    if not self.debug_metadata_serialization(clean_metadata):
                        print(f"⚠️  Chunk {i}: metadata sigue sin ser serializable después de limpieza")
                        failed_inserts += 1
                        continue
                    
                    data = {
                        "content": str(chunk["content"]),
                        "metadata": clean_metadata,
                        "embedding": embedding_vector
                    }
                except Exception as data_error:
                    print(f"❌ Error preparando datos para chunk {i}: {data_error}")
                    failed_inserts += 1
                    continue
                
                # Almacenar en Supabase
                result = supabase.table(table_name).insert(data).execute()
                
                # Verificar que la inserción fue exitosa
                if result.data:
                    articulo_num = chunk['metadata'].get('articulo_numero', 'N/A')
                    successful_inserts += 1
                    if i % 10 == 0 or i == total_chunks:
                        print(f"📝 [{i}/{total_chunks}] Código Comercio - Art. {articulo_num} ✅")
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
        print(f"\n🎯 Resumen del procesamiento Código de Comercio:")
        print(f"   ✅ Chunks exitosos: {successful_inserts}")
        print(f"   ❌ Chunks fallidos: {failed_inserts}")
        print(f"   📊 Total procesados: {total_chunks}")
        
        if successful_inserts > 0:
            print(f"✅ Código de Comercio procesado. {successful_inserts} chunks almacenados en '{table_name}'.")
        else:
            print(f"❌ No se pudo almacenar ningún chunk. Revisar configuración de Supabase.")
        
        return {"successful": successful_inserts, "failed": failed_inserts, "total": total_chunks}

def process_file(file_path: str, table_name: str = "transporte_aereo", creator: str = "System"):
    """Procesa un archivo de texto del Código de Comercio y lo vectoriza en Supabase."""
    print(f"📄 Procesando Código de Comercio: {file_path}")
    
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
    processor = CodigoComercioProcessor(document_text, file_name, creator)
    
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
        print("📋 Uso: python vectorizador_codigo_comercio.py <archivo> [tabla] [creator]")
        print("📋 Ejemplo: python vectorizador_codigo_comercio.py codigo_comercio.txt transporte_aereo Jhonathan")
        
        # Si no se proporciona un archivo, usar el archivo predeterminado si existe
        default_file = "codigo_comercio.txt"
        if os.path.exists(default_file):
            print(f"🔄 Usando archivo por defecto: {default_file}")
            process_file(default_file, "transporte_aereo", "System")
        else:
            print(f"❌ Archivo por defecto '{default_file}' no encontrado.")