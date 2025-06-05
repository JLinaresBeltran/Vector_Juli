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

class EstatutoConsumidorProcessor:
    def __init__(self, text: str, file_name: str = "estatuto_consumidor.txt", creator: str = "System"):
        self.raw_text = text
        self.processed_chunks = []
        self.file_name = file_name
        self.creator = creator
        self.current_titulo = None
        self.current_capitulo = None
        
    def split_into_articles(self) -> List[Dict[str, Any]]:
        """Dividir el documento en artículos y extraer metadatos contextuales."""
        articles = []
        
        # Patrón mejorado para artículos del Estatuto del Consumidor
        article_pattern = r'(ART[IÍ]CULO\s+\d+[oº]?\..*?)(?=ART[IÍ]CULO\s+\d+[oº]?\.|TÍTULO\s+[IVXLCDM]+\.|$)'
        
        for match in re.finditer(article_pattern, self.raw_text, re.DOTALL):
            article_text = match.group(1).strip()
            article_start = match.start()
            
            # Extraer número de artículo (incluyendo variantes como "1o", "2o", etc.)
            article_num_match = re.search(r'ART[IÍ]CULO\s+(\d+)[oº]?', article_text)
            article_num = article_num_match.group(1) if article_num_match else "Unknown"
            
            # Extraer título del artículo (más flexible para el Estatuto del Consumidor)
            article_title_match = re.search(r'ART[IÍ]CULO\s+\d+[oº]?\.?\s*([A-ZÁÉÍÓÚÑ\s\-\,\.]+)\.', article_text)
            article_title = article_title_match.group(1).strip() if article_title_match else ""
            
            # Limpiar el título (remover puntos finales duplicados)
            if article_title.endswith('.'):
                article_title = article_title[:-1]
            
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
        # Buscar el título actual
        titulo_matches = list(re.finditer(r'TÍTULO\s+([IVXLCDM]+)\.?\s*([A-ZÁÉÍÓÚÑ\s\-\,\.]*)', 
                                        self.raw_text[:article_start]))
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
        
        # Buscar el capítulo actual
        capitulo_matches = list(re.finditer(r'CAP[IÍ]TULO\s+([IVXLCDM]+)\.?\s*([A-ZÁÉÍÓÚÑ\s\-\,\.]*)', 
                                          self.raw_text[:article_start]))
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
            "modificaciones": False
        }
        
        # Extraer literales (a), (b), (c), etc.
        literal_matches = re.findall(r'\(([a-z])\)', text)
        elements["literales"] = list(set(literal_matches))
        
        # Extraer numerales 1., 2., 3., etc.
        numeral_matches = re.findall(r'(\d+)\.?\s+([^\.]+)', text)
        elements["numerales"] = [match[0] for match in numeral_matches]
        
        # Extraer parágrafos
        paragrafo_matches = re.findall(r'PAR[ÁA]GRAFO\.?\s*(?:\d+[oº]?)?\s*', text, re.IGNORECASE)
        elements["paragrafos"] = [str(i+1) for i in range(len(paragrafo_matches))]
        
        # Detectar si hay modificaciones/adiciones
        modificacion_matches = re.findall(r'<[^>]*(?:adicionado|modificado|corregido)[^>]*>', text)
        elements["modificaciones"] = len(modificacion_matches) > 0
        
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
        numeral_match = re.search(r'(\d+)\.?\s+', chunk_content)
        if numeral_match:
            elements["numeral_numero"] = numeral_match.group(1)
        
        # Buscar parágrafo específico en el chunk
        paragrafo_match = re.search(r'PAR[ÁA]GRAFO\.?\s*(\d*[oº]?)', chunk_content, re.IGNORECASE)
        if paragrafo_match and paragrafo_match.group(1):
            elements["paragrafo_numero"] = paragrafo_match.group(1).replace('o', '').replace('º', '') or "1"
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
        capitulo_nombre = ""
        if article["jerarquia"].get("capitulo"):
            cap_info = article["jerarquia"]["capitulo"]
            capitulo_nombre = cap_info.get("nombre", "") or ""
            if cap_info.get("numero"):
                capitulo_nombre = f"Capítulo {cap_info['numero']}" + (f" - {capitulo_nombre}" if capitulo_nombre else "")
        elif article["jerarquia"].get("titulo"):
            # Si no hay capítulo, usar el título
            tit_info = article["jerarquia"]["titulo"]
            capitulo_nombre = tit_info.get("nombre", "") or ""
            if tit_info.get("numero"):
                capitulo_nombre = f"Título {tit_info['numero']}" + (f" - {capitulo_nombre}" if capitulo_nombre else "")
        
        # Construir número del capítulo
        capitulo_numero = ""
        if article["jerarquia"].get("capitulo"):
            capitulo_numero = article["jerarquia"]["capitulo"].get("numero", "")
        elif article["jerarquia"].get("titulo"):
            capitulo_numero = article["jerarquia"]["titulo"].get("numero", "")
        
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
            "id_legal": "ECONS",
            "file_name": self.file_name or "estatuto_consumidor.txt",
            "created_at": now,
            "last_modified": now,
            "file_extension": file_ext,
            "tipo_documento": "Estatuto del Consumidor",
            "articulo_numero": article["numero"],  # ← ARTÍCULO REAL del Estatuto (ej: "5", "23", "58")
            "capitulo_nombre": capitulo_nombre or "Estatuto del Consumidor",
            "capitulo_numero": capitulo_numero or "I"
        }
        
        # Agregar campos opcionales solo si tienen valor
        if specific_elements["literal_letra"]:
            metadata["literal_letra"] = specific_elements["literal_letra"]
        
        if specific_elements["numeral_numero"]:
            metadata["numeral_numero"] = specific_elements["numeral_numero"]
        
        if specific_elements["paragrafo_numero"]:
            metadata["paragrafo_numero"] = specific_elements["paragrafo_numero"]
        
        # Limpiar metadata antes de retornar
        return self.clean_metadata_for_json(metadata)
    
    def create_chunks_from_articles(self, articles: List[Dict[str, Any]], 
                                  chunk_size: int = 1000, 
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
    
    def process_document(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """Procesar el documento completo."""
        # Dividir en artículos
        articles = self.split_into_articles()
        print(f"Artículos del Estatuto del Consumidor identificados: {len(articles)}")
        
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
        
        print(f"📊 Iniciando vectorización de {total_chunks} chunks del Estatuto del Consumidor...")
        
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
                        print(f"📝 [{i}/{total_chunks}] Estatuto Consumidor - Art. {articulo_num} ✅")
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
        print(f"\n🎯 Resumen del procesamiento Estatuto del Consumidor:")
        print(f"   ✅ Chunks exitosos: {successful_inserts}")
        print(f"   ❌ Chunks fallidos: {failed_inserts}")
        print(f"   📊 Total procesados: {total_chunks}")
        
        if successful_inserts > 0:
            print(f"✅ Estatuto del Consumidor procesado. {successful_inserts} chunks almacenados en '{table_name}'.")
        else:
            print(f"❌ No se pudo almacenar ningún chunk. Revisar configuración de Supabase.")
        
        return {"successful": successful_inserts, "failed": failed_inserts, "total": total_chunks}

def process_file(file_path: str, table_name: str = "transporte_aereo", creator: str = "System"):
    """Procesa un archivo de texto del Estatuto del Consumidor y lo vectoriza en Supabase."""
    print(f"📄 Procesando Estatuto del Consumidor: {file_path}")
    
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
    processor = EstatutoConsumidorProcessor(document_text, file_name, creator)
    
    # Procesar el documento
    chunks = processor.process_document(chunk_size=1000, chunk_overlap=200)
    
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
        print("📋 Uso: python vectorizador_estatuto_consumidor.py <archivo> [tabla] [creator]")
        print("📋 Ejemplo: python vectorizador_estatuto_consumidor.py estatuto_consumidor.txt transporte_aereo Jhonathan")
        
        # Si no se proporciona un archivo, usar el archivo predeterminado si existe
        default_file = "estatuto_consumidor.txt"
        if os.path.exists(default_file):
            print(f"🔄 Usando archivo por defecto: {default_file}")
            process_file(default_file, "transporte_aereo", "System")
        else:
            print(f"❌ Archivo por defecto '{default_file}' no encontrado.")