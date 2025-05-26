import os
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from supabase import create_client

# Cargar variables de entorno
load_dotenv()

# Configuración de Supabase
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase = create_client(supabase_url, supabase_key)

# Configuración de OpenAI para embeddings
openai_api_key = os.environ.get("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

class EstatutoTransporteProcessor:
    def __init__(self, text: str):
        self.raw_text = text
        self.processed_chunks = []
        self.current_titulo = None
        self.current_capitulo = None
        
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
    
    def create_ubicacion_legible(self, estructura: Dict[str, Any], articulo_numero: str) -> str:
        """Crear una descripción legible de la ubicación del artículo."""
        ubicacion_parts = ["Ley 336 de 1996 - Estatuto Nacional de Transporte"]
        
        if estructura.get("titulo"):
            titulo_info = estructura["titulo"]
            titulo_str = f"Título {titulo_info['numero']}"
            if titulo_info.get("nombre"):
                titulo_str += f" ({titulo_info['nombre']})"
            ubicacion_parts.append(titulo_str)
        
        if estructura.get("capitulo") and estructura["capitulo"].get("numero"):
            cap_info = estructura["capitulo"]
            cap_str = f"Capítulo {cap_info['numero']}"
            if cap_info.get("nombre"):
                cap_str += f" ({cap_info['nombre']})"
            ubicacion_parts.append(cap_str)
        
        ubicacion_parts.append(f"Artículo {articulo_numero}")
        
        return " - ".join(ubicacion_parts)
    
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
            
            # Extraer parágrafos si existen
            paragrafo_pattern = r'PAR[ÁA]GRAFO\.?\s*(?:\d+[.-]?)?\s*'
            paragrafo_matches = re.findall(paragrafo_pattern, article_text)
            paragrafo_count = len(paragrafo_matches)
            
            # Extraer literales/numerales si existen
            literal_patterns = [
                r'([a-z])\.\s+([^\.]+)',  # a. texto
                r'(\d+)\.\s+([^\.]+)',    # 1. texto
                r'([a-z])\)\s+([^\.]+)',  # a) texto
                r'(\d+)\)\s+([^\.]+)'     # 1) texto
            ]
            
            literales_count = 0
            for pattern in literal_patterns:
                matches = re.findall(pattern, article_text)
                literales_count += len(matches)
            
            # Detectar si es un artículo con subnumeración (16-1, 16-2)
            tiene_subnumeracion = '-' in article_num
            
            # Crear estructura del artículo
            article = {
                "numero": article_num,
                "titulo": article_title,
                "texto": article_text,
                "estructura": {
                    "titulo": self.current_titulo,
                    "capitulo": self.current_capitulo
                },
                "paragrafo_count": paragrafo_count,
                "literales_count": literales_count,
                "tiene_subnumeracion": tiene_subnumeracion,
                "start": article_start,
                "end": match.end()
            }
            
            articles.append(article)
        
        return articles
    
    def create_chunks_from_articles(self, articles, chunk_size=1200, chunk_overlap=200):
        """Crear chunks a partir de los artículos identificados con estructura de metadata."""
        for article in articles:
            article_text = article["texto"]
            
            # Crear estructura de jerarquía
            jerarquia = {
                "ley": {
                    "numero": "336",
                    "año": "1996",
                    "nombre": "Estatuto Nacional de Transporte"
                },
                "titulo": article["estructura"]["titulo"],
                "capitulo": article["estructura"]["capitulo"],
                "articulo": {
                    "numero": article["numero"],
                    "titulo": article["titulo"]
                }
            }
            
            # Extraer tags para el artículo
            tags = self.extract_tags_from_article(article_text, article["titulo"])
            
            # Crear ubicación legible
            ubicacion_legible = self.create_ubicacion_legible(article["estructura"], article["numero"])
            
            # Determinar si es un artículo extenso que requiere división
            if len(article_text) > chunk_size * 1.5:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", "PARÁGRAFO", ".", " "]
                )
                chunks = text_splitter.split_text(article_text)
                
                for i, chunk in enumerate(chunks):
                    # Crear metadata según nueva estructura
                    metadata = {
                        "documento": {
                            "tipo": "estatuto_transporte",
                            "titulo": "Ley 336 de 1996 - Estatuto Nacional de Transporte",
                            "fecha": "1996-12-20"
                        },
                        "jerarquia": jerarquia,
                        "chunk": {
                            "es_articulo_completo": False,
                            "indice_chunk": i + 1,
                            "total_chunks": len(chunks),
                            "tamaño_caracteres": len(chunk),
                            "contiene_paragrafo": article["paragrafo_count"] > 0,
                            "numero_parrafos": article["paragrafo_count"],
                            "numero_literales": article["literales_count"],
                            "tiene_subnumeracion": article["tiene_subnumeracion"]
                        },
                        "tags": tags,
                        "ubicacion_legible": f"{ubicacion_legible} (Parte {i+1} de {len(chunks)})",
                        "referencias": {
                            "articulos_relacionados": [],
                            "conceptos_clave": tags[:3],
                            "materias": self.extract_subject_matters(article_text),
                            "modos_transporte": self.extract_transport_modes(article_text)
                        }
                    }
                    
                    self.processed_chunks.append({
                        "content": chunk,
                        "metadata": metadata
                    })
            else:
                # Artículo completo como un solo chunk
                metadata = {
                    "documento": {
                        "tipo": "estatuto_transporte",
                        "titulo": "Ley 336 de 1996 - Estatuto Nacional de Transporte",
                        "fecha": "1996-12-20"
                    },
                    "jerarquia": jerarquia,
                    "chunk": {
                        "es_articulo_completo": True,
                        "indice_chunk": 1,
                        "total_chunks": 1,
                        "tamaño_caracteres": len(article_text),
                        "contiene_paragrafo": article["paragrafo_count"] > 0,
                        "numero_parrafos": article["paragrafo_count"],
                        "numero_literales": article["literales_count"],
                        "tiene_subnumeracion": article["tiene_subnumeracion"]
                    },
                    "tags": tags,
                    "ubicacion_legible": ubicacion_legible,
                    "referencias": {
                        "articulos_relacionados": [],
                        "conceptos_clave": tags[:3],
                        "materias": self.extract_subject_matters(article_text),
                        "modos_transporte": self.extract_transport_modes(article_text)
                    }
                }
                
                self.processed_chunks.append({
                    "content": article_text,
                    "metadata": metadata
                })
        
        return self.processed_chunks
    
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
    
    def process_document(self, chunk_size=1200, chunk_overlap=200):
        """Procesar el documento completo."""
        # Dividir en artículos
        articles = self.split_into_articles()
        print(f"Artículos identificados: {len(articles)}")
        
        # Crear chunks
        self.create_chunks_from_articles(articles, chunk_size, chunk_overlap)
        print(f"Chunks generados: {len(self.processed_chunks)}")
        
        return self.processed_chunks
    
    def vectorize_and_store(self, table_name="transporte_aereo"):
        """Generar embeddings y almacenar en Supabase."""
        total_chunks = len(self.processed_chunks)
        
        for i, chunk in enumerate(self.processed_chunks, 1):
            # Generar el embedding
            embedding_vector = embeddings.embed_query(chunk["content"])
            
            # Almacenar en Supabase
            try:
                result = supabase.table(table_name).insert({
                    "content": chunk["content"],
                    "metadata": chunk["metadata"],
                    "embedding": embedding_vector
                }).execute()
                
                print(f"[{i}/{total_chunks}] Chunk del artículo {chunk['metadata']['jerarquia']['articulo']['numero']} almacenado correctamente en tabla '{table_name}'")
            except Exception as e:
                print(f"Error al almacenar chunk {i}/{total_chunks}: {e}")
                
        print(f"Procesamiento completado. {len(self.processed_chunks)} chunks generados y almacenados en tabla '{table_name}'.")

def process_file(file_path, table_name="transporte_aereo"):
    """Procesa un archivo de texto y lo vectoriza en Supabase."""
    # Leer el archivo
    with open(file_path, "r", encoding="utf-8") as f:
        document_text = f.read()
    
    # Crear el procesador
    processor = EstatutoTransporteProcessor(document_text)
    
    # Procesar el documento (chunk_size más grande para textos del transporte)
    processor.process_document(chunk_size=1200, chunk_overlap=200)
    
    # Vectorizar y almacenar
    processor.vectorize_and_store(table_name)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        table_name = sys.argv[2] if len(sys.argv) > 2 else "transporte_aereo"
        process_file(file_path, table_name)
    else:
        # Si no se proporciona un archivo, usar el archivo predeterminado
        process_file("estatuto_transporte.txt", "transporte_aereo")