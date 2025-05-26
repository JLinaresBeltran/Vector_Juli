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

class EstatutoConsumidorProcessor:
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
        
        # Tags basados en palabras clave específicas del derecho del consumidor
        keywords = [
            'consumidor', 'usuario', 'proveedor', 'productor', 'garantía', 
            'calidad', 'idoneidad', 'seguridad', 'información', 'publicidad',
            'contrato', 'adhesión', 'retracto', 'comercio electrónico', 
            'defectuoso', 'responsabilidad', 'sanción', 'protección',
            'derecho', 'deber', 'obligación', 'cláusula abusiva',
            'promoción', 'oferta', 'precio', 'factura', 'reclamación',
            'superintendencia', 'multa', 'investigación', 'procedimiento'
        ]
        
        text_lower = article_text.lower()
        for keyword in keywords:
            if keyword in text_lower:
                tags.append(keyword.replace(' ', '_'))
        
        # Tags específicos para comercio electrónico
        ecommerce_keywords = [
            'portal de contacto', 'medios electrónicos', 'página web',
            'internet', 'plataforma electrónica', 'transacción electrónica',
            'pago electrónico', 'reversión del pago'
        ]
        
        for keyword in ecommerce_keywords:
            if keyword in text_lower:
                tags.append(keyword.replace(' ', '_'))
        
        # Remover duplicados y retornar máximo 8 tags
        return list(set(tags))[:8]
    
    def create_ubicacion_legible(self, estructura: Dict[str, Any], articulo_numero: str) -> str:
        """Crear una descripción legible de la ubicación del artículo."""
        ubicacion_parts = ["Ley 1480 de 2011 - Estatuto del Consumidor"]
        
        if estructura.get("titulo"):
            ubicacion_parts.append(f"Título {estructura['titulo']}")
        
        if estructura.get("capitulo") and estructura["capitulo"].get("numero"):
            cap_str = f"Capítulo {estructura['capitulo']['numero']}"
            if estructura["capitulo"].get("nombre"):
                cap_str += f" ({estructura['capitulo']['nombre']})"
            ubicacion_parts.append(cap_str)
        
        ubicacion_parts.append(f"Artículo {articulo_numero}")
        
        return " - ".join(ubicacion_parts)
    
    def split_into_articles(self) -> List[Dict[str, Any]]:
        """Dividir el documento en artículos y extraer metadatos contextuales."""
        # Patrón mejorado para artículos del Estatuto del Consumidor
        article_pattern = r'(ART[IÍ]CULO\s+\d+[oº]?\..*?)(?=ART[IÍ]CULO\s+\d+[oº]?\.|TÍTULO\s+[IVXLCDM]+\.|$)'
        articles = []
        
        for match in re.finditer(article_pattern, self.raw_text, re.DOTALL):
            article_text = match.group(1).strip()
            article_start = match.start()
            
            # Extraer número de artículo (incluyendo variantes como "1o", "2o", etc.)
            article_num_match = re.search(r'ART[IÍ]CULO\s+(\d+[oº]?)', article_text)
            article_num = article_num_match.group(1) if article_num_match else "Unknown"
            
            # Extraer título del artículo (más flexible para el Estatuto del Consumidor)
            article_title_match = re.search(r'ART[IÍ]CULO\s+\d+[oº]?\.?\s*([A-ZÁÉÍÓÚÑ\s\-\,\.]+)\.', article_text)
            article_title = article_title_match.group(1).strip() if article_title_match else ""
            
            # Limpiar el título (remover puntos finales duplicados)
            if article_title.endswith('.'):
                article_title = article_title[:-1]
            
            # Determinar el título actual
            titulo_matches = list(re.finditer(r'TÍTULO\s+([IVXLCDM]+)\.?\s*([A-ZÁÉÍÓÚÑ\s\-\,\.]*)', self.raw_text[:article_start]))
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
            capitulo_matches = list(re.finditer(r'CAP[IÍ]TULO\s+([IVXLCDM]+)\.?\s*([A-ZÁÉÍÓÚÑ\s\-\,\.]*)', self.raw_text[:article_start]))
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
            paragrafo_pattern = r'PAR[ÁA]GRAFO\.?\s*(?:\d+[oº]?)?\s*'
            paragrafo_matches = re.findall(paragrafo_pattern, article_text)
            paragrafo_count = len(paragrafo_matches)
            
            # Extraer numerales si existen
            numeral_matches = re.findall(r'(\d+)\.?\s+([^\.]+)', article_text)
            numerales_count = len(numeral_matches)
            
            # Detectar si hay modificaciones/adiciones en parágrafos
            modificacion_matches = re.findall(r'<[^>]*(?:adicionado|modificado|corregido)[^>]*>', article_text)
            tiene_modificaciones = len(modificacion_matches) > 0
            
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
                "numerales_count": numerales_count,
                "tiene_modificaciones": tiene_modificaciones,
                "start": article_start,
                "end": match.end()
            }
            
            articles.append(article)
        
        return articles
    
    def create_chunks_from_articles(self, articles, chunk_size=1000, chunk_overlap=200):
        """Crear chunks a partir de los artículos identificados con estructura de metadata."""
        for article in articles:
            article_text = article["texto"]
            
            # Crear estructura de jerarquía
            jerarquia = {
                "ley": {
                    "numero": "1480",
                    "año": "2011",
                    "nombre": "Estatuto del Consumidor"
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
                            "tipo": "estatuto_consumidor",
                            "titulo": "Ley 1480 de 2011 - Estatuto del Consumidor",
                            "fecha": "2011-10-12"
                        },
                        "jerarquia": jerarquia,
                        "chunk": {
                            "es_articulo_completo": False,
                            "indice_chunk": i + 1,
                            "total_chunks": len(chunks),
                            "tamaño_caracteres": len(chunk),
                            "contiene_paragrafo": article["paragrafo_count"] > 0,
                            "numero_parrafos": article["paragrafo_count"],
                            "numero_numerales": article["numerales_count"],
                            "tiene_modificaciones": article["tiene_modificaciones"]
                        },
                        "tags": tags,
                        "ubicacion_legible": f"{ubicacion_legible} (Parte {i+1} de {len(chunks)})",
                        "referencias": {
                            "articulos_relacionados": [],
                            "conceptos_clave": tags[:3],
                            "materias": self.extract_subject_matters(article_text)
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
                        "tipo": "estatuto_consumidor",
                        "titulo": "Ley 1480 de 2011 - Estatuto del Consumidor",
                        "fecha": "2011-10-12"
                    },
                    "jerarquia": jerarquia,
                    "chunk": {
                        "es_articulo_completo": True,
                        "indice_chunk": 1,
                        "total_chunks": 1,
                        "tamaño_caracteres": len(article_text),
                        "contiene_paragrafo": article["paragrafo_count"] > 0,
                        "numero_parrafos": article["paragrafo_count"],
                        "numero_numerales": article["numerales_count"],
                        "tiene_modificaciones": article["tiene_modificaciones"]
                    },
                    "tags": tags,
                    "ubicacion_legible": ubicacion_legible,
                    "referencias": {
                        "articulos_relacionados": [],
                        "conceptos_clave": tags[:3],
                        "materias": self.extract_subject_matters(article_text)
                    }
                }
                
                self.processed_chunks.append({
                    "content": article_text,
                    "metadata": metadata
                })
        
        return self.processed_chunks
    
    def extract_subject_matters(self, text: str) -> List[str]:
        """Extraer materias específicas del derecho del consumidor."""
        materias = []
        
        subject_patterns = {
            "garantias": r"garant[íi]a",
            "responsabilidad_producto_defectuoso": r"producto defectuoso|responsabilidad por daños",
            "informacion_consumidor": r"información.*consumidor|derecho.*información",
            "publicidad_engañosa": r"publicidad engañosa",
            "comercio_electronico": r"comercio electrónico|ventas.*distancia",
            "contratos_adhesion": r"contrato.*adhesión|cláusula.*abusiva",
            "proteccion_contractual": r"protección contractual",
            "procedimiento_administrativo": r"procedimiento.*administrativo|investigación.*administrativa",
            "sanciones": r"sanción|multa|cierre.*establecimiento",
            "metrologia": r"metrología|unidades.*medida"
        }
        
        text_lower = text.lower()
        for materia, pattern in subject_patterns.items():
            if re.search(pattern, text_lower):
                materias.append(materia)
        
        return materias
    
    def process_document(self, chunk_size=1000, chunk_overlap=200):
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
    processor = EstatutoConsumidorProcessor(document_text)
    
    # Procesar el documento (chunk_size ajustado para textos legales)
    processor.process_document(chunk_size=1000, chunk_overlap=200)
    
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
        process_file("estatuto_consumidor.txt", "transporte_aereo")