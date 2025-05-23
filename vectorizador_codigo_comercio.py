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

class CodigoComercioProcessor:
    def __init__(self, text: str):
        self.raw_text = text
        self.processed_chunks = []
        self.current_parte = None
        self.current_titulo = None
        self.current_capitulo = None
        self.current_seccion = None
        
    def extract_tags_from_article(self, article_text: str, article_title: str) -> List[str]:
        """Extraer tags relevantes del artículo para búsquedas."""
        tags = []
        
        # Tags basados en el título del artículo
        if article_title:
            # Convertir a minúsculas y dividir en palabras
            title_words = re.findall(r'\b[a-záéíóúñ]+\b', article_title.lower())
            tags.extend([word for word in title_words if len(word) > 3])
        
        # Tags basados en palabras clave comunes en el texto
        keywords = [
            'comerciante', 'comercio', 'sociedad', 'empresa', 'contrato', 
            'obligación', 'registro', 'mercantil', 'acto', 'negocio',
            'compraventa', 'transporte', 'seguro', 'banco', 'título valor'
        ]
        
        text_lower = article_text.lower()
        for keyword in keywords:
            if keyword in text_lower:
                tags.append(keyword.replace(' ', '_'))
        
        # Remover duplicados y retornar máximo 5 tags
        return list(set(tags))[:5]
    
    def create_ubicacion_legible(self, estructura: Dict[str, Any], articulo_numero: str) -> str:
        """Crear una descripción legible de la ubicación del artículo."""
        ubicacion_parts = []
        
        if estructura.get("parte"):
            ubicacion_parts.append(f"Parte {estructura['parte']}")
        
        if estructura.get("titulo"):
            ubicacion_parts.append(f"Título {estructura['titulo']}")
        
        if estructura.get("capitulo") and estructura["capitulo"].get("numero"):
            cap_str = f"Capítulo {estructura['capitulo']['numero']}"
            if estructura["capitulo"].get("nombre"):
                cap_str += f" ({estructura['capitulo']['nombre']})"
            ubicacion_parts.append(cap_str)
        
        if estructura.get("seccion"):
            ubicacion_parts.append(f"Sección {estructura['seccion']}")
        
        ubicacion_parts.append(f"Artículo {articulo_numero}")
        
        return " - ".join(ubicacion_parts)
    
    def split_into_articles(self) -> List[Dict[str, Any]]:
        """Dividir el documento en artículos y extraer metadatos contextuales."""
        # Dividir por artículos
        article_pattern = r'(ART[IÍ]CULO\s+\d+\..*?)(?=ART[IÍ]CULO\s+\d+\.|$)'
        articles = []
        
        for match in re.finditer(article_pattern, self.raw_text, re.DOTALL):
            article_text = match.group(1).strip()
            article_start = match.start()
            
            # Extraer número de artículo
            article_num_match = re.search(r'ART[IÍ]CULO\s+(\d+)', article_text)
            article_num = article_num_match.group(1) if article_num_match else "Unknown"
            
            # Extraer título del artículo
            article_title_match = re.search(r'ART[IÍ]CULO\s+\d+\.?\s*([A-ZÁÉÍÓÚÑ\s\-]+)\.', article_text)
            article_title = article_title_match.group(1).strip() if article_title_match else ""
            
            # Determinar la parte actual
            parte_matches = list(re.finditer(r'PARTE\s+([A-ZÁÉÍÓÚÑ]+)\.', self.raw_text[:article_start]))
            if parte_matches:
                self.current_parte = parte_matches[-1].group(1)
            
            # Determinar el título actual
            titulo_matches = list(re.finditer(r'TÍ[TD]ULO\s+([A-ZÁÉÍÓÚÑ\s]+)\.', self.raw_text[:article_start]))
            if titulo_matches:
                self.current_titulo = titulo_matches[-1].group(1).strip()
            
            # Determinar el capítulo actual
            capitulo_matches = list(re.finditer(r'CAP[IÍ]TULO\s+([IVX]+)\.?\s*([A-ZÁÉÍÓÚÑ\s]+)?', self.raw_text[:article_start]))
            if capitulo_matches:
                latest_match = capitulo_matches[-1]
                num_capitulo = latest_match.group(1)
                nombre_capitulo = latest_match.group(2).strip() if latest_match.group(2) else ""
                self.current_capitulo = {
                    "numero": num_capitulo,
                    "nombre": nombre_capitulo
                }
            
            # Determinar la sección actual
            seccion_matches = list(re.finditer(r'SECCI[ÓO]N\s+([IVX]+)\.', self.raw_text[:article_start]))
            if seccion_matches:
                self.current_seccion = seccion_matches[-1].group(1)
            
            # Extraer parágrafos si existen
            paragrafo_exists = bool(re.search(r'PAR[ÁA]GRAFO\.', article_text))
            
            # Extraer literales si existen
            literal_matches = re.findall(r'(\d+)\)\s+([^;\.]+)[;.]', article_text)
            literales_count = len(literal_matches)
            
            # Crear estructura del artículo
            article = {
                "numero": article_num,
                "titulo": article_title,
                "texto": article_text,
                "estructura": {
                    "parte": self.current_parte,
                    "titulo": self.current_titulo,
                    "capitulo": self.current_capitulo,
                    "seccion": self.current_seccion
                },
                "paragrafo": paragrafo_exists,
                "literales_count": literales_count,
                "start": article_start,
                "end": match.end()
            }
            
            articles.append(article)
        
        return articles
    
    def create_chunks_from_articles(self, articles, chunk_size=800, chunk_overlap=150):
        """Crear chunks a partir de los artículos identificados con nueva estructura de metadata."""
        for article in articles:
            article_text = article["texto"]
            
            # Crear estructura de jerarquia
            jerarquia = {
                "parte": article["estructura"]["parte"],
                "titulo": article["estructura"]["titulo"],
                "capitulo": article["estructura"]["capitulo"],
                "seccion": article["estructura"]["seccion"],
                "articulo": {
                    "numero": article["numero"],
                    "titulo": article["titulo"]
                }
            }
            
            # Extraer tags para el artículo
            tags = self.extract_tags_from_article(article_text, article["titulo"])
            
            # Crear ubicación legible
            ubicacion_legible = self.create_ubicacion_legible(article["estructura"], article["numero"])
            
            # Decidir si el artículo debe dividirse en chunks más pequeños
            if len(article_text) > chunk_size * 1.5:  # Si el artículo es significativamente largo
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", ".", " "]
                )
                chunks = text_splitter.split_text(article_text)
                
                for i, chunk in enumerate(chunks):
                    # Crear metadata según nueva estructura
                    metadata = {
                        "documento": {
                            "tipo": "codigo_comercio",
                            "titulo": "Código de Comercio"
                        },
                        "jerarquia": jerarquia,
                        "chunk": {
                            "es_articulo_completo": False,
                            "indice_chunk": i + 1,
                            "total_chunks": len(chunks),
                            "tamaño_caracteres": len(chunk),
                            "contiene_paragrafo": article["paragrafo"],
                            "numero_literales": article["literales_count"]
                        },
                        "tags": tags,
                        "ubicacion_legible": f"{ubicacion_legible} (Parte {i+1} de {len(chunks)})",
                        "referencias": {
                            "articulos_relacionados": [],  # Se puede implementar posteriormente
                            "conceptos_clave": tags[:3]  # Primeros 3 tags como conceptos clave
                        }
                    }
                    
                    self.processed_chunks.append({
                        "content": chunk,
                        "metadata": metadata
                    })
            else:
                # Si el artículo es pequeño, lo mantenemos como un solo chunk
                metadata = {
                    "documento": {
                        "tipo": "codigo_comercio",
                        "titulo": "Código de Comercio"
                    },
                    "jerarquia": jerarquia,
                    "chunk": {
                        "es_articulo_completo": True,
                        "indice_chunk": 1,
                        "total_chunks": 1,
                        "tamaño_caracteres": len(article_text),
                        "contiene_paragrafo": article["paragrafo"],
                        "numero_literales": article["literales_count"]
                    },
                    "tags": tags,
                    "ubicacion_legible": ubicacion_legible,
                    "referencias": {
                        "articulos_relacionados": [],  # Se puede implementar posteriormente
                        "conceptos_clave": tags[:3]  # Primeros 3 tags como conceptos clave
                    }
                }
                
                self.processed_chunks.append({
                    "content": article_text,
                    "metadata": metadata
                })
        
        return self.processed_chunks
    
    def process_document(self, chunk_size=800, chunk_overlap=150):
        """Procesar el documento completo."""
        # Dividir en artículos
        articles = self.split_into_articles()
        print(f"Artículos identificados: {len(articles)}")
        
        # Crear chunks
        self.create_chunks_from_articles(articles, chunk_size, chunk_overlap)
        print(f"Chunks generados: {len(self.processed_chunks)}")
        
        return self.processed_chunks
    
    def vectorize_and_store(self):
        """Generar embeddings y almacenar en Supabase."""
        total_chunks = len(self.processed_chunks)
        
        for i, chunk in enumerate(self.processed_chunks, 1):
            # Generar el embedding
            embedding_vector = embeddings.embed_query(chunk["content"])
            
            # Almacenar en Supabase
            try:
                result = supabase.table("transporte_aereo").insert({
                    "content": chunk["content"],
                    "metadata": chunk["metadata"],
                    "embedding": embedding_vector
                }).execute()
                
                print(f"[{i}/{total_chunks}] Chunk del artículo {chunk['metadata']['jerarquia']['articulo']['numero']} almacenado correctamente")
            except Exception as e:
                print(f"Error al almacenar chunk {i}/{total_chunks}: {e}")
                
        print(f"Procesamiento completado. {len(self.processed_chunks)} chunks generados y almacenados.")

def process_file(file_path):
    """Procesa un archivo de texto y lo vectoriza en Supabase."""
    # Leer el archivo
    with open(file_path, "r", encoding="utf-8") as f:
        document_text = f.read()
    
    # Crear el procesador
    processor = CodigoComercioProcessor(document_text)
    
    # Procesar el documento (chunk_size más pequeño para textos legales)
    processor.process_document(chunk_size=800, chunk_overlap=150)
    
    # Vectorizar y almacenar
    processor.vectorize_and_store()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        process_file(file_path)
    else:
        # Si no se proporciona un archivo, usar el archivo predeterminado
        process_file("codigo_comercio.txt")