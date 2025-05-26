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

class RAC3Processor:
    def __init__(self, text: str):
        self.raw_text = text
        self.processed_chunks = []
        self.current_section_hierarchy = {}
        
    def extract_tags_from_section(self, section_text: str, section_title: str) -> List[str]:
        """Extraer tags relevantes de la sección para búsquedas."""
        tags = []
        
        # Tags basados en el título de la sección
        if section_title:
            # Convertir a minúsculas y dividir en palabras
            title_words = re.findall(r'\b[a-záéíóúñ]+\b', section_title.lower())
            # Filtrar palabras comunes y muy cortas
            stopwords = ['para', 'los', 'las', 'del', 'con', 'por', 'que', 'una']
            tags.extend([word for word in title_words if len(word) > 3 and word not in stopwords])
        
        # Tags basados en palabras clave comunes en el texto
        keywords = [
            'pasajero', 'transportador', 'aerolínea', 'vuelo', 'reserva', 'tiquete',
            'equipaje', 'embarque', 'desembarque', 'itinerario', 'compensación',
            'reembolso', 'demora', 'cancelación', 'sobreventa', 'derechos', 'deberes',
            'seguridad', 'aeropuerto', 'aeronave', 'tripulación', 'slot', 'frecuencia',
            'registro', 'aprobación', 'modificación', 'score', 'ssim', 'temporada'
        ]
        
        text_lower = section_text.lower()
        for keyword in keywords:
            if keyword in text_lower:
                tags.append(keyword)
        
        # Remover duplicados y retornar máximo 5 tags
        unique_tags = list(set(tags))
        return unique_tags[:5]
    
    def create_ubicacion_legible(self, jerarquia: Dict[str, Any]) -> str:
        """Crear una descripción legible de la ubicación en el RAC."""
        ubicacion_parts = []
        
        if jerarquia.get("capitulo"):
            ubicacion_parts.append(f"Capítulo {jerarquia['capitulo']}")
        
        if jerarquia.get("seccion_principal") and jerarquia["seccion_principal"].get("numero"):
            sec = f"Sección {jerarquia['seccion_principal']['numero']}"
            if jerarquia['seccion_principal'].get('titulo'):
                sec += f" - {jerarquia['seccion_principal']['titulo']}"
            ubicacion_parts.append(sec)
        
        if jerarquia.get("subseccion") and jerarquia["subseccion"].get("numero"):
            subsec = f"Subsección {jerarquia['subseccion']['numero']}"
            if jerarquia['subseccion'].get('titulo'):
                subsec += f" - {jerarquia['subseccion']['titulo']}"
            ubicacion_parts.append(subsec)
        
        if jerarquia.get("apartado") and jerarquia["apartado"].get("numero"):
            apart = f"Apartado {jerarquia['apartado']['numero']}"
            if jerarquia['apartado'].get('titulo'):
                apart += f" - {jerarquia['apartado']['titulo']}"
            ubicacion_parts.append(apart)
        
        return " > ".join(ubicacion_parts)
    
    def split_into_sections(self) -> List[Dict[str, Any]]:
        """Dividir el documento RAC en secciones basadas en la numeración jerárquica."""
        sections = []
        
        # Patrón para identificar secciones numeradas del RAC
        # Mejorado para capturar solo el título real, no el contenido
        section_pattern = r'^(3\.\d+(?:\.\d+)*)\s*\.?\s*([^.\n]+?)(?:\.|(?=\s+[A-Z])|\n|$)'
        
        # Primero, identificar todas las secciones con sus posiciones
        matches = list(re.finditer(section_pattern, self.raw_text, re.MULTILINE))
        
        for i, match in enumerate(matches):
            section_num = match.group(1)
            raw_title = match.group(2).strip()
            
            # Para secciones de nivel 3 (como 3.11.2.1), intentar extraer título más específico
            if section_num.count('.') >= 2:
                # Buscar patrones como "Registro y aprobación" antes de contenido largo
                title_match = re.match(r'^([A-Za-záéíóúñÁÉÍÓÚÑ\s]+?)(?:\.|:|\s+[A-Z][a-z])', raw_title)
                if title_match:
                    section_title = title_match.group(1).strip()
                else:
                    section_title = self.clean_section_title(raw_title)
            else:
                section_title = self.clean_section_title(raw_title)
            
            # Si no encontramos un título o es muy largo, usar el mapeo
            if not section_title or len(section_title) > 100:
                mapped_title = self.get_section_title(section_num)
                if mapped_title:
                    section_title = mapped_title
            
            section_start = match.start()
            
            # El contenido de la sección va hasta el inicio de la siguiente sección
            if i < len(matches) - 1:
                section_end = matches[i + 1].start()
            else:
                section_end = len(self.raw_text)
            
            section_text = self.raw_text[section_start:section_end].strip()
            
            # Determinar el nivel de jerarquía basado en el número de puntos
            level = section_num.count('.')
            
            # Construir la jerarquía
            jerarquia = self.build_hierarchy(section_num, section_title)
            
            # Verificar si hay subsecciones con letras (a), (b), etc.
            has_letter_subsections = bool(re.search(r'\([a-z]\)', section_text))
            
            # Contar párrafos
            paragraphs = len(re.findall(r'\n\s*\n', section_text))
            
            section = {
                "numero": section_num,
                "titulo": section_title,
                "texto": section_text,
                "nivel": level,
                "jerarquia": jerarquia,
                "tiene_subsecciones_letras": has_letter_subsections,
                "numero_parrafos": paragraphs,
                "start": section_start,
                "end": section_end
            }
            
            sections.append(section)
        
        return sections
    
    def clean_section_title(self, title: str) -> str:
        """Limpiar y acortar títulos de sección que sean muy largos."""
        # Remover espacios extra
        title = ' '.join(title.split())
        
        # Patrones comunes que indican el fin del título en el RAC
        title_end_patterns = [
            r'^([^.]+?)(?:\.\s*El)',  # Título seguido de ". El"
            r'^([^.]+?)(?:\.\s*La)',  # Título seguido de ". La"
            r'^([^.]+?)(?:\.\s*Los)', # Título seguido de ". Los"
            r'^([^.]+?)(?:\.\s*Las)', # Título seguido de ". Las"
            r'^([^.]+?)(?:\.\s*En)',  # Título seguido de ". En"
            r'^([^.]+?)(?:\.\s*Para)', # Título seguido de ". Para"
            r'^([A-ZÁÉÍÓÚÑ\s]+)(?=\s+[A-Z][a-z])', # Todo mayúsculas seguido de oración normal
            r'^([^.]+?)(?=\s+[A-Z][a-z][^.]*\.)', # Título seguido de oración completa
        ]
        
        # Intentar cada patrón
        for pattern in title_end_patterns:
            match = re.match(pattern, title)
            if match:
                extracted = match.group(1).strip()
                if len(extracted) >= 10:  # Asegurar título mínimo
                    title = extracted
                    break
        
        # Si el título sigue siendo muy largo, cortar en el primer punto o coma
        if len(title) > 100:
            for delimiter in ['. ', ', ', ' - ']:
                pos = title.find(delimiter)
                if 10 < pos < 100:
                    title = title[:pos].strip()
                    break
            
            # Si aún es muy largo, cortar en 80 caracteres
            if len(title) > 100:
                title = title[:80].strip() + "..."
        
        # Capitalizar apropiadamente si está todo en mayúsculas
        if title.isupper() and len(title) > 20:
            # Mantener mayúsculas para siglas conocidas
            siglas = ['RAC', 'SCORE', 'SSIM', 'ATFCM', 'OCS']
            words = title.split()
            title = ' '.join([w if w in siglas else w.title() for w in words])
        
        return title
    
    def build_hierarchy(self, section_num: str, section_title: str) -> Dict[str, Any]:
        """Construir la jerarquía basada en el número de sección."""
        parts = section_num.split('.')
        hierarchy = {"capitulo": "3"}  # Siempre es el capítulo 3 del RAC
        
        # Actualizar la jerarquía actual
        if len(parts) >= 2:
            hierarchy["seccion_principal"] = {
                "numero": f"{parts[0]}.{parts[1]}",
                "titulo": self.get_section_title(f"{parts[0]}.{parts[1]}")
            }
        
        if len(parts) >= 3:
            hierarchy["subseccion"] = {
                "numero": f"{parts[0]}.{parts[1]}.{parts[2]}",
                "titulo": self.get_section_title(f"{parts[0]}.{parts[1]}.{parts[2]}")
            }
        
        if len(parts) >= 4:
            hierarchy["apartado"] = {
                "numero": section_num,
                "titulo": section_title
            }
        
        return hierarchy
    
    def get_section_title(self, section_num: str) -> str:
        """Obtener el título de una sección por su número."""
        # Mapeo manual de las secciones principales del RAC 3
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
    
    def create_chunks_from_sections(self, sections, chunk_size=800, chunk_overlap=150):
        """Crear chunks a partir de las secciones identificadas."""
        for section in sections:
            section_text = section["texto"]
            
            # Extraer tags para la sección
            tags = self.extract_tags_from_section(section_text, section["titulo"])
            
            # Crear ubicación legible
            ubicacion_legible = self.create_ubicacion_legible(section["jerarquia"])
            
            # Decidir si la sección debe dividirse en chunks más pequeños
            if len(section_text) > chunk_size * 1.5:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", ". ", " "],
                    keep_separator=True
                )
                chunks = text_splitter.split_text(section_text)
                
                for i, chunk in enumerate(chunks):
                    # Crear metadata según estructura
                    metadata = {
                        "documento": {
                            "tipo": "reglamento_aeronautico",
                            "titulo": "Reglamentos Aeronáuticos de Colombia - RAC 3",
                            "capitulo": "3 - Actividades Aéreas Civiles"
                        },
                        "jerarquia": section["jerarquia"],
                        "seccion": {
                            "numero": section["numero"],
                            "titulo": section["titulo"],
                            "nivel": section["nivel"]
                        },
                        "chunk": {
                            "es_seccion_completa": False,
                            "indice_chunk": i + 1,
                            "total_chunks": len(chunks),
                            "tamaño_caracteres": len(chunk),
                            "contiene_subsecciones_letras": section["tiene_subsecciones_letras"]
                        },
                        "tags": tags,
                        "ubicacion_legible": f"{ubicacion_legible} (Parte {i+1} de {len(chunks)})",
                        "referencias": {
                            "secciones_relacionadas": [],
                            "conceptos_clave": tags[:3]
                        }
                    }
                    
                    self.processed_chunks.append({
                        "content": chunk,
                        "metadata": metadata
                    })
            else:
                # Si la sección es pequeña, mantenerla como un solo chunk
                metadata = {
                    "documento": {
                        "tipo": "reglamento_aeronautico",
                        "titulo": "Reglamentos Aeronáuticos de Colombia - RAC 3",
                        "capitulo": "3 - Actividades Aéreas Civiles"
                    },
                    "jerarquia": section["jerarquia"],
                    "seccion": {
                        "numero": section["numero"],
                        "titulo": section["titulo"],
                        "nivel": section["nivel"]
                    },
                    "chunk": {
                        "es_seccion_completa": True,
                        "indice_chunk": 1,
                        "total_chunks": 1,
                        "tamaño_caracteres": len(section_text),
                        "contiene_subsecciones_letras": section["tiene_subsecciones_letras"]
                    },
                    "tags": tags,
                    "ubicacion_legible": ubicacion_legible,
                    "referencias": {
                        "secciones_relacionadas": [],
                        "conceptos_clave": tags[:3]
                    }
                }
                
                self.processed_chunks.append({
                    "content": section_text,
                    "metadata": metadata
                })
        
        return self.processed_chunks
    
    def process_document(self, chunk_size=800, chunk_overlap=150):
        """Procesar el documento completo."""
        # Dividir en secciones
        sections = self.split_into_sections()
        print(f"Secciones identificadas: {len(sections)}")
        
        # Imprimir resumen de secciones para verificación
        print("\nResumen de secciones encontradas:")
        for s in sections[:10]:  # Mostrar primeras 10 secciones
            print(f"  {s['numero']}: {s['titulo'][:50]}{'...' if len(s['titulo']) > 50 else ''}")
        
        # Crear chunks
        self.create_chunks_from_sections(sections, chunk_size, chunk_overlap)
        print(f"\nChunks generados: {len(self.processed_chunks)}")
        
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
                
                seccion_num = chunk['metadata']['seccion']['numero']
                print(f"[{i}/{total_chunks}] Chunk de la sección {seccion_num} almacenado correctamente")
            except Exception as e:
                print(f"Error al almacenar chunk {i}/{total_chunks}: {e}")
                
        print(f"Procesamiento completado. {len(self.processed_chunks)} chunks generados y almacenados.")

def process_file(file_path):
    """Procesa un archivo de texto del RAC 3 y lo vectoriza en Supabase."""
    # Leer el archivo
    with open(file_path, "r", encoding="utf-8") as f:
        document_text = f.read()
    
    # Crear el procesador
    processor = RAC3Processor(document_text)
    
    # Procesar el documento
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
        process_file("rac3.txt")