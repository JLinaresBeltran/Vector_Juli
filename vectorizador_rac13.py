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

class RAC13Processor:
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
            stopwords = ['para', 'los', 'las', 'del', 'con', 'por', 'que', 'una', 'son', 'será', 'serán']
            tags.extend([word for word in title_words if len(word) > 3 and word not in stopwords])
        
        # Tags basados en palabras clave específicas del RAC 13 (sanciones)
        keywords = [
            'infracción', 'sanción', 'multa', 'uvt', 'aeronáutico', 'aeronave', 'piloto',
            'licencia', 'permiso', 'suspensión', 'cancelación', 'procedimiento', 'investigación',
            'flagrancia', 'reincidencia', 'atenuante', 'agravante', 'fallo', 'recurso',
            'aeropuerto', 'seguridad', 'operación', 'mantenimiento', 'tripulante', 'explotador',
            'transportador', 'vuelo', 'certificado', 'autorización', 'violación', 'incumplimiento',
            'pliego', 'cargos', 'descargos', 'competencia', 'UAEAC', 'administrativo', 'técnico'
        ]
        
        text_lower = section_text.lower()
        for keyword in keywords:
            if keyword in text_lower:
                tags.append(keyword)
        
        # Remover duplicados y retornar máximo 6 tags
        unique_tags = list(set(tags))
        return unique_tags[:6]
    
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
        
        return " > ".join(ubicacion_parts)
    
    def split_into_sections(self) -> List[Dict[str, Any]]:
        """Dividir el documento RAC 13 en secciones basadas en la numeración jerárquica."""
        sections = []
        
        # Patrón para identificar secciones numeradas del RAC 13
        section_pattern = r'^(13\.\d+(?:\.\d+)*)\s*\.?\s*([^.\n]+?)(?:\.|(?=\s+[A-Z])|\n|$)'
        
        # Identificar todas las secciones con sus posiciones
        matches = list(re.finditer(section_pattern, self.raw_text, re.MULTILINE))
        
        for i, match in enumerate(matches):
            section_num = match.group(1)
            raw_title = match.group(2).strip()
            
            # Limpiar el título
            section_title = self.clean_section_title(raw_title)
            
            # Si no encontramos un título limpio, usar el mapeo
            if not section_title or len(section_title) > 120:
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
        
        # Patrones comunes que indican el fin del título en el RAC 13
        title_end_patterns = [
            r'^([^.]+?)(?:\.\s*\([a-z]\))',  # Título seguido de ". (a)"
            r'^([^.]+?)(?:\.\s*[A-Z][a-z])', # Título seguido de ". Palabra"
            r'^([A-ZÁÉÍÓÚÑ\s]+)(?=\s+[a-z])', # Todo mayúsculas seguido de minúsculas
            r'^([^:]+?)(?=:\s*\([a-z]\))', # Título seguido de ": (a)"
        ]
        
        # Intentar cada patrón
        for pattern in title_end_patterns:
            match = re.match(pattern, title)
            if match:
                extracted = match.group(1).strip()
                if len(extracted) >= 8:  # Asegurar título mínimo
                    title = extracted
                    break
        
        # Si el título sigue siendo muy largo, cortar apropiadamente
        if len(title) > 120:
            for delimiter in ['. ', ', ', ' - ', ': ']:
                pos = title.find(delimiter)
                if 8 < pos < 120:
                    title = title[:pos].strip()
                    break
            
            # Si aún es muy largo, cortar en 100 caracteres
            if len(title) > 120:
                title = title[:100].strip() + "..."
        
        return title
    
    def build_hierarchy(self, section_num: str, section_title: str) -> Dict[str, Any]:
        """Construir la jerarquía basada en el número de sección."""
        parts = section_num.split('.')
        
        # Determinar el capítulo basado en el rango de secciones
        if section_num.startswith('13.0') or section_num.startswith('13.1') or \
           section_num.startswith('13.2') or section_num.startswith('13.3') or \
           section_num.startswith('13.4') or section_num.startswith('13.5') or \
           section_num.startswith('13.6') or section_num.startswith('13.7') or \
           section_num.startswith('13.8') or section_num.startswith('13.9'):
            if int(parts[1]) < 1000:
                capitulo = "A - DE LAS INFRACCIONES Y SANCIONES"
            else:
                capitulo = "B - PROCEDIMIENTO"
        else:
            capitulo = "13 - RÉGIMEN SANCIONATORIO"
        
        hierarchy = {"capitulo": capitulo}
        
        # Construir jerarquía específica
        if len(parts) >= 2:
            hierarchy["seccion_principal"] = {
                "numero": f"{parts[0]}.{parts[1]}",
                "titulo": self.get_section_title(f"{parts[0]}.{parts[1]}")
            }
        
        if len(parts) >= 3:
            hierarchy["subseccion"] = {
                "numero": section_num,
                "titulo": section_title
            }
        
        return hierarchy
    
    def get_section_title(self, section_num: str) -> str:
        """Obtener el título de una sección por su número."""
        # Mapeo manual de las secciones principales del RAC 13
        section_titles = {
            "13.001": "Normas descriptivas de las infracciones y sanciones",
            "13.005": "Ámbito de Aplicación",
            "13.010": "Principios Rectores",
            "13.015": "Facultad Sancionatoria",
            "13.020": "Otras actuaciones",
            "13.100": "De las infracciones",
            "13.105": "Concurso de infracciones",
            "13.110": "Tentativa",
            "13.200": "Reparación del daño",
            "13.300": "De las circunstancias atenuantes y agravantes",
            "13.400": "Inculpabilidad",
            "13.405": "Distribución de las infracciones para su investigación y sanción",
            "13.410": "Actividades de las dependencias a cargo de la investigación y sanción",
            "13.500": "De las sanciones",
            "13.510": "Infracciones a las normas administrativas",
            "13.515": "Multa equivalente a 111 U.V.T.",
            "13.520": "Multa equivalente a 185 U.V.T.",
            "13.525": "Multa equivalente a 370 U.V.T.",
            "13.530": "Multa equivalente a 444 U.V.T.",
            "13.535": "Multa equivalente a 567 U.V.T.",
            "13.540": "Multa equivalente a 740 U.V.T.",
            "13.545": "Multa equivalente a 912 U.V.T.",
            "13.550": "Multa equivalente a 1.109 U.V.T.",
            "13.555": "Multa equivalente a 1.849 U.V.T.",
            "13.560": "Multa equivalente a 3.698 U.V.T.",
            "13.565": "Multa equivalente a 7.395 U.V.T.",
            "13.570": "Multa equivalente a 11.093 U.V.T.",
            "13.575": "Multa equivalente a 18.488 U.V.T.",
            "13.600": "Infracciones a las normas técnicas",
            "13.605": "Multa equivalente a 99 U.V.T.",
            "13.615": "Multa equivalente a 123 U.V.T.",
            "13.625": "Multa equivalente a 247 U.V.T.",
            "13.635": "Multa equivalente a 370 U.V.T.",
            "13.645": "Multa equivalente a 493 U.V.T.",
            "13.660": "Multa equivalente a 740 U.V.T.",
            "13.670": "Multa equivalente a 1.233 U.V.T.",
            "13.680": "Multa equivalente a 2.465 U.V.T.",
            "13.690": "Multa equivalente a 4.930 U.V.T.",
            "13.700": "Multa equivalente a 12.325 U.V.T.",
            "13.1000": "Normas aplicables al procedimiento",
            "13.1005": "Disposiciones generales",
            "13.1010": "Iniciación de la actuación",
            "13.1015": "Denuncia por parte de otros funcionarios",
            "13.1020": "Investigación de accidentes",
            "13.1025": "Sujetos en el proceso sancionatorio",
            "13.1030": "Acumulación de procesos",
            "13.1035": "Conflictos de competencias",
            "13.1045": "Impedimentos y Recusaciones",
            "13.1050": "Pruebas",
            "13.1055": "Providencias",
            "13.1060": "Notificaciones",
            "13.1070": "Recursos",
            "13.1075": "Medidas Preventivas",
            "13.2000": "Procedimiento sancionatorio",
            "13.2005": "Formación de Expediente",
            "13.2010": "Derechos y deberes del Investigado",
            "13.2015": "Caducidad y Prescripción",
            "13.2020": "Normas para lo no previsto",
            "13.2025": "Trámite",
            "13.2030": "Actividades de las dependencias a cargo de la investigación y sanción",
            "13.2035": "Averiguaciones preliminares",
            "13.2040": "Apertura de la investigación",
            "13.2045": "Formulación de Cargos, y descargos del implicado",
            "13.2050": "Actuación de las dependencias y funcionarios designados, frente a casos de Flagrancia",
            "13.2055": "Aviso a terceros",
            "13.2060": "Práctica de pruebas",
            "13.2065": "Período probatorio y alegatos",
            "13.2070": "Decisión",
            "13.2075": "Terminación y archivo del proceso",
            "13.2080": "Terminación por confesión",
            "13.2085": "Reducción de multa por pago anticipado",
            "13.2090": "Comunicación de decisión",
            "13.2095": "Sanción accesoria",
            "13.2100": "Información al implicado",
            "13.2105": "Aplicación a otras actuaciones sancionatorias"
        }
        
        return section_titles.get(section_num, "")
    
    def create_chunks_from_sections(self, sections, chunk_size=1000, chunk_overlap=200):
        """Crear chunks a partir de las secciones identificadas."""
        for section in sections:
            section_text = section["texto"]
            
            # Extraer tags para la sección
            tags = self.extract_tags_from_section(section_text, section["titulo"])
            
            # Crear ubicación legible
            ubicacion_legible = self.create_ubicacion_legible(section["jerarquia"])
            
            # Decidir si la sección debe dividirse en chunks más pequeños
            if len(section_text) > chunk_size * 1.8:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", ". ", " "],
                    keep_separator=True
                )
                chunks = text_splitter.split_text(section_text)
                
                for i, chunk in enumerate(chunks):
                    # Crear metadata específica para RAC 13
                    metadata = {
                        "documento": {
                            "tipo": "reglamento_aeronautico",
                            "titulo": "Reglamentos Aeronáuticos de Colombia - RAC 13",
                            "capitulo": section["jerarquia"]["capitulo"]
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
                        },
                        "tipo_contenido": self.classify_content_type(section["numero"], section["titulo"])
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
                        "titulo": "Reglamentos Aeronáuticos de Colombia - RAC 13",
                        "capitulo": section["jerarquia"]["capitulo"]
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
                    },
                    "tipo_contenido": self.classify_content_type(section["numero"], section["titulo"])
                }
                
                self.processed_chunks.append({
                    "content": section_text,
                    "metadata": metadata
                })
        
        return self.processed_chunks
    
    def classify_content_type(self, section_num: str, section_title: str) -> str:
        """Clasificar el tipo de contenido de la sección."""
        title_lower = section_title.lower()
        
        if "multa" in title_lower or "u.v.t" in title_lower:
            return "sanciones_multas"
        elif "procedimiento" in title_lower or "trámite" in title_lower:
            return "procedimiento"
        elif "infracción" in title_lower:
            return "infracciones"
        elif "recurso" in title_lower or "apelación" in title_lower:
            return "recursos"
        elif "medida preventiva" in title_lower:
            return "medidas_preventivas"
        elif "competencia" in title_lower:
            return "competencia"
        elif section_num.startswith("13.0") or section_num.startswith("13.1"):
            return "disposiciones_generales"
        elif section_num.startswith("13.2"):
            return "procedimiento"
        else:
            return "normativo_general"
    
    def process_document(self, chunk_size=1000, chunk_overlap=200):
        """Procesar el documento completo."""
        # Dividir en secciones
        sections = self.split_into_sections()
        print(f"Secciones identificadas: {len(sections)}")
        
        # Imprimir resumen de secciones para verificación
        print("\nResumen de secciones encontradas:")
        for s in sections[:15]:  # Mostrar primeras 15 secciones
            print(f"  {s['numero']}: {s['titulo'][:60]}{'...' if len(s['titulo']) > 60 else ''}")
        
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
    """Procesa un archivo de texto del RAC 13 y lo vectoriza en Supabase."""
    # Leer el archivo
    with open(file_path, "r", encoding="utf-8") as f:
        document_text = f.read()
    
    # Crear el procesador
    processor = RAC13Processor(document_text)
    
    # Procesar el documento
    processor.process_document(chunk_size=1000, chunk_overlap=200)
    
    # Vectorizar y almacenar
    processor.vectorize_and_store()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        process_file(file_path)
    else:
        # Si no se proporciona un archivo, usar el archivo predeterminado
        process_file("rac13.txt")