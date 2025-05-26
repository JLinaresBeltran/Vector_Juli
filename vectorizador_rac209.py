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

class RAC209Processor:
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
            stopwords = ['para', 'los', 'las', 'del', 'con', 'por', 'que', 'una', 'son', 'será', 'serán', 'sobre', 'desde', 'hasta']
            tags.extend([word for word in title_words if len(word) > 3 and word not in stopwords])
        
        # Tags basados en palabras clave específicas del RAC 209 (facilitación del transporte aéreo)
        keywords = [
            'facilitación', 'transporte', 'aéreo', 'aeropuerto', 'aeronave', 'pasajero', 'tripulación',
            'inspección', 'control', 'fronterizo', 'equipaje', 'carga', 'correo', 'despacho',
            'entrada', 'salida', 'tránsito', 'transferencia', 'embarque', 'desembarque',
            'migración', 'aduana', 'aduanero', 'inmigración', 'visado', 'pasaporte',
            'documento', 'viaje', 'identidad', 'verificación', 'autenticación',
            'terminal', 'instalación', 'servicio', 'infraestructura', 'plataforma',
            'explotador', 'operador', 'línea', 'aérea', 'autoridad', 'competente',
            'internacional', 'nacional', 'bilateral', 'cooperación', 'acuerdo',
            'discapacidad', 'movilidad', 'reducida', 'asistencia', 'especial',
            'emergencia', 'contingencia', 'socorro', 'evacuación', 'desastre',
            'menor', 'acompañado', 'tripulante', 'piloto', 'comandante',
            'zona', 'área', 'pública', 'restringida', 'estéril', 'seguridad',
            'API', 'PNR', 'datos', 'información', 'electrónico', 'sistema',
            'facilitación', 'agilización', 'demora', 'eficiencia', 'calidad'
        ]
        
        text_lower = section_text.lower()
        for keyword in keywords:
            if keyword in text_lower:
                tags.append(keyword)
        
        # Remover duplicados y retornar máximo 8 tags
        unique_tags = list(set(tags))
        return unique_tags[:8]
    
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
        
        return " > ".join(ubicacion_parts)
    
    def split_into_sections(self) -> List[Dict[str, Any]]:
        """Dividir el documento RAC 209 en secciones basadas en la numeración jerárquica."""
        sections = []
        
        # Patrón para identificar secciones numeradas del RAC 209
        section_pattern = r'^(209\.\d+)\s*\.?\s*([^.\n]+?)(?:\s+\([a-z]\)|\n|$)'
        
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
        
        # Patrones comunes que indican el fin del título en el RAC 209
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
        
        # Determinar el capítulo basado en el número de sección
        capitulo = self.get_chapter_from_section(section_num)
        
        hierarchy = {"capitulo": capitulo}
        
        # Para RAC 209, la estructura es más simple
        hierarchy["seccion_principal"] = {
            "numero": section_num,
            "titulo": section_title
        }
        
        return hierarchy
    
    def get_chapter_from_section(self, section_num: str) -> str:
        """Determinar el capítulo basado en el número de sección RAC 209."""
        section_int = int(section_num.split('.')[1])
        
        if section_int <= 15:
            return "A - GENERALIDADES"
        elif section_int <= 580:
            return "ENTRADA Y SALIDA DE PERSONAS Y DE SU EQUIPAJE"
        elif section_int <= 855:
            return "I - INSTALACIONES Y SERVICIOS PARA EL TRÁFICO EN LOS AEROPUERTOS"
        elif section_int <= 910:
            return "J - ATERRIZAJE FUERA DE LOS AEROPUERTOS INTERNACIONALES"
        elif section_int <= 1055:
            return "K - DISPOSICIONES SOBRE FACILITACIÓN DE ASPECTOS ESPECÍFICOS"
        elif section_int <= 1115:
            return "L - SISTEMAS DE INTERCAMBIO DE DATOS SOBRE LOS PASAJEROS"
        else:
            return "DISPOSICIONES VARIAS"
    
    def get_section_title(self, section_num: str) -> str:
        """Obtener el título de una sección por su número."""
        # Mapeo manual de las secciones principales del RAC 209
        section_titles = {
            "209.001": "Objetivo",
            "209.005": "Aplicabilidad",
            "209.010": "Definiciones",
            "209.015": "Abreviaturas",
            "209.500": "Generalidades",
            "209.505": "Documentos requeridos para viajar",
            "209.510": "Seguridad de los documentos de viaje",
            "209.515": "Documentos de viaje",
            "209.520": "Visados de salida",
            "209.525": "Visados de entrada y reingreso",
            "209.530": "Tarjetas de embarque/desembarque",
            "209.535": "Certificados internacionales de vacunación o profilaxis",
            "209.540": "Inspección de documentos de viaje",
            "209.545": "Procedimientos de salida",
            "209.550": "Procedimientos de entrada y responsabilidades",
            "209.555": "Procedimientos y requisitos de tránsito",
            "209.560": "Disposición del equipaje separado de su propietario",
            "209.565": "Identificación y entrada de la tripulación y otro personal",
            "209.570": "Inspectores de la aviación civil",
            "209.575": "Asistencia de emergencia/visados de entrada en casos de fuerza mayor",
            "209.580": "Menores",
            "209.800": "Generalidades",
            "209.805": "Disposiciones relativas al movimiento del tráfico en los aeropuertos",
            "209.810": "Disposiciones relativas al estacionamiento y al servicio de las aeronaves",
            "209.815": "Salida de pasajeros, tripulaciones y equipajes",
            "209.820": "Entrada de pasajeros, tripulaciones y equipajes",
            "209.825": "Tránsito y trasbordo de pasajeros y tripulaciones",
            "209.830": "Instalaciones y servicios varios en los edificios terminales",
            "209.835": "Instalaciones para el manejo y despacho de la carga y el correo",
            "209.840": "Instalaciones y servicios necesarios para implementar las medidas de sanidad pública",
            "209.845": "Instalaciones necesarias para los controles de despacho",
            "209.850": "Pasajeros insubordinados, perturbadores o indisciplinados",
            "209.855": "Comodidades para los pasajeros",
            "209.900": "Generalidades",
            "209.905": "Breve parada-estancia",
            "209.910": "Interrupción del vuelo",
            "209.1035": "Facilitación del transporte de las personas en condición de discapacidad",
            "209.1040": "Acceso a los aeropuertos",
            "209.1045": "Acceso a los servicios aéreos",
            "209.1050": "Asistencia a las víctimas de accidentes de aviación y a sus familiares",
            "209.1055": "Trata de personas",
            "209.1100": "Generalidades",
            "209.1105": "Información Anticipada sobre los Pasajeros (API)",
            "209.1110": "Sistemas Electrónicos de Viaje (ETS)",
            "209.1115": "Datos del Registro de Nombres de los Pasajeros (PNR)"
        }
        
        return section_titles.get(section_num, "")
    
    def create_chunks_from_sections(self, sections, chunk_size=1200, chunk_overlap=200):
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
                    # Crear metadata específica para RAC 209
                    metadata = {
                        "documento": {
                            "tipo": "reglamento_aeronautico",
                            "titulo": "Reglamentos Aeronáuticos de Colombia - RAC 209",
                            "capitulo": section["jerarquia"]["capitulo"]
                        },
                        "jerarquia": section["jerarquia"],
                        "seccion": {
                            "numero": section["numero"],
                            "titulo": section["titulo"]
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
                        "titulo": "Reglamentos Aeronáuticos de Colombia - RAC 209",
                        "capitulo": section["jerarquia"]["capitulo"]
                    },
                    "jerarquia": section["jerarquia"],
                    "seccion": {
                        "numero": section["numero"],
                        "titulo": section["titulo"]
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
        """Clasificar el tipo de contenido de la sección RAC 209."""
        title_lower = section_title.lower()
        section_int = int(section_num.split('.')[1])
        
        if "definicion" in title_lower or "abreviatura" in title_lower:
            return "definiciones_terminologia"
        elif "objetivo" in title_lower or "aplicabilidad" in title_lower:
            return "objetivos_aplicabilidad"
        elif "documento" in title_lower or "pasaporte" in title_lower or "visado" in title_lower:
            return "documentos_viaje"
        elif "entrada" in title_lower or "salida" in title_lower or "procedimientos" in title_lower:
            return "procedimientos_fronterizos"
        elif "equipaje" in title_lower or "carga" in title_lower or "correo" in title_lower:
            return "equipaje_carga"
        elif "instalacion" in title_lower or "servicio" in title_lower or "infraestructura" in title_lower:
            return "instalaciones_servicios"
        elif "tripulación" in title_lower or "personal" in title_lower or "inspector" in title_lower:
            return "personal_tripulacion"
        elif "emergencia" in title_lower or "contingencia" in title_lower or "socorro" in title_lower:
            return "emergencia_contingencia"
        elif "discapacidad" in title_lower or "movilidad" in title_lower or "asistencia" in title_lower:
            return "accesibilidad_asistencia"
        elif "datos" in title_lower or "información" in title_lower or "API" in title_lower or "PNR" in title_lower:
            return "sistemas_datos"
        elif section_int <= 15:
            return "generalidades"
        elif section_int <= 580:
            return "control_fronterizo"
        elif section_int <= 855:
            return "instalaciones_aeroportuarias"
        elif section_int <= 910:
            return "operaciones_especiales"
        elif section_int <= 1055:
            return "facilitacion_especifica"
        elif section_int <= 1115:
            return "intercambio_datos"
        else:
            return "normativo_general"
    
    def process_document(self, chunk_size=1200, chunk_overlap=200):
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
    """Procesa un archivo de texto del RAC 209 y lo vectoriza en Supabase."""
    # Leer el archivo
    with open(file_path, "r", encoding="utf-8") as f:
        document_text = f.read()
    
    # Crear el procesador
    processor = RAC209Processor(document_text)
    
    # Procesar el documento
    processor.process_document(chunk_size=1200, chunk_overlap=200)
    
    # Vectorizar y almacenar
    processor.vectorize_and_store()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        process_file(file_path)
    else:
        # Si no se proporciona un archivo, usar el archivo predeterminado
        process_file("rac209.txt")