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

class RAC209Processor:
    def __init__(self, text: str, file_name: str = "rac209.txt", creator: str = "System"):
        self.raw_text = text
        self.processed_chunks = []
        self.file_name = file_name
        self.creator = creator
        self.current_section_hierarchy = {}
        
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
            
            # Extraer elementos estructurales del contenido
            structural_elements = self.extract_structural_elements(section_text)
            
            section = {
                "numero": section_num,
                "titulo": section_title,
                "texto": section_text,
                "jerarquia": jerarquia,
                "tiene_subsecciones_letras": has_letter_subsections,
                "numero_parrafos": paragraphs,
                "structural_elements": structural_elements,
                "start": section_start,
                "end": section_end,
                "start_line": self.get_line_number(section_start),
                "end_line": self.get_line_number(section_end)
            }
            
            sections.append(section)
        
        return sections
    
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
        
        # Extraer numerales 1., 2., 3., etc.
        numeral_matches = re.findall(r'^(\d+)\.\s', text, re.MULTILINE)
        elements["numerales"] = list(set(numeral_matches))
        
        # Extraer parágrafos
        paragrafo_matches = re.findall(r'PARÁGRAFO\.?\s*(\d*)', text, re.IGNORECASE)
        elements["paragrafos"] = [p for p in paragrafo_matches if p]
        
        return elements
    
    def get_line_number(self, char_position: int) -> int:
        """Calcular el número de línea basado en la posición del carácter."""
        return self.raw_text[:char_position].count('\n') + 1
    
    def determine_subsection_elements(self, section: Dict[str, Any], chunk_content: str) -> Dict[str, Any]:
        """Determinar qué elementos específicos están en este chunk."""
        elements = {"literal_letra": None, "numeral_numero": None, "paragrafo_numero": None}
        
        # Buscar literal específico en el chunk
        literal_match = re.search(r'\(([a-z])\)', chunk_content)
        if literal_match:
            elements["literal_letra"] = literal_match.group(1)
        
        # Buscar numeral específico en el chunk
        numeral_match = re.search(r'^(\d+)\.\s', chunk_content, re.MULTILINE)
        if numeral_match:
            elements["numeral_numero"] = numeral_match.group(1)
        
        # Buscar parágrafo específico en el chunk
        paragrafo_match = re.search(r'PARÁGRAFO\.?\s*(\d*)', chunk_content, re.IGNORECASE)
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
            return "B - ENTRADA Y SALIDA DE PERSONAS Y DE SU EQUIPAJE"
        elif section_int <= 855:
            return "C - INSTALACIONES Y SERVICIOS PARA EL TRÁFICO EN LOS AEROPUERTOS"
        elif section_int <= 910:
            return "D - ATERRIZAJE FUERA DE LOS AEROPUERTOS INTERNACIONALES"
        elif section_int <= 1055:
            return "E - DISPOSICIONES SOBRE FACILITACIÓN DE ASPECTOS ESPECÍFICOS"
        elif section_int <= 1115:
            return "F - SISTEMAS DE INTERCAMBIO DE DATOS SOBRE LOS PASAJEROS"
        else:
            return "G - DISPOSICIONES VARIAS"
    
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
    
    def create_ubicacion_legible(self, jerarquia: Dict[str, Any]) -> str:
        """Crear una descripción legible de la ubicación en el RAC."""
        ubicacion_parts = ["RAC 209 - Facilitación del Transporte Aéreo"]
        
        if jerarquia.get("capitulo"):
            ubicacion_parts.append(jerarquia["capitulo"])
        
        if jerarquia.get("seccion_principal") and jerarquia["seccion_principal"].get("numero"):
            sec = f"Sección {jerarquia['seccion_principal']['numero']}"
            if jerarquia['seccion_principal'].get('titulo'):
                sec += f" - {jerarquia['seccion_principal']['titulo']}"
            ubicacion_parts.append(sec)
        
        return " > ".join(ubicacion_parts)
    
    def create_standardized_metadata(self, section: Dict[str, Any], chunk_content: str = None,
                                   chunk_start_line: int = None, chunk_end_line: int = None) -> Dict[str, Any]:
        """Crear metadata estandarizada compatible con LangChain."""
        
        now = datetime.now().isoformat()
        
        # Determinar elementos específicos del chunk si se proporciona
        specific_elements = {"literal_letra": None, "numeral_numero": None, "paragrafo_numero": None}
        if chunk_content:
            specific_elements = self.determine_subsection_elements(section, chunk_content)
        
        # Asegurar que file_extension no sea vacío
        file_ext = os.path.splitext(self.file_name)[1]
        if not file_ext:
            file_ext = ".txt"
        
        # Construir nombre del capítulo basado en la jerarquía
        capitulo_nombre = section["jerarquia"].get("capitulo", "FACILITACIÓN DEL TRANSPORTE AÉREO")
        capitulo_numero = "A"  # Valor por defecto
        
        # Intentar extraer número del capítulo desde el nombre
        if capitulo_nombre:
            cap_match = re.match(r'^([A-Z]+)\s*-', capitulo_nombre)
            if cap_match:
                capitulo_numero = cap_match.group(1)
        
        # Crear metadata base estandarizada
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
            "id_legal": "RAC209",
            "file_name": self.file_name or "rac209.txt",
            "created_at": now,
            "last_modified": now,
            "file_extension": file_ext,
            "tipo_documento": "Reglamentos Aeronauticos",
            "articulo_numero": section["numero"],  # ← SECCIÓN del RAC 209 (ej: "209.515", "209.1105")
            "capitulo_nombre": capitulo_nombre,
            "capitulo_numero": capitulo_numero
        }
        
        # Agregar campos opcionales solo si tienen valor
        if specific_elements["literal_letra"]:
            metadata["literal_letra"] = specific_elements["literal_letra"]
        
        if specific_elements["numeral_numero"]:
            metadata["numeral_numero"] = specific_elements["numeral_numero"]
        
        if specific_elements["paragrafo_numero"]:
            metadata["paragrafo_numero"] = specific_elements["paragrafo_numero"]
        
        # Campos específicos del RAC 209 (información adicional)
        if chunk_content:
            # Extraer información específica de facilitación del transporte aéreo
            tags = self.extract_tags_from_section(chunk_content, section["titulo"])
            tipo_contenido = self.classify_content_type(section["numero"], section["titulo"])
            ubicacion_legible = self.create_ubicacion_legible(section["jerarquia"])
            
            if tags:
                metadata["tags_facilitacion"] = tags
            if tipo_contenido:
                metadata["tipo_contenido"] = tipo_contenido
            if ubicacion_legible:
                metadata["ubicacion_legible"] = ubicacion_legible
            
            # Información sobre elementos estructurales
            if section["tiene_subsecciones_letras"]:
                metadata["tiene_subsecciones_letras"] = True
        
        # Limpiar metadata antes de retornar
        return self.clean_metadata_for_json(metadata)
    
    def create_chunks_from_sections(self, sections: List[Dict[str, Any]], 
                                  chunk_size: int = 1200, 
                                  chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """Crear chunks a partir de las secciones identificadas."""
        
        for section in sections:
            section_text = section["texto"]
            
            # Decidir si dividir la sección
            if len(section_text) > chunk_size * 1.8:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", ". ", " "],
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
                # Sección completa como un solo chunk
                metadata = self.create_standardized_metadata(section, section_text)
                
                self.processed_chunks.append({
                    "content": section_text,
                    "metadata": metadata
                })
        
        return self.processed_chunks
    
    def process_document(self, chunk_size: int = 1200, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """Procesar el documento completo."""
        # Dividir en secciones
        sections = self.split_into_sections()
        print(f"Secciones RAC 209 identificadas: {len(sections)}")
        
        # Mostrar muestra de secciones encontradas
        if sections:
            print("\nMuestra de secciones encontradas:")
            for s in sections[:5]:
                print(f"  {s['numero']}: {s['titulo'][:60]}{'...' if len(s['titulo']) > 60 else ''}")
        
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
        
        print(f"📊 Iniciando vectorización de {total_chunks} chunks del RAC 209...")
        
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
                    seccion_num = chunk['metadata'].get('articulo_numero', 'N/A')
                    successful_inserts += 1
                    if i % 10 == 0 or i == total_chunks:
                        print(f"📝 [{i}/{total_chunks}] RAC 209 - Sección {seccion_num} ✅")
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
        print(f"\n🎯 Resumen del procesamiento RAC 209:")
        print(f"   ✅ Chunks exitosos: {successful_inserts}")
        print(f"   ❌ Chunks fallidos: {failed_inserts}")
        print(f"   📊 Total procesados: {total_chunks}")
        
        if successful_inserts > 0:
            print(f"✅ RAC 209 procesado. {successful_inserts} chunks almacenados en '{table_name}'.")
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
    """Procesar archivo del RAC 209 y vectorizar en Supabase."""
    print(f"📄 Procesando RAC 209: {file_path}")
    
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
    processor = RAC209Processor(document_text, file_name, creator)
    
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
        print("📋 Uso: python vectorizador_rac209.py <archivo> [tabla] [creator]")
        print("📋 Ejemplo: python vectorizador_rac209.py rac209.txt transporte_aereo Jhonathan")
        
        # Si no se proporciona un archivo, usar el archivo predeterminado si existe
        default_file = "rac209.txt"
        if os.path.exists(default_file):
            print(f"🔄 Usando archivo por defecto: {default_file}")
            process_file(default_file, "transporte_aereo", "System")
        else:
            print(f"❌ Archivo por defecto '{default_file}' no encontrado.")