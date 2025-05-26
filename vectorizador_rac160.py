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

class RAC160Processor:
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
        
        # Tags basados en palabras clave específicas del RAC 160 (seguridad aviación civil)
        keywords = [
            'seguridad', 'aviación', 'aeropuerto', 'aeronave', 'pasajero', 'tripulación',
            'inspección', 'control', 'acceso', 'equipaje', 'carga', 'correo', 'filtros',
            'interferencia', 'ilícita', 'amenaza', 'riesgo', 'evaluación', 'vulnerabilidad',
            'AVSEC', 'OACI', 'aerocivil', 'zona', 'restringida', 'área', 'pública',
            'identificación', 'permiso', 'antecedentes', 'verificación', 'certificación',
            'contingencia', 'emergencia', 'simulacro', 'protocolo', 'procedimiento',
            'tecnología', 'equipo', 'detector', 'rayos', 'explosivos', 'armas',
            'instrucción', 'capacitación', 'personal', 'operador', 'explotador',
            'internacional', 'nacional', 'transbordo', 'tránsito', 'embarque',
            'plataforma', 'terminal', 'hangar', 'combustible', 'mantenimiento',
            'diplomático', 'autoridad', 'policía', 'militar', 'migración', 'aduana'
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
        
        if jerarquia.get("subseccion") and jerarquia["subseccion"].get("numero"):
            subsec = f"Subsección {jerarquia['subseccion']['numero']}"
            if jerarquia['subseccion'].get('titulo'):
                subsec += f" - {jerarquia['subseccion']['titulo']}"
            ubicacion_parts.append(subsec)
        
        return " > ".join(ubicacion_parts)
    
    def split_into_sections(self) -> List[Dict[str, Any]]:
        """Dividir el documento RAC 160 en secciones basadas en la numeración jerárquica."""
        sections = []
        
        # Patrón para identificar secciones numeradas del RAC 160
        section_pattern = r'^(160\.\d+(?:\.\d+)*)\s*\.?\s*([^.\n]+?)(?:\.|(?=\s+[A-Z])|\n|$)'
        
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
        
        # Patrones comunes que indican el fin del título en el RAC 160
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
        
        # Determinar el capítulo basado en el número de sección
        capitulo = self.get_chapter_from_section(section_num)
        
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
    
    def get_chapter_from_section(self, section_num: str) -> str:
        """Determinar el capítulo basado en el número de sección."""
        section_int = int(section_num.split('.')[1])
        
        if section_int <= 5:
            return "A - GENERALIDADES"
        elif section_int <= 145:
            return "B - PRINCIPIOS, OBLIGATORIEDAD Y APLICABILIDAD"
        elif section_int <= 210:
            return "C - AUTORIDADES Y ASIGNACIÓN DE RESPONSABILIDADES"
        elif section_int <= 340:
            return "D - OPERACIONES DE SEGURIDAD DE AVIACIÓN CIVIL EN UN AEROPUERTO"
        elif section_int <= 430:
            return "E - ZONAS O ÁREAS DE SEGURIDAD RESTRINGIDAS Y MEDIDAS RELATIVAS AL CONTROL DE ACCESO"
        elif section_int <= 535:
            return "F - INSPECCIÓN A PERSONAS PREVIO A SU INGRESO A LAS ÁREAS RESTRINGIDAS"
        elif section_int <= 610:
            return "G - CATEGORÍAS ESPECIALES DE PASAJEROS"
        elif section_int <= 700:
            return "H - ACCESO A LAS ZONAS RESTRINGIDAS DESDE PREDIOS DE TENEDORES DE ESPACIO"
        elif section_int <= 835:
            return "I - CONTROL DE ACCESO E INSPECCIÓN DE VEHÍCULO"
        elif section_int <= 910:
            return "J - SISTEMA DE IDENTIFICACIÓN Y VERIFICACIÓN DE ANTECEDENTES"
        elif section_int <= 1005:
            return "K - INFRAESTRUCTURA DE SEGURIDAD"
        elif section_int <= 1125:
            return "L - RESPONSABILIDAD DE LOS EXPLOTADORES DE AERONAVES"
        elif section_int <= 1205:
            return "M - COOPERACIÓN INTERNACIONAL"
        elif section_int <= 1320:
            return "N - AVIACIÓN GENERAL"
        elif section_int <= 1415:
            return "O - MEDIDAS RELATIVAS A LAS ARMAS, SUSTANCIAS EXPLOSIVAS Y MATERIAS PELIGROSAS"
        elif section_int <= 1510:
            return "P - CONTROL DE CALIDAD DE LA SEGURIDAD DE LA AVIACIÓN CIVIL"
        elif section_int <= 1610:
            return "Q - INSTRUCCIÓN DE SEGURIDAD DE LA AVIACIÓN CIVIL"
        elif section_int <= 1755:
            return "R - MÉTODOS PARA HACER FRENTE A LOS ACTOS DE INTERFERENCIA ILÍCITA"
        elif section_int <= 1805:
            return "S - INFORMACIÓN DELICADA RELACIONADA CON LA SEGURIDAD DE LA AVIACIÓN CIVIL"
        elif section_int <= 1900:
            return "T - SEGURIDAD DE LAS INSTALACIONES Y SERVICIOS PARA LA NAVEGACIÓN AÉREA"
        elif section_int <= 2000:
            return "U - CIBER-SEGURIDAD"
        elif section_int <= 2100:
            return "V - INNOVACIÓN, INVESTIGACIÓN Y DESARROLLO"
        elif section_int <= 2200:
            return "W - SISTEMA DE GESTIÓN DE LA SEGURIDAD DE LA AVIACIÓN CIVIL"
        elif section_int <= 2315:
            return "X - EQUIPOS Y SISTEMAS DE APOYO A LA SEGURIDAD DE LA AVIACIÓN CIVIL"
        else:
            return "Y - TRANSPORTE DE MUESTRAS Y ÓRGANOS"
    
    def get_section_title(self, section_num: str) -> str:
        """Obtener el título de una sección por su número."""
        # Mapeo manual de las secciones principales del RAC 160
        section_titles = {
            "160.001": "Objetivo",
            "160.005": "Definiciones, abreviaturas-siglas y acrónimos",
            "160.100": "Principios de la seguridad de la aviación civil",
            "160.105": "Obligatoriedad",
            "160.110": "Campo de aplicación",
            "160.115": "Seguridad y facilitación",
            "160.120": "Facilitación y aplicación de los controles de las autoridades del Estado",
            "160.125": "Aplicación de procedimientos policiales",
            "160.130": "Aplicación de procedimientos migratorios",
            "160.135": "Aplicación de procedimientos de impuestos y aduanas",
            "160.140": "Aplicación de procedimientos fito–zoosanitarios",
            "160.145": "Aplicación de procedimientos de las Fuerzas Militares",
            "160.200": "Comisión Intersectorial de Seguridad de la Aviación civil – CISAC",
            "160.205": "Unidad Administrativa Especial de Aeronáutica Civil - Aerocivil",
            "160.210": "Dependencia con funciones en seguridad de la aviación civil",
            "160.300": "Responsabilidad de la Seguridad de la aviación civil en el aeropuerto",
            "160.305": "Plan de seguridad del aeropuerto – PSA",
            "160.310": "Actualización (enmienda) del plan de seguridad del aeropuerto – PSA",
            "160.315": "Comité de seguridad de aeropuerto – CSA",
            "160.320": "Composición del Comité de seguridad de aeropuerto en un aeropuerto de categoría nacional",
            "160.325": "Composición del Comité de seguridad de aeropuerto en un aeropuerto de categoría internacional",
            "160.330": "Funciones generales del comité de seguridad de aeropuerto",
            "160.335": "Reunión del Comité de Seguridad de Aeropuerto",
            "160.340": "Medidas relativas a la parte pública de los aeropuertos",
            "160.400": "Generalidades",
            "160.405": "Medidas relativas al ingreso de las autoridades colombianas a las áreas de seguridad restringida",
            "160.410": "Medidas relativas al ingreso de los tripulantes a las áreas o zonas de seguridad restringida",
            "160.415": "Medidas relativas al ingreso de inspectores de aviación civil e investigadores de accidentes",
            "160.420": "Medidas relativas al ingreso de pasajeros a las áreas de seguridad restringida",
            "160.500": "Medidas generales de obligatorio cumplimiento",
            "160.505": "Inspección de las personas y sus objetos – Pasajeros y no pasajeros",
            "160.510": "Inspección a servidores públicos de las autoridades de control, vigilancia, policivas, militares o judiciales",
            "160.515": "Oposición a la requisa o a la inspección de la persona, del equipaje u objetos de mano",
            "160.520": "Control de pasajeros y equipajes de mano inspeccionados y personas en general",
            "160.525": "Medida extraordinaria de inspección a tripulaciones, pasajeros, equipaje de mano",
            "160.530": "Restricción especial de equipaje de mano",
            "160.535": "Recurso humano en los puntos de inspección a personas",
            "160.600": "Medidas y procedimientos relativos a categorías especiales de pasajeros",
            "160.605": "Autoridad del piloto al mando",
            "160.610": "Exenciones para las categorías especiales",
            "160.700": "Medidas de seguridad para el acceso a las zonas o áreas restringidas",
            "160.800": "Generalidades",
            "160.805": "Medidas de seguridad para los vehículos de transporte de valores",
            "160.810": "Registro del ingreso de vehículos",
            "160.815": "Control de seguridad de la aviación civil sobre los vehículos destinados a operaciones del aeropuerto",
            "160.820": "Exención a la inspección de vehículos",
            "160.825": "Accesos vehiculares compartidos con la Fuerza Pública",
            "160.830": "Requisitos mínimos de infraestructura, y tecnología en puntos de control de acceso vehicular",
            "160.835": "Ingreso de vehículos de funcionarios de la Aerocivil",
            "160.900": "Sistema de identificación aeroportuario y permiso aeroportuario para ingreso",
            "160.905": "Verificación de antecedentes",
            "160.910": "Documentos de identificación válidos para pasajeros",
            "160.1000": "Requisitos de Infraestructura para la seguridad de la aviación civil",
            "160.1005": "Modelos aeroportuarios para control de personas y requerimientos mínimos de infraestructura",
            "160.1100": "Responsabilidades generales",
            "160.1105": "Medidas relativas a las aeronaves",
            "160.1110": "Plan de seguridad para explotadores de aeronaves de transporte aéreo comercial regular – PSE",
            "160.1115": "Elaboración, enmienda del plan de seguridad de explotador de aeronaves",
            "160.1120": "Medidas relativas al equipaje de bodega",
            "160.1125": "Medidas relativas a la carga, el correo y otros artículos",
            "160.1200": "Solicitud de otros Estados con respecto a la aplicación de controles especiales",
            "160.1205": "Acuerdos bilaterales y/o Regionales",
            "160.1300": "Zonas de aviación general en los aeropuertos",
            "160.1305": "Medidas de seguridad para la operación de aviación general",
            "160.1310": "Ingreso de tripulantes y ocupantes de aeronaves de aviación general",
            "160.1315": "Escoltas en aeronaves privadas",
            "160.1320": "Inspección de pasajeros en el transporte aéreo comercial no regular",
            "160.1400": "Trasporte de armas, sustancias explosivas y materias o mercancías peligrosas",
            "160.1405": "Detección de armas, artículos explosivos, mercancías peligrosas",
            "160.1410": "Medidas especiales de control para el transporte de líquidos, aerosoles y geles",
            "160.1415": "Personal armado a bordo de aeronaves comerciales",
            "160.1500": "Generalidades",
            "160.1505": "Objetivos del Programa Nacional de Control de Calidad",
            "160.1510": "Plan de control de calidad de aeropuerto",
            "160.1600": "Generalidades",
            "160.1605": "Objetivo general del Programa Nacional de Instrucción",
            "160.1610": "Plan de instrucción de la aviación civil",
            "160.1700": "Prevención",
            "160.1705": "Respuesta",
            "160.1710": "Intercambio de información y notificación",
            "160.1715": "Plan Nacional de Contingencia",
            "160.1720": "Plan de contingencia de aeropuerto",
            "160.1725": "Punto de estacionamiento aislado de aeronaves",
            "160.1730": "Tratamiento de artefactos o sustancias sospechosas",
            "160.1735": "Centro de operaciones de emergencia – COE",
            "160.1740": "Ejercicios de seguridad (simulacros)",
            "160.1745": "Notificación de actos de interferencia ilícita",
            "160.1750": "Evaluación del acto de interferencia ilícita y de ejercicios de seguridad",
            "160.1755": "Evaluación de riesgos",
            "160.1800": "Restricción y protección de la información",
            "160.1805": "Uso, conservación, custodia y acceso a los registros o grabaciones del CCTV",
            "160.1900": "Protección de las instalaciones y servicios para la navegación aérea",
            "160.2000": "Medidas relativas a las ciber-seguridad",
            "160.2100": "Medidas relativas a la innovación, investigación y desarrollo",
            "160.2200": "Generalidades",
            "160.2300": "Criterios de implementación de equipos de seguridad",
            "160.2305": "Comprobación de mantenimiento y calibración periódica",
            "160.2310": "Mantenimiento de equipos y sistemas de seguridad",
            "160.2315": "Plan de mantenimiento de los equipos de seguridad",
            "160.2400": "Medidas relativas al transporte de órganos y muestras"
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
                    # Crear metadata específica para RAC 160
                    metadata = {
                        "documento": {
                            "tipo": "reglamento_aeronautico",
                            "titulo": "Reglamentos Aeronáuticos de Colombia - RAC 160",
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
                        "titulo": "Reglamentos Aeronáuticos de Colombia - RAC 160",
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
        section_int = int(section_num.split('.')[1])
        
        if "definicion" in title_lower or "abreviatura" in title_lower:
            return "definiciones_terminologia"
        elif "principio" in title_lower or "objetivo" in title_lower:
            return "principios_objetivos"
        elif "responsabilidad" in title_lower or "autoridad" in title_lower or "comisión" in title_lower:
            return "responsabilidades_autoridades"
        elif "inspección" in title_lower or "control" in title_lower or "filtro" in title_lower:
            return "controles_inspeccion"
        elif "equipaje" in title_lower or "carga" in title_lower or "correo" in title_lower:
            return "equipaje_carga"
        elif "aeronave" in title_lower or "explotador" in title_lower:
            return "aeronaves_explotadores"
        elif "contingencia" in title_lower or "emergencia" in title_lower or "simulacro" in title_lower:
            return "contingencia_emergencia"
        elif "instrucción" in title_lower or "capacitación" in title_lower or "entrenamiento" in title_lower:
            return "instruccion_capacitacion"
        elif "tecnología" in title_lower or "equipo" in title_lower or "sistema" in title_lower:
            return "tecnologia_equipos"
        elif "internacional" in title_lower or "cooperación" in title_lower or "bilateral" in title_lower:
            return "cooperacion_internacional"
        elif section_int <= 5:
            return "generalidades"
        elif section_int <= 145:
            return "principios_aplicabilidad"
        elif section_int <= 340:
            return "operaciones_aeroportuarias"
        elif section_int <= 610:
            return "controles_personas"
        elif section_int <= 910:
            return "acceso_identificacion"
        elif section_int <= 1125:
            return "explotadores_aeronaves"
        elif section_int <= 1755:
            return "contingencia_respuesta"
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
    """Procesa un archivo de texto del RAC 160 y lo vectoriza en Supabase."""
    # Leer el archivo
    with open(file_path, "r", encoding="utf-8") as f:
        document_text = f.read()
    
    # Crear el procesador
    processor = RAC160Processor(document_text)
    
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
        process_file("rac160.txt")