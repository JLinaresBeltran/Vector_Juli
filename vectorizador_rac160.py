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

class RAC160Processor:
    def __init__(self, text: str, file_name: str = "rac160.txt", creator: str = "System"):
        self.raw_text = text
        self.processed_chunks = []
        self.file_name = file_name
        self.creator = creator
        self.current_section_hierarchy = {}
        
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
            
            # Extraer elementos estructurales del contenido
            structural_elements = self.extract_structural_elements(section_text)
            
            section = {
                "numero": section_num,
                "titulo": section_title,
                "texto": section_text,
                "nivel": level,
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
    
    def create_ubicacion_legible(self, jerarquia: Dict[str, Any]) -> str:
        """Crear una descripción legible de la ubicación en el RAC."""
        ubicacion_parts = ["RAC 160 - Seguridad de la Aviación Civil"]
        
        if jerarquia.get("capitulo"):
            ubicacion_parts.append(jerarquia["capitulo"])
        
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
    
    def create_standardized_metadata(self, section: Dict[str, Any], chunk_content: str = None,
                                   chunk_start_line: int = None, chunk_end_line: int = None) -> Dict[str, Any]:
        """Crear metadata estandarizada compatible con LangChain."""
        
        now = datetime.now().isoformat()
        
        # Determinar elementos específicos del chunk si se proporciona
        specific_elements = {"literal_letra": None, "numeral_numero": None, "paragrafo_numero": None}
        if chunk_content:
            specific_elements = self.determine_subsection_elements(section, chunk_content)
        
        # Para subsecciones (ej: 160.1045.1), usar como numeral_numero
        section_parts = section["numero"].split('.')
        subsection_number = None
        if len(section_parts) > 2:  # Es una subsección como 160.1045.1
            subsection_number = '.'.join(section_parts[2:])
        
        # Asegurar que file_extension no sea vacío
        file_ext = os.path.splitext(self.file_name)[1]
        if not file_ext:
            file_ext = ".txt"
        
        # Construir nombre del capítulo basado en la jerarquía
        capitulo_nombre = section["jerarquia"].get("capitulo", "SEGURIDAD DE LA AVIACIÓN CIVIL")
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
            "id_legal": "RAC160",
            "file_name": self.file_name or "rac160.txt",
            "created_at": now,
            "last_modified": now,
            "file_extension": file_ext,
            "tipo_documento": "Reglamentos Aeronauticos",
            "articulo_numero": section["numero"],  # ← SECCIÓN del RAC 160 (ej: "160.515", "160.1045")
            "capitulo_nombre": capitulo_nombre,
            "capitulo_numero": capitulo_numero
        }
        
        # Agregar campos opcionales solo si tienen valor
        if specific_elements["literal_letra"]:
            metadata["literal_letra"] = specific_elements["literal_letra"]
        
        if specific_elements["numeral_numero"] or subsection_number:
            metadata["numeral_numero"] = specific_elements["numeral_numero"] or subsection_number
        
        if specific_elements["paragrafo_numero"]:
            metadata["paragrafo_numero"] = specific_elements["paragrafo_numero"]
        
        # Campos específicos del RAC 160 (información adicional)
        if chunk_content:
            # Extraer información específica de seguridad de aviación civil
            tags = self.extract_tags_from_section(chunk_content, section["titulo"])
            tipo_contenido = self.classify_content_type(section["numero"], section["titulo"])
            ubicacion_legible = self.create_ubicacion_legible(section["jerarquia"])
            
            if tags:
                metadata["tags_seguridad"] = tags
            if tipo_contenido:
                metadata["tipo_contenido"] = tipo_contenido
            if ubicacion_legible:
                metadata["ubicacion_legible"] = ubicacion_legible
            
            # Información sobre el nivel jerárquico
            metadata["nivel_jerarquico"] = section["nivel"]
            
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
        print(f"Secciones RAC 160 identificadas: {len(sections)}")
        
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
        
        print(f"📊 Iniciando vectorización de {total_chunks} chunks del RAC 160...")
        
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
                        print(f"📝 [{i}/{total_chunks}] RAC 160 - Sección {seccion_num} ✅")
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
        print(f"\n🎯 Resumen del procesamiento RAC 160:")
        print(f"   ✅ Chunks exitosos: {successful_inserts}")
        print(f"   ❌ Chunks fallidos: {failed_inserts}")
        print(f"   📊 Total procesados: {total_chunks}")
        
        if successful_inserts > 0:
            print(f"✅ RAC 160 procesado. {successful_inserts} chunks almacenados en '{table_name}'.")
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
    """Procesar archivo del RAC 160 y vectorizar en Supabase."""
    print(f"📄 Procesando RAC 160: {file_path}")
    
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
    processor = RAC160Processor(document_text, file_name, creator)
    
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
        print("📋 Uso: python vectorizador_rac160.py <archivo> [tabla] [creator]")
        print("📋 Ejemplo: python vectorizador_rac160.py rac160.txt transporte_aereo Jhonathan")
        
        # Si no se proporciona un archivo, usar el archivo predeterminado si existe
        default_file = "rac160.txt"
        if os.path.exists(default_file):
            print(f"🔄 Usando archivo por defecto: {default_file}")
            process_file(default_file, "transporte_aereo", "System")
        else:
            print(f"❌ Archivo por defecto '{default_file}' no encontrado.")