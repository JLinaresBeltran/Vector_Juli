import os
import re
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
supabase = create_client(supabase_url, supabase_key)

# Configuración de OpenAI para embeddings
openai_api_key = os.environ.get("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

class SentenciaSICProcessor:
    def __init__(self, text: str):
        self.raw_text = text
        self.processed_chunks = []
        self.sentencia_data = {}
        
    def extract_metadata(self):
        """Extraer metadata básica de la sentencia."""
        # Patrones para extraer información
        patterns = {
            'radicado': r'(?:radicado|No\.|Número de radicado):\s*(\d+-\d+)',
            'fecha': r'Fecha.*?:\s*(\d{1,2}/\d{1,2}/\d{4})',
            'juez': r'Juez:\s*([^\n]+)',
            'demandante': r'Demandante:\s*([^\n]+)',
            'demandado': r'Demandado[ao]?:\s*([^\n]+)',
            'tipo_accion': r'Tipo de acción:\s*([^\n]+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, self.raw_text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                # Convertir fecha al formato ISO
                if key == 'fecha':
                    try:
                        # Intentar parsear fecha en formato DD/MM/YYYY
                        date_obj = datetime.strptime(value, "%d/%m/%Y")
                        value = date_obj.strftime("%Y-%m-%d")
                    except:
                        # Si falla, intentar otros formatos comunes
                        for fmt in ["%Y/%m/%d", "%d-%m-%Y", "%Y-%m-%d"]:
                            try:
                                date_obj = datetime.strptime(value, fmt)
                                value = date_obj.strftime("%Y-%m-%d")
                                break
                            except:
                                continue
                self.sentencia_data[key] = value
    
    def split_into_sections(self) -> List[Dict[str, Any]]:
        """Dividir sentencia en secciones temáticas."""
        sections = []
        
        # Definir secciones y sus patrones
        section_patterns = {
            'tema_juridico': r'Tema Jurídico(.*?)(?=Subtemas|Ratio|Datos|Hechos|$)',
            'subtemas': r'Subtemas[:\s]+(.*?)(?=Ratio|Datos|Hechos|$)',
            'ratio_decidendi': r'Ratio Decidendi(.*?)(?=Datos|Hechos|Fundamentos|$)',
            'datos_basicos': r'Datos Básicos(.*?)(?=Hechos|Fundamentos|$)',
            'hechos': r'Hechos Clave(.*?)(?=Fundamentos|Consideraciones|$)',
            'fundamentos': r'Fundamentos Jurídicos(.*?)(?=Consideraciones|Decisión|$)',
            'consideraciones': r'Consideraciones Principales(.*?)(?=Decisión|Precedente|$)',
            'decision': r'Decisión(.*?)(?=Precedente|$)',
            'precedente': r'Precedente Establecido(.*?)$'
        }
        
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, self.raw_text, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                if content and len(content) > 50:  # Ignorar secciones muy cortas
                    sections.append({
                        'tipo': section_name,
                        'contenido': content,
                        'es_importante': section_name in ['ratio_decidendi', 'decision', 'precedente']
                    })
        
        # Si no se encontraron secciones, tratar el documento completo como una sección
        if not sections:
            sections.append({
                'tipo': 'documento_completo',
                'contenido': self.raw_text,
                'es_importante': True
            })
        
        return sections
    
    def extract_legal_references(self, text: str) -> List[str]:
        """Extraer referencias legales citadas."""
        references = []
        
        # Patrones para diferentes tipos de referencias
        patterns = [
            r'Ley\s+\d+\s+de\s+\d{4}',
            r'Decreto\s+(?:Legislativo\s+)?\d+\s+de\s+\d{4}',
            r'[Aa]rtículo\s+\d+(?:\.\d+)*',
            r'Sentencia\s+[A-Z]-\d+-\d+',
            r'RAC\s+\d+(?:\.\d+)*',
            r'Estatuto del Consumidor',
            r'Código General del Proceso',
            r'Código Civil',
            r'Constitución Política'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            # Normalizar y limpiar las referencias
            for match in matches:
                normalized = re.sub(r'\s+', ' ', match.strip())
                if normalized not in references:
                    references.append(normalized)
        
        return references[:10]  # Máximo 10 referencias
    
    def extract_tags_from_content(self, content: str) -> List[str]:
        """Extraer tags relevantes del contenido."""
        # Lista expandida de términos jurídicos relevantes
        keywords = [
            # Derechos del consumidor
            'consumidor', 'usuario', 'cliente', 'comprador',
            'proveedor', 'productor', 'comerciante', 'vendedor',
            'garantía', 'garantía legal', 'efectividad de la garantía',
            
            # Acciones y remedios
            'reembolso', 'devolución', 'indemnización', 'perjuicios',
            'reparación', 'compensación', 'resarcimiento',
            
            # Derechos específicos
            'información', 'información veraz', 'información clara',
            'publicidad', 'publicidad engañosa', 'propaganda',
            'retracto', 'desistimiento', 'revocación',
            
            # Sectores específicos
            'aerolínea', 'transporte aéreo', 'aviación', 'vuelo',
            'turismo', 'agencia de viajes', 'hotel',
            'servicios públicos', 'telecomunicaciones',
            
            # Contextos especiales
            'covid', 'covid-19', 'pandemia', 'emergencia sanitaria',
            'voucher', 'bono', 'crédito',
            
            # Conceptos jurídicos
            'vulneración', 'infracción', 'incumplimiento',
            'derechos', 'obligaciones', 'responsabilidad',
            'contrato', 'servicio', 'producto', 'bien',
            'solidaridad', 'responsabilidad solidaria',
            'abuso de posición dominante', 'cláusula abusiva'
        ]
        
        tags = set()
        content_lower = content.lower()
        
        for keyword in keywords:
            if keyword in content_lower:
                # Convertir a formato de tag (sin espacios, guiones bajos)
                tag = keyword.replace(' ', '_')
                tags.add(tag)
        
        # Ordenar por relevancia (frecuencia en el texto)
        tag_list = list(tags)
        tag_list.sort(key=lambda x: content_lower.count(x.replace('_', ' ')), reverse=True)
        
        return tag_list[:7]  # Máximo 7 tags
    
    def extract_main_theme(self, content: str) -> str:
        """Extraer tema principal del contenido."""
        content_lower = content.lower()
        
        # Definir temas y sus palabras clave asociadas
        themes = {
            "Responsabilidad en transporte aéreo": ["transporte aéreo", "aerolínea", "vuelo", "pasajero", "tiquete"],
            "Publicidad engañosa": ["publicidad engañosa", "información falsa", "propaganda", "anuncio"],
            "Garantía legal": ["garantía", "efectividad de la garantía", "defecto", "calidad"],
            "Derecho de retracto": ["retracto", "desistimiento", "devolución", "cinco días"],
            "Deber de información": ["información", "deber de informar", "información clara", "información veraz"],
            "Servicios turísticos": ["turismo", "agencia de viajes", "paquete turístico", "hotel"],
            "Emergencia sanitaria COVID-19": ["covid", "pandemia", "emergencia sanitaria", "decreto 482"],
            "Comercio electrónico": ["venta a distancia", "online", "internet", "plataforma digital"],
            "Servicios públicos": ["servicios públicos", "factura", "cobro", "tarifa"],
            "Indemnización de perjuicios": ["perjuicios", "indemnización", "daños", "compensación"]
        }
        
        # Contar coincidencias para cada tema
        theme_scores = {}
        for theme, keywords in themes.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > 0:
                theme_scores[theme] = score
        
        # Retornar el tema con mayor puntuación
        if theme_scores:
            return max(theme_scores, key=theme_scores.get)
        else:
            return "Protección al consumidor"
    
    def create_chunks_from_sections(self, sections: List[Dict[str, Any]], 
                                  chunk_size: int = 1000, 
                                  chunk_overlap: int = 200):
        """Crear chunks a partir de las secciones identificadas."""
        
        for section in sections:
            section_text = section['contenido']
            section_type = section['tipo']
            
            # Decidir si dividir la sección
            if len(section_text) > chunk_size * 1.5:
                # Dividir secciones largas preservando párrafos
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", ". ", "; ", ", ", " "],
                    length_function=len
                )
                chunks = text_splitter.split_text(section_text)
                
                for i, chunk in enumerate(chunks):
                    metadata = self._create_metadata(
                        section_type=section_type,
                        chunk_index=i,
                        total_chunks=len(chunks),
                        content=chunk,
                        is_complete_section=False
                    )
                    
                    self.processed_chunks.append({
                        "content": chunk,
                        "metadata": metadata
                    })
            else:
                # Mantener sección completa
                metadata = self._create_metadata(
                    section_type=section_type,
                    chunk_index=0,
                    total_chunks=1,
                    content=section_text,
                    is_complete_section=True
                )
                
                self.processed_chunks.append({
                    "content": section_text,
                    "metadata": metadata
                })
    
    def _create_metadata(self, section_type: str, chunk_index: int, 
                        total_chunks: int, content: str, 
                        is_complete_section: bool) -> Dict[str, Any]:
        """Crear metadata estructurada para cada chunk."""
        
        # Extraer referencias legales del chunk
        referencias = self.extract_legal_references(content)
        
        # Generar tags basados en el contenido
        tags = self.extract_tags_from_content(content)
        
        # Extraer tema principal
        tema_principal = self.extract_main_theme(content)
        
        # Crear ubicación legible
        radicado = self.sentencia_data.get('radicado', 'Sin número')
        ubicacion = f"Sentencia {radicado} - {section_type.replace('_', ' ').title()}"
        if total_chunks > 1:
            ubicacion += f" (Parte {chunk_index + 1} de {total_chunks})"
        
        metadata = {
            "documento": {
                "tipo": "sentencia_sic",
                "radicado": self.sentencia_data.get('radicado'),
                "fecha": self.sentencia_data.get('fecha'),
                "juez": self.sentencia_data.get('juez'),
                "autoridad": "Superintendencia de Industria y Comercio"
            },
            "partes": {
                "demandante": self.sentencia_data.get('demandante'),
                "demandado": self.sentencia_data.get('demandado'),
                "tipo_accion": self.sentencia_data.get('tipo_accion', 'Acción de protección al consumidor')
            },
            "temas": {
                "tema_principal": tema_principal,
                "tags": tags
            },
            "estructura": {
                "seccion": section_type,
                "es_seccion_completa": is_complete_section,
                "indice_chunk": chunk_index,
                "total_chunks": total_chunks,
                "tamaño_caracteres": len(content)
            },
            "ubicacion_legible": ubicacion,
            "importancia": {
                "es_ratio_decidendi": section_type == "ratio_decidendi",
                "es_decision": section_type == "decision",
                "es_precedente": section_type == "precedente",
                "puntuacion_relevancia": self._calculate_relevance_score(section_type)
            },
            "referencias_legales": referencias
        }
        
        return metadata
    
    def _calculate_relevance_score(self, section_type: str) -> int:
        """Calcular puntuación de relevancia basada en el tipo de sección."""
        relevance_scores = {
            "ratio_decidendi": 10,
            "precedente": 9,
            "decision": 8,
            "consideraciones": 7,
            "tema_juridico": 6,
            "fundamentos": 5,
            "hechos": 4,
            "subtemas": 3,
            "datos_basicos": 2,
            "documento_completo": 5
        }
        return relevance_scores.get(section_type, 1)
    
    def process_document(self) -> List[Dict[str, Any]]:
        """Procesar la sentencia completa."""
        # Extraer metadata básica
        self.extract_metadata()
        print(f"Metadata extraída: {self.sentencia_data}")
        
        # Dividir en secciones
        sections = self.split_into_sections()
        print(f"Secciones identificadas: {len(sections)}")
        for section in sections:
            print(f"  - {section['tipo']}: {len(section['contenido'])} caracteres")
        
        # Crear chunks
        self.create_chunks_from_sections(sections)
        print(f"Chunks generados: {len(self.processed_chunks)}")
        
        return self.processed_chunks
    
    def vectorize_and_store(self, table_name: str = "sentencias_sic"):
        """Generar embeddings y almacenar en Supabase."""
        total_chunks = len(self.processed_chunks)
        
        for i, chunk in enumerate(self.processed_chunks, 1):
            try:
                # Generar el embedding
                embedding_vector = embeddings.embed_query(chunk["content"])
                
                # Preparar datos para insertar
                data = {
                    "content": chunk["content"],
                    "metadata": chunk["metadata"],
                    "embedding": embedding_vector
                }
                
                # Almacenar en Supabase
                result = supabase.table(table_name).insert(data).execute()
                
                seccion = chunk['metadata']['estructura']['seccion']
                radicado = chunk['metadata']['documento']['radicado']
                print(f"[{i}/{total_chunks}] Chunk de {seccion} - Radicado {radicado} almacenado correctamente")
                
            except Exception as e:
                print(f"Error al almacenar chunk {i}/{total_chunks}: {e}")
                # Opcionalmente, guardar chunks fallidos para reprocesar
                with open("failed_chunks.log", "a") as f:
                    f.write(f"{datetime.now()} - Error en chunk {i}: {str(e)}\n")

def process_sentencia_file(file_path: str, table_name: str = "sentencias_sic"):
    """Procesar archivo de sentencia y vectorizar en Supabase."""
    print(f"\nProcesando archivo: {file_path}")
    
    try:
        # Leer el archivo
        with open(file_path, "r", encoding="utf-8") as f:
            document_text = f.read()
        
        # Crear procesador específico para sentencias
        processor = SentenciaSICProcessor(document_text)
        
        # Procesar el documento
        processor.process_document()
        
        # Vectorizar y almacenar
        processor.vectorize_and_store(table_name)
        
        print(f"\nProcesamiento completado para: {file_path}")
        
    except Exception as e:
        print(f"Error procesando archivo {file_path}: {e}")

def process_multiple_sentencias(directory_path: str, table_name: str = "sentencias_sic"):
    """Procesar múltiples archivos de sentencias desde un directorio."""
    # Obtener todos los archivos .txt o .pdf del directorio
    files = [f for f in os.listdir(directory_path) 
             if f.endswith('.txt') or f.endswith('.pdf')]
    
    print(f"Encontrados {len(files)} archivos para procesar")
    
    for i, file_name in enumerate(files, 1):
        print(f"\n{'='*60}")
        print(f"Procesando archivo {i}/{len(files)}: {file_name}")
        print(f"{'='*60}")
        
        file_path = os.path.join(directory_path, file_name)
        process_sentencia_file(file_path, table_name)
        
    print(f"\n{'='*60}")
    print(f"Procesamiento completo. {len(files)} archivos procesados.")
    print(f"{'='*60}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if os.path.isfile(sys.argv[1]):
            # Procesar un archivo único
            process_sentencia_file(sys.argv[1])
        elif os.path.isdir(sys.argv[1]):
            # Procesar un directorio completo
            process_multiple_sentencias(sys.argv[1])
        else:
            print(f"Error: {sys.argv[1]} no es un archivo ni directorio válido")
    else:
        print("Uso:")
        print("  python sentencias_processor.py archivo.txt")
        print("  python sentencias_processor.py /ruta/al/directorio/")