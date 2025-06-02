import streamlit as st
import pandas as pd
import pdf2image
from PIL import Image
import io
import os
import requests
import json
from datetime import datetime
import zipfile
import tempfile
from typing import List, Optional
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFTableExtractor:
    """Clase principal para extracción de tablas de PDFs usando OCR"""
    
    def __init__(self, mistral_api_key: str):
        self.mistral_api_key = mistral_api_key
        self.mistral_url = "https://api.mistral.ai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {mistral_api_key}",
            "Content-Type": "application/json"
        }
    
    def pdf_a_imagenes(self, archivo_pdf) -> List[Image.Image]:
        """
        Convierte un PDF en una lista de imágenes (una por página)
        """
        try:
            # Convertir PDF a imágenes usando pdf2image
            imagenes = pdf2image.convert_from_bytes(
                archivo_pdf.read(),
                dpi=300,  # Alta resolución para mejor OCR
                fmt='PNG'
            )
            logger.info(f"PDF convertido a {len(imagenes)} páginas")
            return imagenes
        except Exception as e:
            logger.error(f"Error al convertir PDF: {str(e)}")
            return []
    
    def imagen_a_base64(self, imagen: Image.Image) -> str:
        """
        Convierte una imagen PIL a base64 para enviar a la API
        """
        buffer = io.BytesIO()
        imagen.save(buffer, format='PNG')
        import base64
        return base64.b64encode(buffer.getvalue()).decode()
    
    def extraer_datos_con_ocr(self, imagen: Image.Image, formato_esperado: str = "tabla") -> List[List[str]]:
        """
        Extrae datos de una imagen usando OCR de Mistral y extrae todas las columnas encontradas por fila.
        """
        try:
            imagen_b64 = self.imagen_a_base64(imagen)
            
            # Prompt modificado para todas las columnas
            prompt = f"""
            Analiza cuidadosamente esta imagen que contiene una tabla o formulario. 
            Extrae todas las FILAS de la tabla, y para cada fila, extrae el valor de cada columna.
            
            Instrucciones específicas:
            1. Identifica filas de datos con varios valores
            2. Para cada fila, extrae una lista con los valores de las columnas, en orden de izquierda a derecha.
            3. Ignora encabezados, títulos, logos o texto descriptivo
            4. Si hay múltiples tablas, procesa todas
            5. Mantén el orden original de las filas
            6. Enfocate de forma especial en extraer los datos de las columnas que tienen fecha con exactitud
            7. No saltes paginas durante el proceso de extraccion de datos
            8. OJO, Toma en consideracion que estamos en el año 2025 para que no lo confundas con el año 2020 durante la extraccion
            
            Responde ÚNICAMENTE con un JSON válido en este formato:
            {{
                "datos": [
                    ["valor_columna_1A", "valor_columna_2A", "valor_columna_3A", ...],
                    ["valor_columna_1B", "valor_columna_2B", "valor_columna_3B", ...],
                    ...
                ]
            }}
            
            Si no encuentras datos tabulares, responde: {{"datos": []}}
            """
            
            payload = {
                "model": "pixtral-12b-2409",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{imagen_b64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0.1
            }
            
            response = requests.post(self.mistral_url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                resultado = response.json()
                contenido = resultado['choices'][0]['message']['content']
                
                # Limpiar y parsear JSON
                contenido = contenido.strip()
                if contenido.startswith('```json'):
                    contenido = contenido[7:-3]
                elif contenido.startswith('```'):
                    contenido = contenido[3:-3]
                
                try:
                    datos_json = json.loads(contenido)
                    # Devuelve listas de valores por fila (todas las columnas)
                    return [list(map(str, fila)) for fila in datos_json.get('datos', [])]
                except json.JSONDecodeError:
                    logger.warning(f"Error al parsear JSON: {contenido}")
                    return []
            else:
                logger.error(f"Error en API Mistral: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error en OCR: {str(e)}")
            return []
    
    def procesar_pdf_completo(self, archivo_pdf, nombre_archivo: str) -> List[List[str]]:
        """
        Procesa un PDF completo, extrayendo datos de todas las páginas
        """
        datos_totales = []
        
        # Convertir PDF a imágenes
        imagenes = self.pdf_a_imagenes(archivo_pdf)
        
        if not imagenes:
            return datos_totales
        
        # Procesar cada página
        for i, imagen in enumerate(imagenes, 1):
            st.write(f"📄 Procesando página {i} de {len(imagenes)} - {nombre_archivo}")
            
            # Extraer datos de la página actual
            datos_pagina = self.extraer_datos_con_ocr(imagen)
            
            if datos_pagina:
                datos_totales.extend(datos_pagina)
                st.success(f"✅ Extraídos {len(datos_pagina)} registros de la página {i}")
            else:
                st.warning(f"⚠️ No se encontraron datos en la página {i}")
        
        return datos_totales
    
    def procesar_imagen_directa(self, imagen: Image.Image, nombre_archivo: str) -> List[List[str]]:
        """
        Procesa una imagen directamente (JPG, PNG)
        """
        st.write(f"🖼️ Procesando imagen: {nombre_archivo}")
        
        datos = self.extraer_datos_con_ocr(imagen)
        
        if datos:
            st.success(f"✅ Extraídos {len(datos)} registros de la imagen")
        else:
            st.warning("⚠️ No se encontraron datos en la imagen")
        
        return datos

def limpiar_y_validar_datos(datos: List[List[str]]) -> List[List[str]]:
    """
    Limpia y valida los datos extraídos
    """
    datos_limpios = []
    for fila in datos:
        fila_limpia = [str(c).strip() for c in fila]
        # Valida que la fila no esté vacía ni llena de 'nan'
        if any(c and c != "nan" for c in fila_limpia):
            datos_limpios.append(fila_limpia)
    return datos_limpios

def crear_dataframe(datos: List[List[str]]) -> pd.DataFrame:
    """
    Convierte los datos a DataFrame de pandas con columnas dinámicas
    """
    if not datos:
        return pd.DataFrame()
    max_cols = max(len(fila) for fila in datos)
    colnames = [f'Columna_{i+1}' for i in range(max_cols)]
    # Completa las filas cortas con ""
    filas_norm = [fila + [""]*(max_cols-len(fila)) for fila in datos]
    df = pd.DataFrame(filas_norm, columns=colnames)
    df['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return df

def guardar_en_excel(df: pd.DataFrame, nombre_archivo: str = "datos_extraidos.xlsx") -> bytes:
    """
    Guarda el DataFrame en formato Excel y retorna los bytes
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Datos_Extraidos', index=False)
    return buffer.getvalue()

def main():
    """Función principal de la aplicación Streamlit"""
    
    # Configuración de la página
    st.set_page_config(
        page_title="Extractor de Tablas PDF con OCR",
        page_icon="📊",
        layout="wide"
    )
    
    # Título y descripción
    st.title("📊 Cleaneitor Fast Extract")
    st.markdown("### Extract Data From PDFs")
    
    # Sidebar para configuración
    with st.sidebar:
        st.image("Clineitor.jpg", width=400)  # Cambia la URL o usa tu logo
        st.header("⚙️ Configuración")
        
        # Campo para API Key de Mistral
        mistral_api_key = st.text_input(
            "Clave API de Mistral:",
            type="password",
            help="Ingresa tu clave API de Mistral para usar el OCR"
        )
        
        if not mistral_api_key:
            st.warning("⚠️ Necesitas una clave API de Mistral para continuar")
            st.stop()
        
        st.success("✅ API Key configurada")
        
        # Sección para cargar imagen de referencia
        st.header("🖼️ Imagen de Referencia")
        imagen_ref = st.file_uploader(
            "Sube una imagen de ejemplo:",
            type=['png', 'jpg', 'jpeg'],
            help="Imagen que muestre el formato de tabla esperado"
        )
        
        if imagen_ref:
            st.image(imagen_ref, caption="Imagen de referencia", use_column_width=True)
    
    # Crear instancia del extractor
    extractor = PDFTableExtractor(mistral_api_key)
    
    # Área principal para carga de archivos
    st.header("📁 Cargar Archivos para Procesar")
    
    archivos = st.file_uploader(
        "Selecciona archivos PDF, JPG o PNG:",
        type=['pdf', 'jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Puedes cargar múltiples archivos. Los PDFs se procesarán página por página."
    )
    
    if archivos:
        st.success(f"📎 {len(archivos)} archivo(s) cargado(s)")
        
        # Botón para iniciar procesamiento
        if st.button("🚀 Iniciar Extracción de Datos", type="primary"):
            
            datos_totales = []
            
            # Barra de progreso
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Procesar cada archivo
            for i, archivo in enumerate(archivos):
                # Actualizar progreso
                progreso = i / len(archivos)
                progress_bar.progress(progreso)
                status_text.text(f"Procesando: {archivo.name}")
                
                try:
                    # Determinar tipo de archivo
                    if archivo.name.lower().endswith('.pdf'):
                        # Procesar PDF
                        datos_archivo = extractor.procesar_pdf_completo(archivo, archivo.name)
                    else:
                        # Procesar imagen
                        imagen = Image.open(archivo)
                        datos_archivo = extractor.procesar_imagen_directa(imagen, archivo.name)
                    
                    # Agregar datos extraídos
                    if datos_archivo:
                        datos_totales.extend(datos_archivo)
                        st.success(f"✅ {archivo.name}: {len(datos_archivo)} registros extraídos")
                    else:
                        st.warning(f"⚠️ {archivo.name}: No se encontraron datos")
                
                except Exception as e:
                    st.error(f"❌ Error procesando {archivo.name}: {str(e)}")
            
            # Completar progreso
            progress_bar.progress(1.0)
            status_text.text("¡Procesamiento completado!")
            
            # Procesar y mostrar resultados
            if datos_totales:
                # Limpiar y validar datos
                st.header("🔍 Procesando Datos Extraídos")
                datos_limpios = limpiar_y_validar_datos(datos_totales)
                st.info(f"📊 Registros después de limpieza: {len(datos_limpios)}")
                
                if datos_limpios:
                    # Crear DataFrame
                    df_final = crear_dataframe(datos_limpios)
                    
                    # Mostrar tabla de resultados
                    st.header("📋 Datos Extraídos")
                    st.dataframe(df_final, use_container_width=True)
                    
                    # Estadísticas
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📊 Total Registros", len(df_final))
                    with col2:
                        st.metric("📁 Archivos Procesados", len(archivos))
                    with col3:
                        st.metric("🕒 Procesado", datetime.now().strftime("%H:%M:%S"))
                    
                    # Generar archivo Excel
                    excel_data = guardar_en_excel(df_final)
                    
                    # Botón de descarga
                    st.download_button(
                        label="📥 Descargar Excel",
                        data=excel_data,
                        file_name=f"datos_extraidos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                else:
                    st.warning("⚠️ No se encontraron datos válidos para exportar")
            else:
                st.error("❌ No se pudieron extraer datos de ningún archivo")
    
    # Información adicional
    with st.expander("ℹ️ Información y Consejos"):
        st.markdown("""
        **Formatos Soportados:**
        - PDF (multipágina)
        - JPG, JPEG, PNG
        
        **Consejos para Mejores Resultados:**
        - Usa imágenes de alta resolución (300 DPI mínimo)
        - Asegúrate de que las tablas estén claramente definidas
        - Evita imágenes borrosas o con mucho ruido
        - Las tablas deben tener estructura columnar clara
        
        **Limitaciones:**
        - Requiere conexión a internet para OCR
        - La precisión depende de la calidad de la imagen
        - Optimizado para tablas de varias columnas
        """)

if __name__ == "__main__":
    main()