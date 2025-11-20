import re
from typing import List

# ======================
# CONFIGURACIÓN DE PERSONA
# ======================

HEALTH_SYSTEM_PROMPT = """
Eres "VitalBot", un asistente experto en salud y bienestar personal.
Tu objetivo es ayudar al usuario a mejorar su estilo de vida mediante consejos prácticos, planes de ejercicio y nutrición.

DIRECTRICES DE COMPORTAMIENTO:
1. Sé empático, motivador y profesional.
2. Tus respuestas deben ser concisas (máximo 2 o 3 oraciones) ya que eres un asistente de voz.
3. Cuando el usuario deba tomar una decisión, preséntale opciones claras numeradas o con letras.
4. NUNCA des diagnósticos médicos graves. Si detectas síntomas serios, recomienda visitar a un doctor.
5. Utiliza un tono conversacional y cercano.

FORMATO DE OPCIONES:
Si ofreces opciones, hazlo así:
- Opción A: [Descripción corta]
- Opción B: [Descripción corta]
"""

# ======================
# PARSER DE OPCIONES
# ======================

def extract_options_from_text(text: str) -> List[str]:
    """
    Analiza el texto generado por el LLM y extrae opciones estructuradas 
    para guardarlas en JSON. Busca patrones como "1.", "A)", "-", etc.
    """
    options = []
    
    # Regex para detectar patrones de lista comunes en respuestas de LLM
    # Ejemplos: "1. Correr", "a) Nadar", "- Yoga", "* Pesas"
    pattern = r'(?:^|\n)(?:\d+\.|[a-zA-Z]\)|\-|\*)\s+(.+?)(?=(?:\n(?:\d+\.|[a-zA-Z]\)|\-|\*)|\n\n|$))'
    
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        # Limpiar espacios extra
        options = [m.strip() for m in matches if len(m.strip()) > 2]
    
    return options