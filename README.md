# VAPI — Voice API Real-Time

Proyecto **VAPI**: servidor de voz en tiempo real que combina **Whisper** (STT), un LLM local (Ollama + *llama3.2:3b*) y síntesis de voz con **ElevenLabs**.  
Este README explica paso a paso cómo instalar dependencias, configurar el entorno y ejecutar el proyecto.

---

## Requisitos previos (generales)
- Python 3.10+ (recomendado).  
- Conexión a internet para descargar modelos y dependencias.
- Espacio en disco suficiente para el modelo `llama3.2:3b` (varía según modelo).
- Cuenta en ElevenLabs (para la API Key y elegir una `voice_id`).

---

## 1) Instalación de herramientas del sistema

### 1.1a Instalar Chocolatey (Windows)
Abre PowerShell **como Administrador** y ejecuta:

```powershell
Get-ExecutionPolicy
```

Si se obtiene en consola `Restricted`, ejecuta `Set-ExecutionPolicy AllSigned` o `Set-ExecutionPolicy Bypass -Scope Process`

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12
iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
```
### 1.1b Instalar Homebrew (macOS)
```Zsh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 1.2 Instalar ffmpeg
Whisper/otros componentes necesitan ffmpeg para manejar archivos de audio.
- Windows (Chocolatey)
```powershell
choco install ffmpeg -y
```
- MacOS (HomeBrew)
```Zsh
choco install ffmpeg -y
```
Verifica la versión:
> ffmpeg --version

## 2) Instalación de Ollama
### 2.1 Instalar Ollama
- [Windows](https://ollama.com/download/OllamaSetup.exe)
- [MacOS](https://ollama.com/download/Ollama.dmg)

Verifica la versión:
> ollama --version

### 2.2 Instalación del modelo
Usa el CLI de Ollama para "pull" del modelo (Pesa 2.0gb):
```bash
ollama pull llama3.2:3b
```

## 3) Preparación del entorno Python
### 3.1 activar entorno virtual
- Windows
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
- MacOS
```Zsh
python -m venv .venv
source .venv/bin/activate
```
### 3.2 Instalar liberias
```bash
pip install -r requirements.txt
```

## 4) Uso de ElevenLabs
1. Regístrate en [ElevenLabs](https://elevenlabs.io/app/developers/api-keys) y crea una cuenta o inicia sesión.

2. En el panel de usuario genera una API Key (generalmente en la sección de API / Settings).

3. Elige una voz en el [dashboard](https://elevenlabs.io/app/voice-lab) y copia su voice id (cada voz tiene un identificador).

4. Configura las variables en el `.env`
```bash
ELEVENLABS_API_KEY=
ELEVENLABS_VOICE_ID=
```
## 5) Funcionalidades del proyecto
- Servidor FastAPI que expone:
  - Interfaz web /voice-chat (HTML/JS) para conversación por voz.
  - WebSocket /ws/voice para enviar audio al servidor y recibir transcripciones, texto de respuesta y audio TTS en base64.
  - Endpoints compatibles con /v1/chat/completions y /v1/audio/transcriptions.
- STT: whisper local para transcribir audio enviado por el cliente.
- LLM: Ollama (llama3.2:3b) como modelo local para generar respuestas.
- TTS: Integración con ElevenLabs para sintetizar la respuesta del LLM en audio (MP3).
- Cliente Web: Página web con grabación desde micrófono, envío de audio al servidor, visualización de conversación y reproducción del audio sintetizado.
- Persistencia de conversación: El cliente mantiene conversationHistory que se envía al servidor para contexto (historial de la sesión).

## 6) Ejecutar el proyecto
### Opción A (Recomendado)
Dado que se utiliza uvicorn, permite actualizarse ante cambios en el código
```bash
uvicorn main:app --reload --port 8000
```
### Opción B
```bash
python main.py
```
## 7) Abrir la interfaz web
### Interfaz web
```bash
http://localhost:8000/voice-chat
```
### Docs (FastAPI)
```bash
http://localhost:8000/docs
```
## 8) Flujo de prueba
1. Asegúrate de que ollama esté instalado y el modelo llama3.2:3b descargado (ollama pull llama3.2:3b).
2. Exporta las variables de entorno de ElevenLabs (API key y voice id) antes de ejecutar el servidor.
3. Inicia el servidor (ver sección 6).
4. Abre http://localhost:8000/voice-chat en el navegador.
5. Observa el estado de conexión en la interfaz:
   - Debe mostrar Conectado - Listo para hablar.
6. Probar la voz:
   - Pulsa "Iniciar Grabación" (el navegador pedirá permiso para usar el micrófono; acepta).
   - Habla con claridad. La interfaz graba el audio y lo envía al servidor cuando pulses "Detener".
7. Detener manualmente:
   - Pulsa "Detener" para enviar el audio grabado al servidor.
8. Procesamiento en servidor:
   - El servidor transcribe con Whisper → envía la transcripción de vuelta (aparecerá en la UI).
   - Envía la transcripción/historial a Ollama → Ollama genera respuesta textual.
   - La respuesta de texto se envía al cliente (se muestra en el chat).
   - El servidor llama a ElevenLabs para sintetizar esa respuesta en audio MP3.
   - El servidor envía el audio sintetizado (base64) por WebSocket.
9. Reproducción:
   - El cliente recibe el mensaje type: 'audio' con format (p. ej. audio/mpeg) y datos base64.
   - El navegador crea un Audio a partir del data URI y lo reproduce.
   - Nota sobre autoplay: los navegadores modernos permiten reproducción automática solo después de que el usuario haya interactuado con la página (p. ej. pulsado el botón de grabación). Si no se reproduce, revisa la consola del navegador por bloqueos de autoplay; interactúa con la página (clic) y vuelve a intentarlo.
10. Verificación visual:
- La conversación se actualiza con el texto transcrito y la respuesta del asistente (texto). Deberías escuchar la voz de ElevenLabs leyendo la respuesta.

