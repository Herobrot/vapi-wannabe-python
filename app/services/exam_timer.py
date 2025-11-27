import asyncio
import time
from typing import Callable, Awaitable, Dict
from enum import Enum
from app.config import config

class TimerState(Enum):
    """Estados explícitos del examen"""
    IDLE = "idle"              # No iniciado
    WAITING_WELCOME = "waiting_welcome"  # Esperando fin de bienvenida
    RUNNING = "running"        # Examen en curso
    PAUSED = "paused"          # Usuario/IA hablando
    FINISHED = "finished"      # Terminado (time_up o todas las preguntas)

class ExamTimer:
    def __init__(self, callback: Callable[[str], Awaitable[None]]):
        self.callback = callback
        self.total_time = config.EXAM_TOTAL_TIME
        self.alert_interval = config.EXAM_QUESTION_TIME
        self.total_questions = config.EXAM_TOTAL_QUESTIONS
        
        # Estado
        self.state = TimerState.IDLE
        self.start_time = 0.0
        self.pause_start = 0.0  # Para acumular tiempo pausado
        self.accumulated_pause = 0.0
        
        self.current_question = 1
        self._question_start_time = 0.0
        self._question_pause_start = 0.0
        self._question_accumulated_pause = 0.0
        
        self._global_task: asyncio.Task | None = None
        self._question_task: asyncio.Task | None = None
        self._alert_30s_fired = False  # Bandera para evitar doble disparo

    # ==========================================
    # CONTROL DE ESTADO (API Pública)
    # ==========================================
    
    def prepare_exam(self):
        """Prepara el examen (crea tasks) pero NO inicia el conteo"""
        self.stop()
        self.state = TimerState.WAITING_WELCOME
        self.current_question = 1
        print("🔧 ExamTimer preparado (esperando welcome)")

    def start_counting(self):
        """Inicia el conteo REAL del examen (llamar después de playback_complete)"""
        if self.state != TimerState.WAITING_WELCOME:
            print(f"⚠️ No se puede iniciar desde estado {self.state}")
            return
        
        self.start_time = time.time()
        self._question_start_time = time.time()
        self.state = TimerState.RUNNING
        
        self._global_task = asyncio.create_task(self._global_timer_loop())
        self._question_task = asyncio.create_task(self._question_timer_loop())
        print("🏁 ExamTimer INICIADO (conteo real)")

    def pause(self):
        """Pausa el conteo (durante audio TTS o transcripción)"""
        if self.state != TimerState.RUNNING:
            return
        
        self.state = TimerState.PAUSED
        self.pause_start = time.time()
        self._question_pause_start = time.time()

    def resume(self):
        """Reanuda el conteo"""
        if self.state != TimerState.PAUSED:
            return
        
        pause_duration = time.time() - self.pause_start
        self.accumulated_pause += pause_duration
        
        question_pause = time.time() - self._question_pause_start
        self._question_accumulated_pause += question_pause
        
        self.state = TimerState.RUNNING

    def stop(self):
        """Detiene completamente el examen"""
        self.state = TimerState.FINISHED
        if self._global_task: 
            self._global_task.cancel()
        if self._question_task: 
            self._question_task.cancel()
        print("🛑 ExamTimer detenido")

    def next_question(self):
        """Avanza a la siguiente pregunta (resetea timer y bandera)"""
        if self.state == TimerState.FINISHED:
            return
        
        if self.current_question < self.total_questions:
            self.current_question += 1
            self._question_start_time = time.time()
            self._question_accumulated_pause = 0.0
            self._alert_30s_fired = False  # RESET crítico
            
            # Recrear task de pregunta limpia
            if self._question_task: 
                self._question_task.cancel()
            self._question_task = asyncio.create_task(self._question_timer_loop())
            print(f"➡️ Avanzando a pregunta {self.current_question}/{self.total_questions}")
        else:
            print("✅ Todas las preguntas completadas")
            self.stop()

    # ==========================================
    # GETTERS
    # ==========================================
    
    def get_stats(self) -> Dict:
        """Estadísticas para el frontend/LLM (descontando pausas)"""
        now = time.time()
        
        # Tiempo total (descontando pausas)
        elapsed_total = int((now - self.start_time) - self.accumulated_pause)
        remaining_total = max(0, self.total_time - elapsed_total)
        
        # Tiempo de pregunta (descontando pausas)
        elapsed_question = int((now - self._question_start_time) - self._question_accumulated_pause)
        
        return {
            "elapsed_total": f"{elapsed_total // 60:02d}:{elapsed_total % 60:02d}",
            "remaining_total": f"{remaining_total // 60:02d}:{remaining_total % 60:02d}",
            "elapsed_question": elapsed_question,
            "current_q": self.current_question,
            "total_q": self.total_questions,
            "is_finished": self.state == TimerState.FINISHED,
            "state": self.state.value
        }

    # ==========================================
    # LOOPS INTERNOS (Private)
    # ==========================================

    async def _emit_event(self, trigger_type: str):
        """Emite evento SOLO si está en estado RUNNING"""
        if self.state != TimerState.RUNNING:
            return
        
        stats = self.get_stats()
        
        message = (
            f"[EXAM_TIMER_EVENT] trigger={trigger_type} | "
            f"remaining={stats['remaining_total']} | "
            f"question={stats['current_q']}/{stats['total_q']} | "
            f"elapsed_q={stats['elapsed_question']}s"
        )
        
        await self.callback(message)

    async def _global_timer_loop(self):
        """Monitor del tiempo total del examen"""
        try:
            while self.state in [TimerState.RUNNING, TimerState.PAUSED]:
                await asyncio.sleep(1)
                
                if self.state != TimerState.RUNNING:
                    continue
                
                stats = self.get_stats()
                elapsed = int(stats["remaining_total"].split(":")[0]) * 60 + int(stats["remaining_total"].split(":")[1])
                
                if elapsed <= 0:
                    print("⏰ TIEMPO GLOBAL AGOTADO")
                    await self._emit_event("time_up")
                    self.stop()
                    break
        except asyncio.CancelledError:
            pass

    async def _question_timer_loop(self):
        """Monitor del tiempo por pregunta (alerta a los 30s)"""
        try:
            while self.state in [TimerState.RUNNING, TimerState.PAUSED]:
                await asyncio.sleep(0.5)  # Check más frecuente
                
                if self.state != TimerState.RUNNING or self._alert_30s_fired:
                    continue
                
                stats = self.get_stats()
                elapsed_q = stats["elapsed_question"]
                
                # Disparar UNA SOLA VEZ a los 30s
                if elapsed_q >= 30:
                    self._alert_30s_fired = True
                    print(f"⏰ ALERTA 30s en pregunta {self.current_question}")
                    await self._emit_event("30s_elapsed")
                    
        except asyncio.CancelledError:
            pass