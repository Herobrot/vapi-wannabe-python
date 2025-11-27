import asyncio
from typing import Callable, Awaitable

class IdleMonitor:
    """
    Gestiona un temporizador asíncrono que ejecuta un callback 
    si no se reinicia antes de que expire el tiempo (TTL).
    """
    def __init__(self, timeout_seconds: int, callback: Callable[[], Awaitable[None]]):
        self.timeout = timeout_seconds
        self.callback = callback
        self._task: asyncio.Task | None = None
        self._is_running = False

    def start(self):
        """Inicia o reinicia el temporizador (Síncrono)"""
        self.cancel() # Cancelar cualquier timer previo
        self._is_running = True
        self._task = asyncio.create_task(self._timer_loop())

    def cancel(self):
        """Detiene el temporizador"""
        if self._task and not self._task.done():
            self._task.cancel()
        self._is_running = False

    def is_active(self):
        return self._is_running

    async def _timer_loop(self):
        """La corrutina que espera y ejecuta el callback"""
        try:
            await asyncio.sleep(self.timeout)
            if self._is_running:
                self._is_running = False
                await self.callback()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"❌ Error en IdleMonitor: {e}")