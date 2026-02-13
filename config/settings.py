"""Gestión de configuración usando variables de entorno."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Cargar archivo .env desde la raíz del proyecto
ruta_env = Path(__file__).parent.parent / ".env"
load_dotenv(ruta_env)


@dataclass(frozen=True)
class Configuracion:
    """Configuración de la aplicación cargada desde variables de entorno."""
    
    # Configuración de API Alpaca
    ALPACA_API_KEY: str = os.getenv("KEY", "")
    ALPACA_SECRET_KEY: str = os.getenv("SECRET", "")
    ALPACA_API_URL: str = os.getenv("API", "https://paper-api.alpaca.markets/v2")
    
    # Configuración de Trading
    SIMBOLO_POR_DEFECTO: str = "BTC/USD"
    PAPER_TRADING: bool = True
    
    # Configuración de IA
    OLLAMA_MODELO: str = "llama3.1:8b"
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    @property
    def es_paper_trading(self) -> bool:
        """Verifica si está usando paper trading."""
        return "paper" in self.ALPACA_API_URL.lower()
    
    def validar(self) -> None:
        """Valida la configuración requerida."""
        if not self.ALPACA_API_KEY or not self.ALPACA_SECRET_KEY:
            raise ValueError("Claves de API de Alpaca no configuradas. Revisa tu archivo .env.")


# Instancia global de configuración
configuracion = Configuracion()
