"""Módulo de obtención de datos de mercado."""
from datetime import datetime
from typing import List, Optional, Union

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.models import BarSet

from config.settings import configuracion
from utils.logger import obtener_logger

logger = obtener_logger(__name__)


class ClienteDatosMercado:
    """Cliente para obtener datos de mercado de criptomonedas."""
    
    def __init__(self):
        """Inicializa el cliente de datos de mercado."""
        # Los datos de cripto no requieren claves API para solicitudes básicas
        self.cliente = CryptoHistoricalDataClient()
        logger.info("ClienteDatosMercado inicializado")
    
    def obtener_barras(
        self,
        simbolos: Union[str, List[str]],
        marco_tiempo: TimeFrame = TimeFrame.Day,
        inicio: Optional[str] = None,
        fin: Optional[str] = None,
        limite: Optional[int] = None
    ) -> BarSet:
        """Obtiene datos históricos de barras para criptomonedas.
        
        Args:
            simbolos: Símbolo único o lista de símbolos (ej. "BTC/USD")
            marco_tiempo: Intervalo de tiempo para las barras (por defecto: Día)
            inicio: Fecha de inicio en formato "YYYY-MM-DD"
            fin: Fecha de fin en formato "YYYY-MM-DD"
            limite: Número máximo de barras a retornar
            
        Returns:
            BarSet con los datos solicitados
        """
        # Normalizar símbolos a lista
        if isinstance(simbolos, str):
            simbolos = [simbolos]
        
        # Por defecto 30 días atrás si no hay fecha de inicio
        if not inicio:
            from datetime import timedelta
            inicio = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        parametros_solicitud = CryptoBarsRequest(
            symbol_or_symbols=simbolos,
            timeframe=marco_tiempo,
            start=inicio,
            end=fin,
            limit=limite
        )
        
        logger.info(f"Obteniendo barras para {simbolos} | Marco: {marco_tiempo} | Inicio: {inicio}")
        
        try:
            barras = self.cliente.get_crypto_bars(parametros_solicitud)
            total_barras = sum(len(barras.data[simbolo]) for simbolo in barras.data)
            logger.info(f"Obtenidas {total_barras} barras totales para {len(barras.data)} símbolos")
            return barras
        except Exception as e:
            logger.error(f"Error obteniendo barras: {e}")
            raise
    
    def obtener_ultima_barra(self, simbolo: str = "BTC/USD") -> dict:
        """Obtiene la última barra para un símbolo.
        
        Args:
            simbolo: Par de trading
            
        Returns:
            Diccionario con datos de la última barra
        """
        barras = self.obtener_barras(simbolos=simbolo, limite=1)
        
        if simbolo in barras.data:
            ultima = barras.data[simbolo][-1]
            return {
                "simbolo": simbolo,
                "marca_tiempo": ultima.timestamp,
                "apertura": ultima.open,
                "maximo": ultima.high,
                "minimo": ultima.low,
                "cierre": ultima.close,
                "volumen": ultima.volume,
                "vwap": ultima.vwap
            }
        
        raise ValueError(f"No hay datos disponibles para {simbolo}")
