"""Módulo de ejecución de órdenes de trading."""
from decimal import Decimal
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.models import Order

from config.settings import configuracion
from utils.logger import obtener_logger

logger = obtener_logger(__name__)


class EjecutorTrading:
    """Ejecutor para colocar y gestionar órdenes de trading."""
    
    def __init__(self):
        """Inicializa el cliente de trading."""
        configuracion.validar()
        
        self.cliente = TradingClient(
            api_key=configuracion.ALPACA_API_KEY,
            secret_key=configuracion.ALPACA_SECRET_KEY,
            paper=configuracion.es_paper_trading
        )
        
        self.cuenta = self.cliente.get_account()
        logger.info(f"EjecutorTrading inicializado | Paper: {configuracion.es_paper_trading}")
        logger.info(f"Cuenta: {self.cuenta.account_number} | Equity: ${self.cuenta.equity}")
    
    def obtener_info_cuenta(self) -> dict:
        """Obtiene información actual de la cuenta.
        
        Returns:
            Diccionario con detalles de la cuenta
        """
        cuenta = self.cliente.get_account()
        return {
            "numero_cuenta": cuenta.account_number,
            "equity": float(cuenta.equity),
            "poder_compra": float(cuenta.buying_power),
            "efectivo": float(cuenta.cash),
            "valor_portafolio": float(cuenta.portfolio_value),
            "estado": cuenta.status
        }
    
    def colocar_orden_mercado(
        self,
        simbolo: str,
        cantidad: float,
        lado: OrderSide = OrderSide.BUY
    ) -> Order:
        """Coloca una orden de mercado.
        
        Args:
            simbolo: Símbolo de trading (ej. "BTC/USD")
            cantidad: Cantidad a operar
            lado: BUY o SELL
            
        Returns:
            Objeto Order colocado
        """
        solicitud_orden = MarketOrderRequest(
            symbol=simbolo,
            qty=Decimal(str(cantidad)),
            side=lado,
            time_in_force=TimeInForce.GTC
        )
        
        logger.info(f"Colocando orden {lado.name} | {simbolo} | Cantidad: {cantidad}")
        
        try:
            orden = self.cliente.submit_order(solicitud_orden)
            logger.info(f"Orden colocada | ID: {orden.id} | Estado: {orden.status}")
            return orden
        except Exception as e:
            logger.error(f"Error colocando orden: {e}")
            raise
    
    def comprar(self, simbolo: str, cantidad: float) -> Order:
        """Coloca una orden de compra a mercado.
        
        Args:
            simbolo: Símbolo de trading
            cantidad: Cantidad a comprar
            
        Returns:
            Objeto Order colocado
        """
        return self.colocar_orden_mercado(simbolo, cantidad, OrderSide.BUY)
    
    def vender(self, simbolo: str, cantidad: float) -> Order:
        """Coloca una orden de venta a mercado.
        
        Args:
            simbolo: Símbolo de trading
            cantidad: Cantidad a vender
            
        Returns:
            Objeto Order colocado
        """
        return self.colocar_orden_mercado(simbolo, cantidad, OrderSide.SELL)
    
    def obtener_posiciones(self) -> list:
        """Obtiene las posiciones abiertas actuales.
        
        Returns:
            Lista de posiciones actuales
        """
        posiciones = self.cliente.get_all_positions()
        return [
            {
                "simbolo": pos.symbol,
                "cantidad": float(pos.qty),
                "valor_mercado": float(pos.market_value),
                "precio_entrada_promedio": float(pos.avg_entry_price),
                "pl_no_realizado": float(pos.unrealized_pl)
            }
            for pos in posiciones
        ]
