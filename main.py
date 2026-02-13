"""Bot de Trading - Punto de Entrada Principal"""
import sys
from pathlib import Path

# Agregar raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import configuracion
from data.market_data import ClienteDatosMercado
from trading.executor import EjecutorTrading
from utils.logger import obtener_logger

logger = obtener_logger("main")


def main():
    """Punto de entrada principal del bot de trading."""
    logger.info("=" * 50)
    logger.info("Bot de Trading Iniciado")
    logger.info("=" * 50)
    
    try:
        # Validar configuración
        configuracion.validar()
        logger.info(f"Entorno: {'Paper' if configuracion.es_paper_trading else 'Live'}")
        
        # Inicializar cliente de datos de mercado
        cliente_datos = ClienteDatosMercado()
        
        # Prueba: Obtener datos más recientes de BTC/USD
        logger.info("Obteniendo datos de mercado más recientes...")
        ultimo = cliente_datos.obtener_ultima_barra("BTC/USD")
        logger.info(f"BTC/USD Último: ${ultimo['cierre']:,.2f}")
        
        # Inicializar ejecutor de trading
        ejecutor = EjecutorTrading()
        
        # Mostrar información de la cuenta
        cuenta = ejecutor.obtener_info_cuenta()
        logger.info(f"Equity de Cuenta: ${cuenta['equity']:,.2f}")
        logger.info(f"Poder de Compra: ${cuenta['poder_compra']:,.2f}")
        
        # Mostrar posiciones
        posiciones = ejecutor.obtener_posiciones()
        if posiciones:
            logger.info(f"Posiciones Abiertas: {len(posiciones)}")
            for pos in posiciones:
                logger.info(f"  - {pos['simbolo']}: {pos['cantidad']} @ ${pos['precio_entrada_promedio']:,.2f}")
        else:
            logger.info("Sin posiciones abiertas")
        
        logger.info("¡Bot inicializado exitosamente!")
        
    except Exception as e:
        logger.error(f"Error fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
