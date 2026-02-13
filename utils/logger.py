"""Configuración de logging para el bot de trading."""
import logging
import sys
from datetime import datetime
from pathlib import Path


def obtener_logger(nombre: str) -> logging.Logger:
    """Obtiene una instancia de logger configurada.
    
    Args:
        nombre: Nombre del logger (típicamente __name__)
        
    Returns:
        Instancia de logger configurada
    """
    logger = logging.getLogger(nombre)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    
    # Manejador de consola
    manejador_consola = logging.StreamHandler(sys.stdout)
    manejador_consola.setLevel(logging.INFO)
    
    # Formato
    formateador = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    manejador_consola.setFormatter(formateador)
    
    logger.addHandler(manejador_consola)
    
    return logger
