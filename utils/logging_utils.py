import logging
import sys

def setup_logger(name="tripendulum", level=logging.INFO):
    """
    Sets up a basic console logger.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
        
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
        datefmt='%H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
