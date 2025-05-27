import logging
import os

def setup_logging():
    LOGLEVEL = os.getenv('LOGLEVEL', 'INFO').upper()
    LOGFILE = os.getenv("LOGFILE", "data/app.log")
    NUMERIC_LEVEL = getattr(logging, LOGLEVEL, logging.INFO)

    logging.basicConfig(
        filename=LOGFILE, 
        filemode='a', 
        encoding='utf-8', 
        level=NUMERIC_LEVEL, # FIXME Changing og level does not work
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True  # ← this forces reconfiguration
    )
    logger = logging.getLogger(__name__)