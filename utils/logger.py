import logging
import os

def setup_logger(save_dir, name="train"):
    """
    Utility for setup logging in the repository.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fh = logging.FileHandler(os.path.join(save_dir, "log.txt"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger
