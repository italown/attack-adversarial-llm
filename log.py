# -*- coding: UTF-8 -*-
import logging
import os
from datetime import datetime

class Log:
    def __init__(self, log_dir, logger_name, debug=False):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.logger_file_name = os.path.join(
            log_dir,
            f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_{logger_name}.log"
        )

        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        self.logger.propagate = False  # evita log duplicado no root logger

        # Evita adicionar múltiplos handlers
        if not self.logger.handlers:
            file_handler = logging.FileHandler(self.logger_file_name, encoding="utf-8")
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)


            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def error(self, msg):
        self.logger.error(msg)

    def header(self, message: str) -> None:
        self.logger.info("="*70)
        self.logger.info('{:^70s}'.format(message))
        self.logger.info("="*70)

    def print_log(self, msg):
        print(msg)
    
    def get_logger_file_name(self):
        return self.logger_file_name
