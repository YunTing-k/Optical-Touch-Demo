import logging
import colorlog


class DemoLogger:
    def __init__(self):
        self.logger = logging.getLogger("logger")
        self.logger.setLevel(logging.INFO)
        self.config = {
            'DEBUG': 'white',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
        self.logger.setLevel(logging.DEBUG)
        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)
        log_format = colorlog.ColoredFormatter(
            fmt='%(log_color)s[%(asctime)s.%(msecs)03d] %(filename)s -> [%(levelname)s] : %(message)s',
            datefmt='%H:%M',
            log_colors=self.config
        )
        sh.setFormatter(log_format)
        self.logger.addHandler(sh)
