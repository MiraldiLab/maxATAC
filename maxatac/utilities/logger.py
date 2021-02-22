import logging
from os import environ

from maxatac.utilities.constants import CPP_LOG_LEVEL


def setup_logger(log_level, log_format):
    for log_handler in logging.root.handlers:
        logging.root.removeHandler(log_handler)

    for log_filter in logging.root.filters:
        logging.root.removeFilter(log_filter)

    logging.basicConfig(level=log_level, format=log_format)

    environ["TF_CPP_MIN_LOG_LEVEL"] = str(CPP_LOG_LEVEL[log_level])
