import logging
import sys
import typing


def get_logger(name, level="INFO", handlers=None, update=False):
    _DEFAULT_LOGGER = "ps.logger"

    _DEFAULT_FORMATTER = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] "
        "[%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
    )

    _ch = logging.StreamHandler(stream=sys.stdout)
    _ch.setFormatter(_DEFAULT_FORMATTER)

    _DEFAULT_HANDLERS = [_ch]

    _LOGGER_CACHE = {}  # type: typing.Dict[str, logging.Logger]

    if name in _LOGGER_CACHE and not update:
        return _LOGGER_CACHE[name]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = handlers or _DEFAULT_HANDLERS
    logger.propagate = False
    return logger