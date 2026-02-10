import logging

from ergaleiothiki._internal.autoimport import expose_subpackages

logging.basicConfig(level=logging.INFO)

expose_subpackages(__name__, __path__)
