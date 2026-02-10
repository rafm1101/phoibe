# fmt: off
"""
# **README**
.. include:: ../README.md
"""
import logging

from ergaleiothiki._internal.autoimport import expose_subpackages

logging.basicConfig(level=logging.INFO)

expose_subpackages(__name__, __path__)

# fmt:on
