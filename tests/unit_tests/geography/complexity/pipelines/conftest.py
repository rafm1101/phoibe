import dataclasses

import pytest

from phoibe.geography.complexity.rix.pipelines.product.schema import PRODUCT_SPECIFICATION


@pytest.fixture
def product_spec():
    return PRODUCT_SPECIFICATION


@pytest.fixture
def make_spec():
    """The product specification with other severities or parameters of its checks."""

    def make(severities=None, check_parameters=None):
        checks = dict(PRODUCT_SPECIFICATION.checks)
        for name, severity in (severities or {}).items():
            checks[name] = dataclasses.replace(checks[name], severity=severity)
        for name, parameters in (check_parameters or {}).items():
            checks[name] = dataclasses.replace(checks[name], parameters=parameters)
        return dataclasses.replace(PRODUCT_SPECIFICATION, checks=checks)

    return make
