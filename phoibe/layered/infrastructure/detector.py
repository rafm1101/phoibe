import re

import pandas as pd


class RegexVariableDetector:
    """Pattern-based veriable identification.

    Notes
    -----
    1. The detector accepts regex patters for each variable name.
    """

    def __init__(self, patterns: dict[str, list[str]]):
        """
        Parameters
        ----------
        patterns
            Mapping signal to possible regex patterns similar to {'variable': ['pattern1', 'pattern2'], ...}.
        """
        self.patterns = {name: [re.compile(p, re.IGNORECASE) for p in patterns] for name, patterns in patterns.items()}

    def detect(self, df: pd.DataFrame) -> dict[str, str | None]:
        detected = {}

        for variable, regex_patterns in self.patterns.items():
            matched = None

            for pattern in regex_patterns:
                for col in df.columns:
                    if pattern.search(col):
                        matched = col
                        break
                if matched:
                    break
            # if matched is None:
            #     self._logger.warning(f"Signal '{signal_name}' not found in columns: {list(df.columns)}")
            # else:
            #     self._logger.debug(f"Signal '{signal_name}' matched to '{matched}'")
            detected[variable] = matched

        return detected
