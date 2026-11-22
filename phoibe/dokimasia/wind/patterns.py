"""How the signals of a wind turbine appear in column names.

Patterns are regular expressions, matched case-insensitively with `re.search`. A pattern for a mean excludes the
words for the other statistics, so `power_min` is never also a candidate for `power_kw`. What still overlaps, such as
`power` and `reactive_power`, is reported by detection as ambiguous and is not decided here.
"""

from phoibe.dokimasia import rules

_OTHER_STATISTICS = "min|max|std"
_QUANTITIES = (
    ("power_kw", "power|production"),
    ("wind_speed", "wind.*speed"),
    ("rotor_speed", "rotor.*speed"),
    ("pitch_angle", "pitch.*angle"),
)


def _build_patterns() -> tuple[rules.SignalPattern, ...]:
    patterns: list[rules.SignalPattern] = []
    for name, quantity in _QUANTITIES:
        patterns.append(rules.SignalPattern(name, (rf"^(?!.*(?:{_OTHER_STATISTICS})).*(?:{quantity})",)))
        patterns.extend(
            rules.SignalPattern(f"{name}_{suffix}", (rf"(?:{quantity}).*{suffix}", rf"{suffix}.*(?:{quantity})"))
            for suffix in ("min", "max")
        )
    return tuple(patterns)


PATTERNS = _build_patterns()
"""The patterns of all signals of the wind catalog."""
