# Claude Development Instructions

This document defines the standards and practices for Claude when working on this codebase. These instructions ensure consistent, senior-level code quality across all contributions.

---

## Table of Contents

1. [Architecture & Design](#architecture--design)
2. [Code Quality Standards](#code-quality-standards)
3. [Testing Requirements](#testing-requirements)
4. [Documentation Standards](#documentation-standards)
5. [Interview & Clarification Process](#interview--clarification-process)
6. [Error Handling & Robustness](#error-handling--robustness)
7. [Performance Considerations](#performance-considerations)
8. [Code Review Checklist](#code-review-checklist)

---

## Architecture & Design

### **1. Clean Architecture Principles**

**ALWAYS:**
- ✅ Follow dependency inversion: High-level modules depend on abstractions (Protocols), not concrete implementations.
- ✅ Separate concerns: Each module has a single, well-defined responsibility.
- ✅ Use dependency injection: Pass dependencies as constructor parameters.
- ✅ Keep business logic in domain layer (core), infrastructure separate.

**NEVER:**
- ❌ Import infrastructure modules from core/domain layer.
- ❌ Put business logic in infrastructure adapters.
- ❌ Use global state or singletons (except for registries).
- ❌ Mix I/O operations with pure business logic.

**Examples
--------**
```python
# ✅ GOOD: Protocol-based dependency
class Validator:
    def __init__(self, loader: DataLoader):  # Protocol, not concrete class
        self.loader = loader

# ❌ BAD: Concrete dependency
class Validator:
    def __init__(self):
        self.loader = PandasDataLoader()  # Hardcoded implementation
```

---

### **2. SOLID Principles**

**Single Responsibility:**
```python
# ✅ GOOD: One reason to change
class CSVLoader:
    """Only loads CSV files"""
    def load(self, path: str) -> pd.DataFrame:
        ...

# ❌ BAD: Multiple responsibilities
class DataHandler:
    """Loads, validates, and saves data"""
    def load(self, path: str): ...
    def validate(self, df): ...
    def save(self, df, path): ...
```

**Open/Closed:**
```python
# ✅ GOOD: Extend via new classes
@RuleRegistry.register("new_rule")
class NewRule(ValidationRule):
    ...

# ❌ BAD: Modify existing factory
class ValidatorFactory:
    def create_validator(self, rule_type):
        if rule_type == "new_rule":  # Don't add cases here
            return NewRule()
```

**Dependency Inversion:**
```python
# ✅ GOOD: Depend on abstraction
def process(loader: DataLoader):  # Protocol
    ...

# ❌ BAD: Depend on concrete class
def process(loader: PandasDataLoader):  # Specific implementation
    ...
```

---

### **3. Design Patterns - When to Use**

| Pattern | Use When | Don't Use When |
|---------|----------|----------------|
| **Factory** | Creating complex objects with dependencies | Simple object creation |
| **Builder** | Constructing complex objects step-by-step | Objects have < 3 parameters |
| **Registry** | Plugin architecture, runtime discovery | Static, known set of classes |
| **Strategy** | Algorithm varies independently | Single algorithm |
| **Adapter** | Wrapping external dependencies | Internal interfaces you control |
| **Decorator** | Adding behavior without inheritance | Core behavior is being changed |

**Example - Registry Pattern:**
```python
# ✅ GOOD: Extensible, no factory modification needed
@RuleRegistry.register("my_rule")
class MyRule(ValidationRule):
    ...

# ❌ BAD: Requires modifying factory for each new rule
class RuleFactory:
    def create_rule(self, name):
        if name == "my_rule":
            return MyRule()
        # ... 50 more if statements
```

---

## Code Quality Standards

### **1. Type Hints - Required**

**Always provide type hints for:**
- All function/method signatures
- Class attributes
- Return values

```python
# ✅ GOOD: Complete type hints
def validate(
        self,
    file_path: str | Path,
    turbine_id: str
) -> LayerReport:
    """Validate turbine data."""
    ...

# ❌ BAD: Missing types
def validate(self, file_path, turbine_id):
    ...
```

**Use modern type syntax (Python 3.10+):**
```python
# ✅ GOOD: Modern syntax
def process(data: list[str] | None) -> dict[str, int]:
    ...

# ❌ BAD: Old syntax
from typing import Optional, List, Dict
def process(data: Optional[List[str]]) -> Dict[str, int]:
    ...
```

---

### **2. Function Design**

**Keep functions focused and testable:**

```python
# ✅ GOOD: Pure function, easy to test
def _calculate_gap_percentage(
    actual_count: int,
    expected_count: int
) -> float:
    """Calculate data gap percentage.
    
    Parameters
    ----------
    actual_count
        Number of actual samples.
    expected_count
        Number of expected samples.
    
    Returns
    -------
    float
        Gap percentage (0.0 to 1.0).
    """
    if expected_count == 0:
        return 0.0
    return max(0.0, (expected_count - actual_count) / expected_count)

# ❌ BAD: Impure, hard to test
def check_gaps(self, df):
    # Mixes I/O, state, and logic
    timestamp_col = self.context.get_column('timestamp')
    df_time = pd.to_datetime(df[timestamp_col])
    expected = self._calculate_expected(df_time)
    gap = (expected - len(df_time)) / expected
    self.logger.info(f"Gap: {gap}")
    return gap < 0.05
```

**Function length guidelines:**
- Aim for < 20 lines.
- Extract complex logic into private helpers.
- Use early returns for error cases.

```python
# ✅ GOOD: Short, clear, early returns
def execute(self, df: pd.DataFrame, context: ValidationContext) -> RuleExecutionResult:
    # 1. Precondition checks (early returns)
    if not self._has_required_column(df, context):
        return self.result_builder.not_checked("Column missing")
    
    # 2. Extract data
    try:
        data = self._extract_data(df, context)
    except Exception as e:
        return self.result_builder.error("Extraction failed", e)
    
    # 3. Core validation (pure function)
    is_valid, metric = self._validate(data, self.threshold)
    
    # 4. Build result
    return self._build_result(is_valid, metric)
```

---

### **3. Error Handling**

**Granular try-except blocks:**

```python
# ✅ GOOD: Specific error handling
def execute(self, df, context):
    # Check preconditions first (no try-except needed)
    column = context.get_column('timestamp')
    if column is None:
        return self.result_builder.not_checked("Column not detected")
    
    # Parse timestamps (specific operation)
    try:
        timestamps = pd.to_datetime(df[column], errors='coerce')
    except Exception as e:
        return self.result_builder.error("Failed to parse timestamps", e)
    
    # Validate data (pure function, shouldn't need try-except)
    is_valid, metric = self._validate_timestamps(timestamps)
    
    return self._build_result(is_valid, metric)

# ❌ BAD: Broad try-except
def execute(self, df, context):
    try:
        # 50 lines of logic
        column = context.get_column('timestamp')
        timestamps = pd.to_datetime(df[column])
        is_valid = self._validate_timestamps(timestamps)
        return self._build_result(is_valid)
    except Exception as e:
        return self.result_builder.error("Something failed", e)  # Too vague!
```

**Error handling hierarchy:**
1. **Precondition checks** → `not_checked` result (no exception).
2. **Expected errors** (parsing, I/O) → `error` result (catch specific exceptions).
3. **Unexpected errors** → Let them propagate (will be caught by validator).

---

### **4. Naming Conventions**

| Element | Convention | Example |
|---------|-----------|---------|
| Classes | PascalCase | `ValidationRule`, `DataLoader` |
| Functions/Methods | snake_case | `execute()`, `get_metadata()` |
| Private methods | _leading_underscore | `_calculate_gaps()`, `_parse_timestamps()` |
| Constants | UPPER_SNAKE_CASE | `MAX_RETRIES`, `DEFAULT_TIMEOUT` |
| Type aliases | PascalCase | `PathLike = str \| Path` |

**Naming guidelines:**
- Use descriptive names (avoid abbreviations unless standard: `df`, `ctx`).
- Boolean variables: `is_valid`, `has_duplicates`, `should_retry`.
- Avoid generic names: `data`, `result`, `temp`, `x`.

```python
# ✅ GOOD: Descriptive names
def calculate_median_resolution(timestamps: pd.Series) -> float:
    time_diffs = timestamps.diff().dropna()
    median_diff = time_diffs.median()
    return median_diff.total_seconds()

# ❌ BAD: Unclear names
def calc(ts):
    d = ts.diff().dropna()
    m = d.median()
    return m.total_seconds()
```

---

## Testing Requirements

### **1. Test Coverage Levels**

**Minimum coverage by layer:**
- Core entities: 100% (critical business logic).
- Application layer: 95% (orchestration).
- Infrastructure: 90% (I/O adapters).
- Rules: 95% (validation logic).

**Test pyramid:**
```
        /\
       /  \  Integration (10%)
      /____\
     /      \  Contract (30%)
    /________\
   /          \ Unit (60%)
  /____________\
```

---

### **2. Contract Tests**

**Purpose:** Ensure components follow protocols/interfaces.

**Required for:**
- All Protocol implementations.
- Abstract base classes.
- Factory-created objects.

```python
class TestDataLoaderContract:
    """Contract: All DataLoader implementations must pass these tests"""
    
    def test_load_returns_dataframe(self, loader):
        """Contract: load() must return pd.DataFrame"""
        result = loader.load("test.csv")
        assert isinstance(result, pd.DataFrame)
    
    def test_load_raises_for_missing_file(self, loader):
        """Contract: Must raise FileNotFoundError for missing files"""
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent.csv")
    
    # 5-10 more contract tests...
```

**Contract test guidelines:**
- Test interface compliance, not implementation details.
- Cover all required methods/properties.
- Test error handling requirements.
- Keep tests simple and focused.

---

### **3. Unit Tests**

**Purpose:** Test business logic and edge cases.

**Structure:**
```python
class TestMyRuleUnit:
    """Unit tests for MyRule business logic"""
    
    @pytest.fixture
    def rule(self):
        """Provide rule instance"""
        return MyRule(threshold=10, points=15)
    
    # =========================================================================
    # Success Cases
    # =========================================================================
    
    def test_passes_when_below_threshold(self, rule):
        """Unit: Passes when metric below threshold"""
        ...
    
    # =========================================================================
    # Failure Cases
    # =========================================================================
    
    def test_fails_when_above_threshold(self, rule):
        """Unit: Fails when metric exceeds threshold"""
        ...
    
    # =========================================================================
    # Edge Cases
    # =========================================================================
    
    def test_handles_empty_data(self, rule):
        """Unit: Handles empty dataset gracefully"""
        ...
    
    # =========================================================================
    # Core Logic (_private methods)
    # =========================================================================
    
    def test_calculate_metric_pure_function(self, rule):
        """Unit: _calculate_metric is pure function"""
        result1 = rule._calculate_metric([1, 2, 3])
        result2 = rule._calculate_metric([1, 2, 3])
        assert result1 == result2  # Deterministic
```

**Unit test guidelines:**
- Test one behavior per test.
- Use descriptive test names: `test__given_`
- Group tests with section comments (`# ===...===`).
- Test pure functions separately from I/O code.
- Aim for 15-30 tests per module.

---

### **4. Test Naming Convention**

```python
# ✅ GOOD: Clear what, when, expected
def test_passes_when_resolution_matches_expected(self):
    """Unit: Rule passes when resolution exactly matches"""
    
def test_fails_when_deviation_exceeds_tolerance(self):
    """Unit: Fails when deviation > tolerance"""

# ❌ BAD: Unclear intent
def test_resolution(self):
    """Test resolution"""
    
def test_case_1(self):
    ...
```

**Docstring format:**
- Start with test type: `Unit:`, `Contract:`, `Integration:`.
- Describe expected behavior clearly.
- Keep to one line.

---

## Documentation Standards

### **1. Google Style Docstrings**

**Required for:**
- All public modules.
- All public classes.
- All public functions/methods.
- Complex private functions.

**Module docstring:**
```python
"""Short one-line summary.

Longer description if needed. Explain module purpose,
key concepts, and usage patterns.

Examples
--------
Basic usage example here::

    from module import MyClass
    obj = MyClass()
    obj.do_something()
"""
```

**Class docstring:**
```python
class DataLoader:
    """Load data from various sources.
    
    Provides unified interface for loading data from files,
    databases, or in-memory sources. Handles encoding detection,
    format parsing, and metadata extraction.
    
    Attributes:
        encoding_priorities: List of encodings to try in order.
        max_file_size: Maximum file size in MB.
    
    Examples
    --------
        Basic usage::
        
            loader = DataLoader()
            df = loader.load('data.csv')
            metadata = loader.get_metadata('data.csv')
    """
```

**Method/Function docstring:**
```python
def validate(
        self,
    file_path: str | Path,
    turbine_id: str,
    strict: bool = False
) -> LayerReport:
    """Validate turbine data from file.
    
    Loads data, detects variables, and executes all configured
    validation rules. Returns comprehensive report with scores
    and status.
    
    Parameters
    ----------
    file_path
        Path to data file (CSV or Excel).
    turbine_id
        Unique identifier for turbine.
    strict
        If True, fail on first critical error. Defaults to False.
    
    Returns
    -------
    LayerReport 
        Report containing validation results, scores, and detected variables.
    
    Raises
    ------
    FileNotFoundError
        If file_path does not exist.
    ValueError
        If file format is unsupported.
    
    Examples
    --------
    >>> validator = LayerValidator(...)
    >>> report = validator.validate('WEA_01.csv', 'WEA_01')
    >>> print(f"Score: {report.percentage}%")
    Score: 95.5%
    """
```

**Sections to include (when relevant):**
- Summary (one line, imperative mood).
- Extended description (optional).
- Args / Parameters.
- Returns.
- Raises.
- Yields (for generators).
- Example.
- Note / Warning (if needed).

---

### **2. Inline Comments**

**When to comment:**
- Non-obvious algorithms or business rules.
- Workarounds for library limitations.
- Performance optimizations.
- Regex patterns.

```python
# ✅ GOOD: Explains why
# Use median instead of mean to handle outliers in SCADA data.
# TR10 spec section 3.2 requires robust statistics.
median_resolution = time_diffs.median()

# Regex pattern matches ISO 8601 with optional timezone
# Examples: 2024-01-01T10:00:00, 2024-01-01T10:00:00+01:00
pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[\+\-]\d{2}:\d{2})?$'

# ❌ BAD: States the obvious
# Calculate median
median = data.median()

# Create list
items = []
```

**Don't comment:**
- What the code does (code should be self-explanatory).
- Obvious operations.
- Redundant parameter descriptions.

---

### **3. Type Hints as Documentation**

Type hints serve as inline documentation:

```python
# ✅ GOOD: Types explain data structure
def aggregate_results(
        reports: list[LayerReport],
    group_by: Literal['turbine', 'status', 'layer'] = 'turbine'
) -> dict[str, list[LayerReport]]:
    """Aggregate validation reports by key."""
    ...

# ❌ BAD: Unclear types
def aggregate_results(reports, group_by='turbine'):
    """Aggregate validation reports by key."""
    ...
```

---

## Interview & Clarification Process

### **1. When to Ask Questions**

**ALWAYS ask before proceeding when:**
- Requirements are ambiguous or contradictory.
- Multiple valid implementation approaches exist.
- Significant architecture decisions are needed.
- Edge case behavior is undefined.
- Performance vs. readability tradeoff is unclear.

**Example questions:**
```markdown
I need clarification on the following before proceeding:

1. **Error Handling Strategy**: When a rule crashes, should we:
   - Option A: Create ERROR result and continue (current behavior)
   - Option B: Stop validation immediately
   - Option C: Retry once before marking as error

2. **Performance Requirement**: For the DataGapRule:
   - Expected dataset size: 1K, 100K, or 1M+ rows?
   - This affects whether we should optimize the gap calculation.

3. **Edge Case**: When all timestamps are NaT (not parseable):
   - Return ERROR result?
   - Return NOT_CHECKED?
   - Raise exception?

Please advise on your preferences.
```

---

### **2. How to Present Alternatives**

When multiple approaches exist:

```markdown
## Problem: How to structure rule registry

### Option 1: Decorator-based (Recommended)
**Pros:**
- Clean syntax: `@RuleRegistry.register("name")`.
- Auto-registration on import.
- No factory modifications needed.

**Cons:**
- Rules must be imported to register.
- Implicit behavior (registration on import).

**Code:**
```python
@RuleRegistry.register("my_rule")
class MyRule(ValidationRule):
    ...
```

### Option 2: Manual registration
**Pros:**
- Explicit registration control.
- No import side effects.

**Cons:**
- Boilerplate code.
- Easy to forget registration.

**Code:**
```python
class MyRule(ValidationRule):
    ...

RuleRegistry.add("my_rule", MyRule)
```

**Recommendation:** Option 1 for better developer experience.
```

---

### **3. Incremental Development**

For large features, propose incremental approach:

```markdown
## Implementation Plan: Parallel Rule Execution

### Phase 1 (MVP - 2 hours):
- Basic ThreadPoolExecutor implementation.
- Thread-safe result collection.
- Tests for 2-5 rules.

### Phase 2 (Optimization - 1 hour):
- Configurable thread pool size.
- Progress reporting.
- Timeout handling.

### Phase 3 (Advanced - 1 hour):
- ProcessPoolExecutor for CPU-bound rules.
- Rule dependency ordering.
- Failure isolation.

Shall I proceed with Phase 1, or would you like to adjust scope?
```

---

## Error Handling & Robustness

### **1. Graceful Degradation**

**Principle:** System should continue operating even when components fail.

```python
# ✅ GOOD: Graceful degradation
for rule in self.rules:
    try:
        result = rule.execute(df, context)
    except Exception as e:
        # Log error but continue
        logger.error(f"Rule {rule.name} crashed: {e}", exc_info=True)
        result = create_error_result(rule, e)
    check_results.append(result)

# ❌ BAD: Fail fast
for rule in self.rules:
    result = rule.execute(df, context)  # Crash stops all validation
    check_results.append(result)
```

---

### **2. Input Validation**

**Validate at boundaries (public APIs):**

```python
def validate(self, file_path: str | Path, turbine_id: str) -> LayerReport:
    """Validate turbine data."""
    # 1. Convert and validate input
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not turbine_id or not isinstance(turbine_id, str):
        raise ValueError(f"Invalid turbine_id: {turbine_id}")
    
    # 2. Proceed with validated inputs
    ...
```

**Don't validate in private methods:**
```python
def _calculate_gap(self, count: int) -> float:
    # Assume count is valid (validated by caller)
    return count / self.total
```

---

### **3. Defensive Programming**

**Guard against common issues:**

```python
# ✅ GOOD: Safe operations
def get_column(self, signal_name: str) -> str | None:
    """Get column name, returns None if not found."""
    return self.detected_signals.get(signal_name)  # Safe

# Empty check before operations
if len(timestamps) == 0:
    return self.result_builder.error("No timestamps found")

# Division by zero protection
percentage = numerator / denominator if denominator != 0 else 0.0

# ❌ BAD: Unsafe
def get_column(self, signal_name: str) -> str:
    return self.detected_signals[signal_name]  # Can raise KeyError

# No empty check
median = timestamps.median()  # Crashes on empty

# No zero check
percentage = numerator / denominator  # ZeroDivisionError
```

---

## Performance Considerations

### **1. When to Optimize**

**Optimize when:**
- Processing > 100K rows regularly.
- User reports slowness.
- Profiling shows clear bottleneck.
- O(n²) or worse complexity identified.

**Don't optimize when:**
- "Might be slow" without evidence.
- Code clarity would suffer significantly.
- Typical dataset is < 10K rows.

---

### **2. Performance Patterns**

**Vectorized operations (pandas/numpy):**
```python
# ✅ GOOD: Vectorized
mask = (df['power'] > 0) & (df['wind_speed'] > 3)
valid_data = df[mask]

# ❌ BAD: Row iteration
valid_data = []
for idx, row in df.iterrows():  # Very slow!
    if row['power'] > 0 and row['wind_speed'] > 3:
        valid_data.append(row)
```

**Lazy evaluation:**
```python
# ✅ GOOD: Generator for large datasets
def iter_chunks(file_path: Path, chunk_size: int = 10000):
    """Yield DataFrame chunks."""
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        yield chunk

# ❌ BAD: Load everything
def load_all(file_path: Path):
    return pd.read_csv(file_path)  # Memory issue for 1GB+ files
```

---

## Code Review Checklist

Before submitting code, verify:

### **Architecture:**
- [ ] Follows clean architecture (no core → infrastructure imports).
- [ ] Single responsibility per class/function.
- [ ] Dependencies injected, not created internally.
- [ ] Protocols used for abstractions.

### **Code Quality:**
- [ ] All public APIs have type hints.
- [ ] Functions < 20 lines (or well-factored).
- [ ] No code duplication (DRY principle).
- [ ] Descriptive names (no abbreviations).
- [ ] Error handling is granular and specific.

### **Testing:**
- [ ] Contract tests for all Protocol implementations.
- [ ] Unit tests for business logic.
- [ ] Edge cases covered.
- [ ] Pure functions tested separately.
- [ ] Test names are descriptive.

### **Documentation:**
- [ ] All public APIs have Google-style docstrings.
- [ ] Args, Returns, Raises documented.
- [ ] Examples provided for complex APIs.
- [ ] Inline comments explain "why", not "what".

### **Robustness:**
- [ ] Input validation at boundaries.
- [ ] Graceful degradation on errors.
- [ ] No division by zero risks.
- [ ] Empty data handled.

### **Performance:**
- [ ] No O(n²) unless necessary.
- [ ] Vectorized operations where possible.
- [ ] No premature optimization.

---

## Quick Reference

### **Common Patterns**

**Creating a new validation rule:**
1. Inherit from `ValidationRule`.
2. Implement `name` property.
3. Implement `execute(df, context)`.
4. Use `self.result_builder` for results.
5. Extract core logic to pure `_private_methods()`.
6. Register with `@RuleRegistry.register("name")`.
7. Write contract + unit tests.

**Adding a new infrastructure adapter:**
1. Create class in `infrastructure/` layer.
2. Implement relevant Protocol from `core/interfaces.py`.
3. Add contract tests (Protocol compliance).
4. Add unit tests (edge cases).
5. Update factory if needed.

**Handling errors in rules:**
1. Check preconditions → `not_checked`.
2. Wrap I/O in try-except → `error`.
3. Use pure functions for logic (no try-except).
4. Build result based on outcome.

---

## Summary

**Remember:**
- 🏗 Architecture: Clean, SOLID, protocol-based.
- 📝 Code: Typed, testable, well-named.
- ✅ Tests: Contract + Unit, 90%+ coverage.
- 📚 Docs: Google-style, examples, "why" not "what".
- 🤔 Clarify: Ask when ambiguous, propose alternatives.
- 🛡 Robust: Validate inputs, handle errors, degrade gracefully.
- ⚡ Performance: Profile first, optimize when needed.

**When in doubt, ask. When certain, document why.**