# Layered data validation

## Idea

## Architecture

### Major components by task

1. Orchestrate:
   1. Validator
2. Register rules and rule to validate:
   1. Rule registry.
   2. Validation rule.
3. I/O:
   1. Input.
   2. Report.
   3. Logging.
4. Configure:
   1. Layer configuration
5. Detector column keys:
   1. Variable detector.

### Major components in structure

1. `application`:
   - `config`: Keeping and reading values for validation layer configuration `ValidationConfiguration`, including read methods.
   - `context`: Keeping values of validation layer context `ValidationContext` to keep additional information.
   - `factory`: Create the orchestrator `ValidatorFactory` and registry for validation rules `RuleRegistry`.
   - `validator`: Orchestrator `LayerValidator`.
2. `core`:
   - `entities`: Keeping values of the report `LayerReport`, metadata about files `FileMetadata`, validation rule executions `RuleExectionResult`, characterizations and states of validation rules `Status` and `Severity`.
   - `interfaces`: Protocols for `DataLoader`, `VariableDetector` and the final `Report`.
3. `infrastructure`:
   - `detector`: Detecting column keys `RegexVariableDetector` via regex patterns.
   - `io`: Getting data `InMemoryDataloader`, `PandasDataloader`, and storing structured reports `YAMLReportRepository`.
4. `logging`:
   - `formatter`: Formatting the JSON output `JSONFormatter`.
   - `handler`: Collection of different logging handler for console `ConsoleHandler`, log files `FileHandler` and JSON files `JSONHandler`.
   - `logging`: `LoggingConfig`, stateless factory for creating logger `LoggerFactory`, context manager for tracking logging messages `ContextualLogger`, context manager for tracking rule execution results `RuleExectionTracker`.
5. `rules`:
   - `rule`: Abstaction of validation rules.


## Usage

### Programmatic usage

### Configuration

### Report: Output format

## Extending the system: Creating and registering new rules

## Logging system: Log outputs

### Console

### File

### JSON audit log

### Logging patterns
