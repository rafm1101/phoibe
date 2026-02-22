# Layered data validation system

## Summary

A modular, extensible validation framework following a progressive issue detection and transformation approach subdivided into layers. The layers address:

- Raw layer: Basic and formal data properties. Aim at standaridising.
- Bronze layer: Rule-based outlier and data issue detection. Aim at identifying potentially relevant data.
- Silver layer: Sophisticated quality detection and imputation. Aim at preparing for value extraction.
- Gold layer: Advanced analytics for the use case.

## Architecture

### Major component responsibilities

1. Core:
   - Entities, protocols, enums.
   - Domain models, interface definitions.
2. Application:
   - Validator, factory, config.
   - Orchestration, dependency injection.
3. Infrastructure:
   - Loaders, detectors, repos.
   - External integrations with files and parsing.
4. Rules:
   - Validation rules.
   - Logic for validation checks.

### Key concepts

1. Validation rule: Single rule for repsonsible for checking one aspect of data quality.
2. Rule registry: Central registry for validation rules using decorators. Validation rules register on import.
3. Validation context: Context passed to every rule.
4. Layer report: Complete validation result for one dataset.

### Key features

1. Config-driven: Define layer rules in configuration files.
1. Extensibility: Add new rules directly.
1. Error-handling: Log steps, degrade gracefully.

### Layers

1. `application` - application layer:
   - `validator`: Orchestrator `LayerValidator`.
   - `factory`: Create the orchestrator `ValidatorFactory` and registry for validation rules `RuleRegistry`.
   - `config`: Keeping and reading values for validation layer configuration `ValidationConfiguration`, including read methods.
   - `context`: Keeping values of validation layer context `ValidationContext` to keep additional information.
2. `core` - core layer:
   - `entities`: Keeping values of the report `LayerReport`, metadata about files `FileMetadata`, validation rule executions `RuleExectionResult`, characterizations and states of validation rules `Status` and `Severity`.
   - `interfaces`: Protocols for `DataLoader`, `VariableDetector` and the final `Report`.
3. `infrastructure` - infrastructure layer:
   - `detector`: Detecting column keys `RegexVariableDetector` via regex patterns.
   - `io`: Getting data `InMemoryDataloader`, `PandasDataloader`, and storing structured reports `YAMLReportRepository`.
4. `rules` - rule layer:
   - `rule`: Abstaction of validation rules and their structured results `RuleExecutionBuilder`.
4. `logging`:
   - `formatter`: Formatting the JSON output `JSONFormatter`.
   - `handler`: Collection of different logging handler for console `ConsoleHandler`, log files `FileHandler` and JSON files `JSONHandler`.
   - `logging`: `LoggingConfig`, stateless factory for creating logger `LoggerFactory`, context manager for tracking logging messages `ContextualLogger`, context manager for tracking rule execution results `RuleExectionTracker`.


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
