import datetime
import pathlib

from phoibe.layered.application.context import ValidationContext
from phoibe.layered.core.entities import LayerReport
from phoibe.layered.core.entities import RuleExecutionResult
from phoibe.layered.core.entities import Severity
from phoibe.layered.core.entities import Status
from phoibe.layered.core.interfaces import DataLoader
from phoibe.layered.core.interfaces import VariableDetector
from phoibe.layered.logging.logging import RuleExecutionTracker
from phoibe.layered.logging.logging import get_logger
from phoibe.layered.rules.rule import ValidationRule


class LayerValidator:
    """Orchestrator of the validation rules for a single data layer.

    Coordinates:
    - Data loading
    - Signal/variable detection
    - Rule execution with timing
    - Result aggregation
    """

    layer_name: str
    """Name of the current layer (raw, bronze, silver, gold)."""
    data_loader: DataLoader
    """Loader for the data (file or memory)."""
    variable_detector: VariableDetector
    """Detector of variable-to-column-key mapping."""
    rules: list
    """List of validation rules."""

    def __init__(
        self, layer_name: str, data_loader: DataLoader, variable_detector: VariableDetector, rules: list[ValidationRule]
    ):
        self.layer_name = layer_name
        self.data_loader = data_loader
        self.variable_detector = variable_detector
        self.rules = rules
        self._logger = get_logger(f"validator.{layer_name}")

    def validate(self, file_path: str | pathlib.Path, turbine_id: str) -> LayerReport:
        """Validate data from file or memory.

        Parameters
        ----------
        file_path
            Path to the data file. Ignored for in-memory loaders.
        turbine_id
            Identifier of the turbine.

        Return
        ------
        report
            `LayerReport` with validation results.
        """
        file_path = pathlib.Path(file_path) if isinstance(file_path, str) else file_path
        self._logger.info(f"Starting validation for {turbine_id}.")

        file_metadata = self.data_loader.get_metadata(file_path)
        self._logger.debug(f"File metadata: {file_metadata.filename} ({file_metadata.size_mb} MB).")

        df = self.data_loader.load(file_path)
        self._logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

        detected_variables = self.variable_detector.detect(df)
        detected_count = sum(1 for v in detected_variables.values() if v is not None)
        self._logger.info(f"Detected {detected_count}/{len(detected_variables)} variables.")

        # Note: The draft passes a configuration as an additional field `config` at this point.
        context = ValidationContext(
            detected_variables=detected_variables, turbine_id=turbine_id, layer_name=self.layer_name
        )

        rule_execution_results = []
        for rule in self.rules:
            with RuleExecutionTracker(self._logger, rule.name, turbine_id):
                try:
                    result = rule.execute(df, context)
                    rule_execution_results.append(result)
                except Exception as exception:
                    self._logger.error(
                        f"Rule '{rule.name}' crashed: {exception}",
                        exc_info=True,
                        extra={"rule_name": rule.name, "turbine_id": turbine_id},
                    )
                    error_result = RuleExecutionResult(
                        rule_name=rule.name,
                        status=Status.ERROR,
                        severity=Severity.CRITICAL,
                        required="N/A",
                        actual=None,
                        points_max=getattr(rule, "points", 0),
                        points_achieved=0,
                        message=f"Rule execution failed: {str(exception)}",
                    )
                    rule_execution_results.append(error_result)

        report = LayerReport(
            layer_name=self.layer_name,
            turbine_id=turbine_id,
            timestamp=datetime.datetime.now(),
            file_metadata=file_metadata,
            detected_variables=detected_variables,
            rule_execution_results=rule_execution_results,
        )

        self._logger.info(
            f"Validation complete: {report.overall_status.value} "
            f"({report.percentage:.1f}%, {report.score_achieved}/{report.score_max} points)."
        )

        return report


__all__ = ["LayerValidator"]
