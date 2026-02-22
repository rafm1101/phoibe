import datetime
import pathlib
import sys

import pandas as pd
import xarray
import yaml

from ..core.entities import FileMetadata
from ..core.entities import LayerReport
from ..core.entities import RuleExecutionResult
from ..core.entities import Severity
from ..core.entities import Status
from ..logging.logging import get_logger

LOGGER = get_logger(__name__)


class InMemoryDataLoader:
    """Loader for in-memory data such as DataFrames and xarrays."""

    def __init__(
        self, data: pd.DataFrame | xarray.Dataset | xarray.DataArray, filename: pathlib.Path | str = "in_memory_data"
    ):
        """
        Parameters
        ----------
        data
            DataFrame oder xarray Dataset/DataArray.
        filename
            Virtual filename used to identify in the report.
        """
        self.filename = filename
        self._logger = get_logger("adapters.memory")

        if isinstance(data, (xarray.Dataset, xarray.DataArray)):
            self._logger.debug(f"Converting xarray {type(data).__name__} to DataFrame.")
            self.data = data.to_dataframe().reset_index()
        elif isinstance(data, pd.DataFrame):
            self.data = data
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def load(self, file_path: pathlib.Path | str = "") -> pd.DataFrame:
        self._logger.debug(f"Using in-memory data: {type(self.data).__name__}")

        self._logger.info(f"In-memory data loaded: {len(self.data)} rows, {len(self.data.columns)} columns")
        return self.data

    def get_metadata(self, file_path: pathlib.Path | str = "") -> FileMetadata:
        if isinstance(self.data, pd.DataFrame):
            size_bytes = self.data.memory_usage(deep=True).sum()
        else:
            size_bytes = sys.getsizeof(self.data)

        return FileMetadata(
            filename=str(self.filename), size_bytes=size_bytes, format="in_memory", modified_at=datetime.datetime.now()
        )


class PandasDataLoader:
    """Load data from csv or excel files using pandas."""

    ENCODINGS = ["utf-8", "latin1", "iso-8859-1"]

    def load(self, file_path: pathlib.Path | str) -> pd.DataFrame:
        path = pathlib.Path(file_path)

        if not path.exists() or not path.is_file():
            LOGGER.error(f"File not found: {path}")
            raise FileNotFoundError(f"File not found: {path}")

        LOGGER.debug(f"Loading file: {path.name} (format: {path.suffix}).")

        if path.suffix.lower() == ".csv":
            df = self._load_csv(path)
        elif path.suffix.lower() in [".xlsx", ".xls"]:
            df = self._load_excel(path)
        else:
            LOGGER.error(f"Unsupported file format: {path.suffix}")
            raise ValueError(f"Unsupported file format: {path.suffix}.")

        LOGGER.info(f"Loaded {len(df)} rows, {len(df.columns)} columns from {path.name}")
        return df

    def _load_csv(self, file_path: pathlib.Path | str) -> pd.DataFrame:
        for encoding in self.ENCODINGS:
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except UnicodeDecodeError:
                continue

        LOGGER.error(f"Unsupported encoding of: {file_path}.")
        raise ValueError(f"Could not decode {file_path} with any of the encodings {', '.join(self.ENCODINGS)}.")

    def _load_excel(self, file_path: pathlib.Path | str) -> pd.DataFrame:
        try:
            return pd.read_excel(file_path)
        except Exception:
            LOGGER.error(f"Failed to load Excel: {file_path}.")
            raise ValueError(f"Failed to load Excel {file_path}")

    def get_metadata(self, file_path: pathlib.Path | str) -> FileMetadata:
        path = pathlib.Path(file_path)

        if not path.exists():
            LOGGER.error(f"File not found: {path}")
            raise FileNotFoundError(f"File not found: {path}")

        stat = path.stat()

        return FileMetadata(
            filename=path.name,
            size_bytes=stat.st_size,
            format=path.suffix[1:].lower() if path.suffix else "unknown",
            modified_at=datetime.datetime.fromtimestamp(stat.st_mtime),
        )


class YAMLReportRepository:
    """Save validation results as yaml file."""

    def save(self, reported: list[LayerReport], output_path: pathlib.Path | str) -> None:
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "run_metadata": self._create_metadata(reported),
            "turbines": {item.turbine_id: self._report_to_dict(item) for item in reported},
        }

        with open(output_path, "w", encoding="utf-8") as filestream:
            yaml.dump(report, filestream, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def _create_metadata(self, reported: list[LayerReport]) -> dict:
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_turbines": len(reported),
            "passed": sum(1 for item in reported if item.overall_status == Status.PASSED),
            "failed": sum(1 for item in reported if item.overall_status == Status.FAILED),
            "warnings": sum(1 for item in reported if item.overall_status == Status.WARNING),
            "errors": sum(1 for item in reported if item.overall_status == Status.ERROR),
        }

    def _report_to_dict(self, reported: LayerReport) -> dict:
        return {
            "turbine_id": reported.turbine_id,
            "timestamp": reported.timestamp.isoformat(),
            "layer": reported.layer_name,
            "file_info": {
                "filename": str(reported.file_metadata.filename),
                "size_mb": float(reported.file_metadata.size_mb),
                "format": str(reported.file_metadata.format),
                "modified": reported.file_metadata.modified_at.isoformat(),
            },
            "detected_variables": reported.detected_variables,
            "rules": [self._rule_execution_result_to_dict(result=result) for result in reported.rule_execution_results],
            "scoring": {
                "achieved": reported.score_achieved,
                "max": reported.score_max,
                "percentage": round(reported.percentage, 2),
                "status": reported.overall_status.value,
            },
            "summary": {"critical_failures": reported.critical_failures, "warnings": reported.warnings},
        }

    def _rule_execution_result_to_dict(self, result: RuleExecutionResult) -> dict:
        return {
            "name": result.rule_name,
            "status": result.status.value,
            "severity": result.severity.value,
            "required": str(result.required),
            "actual": str(result.actual),
            "points": f"{result.points_achieved}/{result.points_max}",
            "message": result.message,
            "details": result.details,
        }

    def _rule_execution_results_from_dict(self, d) -> RuleExecutionResult:
        result = RuleExecutionResult(
            rule_name=d["name"],
            status=Status[d["status"].upper()],
            severity=Severity[d["severity"].upper()],
            required=d["required"],
            actual=d["actual"],
            points_max=int(d["points"].split("/")[1]),
            points_achieved=int(d["points"].split("/")[0]),
            message=d["message"],
            details=d["details"],
        )
        return result

    def _layer_report_from_dict(self, d) -> LayerReport:
        fm = d["file_info"]
        fm = FileMetadata(
            filename=fm["filename"],
            size_bytes=fm["size_mb"],
            format=fm["format"],
            modified_at=datetime.datetime.fromisoformat(fm["modified"]),
        )
        reported = LayerReport(
            layer_name=d["layer"],
            turbine_id=d["turbine_id"],
            timestamp=datetime.datetime.fromisoformat(d["timestamp"]),
            file_metadata=fm,
            detected_variables=d["detected_variables"],
            rule_execution_results=[self._rule_execution_results_from_dict(rule) for rule in d["rules"]],
        )
        return reported

    def load(self, input_path: pathlib.Path | str) -> list[LayerReport]:
        path = pathlib.Path(input_path)
        if not path.exists():
            LOGGER.error(f"File not found: {path}")
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r") as filestream:
            contents = yaml.safe_load(filestream)

        layer_reports = [self._layer_report_from_dict(content) for content in contents["turbines"].values()]
        return layer_reports
