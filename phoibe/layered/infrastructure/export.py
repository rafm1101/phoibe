import datetime
import json
import pathlib
from typing import Any

import yaml

from phoibe.layered.core.entities import LayerReport
from phoibe.layered.core.metadata import DataLineage
from phoibe.layered.core.metadata import DeviceSpecification
from phoibe.layered.core.metadata import SiteMetadata
from phoibe.layered.core.metadata import TemporalSpecification
from phoibe.layered.core.metadata import VariableMetadata


class ProfileExporter:
    """Export validation profiles in standard formats.

    Generates ODCS-compatible profile fragments from validation reports
    with optional WRA metadata for documentation and contract creation.

    Example
    -------
    1. Basic profile export:

       > exporter = ProfileExporter()
       > profile = exporter.to_odcs_fragment(report)
       > exporter.export_to_yaml(report, "profiles/WEA_01_raw.yaml")

    2. With WRA metadata:

       > profile = exporter.to_odcs_fragment(
       >     report,
       >     site=SiteMetadata(site_id="SITE_001", ...),
       >     device=DeviceSpecification(device_id="WEA_01", ...),
       >     variables=[VariableMetadata(name="wind_speed", units="m/s", ...)]
       > )
    """

    def to_odcs_fragment(
        self,
        report: LayerReport,
        *,
        site: SiteMetadata | None = None,
        device: DeviceSpecification | None = None,
        variables: list[VariableMetadata] | None = None,
        temporal: TemporalSpecification | None = None,
    ) -> dict[str, Any]:
        """Export report as ODCS-compatible profile fragment.

        Creates discoveredSchema and discoveredQuality sections
        that can be used as input for contract creation.

        Parameters
        ----------
        report
            Validation report from profiling.
        site
            Optional site metadata (WRA compliance).
        device
            Optional device specifications (WRA compliance).
        variables
            Optional variable metadata list (WRA compliance).
        temporal
            Optional temporal specification (WRA compliance).

        Return
        ------
        dict
            ODCS-compatible profile structure with WRA metadata.
        """
        profile = {
            "profileMetadata": {
                "generatedBy": "phoibe-validator",
                "turbineId": report.turbine_id,
                "layer": report.layer_name,
                "profiledAt": report.timestamp.isoformat(),
                "validationScore": report.percentage,
            },
            "discoveredSchema": self._extract_schema(report, variables),
            "discoveredQuality": self._extract_quality_metrics(report),
            "detectedVariables": report.detected_variables,
        }

        if site:
            profile["site"] = site.to_dict()

        if device:
            profile["device"] = device.to_dict()

        if temporal:
            profile["temporal"] = temporal.to_dict()

        return profile

    def _extract_schema(
        self,
        report: LayerReport,
        variables: list[VariableMetadata] | None = None,
    ) -> dict[str, Any]:
        """Extract schema information from report.

        Parameters
        ----------
        report
            Validation report.
        variables
            Optional variable metadata for enrichment.

        Return
        ------
        dict
            Schema dictionary.
        """
        schema: dict[str, Any] = {}

        var_metadata = {}
        if variables:
            var_metadata = {v.name: v for v in variables}

        for variable_name, column_name in report.detected_variables.items():
            var_info: dict[str, Any] = {"actualColumn": column_name, "detected": column_name is not None}

            if variable_name in var_metadata:
                meta = var_metadata[variable_name]
                if meta.units:
                    var_info["units"] = meta.units
                if meta.standard_name:
                    var_info["standard_name"] = meta.standard_name
                if meta.height is not None:
                    var_info["height"] = meta.height
                if meta.description:
                    var_info["description"] = meta.description

            schema[variable_name] = var_info

        for result in report.rule_execution_results:
            if "ranges" in result.rule_name or "essential_ranges" in result.rule_name:
                if result.details and "checked" in result.details:
                    for var, range_vals in result.details["checked"].items():
                        if var not in schema:
                            schema[var] = {}
                        schema[var]["discoveredRange"] = range_vals

        return schema

    def _extract_quality_metrics(self, report: LayerReport) -> dict[str, Any]:
        """Extract quality metrics from report.

        Parameters
        ----------
        report
            Validation report.

        Return
        ------
        dict
            Quality metrics dictionary.
        """
        metrics: dict[str, Any] = {
            "overallScore": report.percentage,
            "overallStatus": report.overall_status.value,
            "criticalFailures": report.critical_failures,
            "warnings": report.warnings,
        }

        for result in report.rule_execution_results:
            if result.details:
                if result.rule_name == "temporal_attributes":
                    metrics["temporal"] = {
                        "frequency": result.details.get("frequency"),
                        "hasDuplicates": result.details.get("has_duplicates"),
                        "isSorted": result.details.get("is_sorted"),
                        "start": result.details.get("start"),
                        "end": result.details.get("end"),
                        "timezone": result.details.get("tzinfo"),
                        "oversampling": result.details.get("oversampling"),
                    }

                elif result.rule_name == "data_gaps":
                    metrics["completeness"] = {
                        "gapCount": result.details.get("gap_count"),
                        "longestGap": result.details.get("gap_length_max"),
                        "meanGap": result.details.get("gap_length_mean"),
                        "totalGapLength": result.details.get("gap_length_total"),
                    }

                elif result.rule_name == "availability":
                    metrics["availability"] = result.details.get("availability_data")

        return metrics

    def export_to_yaml(
        self,
        report: LayerReport,
        output_path: str | pathlib.Path,
        **kwargs,
    ) -> None:
        """Export profile to YAML file.

        Parameters
        ----------
        report
            Validation report.
        output_path
            Path to output YAML file.
        **kwargs
            Additional arguments passed to to_odcs_fragment()
            (site, device, variables, temporal).
        """
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        profile = self.to_odcs_fragment(report, **kwargs)

        with open(output_path, "w") as f:
            yaml.dump(profile, f, default_flow_style=False, sort_keys=False)

    def export_to_json(
        self,
        report: LayerReport,
        output_path: str | pathlib.Path,
        **kwargs,
    ) -> None:
        """Export profile to JSON file.

        Parameters
        ----------
        report
            Validation report.
        output_path
            Path to output JSON file.
        **kwargs
            Additional arguments passed to to_odcs_fragment()
            (site, device, variables, temporal).
        """
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        profile = self.to_odcs_fragment(report, **kwargs)

        with open(output_path, "w") as f:
            json.dump(profile, f, indent=2, default=str)


class QualityAttestationWriter:
    """Write quality attestations for data products.

    Creates ODPS-compatible quality certificates with optional WRA metadata
    for IEA Wind Task 43 compliance.

    Example
    -------
    1. Basic attestation:

       > writer = QualityAttestationWriter()
       > attestation = writer.create_attestation(
       >     report=bronze_report,
       >     product_id="scada-bronze-10min",
       >     version="1.2.0"
       > )

    2. With WRA metadata:

       > attestation = writer.create_attestation(
       >     report=bronze_report,
       >     product_id="scada-bronze-10min",
       >     version="1.2.0",
       >     site=SiteMetadata(site_id="SITE_001", latitude=52.52, ...),
       >     device=DeviceSpecification(device_id="WEA_01", manufacturer="Vestas", ...),
       >     variables=[VariableMetadata(name="wind_speed", units="m/s", ...)],
       >     temporal=TemporalSpecification(sampling_interval="PT10M")
       > )
    """

    def create_attestation(
        self,
        report: LayerReport,
        product_id: str,
        version: str,
        *,
        contract_id: str | None = None,
        site: SiteMetadata | None = None,
        device: DeviceSpecification | None = None,
        variables: list[VariableMetadata] | None = None,
        temporal: TemporalSpecification | None = None,
        lineage: DataLineage | None = None,
    ) -> dict[str, Any]:
        """Create quality attestation from validation report.

        Parameters
        ----------
        report
            Validation report from contract validation.
        product_id
            Data product identifier.
        version
            Product version.
        contract_id
            Optional contract identifier.
        site
            Optional site metadata (WRA compliance).
        device
            Optional device specifications (WRA compliance).
        variables
            Optional variable metadata list (WRA compliance).
        temporal
            Optional temporal specification (WRA compliance).
        lineage
            Optional data lineage / provenance (WRA compliance).

        Return
        ------
        dict
            ODPS-compatible attestation with WRA metadata.
        """
        attestation: dict[str, Any] = {
            "productId": product_id,
            "version": version,
            "attestation": {
                "attestedAt": datetime.datetime.now().isoformat(),
                "attestedBy": "phoibe-layered",
                "validationLayer": report.layer_name,
                "turbineId": report.turbine_id,
                "qualityMetrics": {
                    "score": report.percentage,
                    "scoreAchieved": report.score_achieved,
                    "scoreMaximum": report.score_max,
                    "overallStatus": report.overall_status.value,
                },
                "compliance": {
                    "criticalFailures": report.critical_failures,
                    "warnings": report.warnings,
                    "totalChecks": len(report.rule_execution_results),
                    "passedChecks": sum(1 for r in report.rule_execution_results if r.status.value == "passed"),
                },
                "detailedMetrics": self._extract_detailed_metrics(report),
                "validationReport": {
                    "fileMetadata": {
                        "filename": report.file_metadata.filename,
                        "sizeMB": round(float(report.file_metadata.size_mb), 2),
                        "format": report.file_metadata.format,
                    },
                    "timestamp": report.timestamp.isoformat(),
                },
            },
        }

        if contract_id:
            attestation["attestation"]["contractId"] = contract_id

        if site:
            attestation["site"] = site.to_dict()

        if device:
            attestation["device"] = device.to_dict()

        if variables:
            attestation["variables"] = [v.to_dict() for v in variables]

        if temporal:
            attestation["temporal"] = temporal.to_dict()

        if lineage:
            attestation["lineage"] = lineage.to_dict()

        return attestation

    def _extract_detailed_metrics(self, report: LayerReport) -> dict[str, Any]:
        """Extract detailed metrics from report.

        Parameters
        ----------
        report
            Validation report.

        Return
        ------
        dict
            Detailed metrics dictionary.
        """
        metrics = {}

        for result in report.rule_execution_results:
            if result.details:
                if result.rule_name == "availability":
                    metrics["availability"] = result.details.get("availability_data")

                elif result.rule_name == "data_gaps":
                    gap_count = result.details.get("gap_count", 0)
                    metrics["gapCount"] = gap_count
                    metrics["gapLengthMax"] = result.details.get("gap_length_max")
                    metrics["gapLengthMean"] = result.details.get("gap_length_mean")

                elif result.rule_name == "temporal_attributes":
                    metrics["temporal"] = {
                        "frequency": result.details.get("frequency"),
                        "sorted": result.details.get("is_sorted"),
                        "duplicates": result.details.get("has_duplicates"),
                        "oversampling": result.details.get("oversampling"),
                    }

        return metrics

    def write_attestation(
        self,
        attestation: dict[str, Any],
        output_path: str | pathlib.Path,
    ) -> None:
        """Write attestation to JSON file.

        Typically saved alongside data product as:
        - _quality_attestation.json
        - metadata/quality_attestation_.json

        Parameters
        ----------
        attestation
            Attestation dictionary.
        output_path
            Path to output file.
        """
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(attestation, f, indent=2, default=str)

    def create_and_write(
        self,
        report: LayerReport,
        product_id: str,
        version: str,
        output_path: str | pathlib.Path,
        **kwargs,
    ) -> dict[str, Any]:
        """Convenience method: Create and write attestation in one call.

        Parameters
        ----------
        report
            Validation report.
        product_id
            Data product ID.
        version
            Product version.
        output_path
            Path to output file.
        **kwargs
            Additional arguments passed to create_attestation()
            (contract_id, site, device, variables, temporal, lineage).

        Return
        ------
        dict
            Created attestation dictionary.
        """
        attestation = self.create_attestation(report, product_id, version, **kwargs)
        self.write_attestation(attestation, output_path)
        return attestation


__all__ = ["ProfileExporter", "QualityAttestationWriter"]
