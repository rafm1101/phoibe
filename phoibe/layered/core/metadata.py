from __future__ import annotations

import dataclasses
import datetime
from typing import Any


@dataclasses.dataclass
class SiteMetadata:
    """Site location and identification metadata.

    Attributes
    ----------
    site_id
        Unique site identifier.
    site_name
        Human-readable site name.
    latitude
        Site latitude in decimal degrees (WGS84).
    longitude
        Site longitude in decimal degrees (WGS84).
    elevation
        Site elevation above sea level in meters.
    timezone
        IANA timezone identifier (e.g. 'UTC', 'Europe/Berlin').
    """

    site_id: str
    site_name: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    elevation: float | None = None
    timezone: str = "UTC"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.site_id,
            "name": self.site_name,
            "location": (
                {
                    "latitude": self.latitude,
                    "longitude": self.longitude,
                    "elevation": self.elevation,
                }
                if self.latitude is not None
                else None
            ),
            "timezone": self.timezone,
        }


@dataclasses.dataclass
class DeviceSpecification:
    """Wind turbine device specifications.

    Attributes
    ----------
    device_id
        Unique device identifier (e.g. turbine ID).
    device_type
        Type of device ('wind_turbine', 'met_mast', 'lidar', etc.).
    manufacturer
        Device manufacturer.
    model
        Device model designation.
    hub_height
        Hub height in meters (for wind turbines).
    rotor_diameter
        Rotor diameter in meters (for wind turbines).
    rated_power
        Rated power in kW (for wind turbines).
    """

    device_id: str
    device_type: str = "wind_turbine"
    manufacturer: str | None = None
    model: str | None = None
    hub_height: float | None = None
    rotor_diameter: float | None = None
    rated_power: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "device_id": self.device_id,
            "device_type": self.device_type,
        }

        if self.manufacturer:
            result["manufacturer"] = self.manufacturer
        if self.model:
            result["model"] = self.model
        if self.hub_height is not None:
            result["hub_height"] = self.hub_height
        if self.rotor_diameter is not None:
            result["rotor_diameter"] = self.rotor_diameter
        if self.rated_power is not None:
            result["rated_power"] = self.rated_power

        return result


@dataclasses.dataclass
class VariableMetadata:
    """Metadata for a measured variable.

    Attributes
    ----------
    name
        Variable name (e.g. 'wind_speed', 'power').
    standard_name
        CF Convention standard name (e.g. 'wind_speed_at_hub_height').
    units
        Units of measurement (e.g. 'm/s', 'kW', '°C').
    height
        Measurement height in meters (if applicable).
    description
        Human-readable description.
    """

    name: str
    standard_name: str | None = None
    units: str | None = None
    height: float | None = None
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {"name": self.name}

        if self.standard_name:
            result["standard_name"] = self.standard_name
        if self.units:
            result["units"] = self.units
        if self.height is not None:
            result["height"] = self.height
        if self.description:
            result["description"] = self.description

        return result


@dataclasses.dataclass
class TemporalSpecification:
    """Temporal metadata with ISO 8601 compliance.

    Attributes
    ----------
    sampling_interval
        ISO 8601 duration (e.g. 'PT10M' for 10 minutes).
    reference_time
        'start_of_interval' or 'end_of_interval'.
    start_time
        Start of measurement period (ISO 8601).
    end_time
        End of measurement period (ISO 8601).
    timezone
        IANA timezone identifier.
    """

    sampling_interval: str
    reference_time: str = "end_of_interval"
    start_time: datetime.datetime | None = None
    end_time: datetime.datetime | None = None
    timezone: str = "UTC"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "sampling_interval": self.sampling_interval,
            "reference_time": self.reference_time,
            "timezone": self.timezone,
        }

        if self.start_time:
            result["start_time"] = self.start_time.isoformat()
        if self.end_time:
            result["end_time"] = self.end_time.isoformat()

        return result

    @staticmethod
    def from_minutes(minutes: int) -> str:
        """Convert minutes to ISO 8601 duration.

        Parameters
        ----------
        minutes
            Number of minutes.

        Return
        ------
        str
            ISO 8601 duration (e.g., 'PT10M').

        Example
        -------
        >>> TemporalSpecification.from_minutes(10)
        'PT10M'
        >>> TemporalSpecification.from_minutes(60)
        'PT1H'
        """
        if minutes >= 60 and minutes % 60 == 0:
            hours = minutes // 60
            return f"PT{hours}H"
        return f"PT{minutes}M"


@dataclasses.dataclass
class LineageNode:
    """Processing step in data lineage.

    Attributes
    ----------
    step
        Step number in processing chain.
    process
        Process name (e.g. 'raw_data_acquisition', 'quality_control').
    timestamp
        When this step was executed.
    software
        Software name and version.
    validation_report
        Path to validation report (if applicable).
    contract_id
        Data contract ID (if applicable).
    """

    step: int
    process: str
    timestamp: datetime.datetime
    software: str | None = None
    validation_report: str | None = None
    contract_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "step": self.step,
            "process": self.process,
            "timestamp": self.timestamp.isoformat(),
        }

        if self.software:
            result["software"] = self.software
        if self.validation_report:
            result["validation_report"] = self.validation_report
        if self.contract_id:
            result["contract_id"] = self.contract_id

        return result


@dataclasses.dataclass
class DataLineage:
    """Complete data lineage / provenance.

    Attributes
    ----------
    parent_nodes
        List of parent data product IDs.
    processing_chain
        Ordered list of processing steps.
    source_organization
        Organization that produced the data.
    contact
        Contact information.
    data_policy
        Data usage policy / license (e.g. 'CC-BY-4.0').
    """

    parent_nodes: list[str]
    processing_chain: list[LineageNode]
    source_organization: str | None = None
    contact: str | None = None
    data_policy: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "parent_nodes": self.parent_nodes,
            "processing_chain": [node.to_dict() for node in self.processing_chain],
        }

        if self.source_organization:
            result["source_organization"] = self.source_organization
        if self.contact:
            result["contact"] = self.contact
        if self.data_policy:
            result["data_policy"] = self.data_policy

        return result


__all__ = [
    "SiteMetadata",
    "DeviceSpecification",
    "VariableMetadata",
    "TemporalSpecification",
    "LineageNode",
    "DataLineage",
]
