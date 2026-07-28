"""Phase-specific scan execution errors used for conservative failure records."""


class ScanConfigurationError(ValueError):
    """The declared experiment could not be converted into executable inputs."""


class ResultIntegrityError(RuntimeError):
    """A returned trajectory or event record was internally inconsistent."""


class OutputSerializationError(RuntimeError):
    """A valid in-memory result could not be serialized as declared."""
