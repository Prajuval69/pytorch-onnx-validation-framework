"""Structured failure categories; only InvalidInput is an expected negative-test rejection."""


class ValidationError(Exception):
    """Base for user-visible engineering failures."""


class ConfigError(ValidationError):
    pass


class ExportFailure(ValidationError):
    pass


class InvalidInput(ValidationError):
    def __init__(self, message: str, reason: str = "unknown") -> None:
        super().__init__(message)
        self.reason = reason


class RuntimeFailure(ValidationError):
    pass


class NumericalMismatch(ValidationError):
    pass


class ArtifactError(ValidationError):
    pass
