"""
Home to exceptions used inside of wuzup.
"""


class ModelNotSetError(ValueError):
    """Raised when a required model is not set."""

    pass


class NonBooleanAIResponseError(ValueError):
    """Raised when an AI response expected to be boolean is not."""

    pass


class MissingRequiredKeyError(KeyError):
    """Raised when a required key is missing."""

    pass
