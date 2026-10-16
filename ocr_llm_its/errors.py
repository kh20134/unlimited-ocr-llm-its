class PipelineError(Exception):
    """User-facing failure of the OCR or LLM pipeline."""


class CudaRequiredError(PipelineError):
    """Local Unlimited-OCR was requested, but CUDA is not available."""


class FixtureNotFoundError(PipelineError):
    """No precomputed OCR markdown exists for this input file."""


class LlmConfigError(PipelineError):
    """The selected LLM backend is missing a host, key, or model."""
