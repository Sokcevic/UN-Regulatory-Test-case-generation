"""Pipeline and LLM configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Settings for the LLM backend — any OpenAI-compatible endpoint (hosted or self-hosted vLLM)."""

    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4.1-mini"
    api_key: str = "EMPTY"  # vLLM accepts any non-empty string; set a real key for hosted providers
    temperature: float = 0.1
    max_tokens: int = 32768
    max_loops: int = 15  # maximum ReAct iterations per clause


class PipelineConfig(BaseModel):
    """Top-level pipeline settings."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    output_dir: str = "output"
    workers: int = 4  # parallel agent calls during test generation

    # Ablation: set to False to skip LLM classification and treat every non-pseudo
    # clause as a testable candidate. Useful for measuring the contribution of the
    # semantic filtering stage. When True (default), only obligation /
    # test_procedure / performance_data clauses are passed to the generator.
    use_llm_classification: bool = True
