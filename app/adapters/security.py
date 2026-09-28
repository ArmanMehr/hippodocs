import re
from typing import ClassVar

import structlog
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities.engine.recognizer_result import (
    RecognizerResult as AnonymizerRecognizerResult,
)

from app.exceptions import (
    SuspiciousInputError,
)
from app.observability import observe

logger = structlog.get_logger(__name__)


class RegexInputSanitizer:
    INJECTION_PATTERNS: ClassVar[list[tuple[str, str]]] = [
        ("ignore_previous_instructions", r"ignore\s+(all\s+)?previous\s+instructions"),
        ("forget_previous", r"forget\s+(all\s+)?previous"),
        ("new_instructions", r"new\s+instructions:"),
        ("system_prompt", r"system\s*prompt"),
        ("end_prompt", r"---\s*end\s*(of)?\s*prompt"),
        ("pretend_you_are", r"pretend\s+you\s+are"),
        ("act_as_if_you", r"act\s+as\s+(if\s+)?you"),
        ("bypass_restrictions", r"bypass\s+(all\s+)?restrictions"),
    ]

    def __init__(self) -> None:
        self._patterns = [
            (name, re.compile(pattern, re.IGNORECASE))
            for name, pattern in self.INJECTION_PATTERNS
        ]

    def is_suspicious(self, text: str) -> bool:
        return any(pattern.search(text) for _, pattern in self._patterns)

    def _suspicious_pattern(self, text: str) -> str | None:
        for name, pattern in self._patterns:
            if pattern.search(text):
                return name
        return None

    @observe(name="input-sanitize", as_type="chain")
    def sanitize(self, text: str) -> str:
        pattern_name = self._suspicious_pattern(text)
        if pattern_name is not None:
            logger.warning(
                "suspicious_information_input",
                pattern=pattern_name,
            )
            raise SuspiciousInputError()

        text = re.sub(r"[-]{3,}", "", text)
        text = re.sub(r"[=]{3,}", "", text)
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = text.replace("{{", "{ {").replace("}}", "} }")

        return text


class LocalPresidioPIIRedactor:
    PII_ENTITIES: ClassVar[list[str]] = [
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "CREDIT_CARD",
        "IP_ADDRESS",
        "LOCATION",
    ]

    def __init__(self) -> None:
        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()

    @observe(name="pii-redact", as_type="chain")
    def redact(self, text: str) -> str:
        results = self._analyzer.analyze(
            text=text,
            language="en",
            entities=self.PII_ENTITIES,
        )
        anonymized = self._anonymizer.anonymize(
            text=text,
            analyzer_results=[
                AnonymizerRecognizerResult(
                    entity_type=result.entity_type,
                    start=result.start,
                    end=result.end,
                    score=result.score,
                )
                for result in results
            ],
        )
        return anonymized.text


class LocalPresidioRegexOutputValidator:
    SECRET_PATTERNS: ClassVar[list[tuple[str, str]]] = [
        ("openai_api_key", r"sk-[A-Za-z0-9]{20,}"),
        ("aws_access_key", r"AKIA[0-9A-Z]{16}"),
        ("private_key", r"-----BEGIN .* PRIVATE KEY-----"),
        ("attack_instructions", r"here('s| is) (how|the way) to (hack|steal|attack)"),
        ("password_disclosure", r"password is"),
        ("api_key_disclosure", r"api[_\s]?(key|token|secret)"),
    ]

    PII_ENTITIES: ClassVar[list[str]] = [
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "CREDIT_CARD",
    ]

    def __init__(self) -> None:
        self._analyzer = AnalyzerEngine()
        self._secret_patterns = [
            (name, re.compile(pattern, re.IGNORECASE))
            for name, pattern in self.SECRET_PATTERNS
        ]

    @observe(name="output-validate", as_type="chain")
    def validate(self, answer: str) -> str:
        suspicious_patterns = []
        for name, pattern in self._secret_patterns:
            if pattern.search(answer):
                suspicious_patterns.append(name)

        if suspicious_patterns:
            logger.warning(
                "suspicious_information_in_model_output",
                patterns=sorted(set(suspicious_patterns)),
            )
            return "[CONTENT BLOCKED]"

        results = self._analyzer.analyze(
            text=answer, language="en", entities=self.PII_ENTITIES
        )

        if results:
            logger.warning(
                "pii_detected_in_model_output",
                entity_types=sorted({result.entity_type for result in results}),
            )
            return "[CONTENT BLOCKED]"

        return answer
