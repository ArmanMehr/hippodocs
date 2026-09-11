import re
from logging import getLogger
from typing import ClassVar

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities.engine.recognizer_result import (
    RecognizerResult as AnonymizerRecognizerResult,
)

from app.exceptions import (
    SuspiciousInputError,
    SuspiciousOutputError,
)

logger = getLogger(__name__)


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

    def __init__(self):
        self._patterns = [
            (name, re.compile(p, re.IGNORECASE)) for name, p in self.INJECTION_PATTERNS
        ]

    def is_suspicious(self, text: str) -> bool:
        for name, pattern in self._patterns:
            if pattern.search(text):
                logger.warning("Suspicious pattern detected: %s", name)
                return True
        return False

    def sanitize(self, text: str) -> str:
        if self.is_suspicious(text):
            raise SuspiciousInputError("Suspicious input detected")
        text = re.sub(r"[-]{3,}", "", text)
        text = re.sub(r"[=]{3,}", "", text)
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = text.replace("{{", "{ {").replace("}}", "} }")
        return text.strip()


class LocalPresidioPIIRedactor:
    PII_ENTITIES: ClassVar[list[str]] = [
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "CREDIT_CARD",
        "IP_ADDRESS",
        "LOCATION",
    ]

    def __init__(self):
        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()

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
                    entity_type=r.entity_type,
                    start=r.start,
                    end=r.end,
                    score=r.score,
                )
                for r in results
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

    def __init__(self):
        self._analyzer = AnalyzerEngine()
        self._secret_patterns = [
            (name, re.compile(p, re.IGNORECASE)) for name, p in self.SECRET_PATTERNS
        ]

    def validate(self, answer: str) -> str:
        for name, pattern in self._secret_patterns:
            if pattern.search(answer):
                logger.error("Secret pattern detected: %s", name)
                raise SuspiciousOutputError("Potential secret detected in model output")

        results = self._analyzer.analyze(
            text=answer,
            language="en",
            entities=self.PII_ENTITIES,
        )

        if results:
            logger.error("PII detected in model output")
            return "[CONTENT BLOCKED]"

        return answer
