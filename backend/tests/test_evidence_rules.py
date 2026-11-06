"""Regression tests for conservative resume evidence handling."""

import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.nlp_service import (  # noqa: E402
    _build_cv_evidence,
    _chunk_raw_resume_text,
    _evidence_support_level,
    _evidence_supports_requirement,
    _looks_like_english_resume,
    _quote_is_in_source,
)


class EvidenceRuleTests(unittest.TestCase):
    def test_business_sales_advisor_is_contextual_communication_evidence(self):
        item = {
            "label": "Business Sales Advisor — Vodafone",
            "quote": "Business Sales Advisor",
            "type": "experience",
        }
        self.assertEqual(
            _evidence_support_level("Confidence in speaking to users directly", item),
            "contextual",
        )

    def test_education_heading_is_not_english_proficiency_evidence(self):
        self.assertFalse(
            _evidence_supports_requirement(
                "Excellent understanding of English – spoken and written",
                "Education",
            )
        )

    def test_explicit_english_level_is_direct_evidence(self):
        item = {
            "label": "Languages",
            "quote": "English — Advanced; spoken and written",
            "type": "language",
        }
        self.assertEqual(
            _evidence_support_level(
                "Excellent understanding of English – spoken and written",
                item,
            ),
            "direct",
        )

    def test_english_cv_text_is_only_contextual_written_evidence(self):
        english_text = (
            "The candidate worked with customers and the team. The role was focused "
            "on helping users, answering questions, and supporting the service. "
        ) * 8
        self.assertTrue(_looks_like_english_resume(english_text))

        evidence = _build_cv_evidence({}, english_text)
        samples = [item for item in evidence if item["type"] == "language_context"]
        self.assertTrue(samples)
        self.assertTrue(_quote_is_in_source(samples[0]["quote"], english_text))
        self.assertEqual(
            _evidence_support_level(
                "Excellent understanding of English – spoken and written",
                samples[0],
            ),
            "contextual",
        )

    def test_github_tools_do_not_prove_networking(self):
        item = {
            "label": "Tools",
            "quote": "Git, GitHub, GitLab",
            "type": "tool",
        }
        self.assertEqual(
            _evidence_support_level("Basic networking skills required", item),
            "none",
        )

    def test_internship_title_does_not_prove_working_under_pressure(self):
        item = {
            "label": "Software Engineer Intern — Continental Automotive",
            "quote": "Software Engineer Intern July 2023 – Sept 2023",
            "type": "experience",
        }
        self.assertEqual(
            _evidence_support_level("Ability to work under pressure", item),
            "none",
        )

    def test_context_windows_keep_role_title_near_responsibilities(self):
        source = (
            "Professional Experience\n"
            "Business Sales Advisor — Vodafone\n"
            "February 2025 – September 2025\n"
            "Assisted customers with mobile plans and account questions."
        )
        chunks = _chunk_raw_resume_text(source)
        self.assertTrue(
            any(
                "Business Sales Advisor" in chunk
                and "Assisted customers" in chunk
                for chunk in chunks
            )
        )


if __name__ == "__main__":
    unittest.main()
