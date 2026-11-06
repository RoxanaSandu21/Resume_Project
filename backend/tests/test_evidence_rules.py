"""Regression tests for conservative resume evidence handling."""

import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.nlp_service import (  # noqa: E402
    _build_cv_evidence,
    PRIORITY_WEIGHTS,
    _calculate_evidence_coverage,
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
        samples = [item for item in evidence if item.get("is_english_writing_sample")]
        self.assertTrue(samples)
        self.assertTrue(_quote_is_in_source(samples[0]["quote"], english_text))
        self.assertEqual(
            _evidence_support_level(
                "Excellent understanding of English – spoken and written",
                samples[0],
            ),
            "contextual",
        )

    def test_short_english_cv_uses_verbatim_writing_sample(self):
        short_cv = (
            "Professional Experience\n"
            "Junior Developer\n"
            "Worked with the engineering team to build and maintain web applications "
            "for internal users. Reviewed existing features, fixed issues, and wrote "
            "clear documentation for users. Collaborated with colleagues across the "
            "team and helped improve application usability. Created updates for the "
            "product and communicated progress with the team.\n"
            "Education\n"
            "Bachelor of Computer Science\n"
        )
        self.assertLess(len(short_cv.split()), 80)
        self.assertTrue(_looks_like_english_resume(short_cv))
        evidence = _build_cv_evidence({}, short_cv)
        samples = [item for item in evidence if item.get("is_english_writing_sample")]
        self.assertEqual(len(samples), 1)
        self.assertTrue(_quote_is_in_source(samples[0]["quote"], short_cv))
        self.assertEqual(
            _evidence_support_level(
                "Excellent understanding of English – spoken and written",
                samples[0],
            ),
            "contextual",
        )

    def test_extraction_prompt_requires_all_roles_not_only_latest_role(self):
        from services.nlp_service import _CV_SYSTEM_PROMPT

        self.assertIn("extract EVERY distinct work/experience", _CV_SYSTEM_PROMPT)
        self.assertIn("Do not stop after the current or most recent job", _CV_SYSTEM_PROMPT)

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

    def test_source_fallback_extracts_individual_skills(self):
        source = (
            "Technical Skills\n"
            "Programming Languages: C++, C, Java, JavaScript, TypeScript, Python, PHP, HTML, CSS\n"
            "Frameworks & Libraries: SpringBoot, React.js, React Native, Node.js, Express\n"
            "Databases: SQL, MongoDB, Oracle Database, PostgreSQL\n"
            "Tools: Git, GitHub, GitLab"
        )
        from services.nlp_service import _recover_skills_from_source

        recovered = _recover_skills_from_source(source)
        names = {item["name"] for item in recovered}
        self.assertIn("C++", names)
        self.assertIn("C", names)
        self.assertIn("Java", names)
        self.assertIn("Python", names)
        self.assertIn("React Native", names)
        self.assertIn("PostgreSQL", names)
        self.assertIn("GitHub", names)
        self.assertNotIn("Programming Languages", names)
        self.assertTrue(all(_quote_is_in_source(item["evidence"], source) for item in recovered))

    def test_source_fallback_recovers_dated_roles_and_excludes_education(self):
        source = (
            "Professional Experience\n"
            "Business Sales Advisor Feb 2025 – Sep 2025\n"
            "Assisted customers with account questions.\n"
            "Software Engineer Intern Jul 2023 – Sep 2023\n"
            "Worked on an engineering project.\n"
            "Education\n"
            "Bachelor's in Mathematics and Computer Science Oct 2021 – Jul 2025"
        )
        from services.nlp_service import _recover_experience_from_source

        recovered = _recover_experience_from_source(source)
        roles = {item["role"] for item in recovered}
        self.assertIn("Business Sales Advisor", roles)
        self.assertIn("Software Engineer Intern", roles)
        self.assertFalse(any("Bachelor" in role for role in roles))
        self.assertTrue(all(_quote_is_in_source(item["evidence"], source) for item in recovered))

    def test_empty_llm_sections_are_filled_from_source_grounded_fallbacks(self):
        source = (
            "Programming Languages: Java, Python, C++\n"
            "Professional Experience\n"
            "Business Sales Advisor Feb 2025 – Sep 2025\n"
            "Assisted customers with account questions."
        )
        from services.nlp_service import _merge_source_grounded_fallbacks

        skills, experience = _merge_source_grounded_fallbacks(source, [], [])
        self.assertTrue(any(item["name"] == "Java" for item in skills))
        self.assertTrue(any(item["role"] == "Business Sales Advisor" for item in experience))
        self.assertTrue(all(_quote_is_in_source(item["evidence"], source) for item in skills + experience))

    def test_coverage_metrics_distinguish_partial_from_fully_matched(self):
        metrics = _calculate_evidence_coverage([
            {"status": "partial", "priority": "required", "evidence": [{"quote": "English CV text"}]},
            {"status": "partial", "priority": "required", "evidence": [{"quote": "Business Sales Advisor"}]},
            {"status": "missing", "priority": "required", "evidence": []},
            {"status": "missing", "priority": "required", "evidence": []},
            {"status": "missing", "priority": "required", "evidence": []},
            {"status": "missing", "priority": "preferred", "evidence": []},
        ])
        self.assertEqual(metrics["evidence_covered_count"], 2)
        self.assertEqual(metrics["requirements_total_count"], 6)
        self.assertEqual(metrics["evidence_coverage_percent"], 33.33)
        self.assertEqual(metrics["fully_matched_count"], 0)
        self.assertEqual(metrics["required_missing_count"], 3)
        self.assertEqual(metrics["required_criteria_score"], 20.0)
        self.assertEqual(metrics["required_criteria_count"], 5)
        self.assertEqual(PRIORITY_WEIGHTS["preferred"], 1.5)
        self.assertEqual(PRIORITY_WEIGHTS["required"], 3.0)

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
