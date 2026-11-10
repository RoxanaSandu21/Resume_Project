"""Regression tests for conservative resume evidence handling."""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.nlp_service import (  # noqa: E402
    _build_cv_evidence,
    CVExtraction,
    CVSkill,
    MatchAssessmentOutput,
    RequirementAssessment,
    _extract_cv_cached,
    _llm_assess_requirements,
    extract_cv_information,
    PRIORITY_WEIGHTS,
    _calculate_evidence_coverage,
    _calculate_weighted_match_score,
    _chunk_raw_resume_text,
    _deterministic_requirement_assessment,
    _is_skill_or_project_only_evidence,
    _evidence_support_level,
    _evidence_supports_requirement,
    _looks_like_english_resume,
    _merge_source_grounded_fallbacks,
    _quote_is_in_source,
    _recover_experience_from_source,
    _recover_projects_from_source,
    _recover_education_from_source,
    _merge_source_grounded_projects,
    _merge_source_grounded_education,
    _retrieve_evidence,
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

    def test_software_debugging_is_contextual_not_direct_it_support_evidence(self):
        requirement = "Troubleshooting experience in a support environment"
        debugging_item = {
            "label": "Junior Java Developer | Responsibility",
            "quote": "Debugged Java application defects and fixed software bugs.",
            "type": "experience_responsibility",
        }
        support_item = {
            "label": "IT Support Assistant | Responsibility",
            "quote": "Worked on the IT help desk and resolved user support tickets.",
            "type": "experience_responsibility",
        }

        self.assertEqual(
            _evidence_support_level(requirement, debugging_item),
            "contextual",
        )
        self.assertEqual(
            _evidence_support_level(requirement, support_item),
            "direct",
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

    def test_experience_parser_recovers_organization_and_responsibilities(self):
        source = (
            "Professional Experience\n"
            "Junior Java Developer Sep 2025 – present\n"
            "Endava\n"
            "• Resolved software incidents.\n"
            "• Managed multiple priorities under tight deadlines.\n"
            "Business Sales Advisor Feb 2025 – Sep 2025\n"
            "Vodafone\n"
            "• Assisted customers with account questions.\n"
            "IT Support Assistant Jan 2022 – Jun 2022\n"
            "Acme Support\n"
            "Provide first-level support to colleagues.\n"
            "Mathematics and Computer Science Teacher Dec 2021 – June 2024\n"
            "Matematica.md\n"
            "• Taught mathematics and explained difficult concepts to students.\n"
            "Technologies\n"
            "Languages: C++, Java, Python\n"
            "Education\n"
            "Bachelor's in Mathematics and Computer Science Oct 2021 – Jul 2025"
        )

        recovered = _recover_experience_from_source(source)
        by_role = {item["role"]: item for item in recovered}

        self.assertEqual(by_role["Junior Java Developer"]["organization"], "Endava")
        self.assertEqual(
            by_role["Junior Java Developer"]["responsibilities"],
            [
                "Resolved software incidents.",
                "Managed multiple priorities under tight deadlines.",
            ],
        )
        self.assertEqual(by_role["Business Sales Advisor"]["organization"], "Vodafone")
        self.assertEqual(
            by_role["Business Sales Advisor"]["responsibilities"],
            ["Assisted customers with account questions."],
        )
        self.assertEqual(by_role["IT Support Assistant"]["organization"], "Acme Support")
        self.assertEqual(
            by_role["IT Support Assistant"]["responsibilities"],
            ["Provide first-level support to colleagues."],
        )
        teacher = by_role["Mathematics and Computer Science Teacher"]
        self.assertEqual(teacher["organization"], "Matematica.md")
        self.assertEqual(
            teacher["responsibilities"],
            ["Taught mathematics and explained difficult concepts to students."],
        )
        self.assertNotIn("Technologies", teacher["evidence"])
        self.assertNotIn("Languages:", teacher["evidence"])
        self.assertTrue(all(_quote_is_in_source(item["evidence"], source) for item in recovered))

    def test_existing_llm_experience_is_enriched_not_duplicated(self):
        source = (
            "Professional Experience\n"
            "Business Sales Advisor Feb 2025 – Sep 2025\n"
            "Vodafone\n"
            "• Assisted customers with account questions.\n"
            "Technologies\n"
            "Languages: C++, Java, Python"
        )
        existing_llm_entry = {
            "role": "Business Sales Advisor",
            "organization": "",
            "dates": "Feb 2025 – Sep 2025",
            "responsibilities": [],
            "evidence": "Business Sales Advisor Feb 2025 – Sep 2025",
        }

        _, merged = _merge_source_grounded_fallbacks(source, [], [existing_llm_entry])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["organization"], "Vodafone")
        self.assertEqual(
            merged[0]["responsibilities"],
            ["Assisted customers with account questions."],
        )
        self.assertIn("Vodafone", merged[0]["evidence"])
        self.assertNotIn("Technologies", merged[0]["evidence"])
        self.assertTrue(_quote_is_in_source(merged[0]["evidence"], source))

    def test_responsibilities_become_separate_source_grounded_evidence(self):
        source = (
            "Business Sales Advisor Feb 2025 – Sep 2025\n"
            "Vodafone\n"
            "• Assisted customers with account questions."
        )
        cv_data = {
            "experience": [{
                "role": "Business Sales Advisor",
                "organization": "Vodafone",
                "dates": "Feb 2025 – Sep 2025",
                "evidence": "Business Sales Advisor Feb 2025 – Sep 2025\nVodafone",
                "responsibilities": ["Assisted customers with account questions."],
            }]
        }

        evidence = _build_cv_evidence(cv_data, source)
        responsibility_items = [
            item for item in evidence
            if item["type"] == "experience_responsibility"
        ]

        self.assertEqual(len(responsibility_items), 1)
        self.assertEqual(
            responsibility_items[0]["quote"],
            "Assisted customers with account questions.",
        )
        self.assertTrue(_quote_is_in_source(responsibility_items[0]["quote"], source))
        self.assertEqual(
            _evidence_support_level(
                "Confidence in speaking to users directly",
                responsibility_items[0],
            ),
            "direct",
        )

    def test_retrieval_keeps_relevant_responsibilities_even_with_low_similarity(self):
        import numpy as np
        from unittest.mock import patch

        requirements = [
            {"text": "Confidence in speaking to users directly", "source_text": "Confidence in speaking to users directly"},
            {"text": "Troubleshooting experience in a support environment", "source_text": "Troubleshooting experience in a support environment"},
            {"text": "Ability to work under pressure", "source_text": "Ability to work under pressure"},
        ]
        evidence_items = [
            {"id": "E1", "type": "experience", "label": "Business Sales Advisor", "text": "Business Sales Advisor", "quote": "Business Sales Advisor"},
            {"id": "E2", "type": "experience_responsibility", "label": "Business Sales Advisor | Responsibility", "text": "Assisted customers with account questions.", "quote": "Assisted customers with account questions."},
            {"id": "E3", "type": "experience_responsibility", "label": "Junior Java Developer | Responsibility", "text": "Debugged Java application defects and fixed software bugs.", "quote": "Debugged Java application defects and fixed software bugs."},
            {"id": "E4", "type": "experience_responsibility", "label": "Junior Java Developer | Responsibility", "text": "Managed multiple priorities under tight deadlines.", "quote": "Managed multiple priorities under tight deadlines."},
        ]

        class ZeroSimilarityModel:
            def encode(self, texts, **kwargs):
                if len(texts) == len(requirements):
                    return np.tile(np.array([[1.0, 0.0]]), (len(texts), 1))
                return np.tile(np.array([[0.0, 1.0]]), (len(texts), 1))

        with patch("services.nlp_service._get_embedding_model", return_value=ZeroSimilarityModel()):
            retrieved, _ = _retrieve_evidence(requirements, evidence_items)

        retrieved_ids = [{item["id"] for item in candidates} for candidates in retrieved]
        self.assertIn("E2", retrieved_ids[0])
        self.assertIn("E3", retrieved_ids[1])
        self.assertIn("E4", retrieved_ids[2])

    def test_coverage_metrics_distinguish_partial_from_fully_matched(self):
        metrics = _calculate_evidence_coverage([
            {"status": "partial", "score": 50, "priority": "required", "evidence": [{"quote": "English CV text"}]},
            {"status": "partial", "score": 50, "priority": "required", "evidence": [{"quote": "Business Sales Advisor"}]},
            {"status": "missing", "score": 0, "priority": "required", "evidence": []},
            {"status": "missing", "score": 0, "priority": "required", "evidence": []},
            {"status": "missing", "score": 0, "priority": "required", "evidence": []},
            {"status": "missing", "score": 0, "priority": "preferred", "evidence": []},
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

    def test_preferred_criterion_has_half_weight_and_weighted_score_is_transparent(self):
        details = [
            {"status": "partial", "score": 50, "priority": "required", "evidence": [{"quote": "English CV text"}]},
            {"status": "partial", "score": 50, "priority": "required", "evidence": [{"quote": "Business Sales Advisor"}]},
            {"status": "missing", "score": 0, "priority": "required", "evidence": []},
            {"status": "missing", "score": 0, "priority": "required", "evidence": []},
            {"status": "missing", "score": 0, "priority": "required", "evidence": []},
            {"status": "missing", "score": 0, "priority": "preferred", "evidence": []},
        ]
        self.assertEqual(_calculate_weighted_match_score(details), 18.18)

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

    def test_cv_prompt_explicitly_extracts_projects_separately(self):
        from services.nlp_service import _CV_SYSTEM_PROMPT

        self.assertIn("Projects are NOT employment entries", _CV_SYSTEM_PROMPT)
        self.assertIn("Do not omit this section", _CV_SYSTEM_PROMPT)

    def test_project_parser_recovers_all_project_descriptions_and_technologies(self):
        source = (
            "Education\n"
            "Faculty of Computer Science Oct 2021 – July 2025\n"
            "Alexandru Ioan Cuza University\n"
            "Projects\n"
            "Interview Management – Recruiter Support Web Application github 2\n"
            "◦ Built a web application for recruiters to schedule and manage interviews.\n"
            "◦ Tools Used: Java 21, Spring Boot, HTML, CSS, JavaScript\n"
            "iNeed - Mental Wellness & Habit Tracking App demo 2\n"
            "◦ Developed a mobile app for users to monitor their basic needs.\n"
            "◦ Tools Used: React Native, Node.js, Express, MongoDB, TypeScript\n"
            "Bubble Buster – Arcade Game github 2\n"
            "◦ Created a game that simulates classic Bubble Shooter mechanics.\n"
            "◦ Tools Used: Python, Pygame\n"
            "Electronic Circuit Viewer\n"
            "◦ Built a web application in C to explore electronic circuits.\n"
            "Escape Room Game\n"
            "◦ Worked in a small team to create a 3D escape room game.\n"
            "◦ Tools Used: Unreal Engine 5, Blueprints"
        )

        projects = _recover_projects_from_source(source)
        by_name = {item["name"]: item for item in projects}

        self.assertEqual(len(projects), 5)
        self.assertIn("MongoDB", by_name["iNeed - Mental Wellness & Habit Tracking App"]["technologies"])
        self.assertIn("Python", by_name["Bubble Buster – Arcade Game"]["technologies"])
        self.assertEqual(
            by_name["Escape Room Game"]["description"],
            ["Worked in a small team to create a 3D escape room game."],
        )
        self.assertTrue(all(_quote_is_in_source(item["evidence"], source) for item in projects))

    def test_project_merge_fills_llm_omissions_without_inventing_content(self):
        source = (
            "Projects\n"
            "Bubble Buster – Arcade Game\n"
            "◦ Created a game using Python.\n"
            "◦ Tools Used: Python, Pygame"
        )
        merged = _merge_source_grounded_projects(
            source,
            [{
                "name": "Bubble Buster – Arcade Game",
                "description": [],
                "technologies": [],
                "evidence": "Bubble Buster – Arcade Game",
            }],
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["technologies"], ["Python", "Pygame"])
        self.assertEqual(merged[0]["description"], ["Created a game using Python."])
        self.assertTrue(_quote_is_in_source(merged[0]["evidence"], source))

    def test_education_parser_does_not_invent_a_bachelor_degree(self):
        source = (
            "Education\n"
            "Masters in Software Development Oct 2025 - present\n"
            "Faculty of Economics and Business Administration\n"
            "Faculty of Computer Science Iasi Oct 2021 – July 2025\n"
            "Alexandru Ioan Cuza University\n"
            "Experience\n"
            "Junior Java Developer Sep 2025 – present\n"
            "Endava"
        )

        recovered = _recover_education_from_source(source)
        computer_science = next(
            item for item in recovered
            if "Computer Science" in item["qualification"]
        )

        self.assertEqual(computer_science["qualification"], "Faculty of Computer Science Iasi")
        self.assertEqual(computer_science["institution"], "Alexandru Ioan Cuza University")
        self.assertNotIn("Bachelor", computer_science["qualification"])
        self.assertTrue(_quote_is_in_source(computer_science["evidence"], source))

    def test_education_requirement_gets_partial_for_relevant_study_without_explicit_degree(self):
        source = (
            "Faculty of Computer Science Iasi Oct 2021 – July 2025\n"
            "Alexandru Ioan Cuza University"
        )
        evidence = _build_cv_evidence(
            {
                "education": [{
                    "qualification": "Faculty of Computer Science Iasi",
                    "institution": "Alexandru Ioan Cuza University",
                    "dates": "Oct 2021 – July 2025",
                    "evidence": source,
                }]
            },
            source,
        )
        assessment = _deterministic_requirement_assessment(
            {
                "text": "BS Degree in Computer Science or applicable programming area",
                "source_text": "BS Degree in Computer Science or applicable programming area of study",
                "category": "education",
            },
            evidence,
        )

        self.assertIsNotNone(assessment)
        self.assertEqual(assessment["status"], "partial")
        self.assertEqual(assessment["score"], 50)
        self.assertTrue(assessment["evidence_ids"])

    def test_project_technology_supports_partial_not_full_experience_match(self):
        evidence = [{
            "id": "E8",
            "type": "project_technology",
            "label": "iNeed | Project | Technologies used",
            "text": "iNeed. Tools Used: React Native, Node.js, Express, MongoDB",
            "quote": "◦ Tools Used: React Native, Node.js, Express, MongoDB",
            "skill_names": ["React Native", "Node.js", "Express", "MongoDB"],
        }]
        assessment = _deterministic_requirement_assessment(
            {
                "text": "Experience working with SQL and NoSQL databases like Oracle, MySQL, Redis, PostgreSQL, MongoDB",
                "source_text": "Experience working with SQL and NoSQL databases like Oracle, MySQL, Redis, PostgreSQL, MongoDB",
                "category": "experience",
            },
            evidence,
        )

        self.assertEqual(assessment["status"], "partial")
        self.assertEqual(assessment["score"], 50)
        self.assertEqual(assessment["evidence_ids"], ["E8"])
        self.assertIn("does not establish", assessment["explanation"])

    def test_project_development_and_teamwork_can_support_partial_evidence(self):
        evidence = [
            {
                "id": "E9",
                "type": "project_description",
                "label": "Interview Management | Project",
                "text": "Interview Management. Built a web application for recruiters.",
                "quote": "Built a web application for recruiters.",
            },
            {
                "id": "E10",
                "type": "project_description",
                "label": "Escape Room Game | Project",
                "text": "Escape Room Game. Worked in a small team to create a 3D escape room game.",
                "quote": "Worked in a small team to create a 3D escape room game.",
            },
        ]

        software_assessment = _deterministic_requirement_assessment(
            {
                "text": "Foundation of software development knowledge",
                "source_text": "Foundation of software development knowledge",
                "category": "experience",
            },
            evidence,
        )
        team_assessment = _deterministic_requirement_assessment(
            {
                "text": "Ability to work in an open and dynamic team environment",
                "source_text": "Ability to work in an open and dynamic team environment",
                "category": "experience",
            },
            evidence,
        )

        self.assertEqual(software_assessment["status"], "partial")
        self.assertEqual(software_assessment["evidence_ids"], ["E9"])
        self.assertEqual(team_assessment["status"], "partial")
        self.assertEqual(team_assessment["evidence_ids"], ["E10"])

    def test_retrieval_forces_exact_project_technology_overlap(self):
        import numpy as np

        requirements = [{
            "text": "Experience working with MongoDB databases",
            "source_text": "Experience working with MongoDB databases",
            "category": "experience",
        }]
        evidence_items = [{
            "id": "E11",
            "type": "project_technology",
            "label": "iNeed | Technologies used",
            "text": "iNeed. Tools Used: React Native, Node.js, Express, MongoDB",
            "quote": "Tools Used: React Native, Node.js, Express, MongoDB",
            "skill_names": ["React Native", "Node.js", "Express", "MongoDB"],
        }]

        class ZeroSimilarityModel:
            def encode(self, texts, **kwargs):
                if len(texts) == len(requirements):
                    return np.tile(np.array([[1.0, 0.0]]), (len(texts), 1))
                return np.tile(np.array([[0.0, 1.0]]), (len(texts), 1))

        with patch("services.nlp_service._get_embedding_model", return_value=ZeroSimilarityModel()):
            retrieved, _ = _retrieve_evidence(requirements, evidence_items)

        self.assertEqual([item["id"] for item in retrieved[0]], ["E11"])

    def test_raw_resume_skill_and_tools_lines_cannot_prove_professional_tenure(self):
        skill_excerpt = {
            "type": "resume_text",
            "quote": "Databases: SQL, MongoDB, Oracle Database, PostgreSQL",
        }
        project_tools_excerpt = {
            "type": "resume_text",
            "quote": "iNeed - Mobile App\n◦ Tools Used: React Native, Node.js, MongoDB",
        }
        professional_duty = {
            "type": "experience_responsibility",
            "quote": "Administered MongoDB clusters for a production service.",
        }

        self.assertTrue(_is_skill_or_project_only_evidence(skill_excerpt))
        self.assertTrue(_is_skill_or_project_only_evidence(project_tools_excerpt))
        self.assertFalse(_is_skill_or_project_only_evidence(professional_duty))

    def test_skill_list_can_support_partial_but_not_full_experience_match(self):
        evidence = [{
            "id": "E12",
            "type": "technical_skill",
            "label": "MongoDB",
            "text": "MongoDB. Databases: SQL, MongoDB, Oracle Database, PostgreSQL",
            "quote": "Databases: SQL, MongoDB, Oracle Database, PostgreSQL",
            "skill_names": ["SQL", "MongoDB", "Oracle Database", "PostgreSQL"],
        }]
        assessment = _deterministic_requirement_assessment(
            {
                "text": "Minimum two years' experience working with MongoDB",
                "source_text": "Minimum two years' experience working with MongoDB",
                "category": "experience",
            },
            evidence,
        )

        self.assertEqual(assessment["status"], "partial")
        self.assertEqual(assessment["score"], 50)
        self.assertEqual(assessment["evidence_ids"], ["E12"])
        self.assertIn("does not establish", assessment["explanation"])

    def test_unrelated_completed_degree_cannot_upgrade_computer_science_study_to_full_match(self):
        evidence = [
            {
                "id": "E13",
                "type": "education",
                "label": "Masters in Business Administration",
                "text": "Masters in Business Administration Oct 2020 – May 2022",
                "quote": "Masters in Business Administration Oct 2020 – May 2022",
            },
            {
                "id": "E14",
                "type": "education",
                "label": "Faculty of Computer Science",
                "text": "Faculty of Computer Science Oct 2021 – July 2025",
                "quote": "Faculty of Computer Science Oct 2021 – July 2025",
            },
        ]
        assessment = _deterministic_requirement_assessment(
            {
                "text": "BS Degree in Computer Science",
                "source_text": "BS Degree in Computer Science",
                "category": "education",
            },
            evidence,
        )

        self.assertEqual(assessment["status"], "partial")
        self.assertEqual(assessment["score"], 50)
        self.assertEqual(assessment["evidence_ids"], ["E14"])

    def test_education_merge_rejects_degree_name_not_present_in_source_quote(self):
        source = (
            "Education\n"
            "Faculty of Computer Science Iasi Oct 2021 – July 2025\n"
            "Alexandru Ioan Cuza University"
        )
        merged = _merge_source_grounded_education(
            source,
            [{
                "qualification": "Bachelor's in Mathematics and Computer Science",
                "institution": "Faculty of Computer Science Iasi",
                "dates": "Oct 2021 – July 2025",
                "evidence": (
                    "Faculty of Computer Science Iasi Oct 2021 – July 2025\n"
                    "Alexandru Ioan Cuza University"
                ),
            }],
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["qualification"], "Faculty of Computer Science Iasi")
        self.assertEqual(merged[0]["institution"], "Alexandru Ioan Cuza University")
        self.assertNotIn("Bachelor", merged[0]["qualification"])

    def test_project_evidence_builder_keeps_description_and_named_tools_separate(self):
        source = (
            "Projects\n"
            "iNeed - Mental Wellness App\n"
            "◦ Developed a mobile app to monitor basic needs.\n"
            "◦ Tools Used: React Native, Node.js, MongoDB"
        )
        projects = _recover_projects_from_source(source)
        evidence = _build_cv_evidence({"projects": projects}, source)

        descriptions = [item for item in evidence if item["type"] == "project_description"]
        technologies = [item for item in evidence if item["type"] == "project_technology"]

        self.assertEqual(len(descriptions), 1)
        self.assertEqual(descriptions[0]["quote"], "Developed a mobile app to monitor basic needs.")
        self.assertEqual(len(technologies), 1)
        self.assertEqual(technologies[0]["skill_names"], ["React Native", "Node.js", "MongoDB"])
        self.assertTrue(_quote_is_in_source(technologies[0]["quote"], source))

    def test_assessment_prompt_keeps_exact_evidence_text_without_duplicate_fields(self):
        requirement = {
            "text": "Confidence in speaking to users directly",
            "source_text": "Confidence in speaking to users directly;",
            "priority": "required",
            "category": "experience",
        }
        evidence = {
            "id": "E7",
            "type": "experience",
            "label": "Business Sales Advisor — Vodafone",
            "text": "Business Sales Advisor — Vodafone. Business Sales Advisor Feb 2025 – Sep 2025",
            "quote": "Business Sales Advisor Feb 2025 – Sep 2025",
            "semantic_similarity": 0.2703,
        }
        fake_result = MatchAssessmentOutput(
            assessments=[
                RequirementAssessment(
                    requirement_index=0,
                    status="partial",
                    score=50,
                    evidence_ids=["E7"],
                    explanation="People-facing experience is transferable but not direct IT support.",
                )
            ]
        )

        with patch("services.nlp_service._ollama_extract", return_value=fake_result) as ollama_call:
            result = _llm_assess_requirements([requirement], [[evidence]])

        prompt = ollama_call.call_args.args[2]
        payload = json.loads(prompt.split("DATA:\n", 1)[1])
        candidate = payload[0]["candidate_evidence"][0]

        self.assertEqual(result[0]["status"], "partial")
        self.assertEqual(candidate["id"], evidence["id"])
        self.assertEqual(candidate["type"], evidence["type"])
        self.assertEqual(candidate["label"], evidence["label"])
        self.assertEqual(candidate["text"], evidence["text"])
        self.assertIn(evidence["quote"], candidate["text"])
        self.assertNotIn("source_quote", candidate)
        self.assertEqual(candidate["retrieval_similarity"], evidence["semantic_similarity"])

    def test_identical_cv_uses_cached_extraction_without_mutating_cached_result(self):
        _extract_cv_cached.cache_clear()
        fake_extraction = CVExtraction(
            full_name="Roxana Sandu",
            summary="Software development experience.",
            skills=[
                CVSkill(
                    name="Python",
                    category="technical_skill",
                    evidence="Skills: Python",
                )
            ],
            experience=[],
            education=[],
            certifications=[],
            languages=[],
        )

        try:
            with patch("services.nlp_service._ollama_extract", return_value=fake_extraction) as ollama_call:
                first = extract_cv_information("Skills: Python")
                first["skills"].clear()
                second = extract_cv_information("Skills: Python")

            self.assertEqual(ollama_call.call_count, 1)
            self.assertEqual([item["name"] for item in second["skills"]], ["Python"])
        finally:
            _extract_cv_cached.cache_clear()


if __name__ == "__main__":
    unittest.main()
