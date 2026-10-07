"""
agents/compliance_checker.py
-----------------------------
Agent 2 – Compliance Checker
Cross-checks contract clauses against a statutory rule-set and flags
potential violations of common legal standards.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from utils.granite import get_client

# ---------------------------------------------------------------------------
# Statutory rule knowledge-base (static, no vector DB needed here)
# ---------------------------------------------------------------------------

@dataclass
class StatutoryRule:
    rule_id: str
    jurisdiction: str
    statute: str
    description: str
    trigger_keywords: List[str]
    guidance: str


STATUTORY_RULES: List[StatutoryRule] = [
    StatutoryRule(
        rule_id="NLRA-001",
        jurisdiction="USA",
        statute="National Labor Relations Act (NLRA) §7",
        description="Employees have the right to discuss wages, hours, and working conditions.",
        trigger_keywords=["non-disclosure", "confidential", "salary", "wages", "compensation"],
        guidance=(
            "Overly broad confidentiality clauses that prohibit discussion of wages or "
            "working conditions may violate NLRA §7 protections for employees."
        ),
    ),
    StatutoryRule(
        rule_id="FTC-NC-001",
        jurisdiction="USA",
        statute="FTC Non-Compete Clause Rule (2024) / State Laws",
        description="Non-compete agreements face significant restrictions at federal and state levels.",
        trigger_keywords=["non-compete", "non compete", "noncompete", "competing business"],
        guidance=(
            "The FTC has proposed broad restrictions on non-compete clauses. Several states "
            "(California, North Dakota, Oklahoma, Minnesota) ban them outright. "
            "Enforceability depends heavily on jurisdiction, scope, and duration."
        ),
    ),
    StatutoryRule(
        rule_id="UCC-2-316",
        jurisdiction="USA",
        statute="UCC §2-316 / Magnuson-Moss Warranty Act",
        description="Warranty disclaimers must be conspicuous and may be restricted in consumer contracts.",
        trigger_keywords=["warranty", "as is", "limitation of liability", "disclaims"],
        guidance=(
            "Warranty exclusions in consumer contracts may be void under Magnuson-Moss. "
            "All limitations must be conspicuously displayed."
        ),
    ),
    StatutoryRule(
        rule_id="GDPR-001",
        jurisdiction="EU/UK",
        statute="GDPR Article 5 / UK GDPR",
        description="Personal data must be processed lawfully, fairly, and transparently.",
        trigger_keywords=["personal data", "data processing", "privacy", "gdpr"],
        guidance=(
            "Any clause referencing personal data processing must include lawful basis, "
            "purpose limitation, and data subject rights under GDPR/UK GDPR."
        ),
    ),
    StatutoryRule(
        rule_id="CCPA-001",
        jurisdiction="USA (California)",
        statute="California Consumer Privacy Act (CCPA/CPRA)",
        description="California residents have rights over their personal information.",
        trigger_keywords=["california", "personal information", "consumer data", "opt-out"],
        guidance=(
            "Contracts involving California residents must comply with CCPA/CPRA. "
            "Blanket waiver of privacy rights may be unenforceable."
        ),
    ),
    StatutoryRule(
        rule_id="ARB-FAA-001",
        jurisdiction="USA",
        statute="Federal Arbitration Act (FAA) / State Consumer Protection Laws",
        description="Arbitration clauses have specific enforceability requirements.",
        trigger_keywords=["arbitration", "binding arbitration", "dispute resolution"],
        guidance=(
            "While the FAA generally favours arbitration, class action waivers in consumer "
            "contracts face scrutiny. Some states restrict mandatory arbitration in "
            "employment and consumer contracts."
        ),
    ),
    StatutoryRule(
        rule_id="UCTA-001",
        jurisdiction="UK/EU",
        statute="Unfair Contract Terms Act 1977 / Consumer Rights Act 2015",
        description="Unfair limitation of liability clauses may be void against consumers.",
        trigger_keywords=["limitation of liability", "shall not be liable", "no event shall"],
        guidance=(
            "Under UCTA 1977 and CRA 2015, clauses that exclude liability for negligence "
            "causing personal injury/death are void. Unreasonable limitation clauses are "
            "unenforceable against consumers."
        ),
    ),
    StatutoryRule(
        rule_id="TRADE-SECRETS-001",
        jurisdiction="USA",
        statute="Defend Trade Secrets Act (DTSA) / Uniform Trade Secrets Act (UTSA)",
        description="Trade secret protections and NDA enforceability standards.",
        trigger_keywords=["trade secret", "confidential information", "proprietary"],
        guidance=(
            "NDAs must be reasonable in scope and duration to be enforceable under DTSA/UTSA. "
            "Overly broad definitions of 'confidential information' may render the NDA void."
        ),
    ),
    StatutoryRule(
        rule_id="IP-ASSIGN-001",
        jurisdiction="USA/UK",
        statute="Copyright Act §101 / Patents Act 1977",
        description="Automatic IP assignment clauses may be invalid for independent contractors.",
        trigger_keywords=["intellectual property", "ip assignment", "work for hire", "assign"],
        guidance=(
            "Blanket IP assignment clauses for independent contractors may not be enforceable "
            "under 'work for hire' doctrine. Specific written assignments are required for "
            "copyrightable works."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ComplianceIssue:
    rule: StatutoryRule
    triggered_by: str  # the keyword or clause that triggered this
    severity: str  # VIOLATION | WARNING | NOTE
    llm_analysis: str = ""


@dataclass
class ComplianceResult:
    issues: List[ComplianceIssue] = field(default_factory=list)
    compliant_rules: List[StatutoryRule] = field(default_factory=list)
    overall_status: str = "COMPLIANT"  # COMPLIANT | WARNINGS | NON_COMPLIANT
    raw_llm_output: str = ""

    def issue_counts(self) -> dict:
        counts = {"VIOLATION": 0, "WARNING": 0, "NOTE": 0}
        for issue in self.issues:
            if issue.severity in counts:
                counts[issue.severity] += 1
        return counts


# ---------------------------------------------------------------------------
# Heuristic compliance scan
# ---------------------------------------------------------------------------

def _heuristic_compliance_scan(text: str) -> List[ComplianceIssue]:
    """Match statutory rules against contract text using keyword scanning."""
    text_lower = text.lower()
    issues: List[ComplianceIssue] = []

    for rule in STATUTORY_RULES:
        matched_keywords = [kw for kw in rule.trigger_keywords if kw.lower() in text_lower]
        if matched_keywords:
            # Determine rough severity from context
            severity = "WARNING"
            # Escalate if explicit problematic patterns exist
            if rule.rule_id in ("FTC-NC-001", "NLRA-001") and any(
                kw in text_lower for kw in ["non-compete", "non compete", "noncompete"]
            ):
                severity = "VIOLATION"
            elif rule.rule_id == "UCTA-001" and re.search(
                r"shall not exceed \$\d+", text_lower
            ):
                severity = "VIOLATION"

            issues.append(
                ComplianceIssue(
                    rule=rule,
                    triggered_by=", ".join(matched_keywords[:3]),
                    severity=severity,
                )
            )

    return issues


# ---------------------------------------------------------------------------
# LLM compliance analysis
# ---------------------------------------------------------------------------

_COMPLIANCE_SYSTEM_PROMPT = """
You are a legal compliance specialist. You review contracts against statutory requirements.

For each compliance concern, output:
---ISSUE---
RULE_ID: <from the provided list>
SEVERITY: <VIOLATION|WARNING|NOTE>
TRIGGERED_BY: <specific clause or text excerpt>
ANALYSIS: <2-3 sentences explaining the compliance concern>
---END_ISSUE---

After all issues:
OVERALL_STATUS: <NON_COMPLIANT|WARNINGS|COMPLIANT>

Be specific, cite the relevant statute, and focus on practical legal risk.
"""


def _parse_compliance_llm_output(raw: str, issues: List[ComplianceIssue]) -> tuple[List[ComplianceIssue], str]:
    """Enrich existing issues with LLM analysis."""
    rule_map = {r.rule_id: r for r in STATUTORY_RULES}
    llm_issues: List[ComplianceIssue] = []

    blocks = re.findall(r"---ISSUE---(.*?)---END_ISSUE---", raw, re.DOTALL)
    for block in blocks:
        def _f(name: str) -> str:
            m = re.search(rf"{name}:\s*(.+)", block)
            return m.group(1).strip() if m else ""

        rule_id = _f("RULE_ID")
        severity = _f("SEVERITY").upper()
        triggered_by = _f("TRIGGERED_BY")
        analysis = _f("ANALYSIS")

        if rule_id in rule_map and severity in ("VIOLATION", "WARNING", "NOTE"):
            llm_issues.append(
                ComplianceIssue(
                    rule=rule_map[rule_id],
                    triggered_by=triggered_by,
                    severity=severity,
                    llm_analysis=analysis,
                )
            )

    # Enrich heuristic issues with LLM analysis where available
    llm_rule_ids = {i.rule.rule_id for i in llm_issues}
    for issue in issues:
        if issue.rule.rule_id in llm_rule_ids:
            match = next(i for i in llm_issues if i.rule.rule_id == issue.rule.rule_id)
            issue.llm_analysis = match.llm_analysis
            issue.severity = match.severity  # trust LLM severity over heuristic

    # Add any LLM issues not found by heuristics
    heuristic_rule_ids = {i.rule.rule_id for i in issues}
    for li in llm_issues:
        if li.rule.rule_id not in heuristic_rule_ids:
            issues.append(li)

    overall_match = re.search(
        r"OVERALL_STATUS:\s*(NON_COMPLIANT|WARNINGS|COMPLIANT)", raw, re.IGNORECASE
    )
    overall = overall_match.group(1).upper() if overall_match else ""
    return issues, overall


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_compliance(text: str) -> ComplianceResult:
    """
    Run the full compliance check pipeline on *text*.

    Steps:
    1. Heuristic keyword scan against statutory rules.
    2. LLM deep analysis for context-aware compliance interpretation.
    3. Determine overall compliance status.
    """
    client = get_client()

    # Step 1: heuristic scan
    issues = _heuristic_compliance_scan(text)
    triggered_rule_ids = {i.rule.rule_id for i in issues}
    compliant_rules = [r for r in STATUTORY_RULES if r.rule_id not in triggered_rule_ids]

    # Step 2: LLM analysis
    rule_list = "\n".join(
        f"- {r.rule_id}: {r.statute} ({r.jurisdiction})"
        for r in STATUTORY_RULES
    )
    user_prompt = (
        f"Review the contract below against these statutory rules:\n{rule_list}\n\n"
        "=== CONTRACT TEXT ===\n"
        f"{text[:5000]}\n"
        "=== END CONTRACT TEXT ==="
    )

    raw_output = client.generate_with_system(_COMPLIANCE_SYSTEM_PROMPT, user_prompt)
    issues, llm_overall = _parse_compliance_llm_output(raw_output, issues)

    # Step 3: overall status
    if llm_overall:
        overall_status = llm_overall
    elif any(i.severity == "VIOLATION" for i in issues):
        overall_status = "NON_COMPLIANT"
    elif any(i.severity == "WARNING" for i in issues):
        overall_status = "WARNINGS"
    else:
        overall_status = "COMPLIANT"

    # Sort: VIOLATION first
    sev_order = {"VIOLATION": 0, "WARNING": 1, "NOTE": 2}
    issues.sort(key=lambda i: sev_order.get(i.severity, 9))

    return ComplianceResult(
        issues=issues,
        compliant_rules=compliant_rules,
        overall_status=overall_status,
        raw_llm_output=raw_output,
    )
