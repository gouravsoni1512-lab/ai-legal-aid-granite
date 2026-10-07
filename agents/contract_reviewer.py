"""
agents/contract_reviewer.py
----------------------------
Agent 1 – Contract Reviewer
Parses legal agreement text and flags risky or one-sided clauses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from utils.granite import get_client

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

RISK_LEVELS = ("HIGH", "MEDIUM", "LOW", "INFO")

RISK_COLORS = {
    "HIGH": "#d9534f",
    "MEDIUM": "#f0ad4e",
    "LOW": "#5cb85c",
    "INFO": "#5bc0de",
}


@dataclass
class RiskyClause:
    clause_title: str
    excerpt: str
    risk_level: str  # HIGH | MEDIUM | LOW | INFO
    explanation: str

    @property
    def color(self) -> str:
        return RISK_COLORS.get(self.risk_level, "#aaa")


@dataclass
class ContractReviewResult:
    risky_clauses: List[RiskyClause] = field(default_factory=list)
    overall_risk: str = "LOW"
    summary: str = ""
    raw_llm_output: str = ""

    def risk_counts(self) -> dict:
        counts: dict = {r: 0 for r in RISK_LEVELS}
        for c in self.risky_clauses:
            if c.risk_level in counts:
                counts[c.risk_level] += 1
        return counts


# ---------------------------------------------------------------------------
# Heuristic pre-scan (works without LLM, enriches context)
# ---------------------------------------------------------------------------

_HEURISTIC_PATTERNS = [
    (
        "Unilateral amendment",
        r"(sole\s+discretion|unilateral(ly)?|without\s+prior\s+notice).{0,80}(modif|amend|chang)",
        "HIGH",
        "One party can change terms without notifying or getting consent from the other party.",
    ),
    (
        "Broad non-compete",
        r"non.?compet.{0,120}(year|month|mile|km|radius)",
        "HIGH",
        "Non-compete clauses that are overly broad in geography or duration may be unenforceable "
        "and can significantly restrict the receiving party's future employment or business activities.",
    ),
    (
        "Jury trial waiver",
        r"waive.{0,40}(jury|trial)",
        "HIGH",
        "Waiving the right to a jury trial removes an important legal protection for the signing party.",
    ),
    (
        "Liability cap / indemnification",
        r"(liability|damages).{0,80}(shall not exceed|limited to|cap)",
        "MEDIUM",
        "Capped liability clauses limit the compensation you can claim even if significant harm occurs.",
    ),
    (
        "Binding arbitration",
        r"binding\s+arbitration|mandatory\s+arbitration",
        "MEDIUM",
        "Mandatory arbitration may prevent access to courts and limit appeal rights.",
    ),
    (
        "No-poaching / non-solicitation",
        r"non.?solicit|no.?poach|shall not.{0,40}(hire|employ|recruit)",
        "MEDIUM",
        "Broad non-solicitation clauses can restrict normal business recruiting activities.",
    ),
    (
        "Excessive confidentiality term",
        r"(five|6|seven|ten|\d{2})\s+(year|yr).{0,60}(confidential|nda|secret)",
        "MEDIUM",
        "Confidentiality obligations exceeding 2–3 years are often considered excessive.",
    ),
    (
        "Immediate injunctive relief",
        r"without.{0,30}(bond|actual\s+damages|proof).{0,60}(injunction|equitable)",
        "MEDIUM",
        "Allowing injunctive relief without requiring proof of actual harm gives one party broad power "
        "to obtain court orders quickly.",
    ),
    (
        "Governing law / jurisdiction",
        r"governed by.{0,60}(law|laws).{0,40}(state|province|country)",
        "INFO",
        "Note which jurisdiction's law governs disputes — this affects your legal rights and remedies.",
    ),
    (
        "Entire agreement / merger clause",
        r"entire\s+agreement|supersedes.{0,60}(prior|previous|all)",
        "INFO",
        "Merger clauses mean any prior oral or written promises not in this document are not binding.",
    ),
]


def _heuristic_scan(text: str) -> List[RiskyClause]:
    """Fast regex-based scan that flags known risky patterns."""
    text_lower = text.lower()
    findings: List[RiskyClause] = []
    seen: set = set()

    for title, pattern, risk, explanation in _HEURISTIC_PATTERNS:
        match = re.search(pattern, text_lower)
        if match and title not in seen:
            # Extract a short excerpt from the original (non-lowercased) text
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 80)
            excerpt = "…" + text[start:end].strip() + "…"
            findings.append(
                RiskyClause(
                    clause_title=title,
                    excerpt=excerpt,
                    risk_level=risk,
                    explanation=explanation,
                )
            )
            seen.add(title)

    return findings


# ---------------------------------------------------------------------------
# LLM-based deep review
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """
You are an expert contract attorney specialising in identifying risky, unfair,
or one-sided clauses in legal agreements.

When given contract text, you will:
1. Identify every clause that could disadvantage the signing (Receiving) party.
2. For each clause provide:
   - CLAUSE_TITLE: short name (e.g. "Unlimited Liability", "Perpetual IP Assignment")
   - RISK_LEVEL: one of HIGH / MEDIUM / LOW / INFO
   - EXCERPT: a verbatim short quote from the contract (max 30 words)
   - EXPLANATION: plain-English explanation of why this is risky (2-3 sentences)
3. End with OVERALL_RISK: HIGH / MEDIUM / LOW

Use the following format strictly for each clause:
---CLAUSE---
CLAUSE_TITLE: <title>
RISK_LEVEL: <HIGH|MEDIUM|LOW|INFO>
EXCERPT: <verbatim excerpt>
EXPLANATION: <plain-English explanation>
---END_CLAUSE---

After all clauses:
OVERALL_RISK: <HIGH|MEDIUM|LOW>
"""


def _parse_llm_output(raw: str) -> tuple[List[RiskyClause], str]:
    """Parse structured LLM output into RiskyClause objects."""
    clauses: List[RiskyClause] = []

    clause_blocks = re.findall(r"---CLAUSE---(.*?)---END_CLAUSE---", raw, re.DOTALL)
    for block in clause_blocks:
        def _field(name: str) -> str:
            m = re.search(rf"{name}:\s*(.+)", block)
            return m.group(1).strip() if m else ""

        title = _field("CLAUSE_TITLE")
        risk = _field("RISK_LEVEL").upper()
        excerpt = _field("EXCERPT")
        explanation = _field("EXPLANATION")

        if title and risk in RISK_LEVELS:
            clauses.append(
                RiskyClause(
                    clause_title=title,
                    excerpt=excerpt,
                    risk_level=risk,
                    explanation=explanation,
                )
            )

    overall_match = re.search(r"OVERALL_RISK:\s*(HIGH|MEDIUM|LOW)", raw, re.IGNORECASE)
    overall = overall_match.group(1).upper() if overall_match else "MEDIUM"
    return clauses, overall


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def review_contract(text: str) -> ContractReviewResult:
    """
    Run the full contract review pipeline on *text*.

    Steps:
    1. Heuristic pre-scan (fast, always runs).
    2. LLM deep review (Granite or mock fallback).
    3. Merge and deduplicate findings.
    """
    client = get_client()

    # --- Step 1: heuristic scan ---
    heuristic_clauses = _heuristic_scan(text)

    # --- Step 2: LLM review ---
    user_prompt = (
        "Please review the following legal agreement and identify all risky clauses "
        "according to the instructions.\n\n"
        "=== CONTRACT TEXT ===\n"
        f"{text[:6000]}\n"  # Limit to avoid token overflow
        "=== END CONTRACT TEXT ==="
    )

    raw_output = client.generate_with_system(_SYSTEM_PROMPT, user_prompt)

    llm_clauses, overall_risk = _parse_llm_output(raw_output)

    # --- Step 3: Merge (LLM takes priority; heuristics fill gaps) ---
    llm_titles = {c.clause_title.lower() for c in llm_clauses}
    merged: List[RiskyClause] = list(llm_clauses)
    for hc in heuristic_clauses:
        if hc.clause_title.lower() not in llm_titles:
            merged.append(hc)

    # Recalculate overall risk if LLM didn't provide one
    if not overall_risk and merged:
        if any(c.risk_level == "HIGH" for c in merged):
            overall_risk = "HIGH"
        elif any(c.risk_level == "MEDIUM" for c in merged):
            overall_risk = "MEDIUM"
        else:
            overall_risk = "LOW"

    # Sort by risk level
    order = {r: i for i, r in enumerate(RISK_LEVELS)}
    merged.sort(key=lambda c: order.get(c.risk_level, 99))

    return ContractReviewResult(
        risky_clauses=merged,
        overall_risk=overall_risk or "LOW",
        summary=f"Found {len(merged)} notable clauses. Overall risk: {overall_risk}.",
        raw_llm_output=raw_output,
    )
