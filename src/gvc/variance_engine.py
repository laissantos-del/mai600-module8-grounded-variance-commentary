"""
Deterministic variance layer: the "numbers" authority.

No language model ever touches this module. It computes every figure the system
reports and the materiality decision those figures imply, then hands them to the
generator as verified fact.

Ported from mai600_module6/build/variance_engine.py, which passes 9/9 self-checks.
Module 8 change: `verified_figures()` now also emits the computed materiality
thresholds, so the model never has cause to derive one itself. That closes the C3
rule violation observed in Module 7, where the model wrote "5% of budget
(USD 33,750)" because the figure was not supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Materiality policy, mirroring policy document D01 section 4.2.
DOLLAR_AND_PCT_USD = 250_000      # requires BOTH this ...
DOLLAR_AND_PCT_PCT = 5.0          # ... AND this
DOLLAR_ALONE_USD = 500_000        # OR this alone

# Non-dollar trigger, mirroring D01 section 4.3. Commentary is required, whatever the
# dollar amount, when a movement brings a lending-covenant metric within this margin
# of its limit. This matters because a small interest movement can be immaterial in
# dollars while still being the most reportable event of the period.
COVENANT_HEADROOM_TRIGGER = 0.25  # turns of EBITDA


@dataclass
class VarianceRow:
    case_id: str
    period: str
    account: str
    cost_centre: str
    region: str
    product_line: str
    actual: float
    budget: float

    # optional flexible-budget decomposition (revenue rows only)
    price_effect: Optional[float] = None
    volume_effect: Optional[float] = None
    mix_effect: Optional[float] = None

    # covenant headroom in turns of EBITDA, where supplied. Drives the D01 4.3 rule.
    covenant_headroom: Optional[float] = None
    covenant_limit: Optional[float] = None

    # ground truth, used for scoring only and NEVER placed in a prompt
    true_cause: str = ""
    gold_doc: Optional[str] = None
    stratum: str = "documented"

    # computed
    variance_usd: float = field(init=False)
    variance_pct: float = field(init=False)
    commentary_required: bool = field(init=False)
    materiality_trigger: str = field(init=False)

    def __post_init__(self) -> None:
        self.variance_usd = round(self.actual - self.budget, 2)
        self.variance_pct = (
            round(100.0 * self.variance_usd / self.budget, 1) if self.budget else 0.0
        )
        self._apply_materiality()

    def _apply_materiality(self) -> None:
        a_usd = abs(self.variance_usd)
        a_pct = abs(self.variance_pct)
        if (self.covenant_headroom is not None
                and self.covenant_headroom <= COVENANT_HEADROOM_TRIGGER):
            self.commentary_required = True
            self.materiality_trigger = "covenant"
        elif a_usd >= DOLLAR_ALONE_USD:
            self.commentary_required = True
            self.materiality_trigger = "dollar_alone"
        elif a_usd >= DOLLAR_AND_PCT_USD and a_pct >= DOLLAR_AND_PCT_PCT:
            self.commentary_required = True
            self.materiality_trigger = "dollar_and_pct"
        else:
            self.commentary_required = False
            self.materiality_trigger = "none"

    def verified_figures(self) -> dict:
        """The block handed to every arm. Ground-truth fields are excluded by design.

        `pct_threshold_usd` is the dollar value of the 5% test for this row. Supplying
        it is what stops the model computing it, which was the C3 failure in Module 7.
        """
        vf = {
            "period": self.period,
            "account": self.account,
            "cost_centre": self.cost_centre,
            "region": self.region,
            "product_line": self.product_line,
            "actual": self.actual,
            "budget": self.budget,
            "variance_usd": self.variance_usd,
            "variance_pct": self.variance_pct,
            "materiality_dollar_threshold": DOLLAR_AND_PCT_USD,
            "materiality_pct_threshold": DOLLAR_AND_PCT_PCT,
            "pct_threshold_usd": round(self.budget * DOLLAR_AND_PCT_PCT / 100.0, 2),
            "dollar_alone_threshold": DOLLAR_ALONE_USD,
        }
        for k in ("price_effect", "volume_effect", "mix_effect"):
            v = getattr(self, k)
            if v is not None:
                vf[k] = v
        if self.covenant_headroom is not None:
            vf["covenant_headroom_turns"] = self.covenant_headroom
            vf["covenant_limit_turns"] = self.covenant_limit
            vf["covenant_disclosure_trigger_turns"] = COVENANT_HEADROOM_TRIGGER
        return vf


def self_check(cases: list[VarianceRow]) -> tuple[bool, list[tuple[str, bool]]]:
    """Assertions that must hold or the evaluation is contaminated."""
    by = {c.case_id: c for c in cases}
    checks: list[tuple[str, bool]] = []

    for c in cases:
        if c.stratum == "documented":
            checks.append((f"{c.case_id} documented is material", c.commentary_required))
            checks.append((f"{c.case_id} documented has a gold doc", c.gold_doc is not None))
        elif c.stratum == "evidence_less":
            # must be MATERIAL, else "abstain" collapses into "no comment"
            checks.append(
                (f"{c.case_id} evidence-less is material and unsourced",
                 c.commentary_required and c.gold_doc is None)
            )
        elif c.stratum == "immaterial":
            checks.append((f"{c.case_id} immaterial requires no commentary",
                           not c.commentary_required))
            # noise must never cross the threshold, or over-generation is unmeasurable
            checks.append((f"{c.case_id} noise stays below threshold",
                           abs(c.variance_usd) < DOLLAR_AND_PCT_USD))

        elif c.stratum == "covenant":
            checks.append(
                (f"{c.case_id} covenant-triggered despite sub-threshold dollars",
                 c.commentary_required and c.materiality_trigger == "covenant")
            )

    if "C7" in by:
        checks.append(("C7 reflects the approved +6% not the proposed +8%",
                       abs(by["C7"].variance_pct - 6.0) < 0.05))

    return all(ok for _, ok in checks), checks
