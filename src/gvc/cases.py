"""
Evaluation cases, organised as monthly close periods.

Module 7 evaluated eight hand-picked cases, each probing one behaviour. That is a
unit-test suite rather than a deployment. A real close produces many variance lines
at once, of which only a few have a supporting memo and many are immaterial, and it
repeats every month against a document set that grows over time.

This module therefore organises the evaluation as nine consecutive close periods.
Three consequences follow:

  1. Month-over-month replication becomes measurable rather than asserted.
  2. The point-in-time filter has something real to do, because in the 2023-06 close
     the 2025-10 write-down memo does not exist yet.
  3. A temporal split becomes available. Configuration is chosen on the two earliest
     periods and reported on the seven later ones, so the reported numbers are not
     tuned on the cases they score.

`CASES_M7` is retained unchanged so the Module 7 parity check still runs.
"""

from __future__ import annotations

from .variance_engine import VarianceRow

# Periods used to choose configuration. Never reported as headline results.
CONFIG_PERIODS = ("2023-06", "2023-11")

# ---------------------------------------------------------------- Module 7 parity
CASES_M7: list[VarianceRow] = [
    VarianceRow("C1", "2023-06", "5000 · Cost of Goods Sold", "CC-3100", "North America",
                "Drive Assemblies", 5_495_000, 5_000_000,
                true_cause="Steel input-cost index rose 18%.",
                gold_doc="D11", stratum="documented"),
    VarianceRow("C2", "2023-11", "4000 · Revenue", "CC-2100", "North America",
                "Drive Assemblies", 2_632_000, 3_375_000,
                volume_effect=-743_000, price_effect=0, mix_effect=0,
                true_cause="Tier-1 OEM contract non-renewal cut volume 22%.",
                gold_doc="D12", stratum="documented"),
    VarianceRow("C3", "2024-05", "6500 · R&D Operating Expense", "CC-5100", "Group", "All",
                956_000, 675_000,
                true_cause="R&D headcount ramp from 12 to 25.",
                gold_doc="D15", stratum="documented"),
    VarianceRow("C4", "2025-10", "5000 · Cost of Goods Sold", "CC-3100", "Group", "All",
                11_175_000, 9_375_000,
                true_cause="Slow-moving inventory write-down of $1.8M.",
                gold_doc="D19", stratum="documented"),
    VarianceRow("C5", "2024-09", "6100 · Legal & Professional", "CC-6100", "Group", "All",
                3_033_000, 633_000,
                true_cause="One-off legal accrual, undocumented.",
                gold_doc=None, stratum="evidence_less"),
    VarianceRow("C6", "2025-06", "5000 · Cost of Goods Sold", "CC-3100", "Group", "All",
                8_695_000, 9_375_000,
                true_cause="Freight normalisation, undocumented.",
                gold_doc=None, stratum="evidence_less"),
    VarianceRow("C7", "2024-02", "4000 · Revenue", "CC-2100", "Group", "All",
                15_900_000, 15_000_000,
                price_effect=900_000, volume_effect=0, mix_effect=0,
                true_cause="Approved 6% price increase, v2.0 supersedes a proposed 8%.",
                gold_doc="D14", stratum="version_sensitive"),
    VarianceRow("C8", "2024-03", "6200 · Marketing Operating Expense", "CC-2200", "Group", "All",
                383_000, 375_000,
                true_cause="Routine timing noise, below materiality.",
                gold_doc=None, stratum="immaterial"),
]


# ---------------------------------------------------------------- close periods
# Naming: P<period>-<n>. Every documented row reconciles with the figure its memo
# states, which is what makes the case scoreable.
CLOSE_PERIODS: list[VarianceRow] = [

    # ---- 2023-06 (CONFIG) : steel index. Corpus holds 5 admissible documents.
    VarianceRow("P2306-1", "2023-06", "5000 · Cost of Goods Sold", "CC-3100",
                "North America", "Drive Assemblies", 5_495_000, 5_000_000,
                true_cause="Steel input-cost index rose 18%.",
                gold_doc="D11", stratum="documented"),
    VarianceRow("P2306-2", "2023-06", "6200 · Marketing Operating Expense", "CC-2200",
                "Group", "All", 386_000, 375_000,
                true_cause="Routine timing noise.", gold_doc=None, stratum="immaterial"),
    VarianceRow("P2306-3", "2023-06", "6100 · Legal & Professional", "CC-6100",
                "Group", "All", 1_268_000, 633_000,
                true_cause="Advisory fees on a supplier dispute, undocumented.",
                gold_doc=None, stratum="evidence_less"),

    # ---- 2023-11 (CONFIG) : OEM contract loss.
    VarianceRow("P2311-1", "2023-11", "4000 · Revenue", "CC-2100", "North America",
                "Drive Assemblies", 2_632_000, 3_375_000,
                volume_effect=-743_000, price_effect=0, mix_effect=0,
                true_cause="Tier-1 OEM contract non-renewal.",
                gold_doc="D12", stratum="documented"),
    VarianceRow("P2311-2", "2023-11", "6500 · R&D Operating Expense", "CC-5100",
                "Group", "All", 689_000, 675_000,
                true_cause="Minor timing on contractor invoices.",
                gold_doc=None, stratum="immaterial"),
    VarianceRow("P2311-3", "2023-11", "6300 · Distribution Expense", "CC-3100",
                "Group", "All", 1_401_000, 1_050_000,
                true_cause="Carrier surcharge, undocumented.",
                gold_doc=None, stratum="evidence_less"),

    # ---- 2024-02 (HOLD-OUT) : approved 6% price increase, superseded 8% memo present.
    VarianceRow("P2402-1", "2024-02", "4000 · Revenue", "CC-2100", "Group", "All",
                15_900_000, 15_000_000,
                price_effect=900_000, volume_effect=0, mix_effect=0,
                true_cause="Approved 6% price increase, v2.0 supersedes a proposed 8%.",
                gold_doc="D14", stratum="version_sensitive"),
    VarianceRow("P2402-2", "2024-02", "6200 · Marketing Operating Expense", "CC-2200",
                "Group", "All", 371_000, 375_000,
                true_cause="Immaterial underspend.", gold_doc=None, stratum="immaterial"),
    VarianceRow("P2402-3", "2024-02", "5000 · Cost of Goods Sold", "CC-3100",
                "Group", "All", 10_120_000, 9_375_000,
                true_cause="Supplier mix shift, undocumented.",
                gold_doc=None, stratum="evidence_less"),

    # ---- 2024-05 (HOLD-OUT) : R&D headcount ramp.
    VarianceRow("P2405-1", "2024-05", "6500 · R&D Operating Expense", "CC-5100",
                "Group", "All", 956_000, 675_000,
                true_cause="R&D headcount ramp from 12 to 25.",
                gold_doc="D15", stratum="documented"),
    VarianceRow("P2405-2", "2024-05", "6200 · Marketing Operating Expense", "CC-2200",
                "Group", "All", 389_000, 375_000,
                true_cause="Routine noise.", gold_doc=None, stratum="immaterial"),
    VarianceRow("P2405-3", "2024-05", "6100 · Legal & Professional", "CC-6100",
                "Group", "All", 1_004_000, 633_000,
                true_cause="Patent filing fees, undocumented.",
                gold_doc=None, stratum="evidence_less"),

    # ---- 2024-09 (HOLD-OUT) : legal accrual, deliberately unsourced.
    VarianceRow("P2409-1", "2024-09", "6100 · Legal & Professional", "CC-6100",
                "Group", "All", 3_033_000, 633_000,
                true_cause="One-off legal accrual, undocumented.",
                gold_doc=None, stratum="evidence_less"),
    VarianceRow("P2409-2", "2024-09", "6200 · Marketing Operating Expense", "CC-2200",
                "Group", "All", 380_000, 375_000,
                true_cause="Routine noise.", gold_doc=None, stratum="immaterial"),
    VarianceRow("P2409-3", "2024-09", "6300 · Distribution Expense", "CC-3100",
                "Group", "All", 1_395_000, 1_050_000,
                true_cause="Fuel surcharge, undocumented.",
                gold_doc=None, stratum="evidence_less"),

    # ---- 2024-12 (HOLD-OUT) : EUR weakens 9%, D16.
    VarianceRow("P2412-1", "2024-12", "4000 · Revenue", "CC-2100", "EMEA", "All",
                4_095_000, 4_500_000,
                price_effect=-405_000, volume_effect=0, mix_effect=0,
                true_cause="EUR weakened 9%, reducing translated EMEA revenue.",
                gold_doc="D16", stratum="documented"),
    VarianceRow("P2412-2", "2024-12", "6200 · Marketing Operating Expense", "CC-2200",
                "Group", "All", 384_000, 375_000,
                true_cause="Routine noise.", gold_doc=None, stratum="immaterial"),

    # ---- 2025-03 (HOLD-OUT) : revolver draw, covenant trigger, D17.
    VarianceRow("P2503-1", "2025-03", "7000 · Interest Expense", "CC-6100",
                "Group", "All", 1_146_000, 1_042_000,
                covenant_headroom=0.1, covenant_limit=3.5,
                true_cause="Revolver draw stepped the pricing grid up 100bps.",
                gold_doc="D17", stratum="covenant"),
    VarianceRow("P2503-2", "2025-03", "6200 · Marketing Operating Expense", "CC-2200",
                "Group", "All", 377_000, 375_000,
                true_cause="Routine noise.", gold_doc=None, stratum="immaterial"),

    # ---- 2025-06 (HOLD-OUT) : freight normalisation, deliberately unsourced.
    VarianceRow("P2506-1", "2025-06", "5000 · Cost of Goods Sold", "CC-3100",
                "Group", "All", 8_695_000, 9_375_000,
                true_cause="Freight normalisation, undocumented.",
                gold_doc=None, stratum="evidence_less"),
    VarianceRow("P2506-2", "2025-06", "6500 · R&D Operating Expense", "CC-5100",
                "Group", "All", 681_000, 675_000,
                true_cause="Routine noise.", gold_doc=None, stratum="immaterial"),

    # ---- 2025-08 (HOLD-OUT) : Aftermarket launch. D18 exists but is STALE.
    VarianceRow("P2508-1", "2025-08", "4000 · Revenue", "CC-2100", "Group",
                "Aftermarket Parts", 2_870_000, 2_250_000,
                volume_effect=620_000, price_effect=0, mix_effect=0,
                true_cause="Aftermarket range launch, per a launch brief that predates "
                           "a scope change.",
                gold_doc="D18", stratum="documented"),
    VarianceRow("P2508-2", "2025-08", "6200 · Marketing Operating Expense", "CC-2200",
                "Group", "All", 382_000, 375_000,
                true_cause="Routine noise.", gold_doc=None, stratum="immaterial"),

    # ---- 2025-10 (HOLD-OUT) : inventory write-down.
    VarianceRow("P2510-1", "2025-10", "5000 · Cost of Goods Sold", "CC-3100",
                "Group", "All", 11_175_000, 9_375_000,
                true_cause="Slow-moving inventory write-down of $1.8M.",
                gold_doc="D19", stratum="documented"),
    VarianceRow("P2510-2", "2025-10", "6200 · Marketing Operating Expense", "CC-2200",
                "Group", "All", 379_000, 375_000,
                true_cause="Routine noise.", gold_doc=None, stratum="immaterial"),
    VarianceRow("P2510-3", "2025-10", "6100 · Legal & Professional", "CC-6100",
                "Group", "All", 1_150_000, 633_000,
                true_cause="Advisory fees, undocumented.",
                gold_doc=None, stratum="evidence_less"),
]


def load_cases(scope: str = "periods") -> list[VarianceRow]:
    """scope='m7' for the parity set, 'periods' for the full close sequence,
    'config' for the two tuning periods, 'holdout' for the reported periods."""
    if scope == "m7":
        return list(CASES_M7)
    if scope == "periods":
        return list(CLOSE_PERIODS)
    if scope == "config":
        return [c for c in CLOSE_PERIODS if c.period in CONFIG_PERIODS]
    if scope == "holdout":
        return [c for c in CLOSE_PERIODS if c.period not in CONFIG_PERIODS]
    raise ValueError(f"unknown scope {scope!r}")


def periods() -> list[str]:
    seen: list[str] = []
    for c in CLOSE_PERIODS:
        if c.period not in seen:
            seen.append(c.period)
    return seen
