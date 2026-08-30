"""
Add the three driver memos specified in the Module 6 corpus design but never built:
D16 (FX), D17 (revolver and covenant), D18 (Aftermarket launch, deliberately stale).

Each is written to match the planted event in data_design.md, so the figure the memo
states reconciles with the figure the variance engine computes. That reconciliation
is what makes the case scoreable.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus_documents.json"

NEW = [
    {
        "doc_id": "D16",
        "doc_title": "Treasury Memo — EUR Exposure Update",
        "doc_type": "memo",
        "category": "causes",
        "version": "v1.0",
        "published_date": "2024-12-09",
        "effective_from": "2024-12-01",
        "effective_to": None,
        "supersedes": None,
        "superseded_by": None,
        "owner": "Treasury",
        "entity_scope": ["EMEA"],
        "topic": ["fx", "revenue", "currency"],
        "sensitivity": "internal",
        "sections": [
            {
                "section": "§1 Exposure Summary",
                "text": (
                    "To: Group Financial Controller. From: Treasury. Date: 9 December 2024. "
                    "Subject: EUR exposure and the December reporting impact. EMEA revenue is "
                    "invoiced in euro and translated to US dollars at the average monthly rate. "
                    "The euro weakened approximately 9% against the dollar during the month."
                ),
            },
            {
                "section": "§2 Rate Movement and Revenue Impact",
                "text": (
                    "The 9% adverse rate movement reduces reported EMEA revenue by approximately "
                    "USD 405,000 in December versus the frozen budget, which assumed the prior-year "
                    "average rate. The movement is translation only and does not reflect a change in "
                    "underlying EMEA volume or pricing. No hedge was in force for the period."
                ),
            },
        ],
    },
    {
        "doc_id": "D17",
        "doc_title": "Treasury Memo — Revolver Draw and Leverage Grid",
        "doc_type": "memo",
        "category": "causes",
        "version": "v1.0",
        "published_date": "2025-03-11",
        "effective_from": "2025-03-01",
        "effective_to": None,
        "supersedes": None,
        "superseded_by": None,
        "owner": "Treasury",
        "entity_scope": ["Group"],
        "topic": ["interest", "revolver", "covenant", "leverage"],
        "sensitivity": "internal",
        "sections": [
            {
                "section": "§1 Draw Summary",
                "text": (
                    "To: Group Financial Controller. From: Treasury. Date: 11 March 2025. "
                    "Subject: Revolver draw and pricing grid movement. A further USD 15,000,000 was "
                    "drawn on the revolving facility in March to fund seasonal working capital, "
                    "bringing total drawings to USD 50,000,000 against the USD 60,000,000 facility."
                ),
            },
            {
                "section": "§3 Pricing Grid and Interest Impact",
                "text": (
                    "The additional draw moves net leverage to 3.4 times trailing EBITDA against the "
                    "3.5 times covenant limit, leaving headroom of 0.1 times. Under the pricing grid "
                    "this steps the applicable margin up by 100 basis points, increasing March "
                    "interest expense by approximately USD 104,000. Because headroom is below the "
                    "0.25 times disclosure trigger, covenant commentary is required this period "
                    "irrespective of the dollar materiality thresholds."
                ),
            },
        ],
    },
    {
        "doc_id": "D18",
        "doc_title": "Aftermarket Launch Brief",
        "doc_type": "memo",
        "category": "causes",
        "version": "v1.0",
        "published_date": "2025-06-18",
        "effective_from": "2025-06-18",
        "effective_to": None,
        "supersedes": None,
        "superseded_by": None,
        "owner": "Product",
        "entity_scope": ["Aftermarket Parts"],
        "topic": ["launch", "revenue", "opex", "product"],
        "sensitivity": "internal",
        "sections": [
            {
                "section": "§2 Launch Plan",
                "text": (
                    "To: Commercial Director. From: Product. Date: 18 June 2025. Subject: Aftermarket "
                    "range launch, planning basis. The Aftermarket Parts range launches in August. "
                    "The planning basis assumes an incremental USD 620,000 of revenue in the launch "
                    "month, with associated launch costs of approximately USD 400,000 in marketing "
                    "and channel support."
                ),
            },
            {
                "section": "§4 Planning Caveat",
                "text": (
                    "These figures reflect the launch scope agreed in June. Any subsequent change to "
                    "the launch scope, channel mix, or promotional calendar will change the revenue "
                    "and cost profile, and this brief will not be reissued. Confirm the current scope "
                    "with the Product team before relying on these figures for reporting."
                ),
            },
        ],
    },
]


def main() -> int:
    docs = json.loads(CORPUS.read_text())
    existing = {d["doc_id"] for d in docs}
    added = []
    for m in NEW:
        if m["doc_id"] in existing:
            print(f"  skip {m['doc_id']}, already present")
            continue
        docs.append(m)
        added.append(m["doc_id"])
    docs.sort(key=lambda d: d["doc_id"])
    CORPUS.write_text(json.dumps(docs, indent=2, ensure_ascii=False) + "\n")

    print(f"  added {added}")
    print(f"  corpus now {len(docs)} documents, "
          f"{sum(len(d['sections']) for d in docs)} sections")
    cats: dict[str, int] = {}
    for d in docs:
        cats[d["category"]] = cats.get(d["category"], 0) + 1
    print(f"  by category: {cats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
