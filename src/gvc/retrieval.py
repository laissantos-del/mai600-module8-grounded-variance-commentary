"""
Two-axis, point-in-time filtered retrieval.

The rules axis and the causes axis search disjoint sub-corpora with separate top-k
budgets, so a strong policy match cannot crowd out a driver memo (or the reverse).
Retrieved documents are de-duplicated, because one document's several sections
would otherwise consume the whole budget.

Module 8 adds two configurable controls, both motivated by measured Module 7
failures rather than speculation:

  exclude_precedent  The prior-period commentary pack D20 topped the causes axis in
                     3 of 8 cases and appeared in 6. Excluding the `precedent`
                     category is the single highest-value retrieval fix available.

  sufficiency        Module 7 proposed abstaining when no chunk clears tau = 0.45.
                     On C6 an irrelevant document scored 0.507, clearing it. So an
                     absolute threshold is not sufficient on its own; a relative
                     margin and an entity-scope requirement are offered instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import faiss
import numpy as np
import pandas as pd

from .corpus import CAUSE_CATS, RULES_CATS, admissible, month_end


@dataclass
class RetrievalConfig:
    k_rules: int = 2
    k_causes: int = 3
    exclude_precedent: bool = False      # Stage 2 fix for the D20 trap
    tau: float = 0.45                    # absolute sufficiency floor
    min_margin: float = 0.0              # top score must beat the next by this
    require_scope_match: bool = False     # top cause must match entity scope


class Retriever:
    def __init__(self, chunks: pd.DataFrame, embed_model, config: RetrievalConfig | None = None):
        self.chunks = chunks.reset_index(drop=True)
        self.embed = embed_model
        self.config = config or RetrievalConfig()
        self.embeddings = self.embed.encode(
            self.chunks["chunk_text"].tolist(),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    # ---------------- core search ----------------
    def _mask(self, close: date, categories: set[str] | None) -> np.ndarray:
        mask = np.array([admissible(r, close) for _, r in self.chunks.iterrows()])
        if categories:
            mask &= self.chunks["category"].isin(categories).values
        return mask

    def search(self, query: str, close: date, top_k: int,
               categories: set[str] | None = None) -> pd.DataFrame:
        idx = np.where(self._mask(close, categories))[0]
        cols = list(self.chunks.columns) + ["rank", "score"]
        if len(idx) == 0:
            return pd.DataFrame(columns=cols)

        index = faiss.IndexFlatIP(self.embeddings.shape[1])
        index.add(self.embeddings[idx].astype("float32"))
        q = self.embed.encode([query], convert_to_numpy=True,
                              normalize_embeddings=True).astype("float32")
        scores, local = index.search(q, len(idx))

        seen, rows = set(), []
        for rank, li in enumerate(local[0], start=1):
            row = self.chunks.iloc[idx[li]].copy()
            if row["doc_id"] in seen:          # one slot per document
                continue
            seen.add(row["doc_id"])
            row["score"] = float(scores[0][rank - 1])
            rows.append(row)
            if len(rows) >= top_k:
                break

        out = pd.DataFrame(rows).reset_index(drop=True)
        out["rank"] = range(1, len(out) + 1)
        return out

    # ---------------- per-case retrieval ----------------
    @staticmethod
    def queries_for(row) -> tuple[str, str]:
        q_rules = (
            f"materiality threshold commentary requirement owner "
            f"{row.account} {row.cost_centre} variance"
        )
        q_causes = (
            f"{row.account} {row.cost_centre} {row.region} {row.product_line} "
            f"{row.period} driver cause change explanation"
        )
        return q_rules, q_causes

    def retrieve(self, row) -> tuple[pd.DataFrame, pd.DataFrame]:
        close = month_end(row.period)
        q_rules, q_causes = self.queries_for(row)
        cause_cats = set(CAUSE_CATS)
        if self.config.exclude_precedent:
            cause_cats.discard("precedent")
        rules = self.search(q_rules, close, self.config.k_rules, RULES_CATS)
        causes = self.search(q_causes, close, self.config.k_causes, cause_cats)
        return rules, causes

    # ---------------- sufficiency gate ----------------
    def sufficient(self, row, causes: pd.DataFrame) -> tuple[bool, str]:
        """Decide, before generating, whether the retrieved causes are usable.

        Returns (is_sufficient, reason). Deciding here is more reliable than asking
        the model to abstain, because a generator handed a "why" question will
        almost always produce an answer.
        """
        cfg = self.config
        drivers = causes[causes["category"] == "causes"]
        if drivers.empty:
            return False, "no driver-category evidence retrieved"

        def scope_ok(chunk) -> bool:
            if not cfg.require_scope_match:
                return True
            scope = str(chunk.get("entity_scope", ""))
            if "All" in scope:
                return True
            targets = {str(row.cost_centre), str(row.product_line), str(row.region)}
            return any(t and t in scope for t in targets)

        # Scan every retrieved driver, not only the top-ranked one. The gold document
        # can sit at rank 2 or 3 behind a higher-scoring but irrelevant memo, and the
        # question the gate must answer is whether ANY usable evidence is present.
        best = None
        for _, chunk in drivers.iterrows():
            if float(chunk["score"]) < cfg.tau:
                continue
            if not scope_ok(chunk):
                continue
            best = chunk
            break

        if best is None:
            top = drivers.iloc[0]
            if float(top["score"]) < cfg.tau:
                return False, f"no driver clears tau {cfg.tau} (best {float(top['score']):.3f})"
            return False, "no retrieved driver matches the row's entity scope"

        if cfg.min_margin > 0 and len(causes) > 1:
            margin = float(best["score"]) - float(causes.iloc[1]["score"])
            if margin < cfg.min_margin:
                return False, f"margin {margin:.3f} below {cfg.min_margin}"

        return True, f"sufficient ({best['doc_id']} at {float(best['score']):.3f})"
