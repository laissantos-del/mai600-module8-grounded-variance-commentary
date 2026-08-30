# AI Usage Disclosure

**Author:** Lais Santos Silva
**Course:** MAI 600 - Module 8 Final Project

## AI Tools Used

- **Claude:** Used throughout design, drafting, code generation, and verification, under my
  direction and review.
- **Claude Code:** The same tool as "Claude" above, used for the local Python package, the
  experiment scripts, the Streamlit interface, the notebook, and the figures.
- **Ollama with `qwen3:4b-instruct`:** The local model runtime. This is the system under
  evaluation, not a drafting aid.
- **sentence-transformers and FAISS:** The retrieval stack under test, not AI-assistance
  tools.

## How I Used AI

Module 8 moved the project off Google Colab and onto my own machine, and most of the AI
assistance went into that port and into the evaluation that followed it.

Claude helped restructure the Module 7 notebook into a proper Python package
(`src/gvc/`), implement the five comparison arms B0 to B4, extend the evaluation from
eight hand-picked probes to twenty-six cases across ten consecutive close periods, add the
covenant materiality rule and three new driver memos, build the Streamlit interface,
generate the architecture and results figures, and draft this repository's written
material from the executed output files.

Claude also proposed the temporal split, which I adopted after the earlier design let a
configuration be chosen on the same cases it was scored on.

## Prompts Used

Important prompt categories included:

- Porting the Colab notebook into a package whose first job is to reproduce the Module 7
  result exactly, before any new measurement is attempted.
- Designing an evaluation shaped like consecutive monthly closes rather than a set of unit
  tests, so that month-over-month replication could be measured instead of asserted.
- Building a baseline ladder that includes controls capable of embarrassing the proposed
  system, specifically a deterministic template and an unfiltered long-context arm.
- Reviewing metric definitions for scopes that flatter the system, and reporting the
  honest pair instead.
- Drafting the README, the notebook narrative, and the article strictly from the executed
  result files, and correcting them wherever a draft disagreed with those files.

## What I Verified Myself

**Parity before anything else.** `tests/test_parity.py` reproduces the Module 7 Colab
result on my machine before any new number is trusted. It caught a real bug immediately:
under the newer pandas in my local environment, Arrow-backed string columns represent a
JSON null as a float NaN, which is truthy, so the date parser received NaN where Colab had
received None. Every point-in-time filter would have been silently wrong. That failure is
the reason the parity check exists.

**Metric scope.** The summary originally reported correct abstention without
over-abstention. That hid a candidate configuration which scored a perfect 4 out of 4 on
the configuration periods and then abstained on 100% of the hold-out cases. I added the
paired metric, and then narrowed it further so that abstaining when the evidence was never
retrieved is not counted against the system.

**A second scope problem I found later.** Numeric fidelity was measured only on documented
cases, which cannot see a figure invented on a case that abstained. Widening it to every
case that produced text changed two conclusions in the paper: the long-context arm fell
from a perfect score to 50%, and B4 turned out to beat B3 rather than tie it. Both are
reported at both scopes.

**The gate logic.** Two cases were abstaining despite the correct memo being retrieved,
because the sufficiency check inspected only the top-ranked driver. The supporting memo
frequently sits at rank 2 or 3 behind a higher-scoring irrelevant one. I had Claude fix it
to scan all retrieved drivers and re-verified on the configuration periods.

**Chart honesty.** An early improvement chart showed correct abstention as 100% before and
100% after, which suggested nothing had changed. The measurement had actually grown from
two cases to eight. The chart now carries the raw fractions and a caption saying so.

Every number in the README, the notebook, and the article traces to a file in `results/`,
and every one of those files is written by a script in `build/` or `tests/`. Nothing is
transcribed by hand, because a hand-written table slipped into a Module 7 draft once and
missed a real failure.

I ran the full pipeline on my own machine, executed the notebook top to bottom, and drove
the Streamlit interface through both a documented case and an evidence-less case to
confirm the abstention state renders.

## Failures or Limitations

Claude's first sufficiency gate used an absolute similarity threshold of 0.45, carried
over from a Module 7 suggestion. On this corpus the scores cluster between 0.34 and 0.45,
so the gate fired on every case. It scored perfectly on the configuration periods and then
failed on 100% of hold-out. The tell was a mean response time of 0.00 seconds, which I
noticed only because the arm was faster than physically possible. Reviewing generated code
by reading it would not have caught this; running it and looking at an implausible number
did.

Claude also initially proposed scaling the corpus from 15 documents to 38. I did not do
that, because corpus authoring is the largest time cost in the project and the assignment
awards no points for it.

## Academic Integrity Statement

I confirm that AI was used as a learning, development, and drafting support tool under my
direction. I made the design decisions, ran every experiment on my own machine, and
verified every reported result against the output files that those runs produced. I take
responsibility for the final submitted work.
