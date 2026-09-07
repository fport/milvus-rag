# Rung 6 — GraphRAG

!!! done "Not built here"

    This project stops at rung 5. This page is what rung 6 is, when it is worth it, and
    what would be built first if it were — written to the same standard as the rest,
    which means the numbers below are other people's, and labelled as such.

## The question that top-k cannot answer

Everything up to here retrieves *passages*. Some questions have no answer in any
passage:

> *"What are the main themes in this corpus?"*
> *"Which teams does this change affect?"*
> *"What breaks if I change this interface?"*

Not because ranking is bad — because the answer is not written anywhere. It is a
property of how the pieces relate to each other, and you would have to read all of them
to construct it. Microsoft's GraphRAG paper names this the difference between **local**
questions (answered by a few passages) and **global** ones (answered by the corpus as a
whole).

## What it does

The text version, roughly:

1. **Extract.** An LLM reads every chunk and pulls out entities and relations —
   `(PaymentService) --[calls]--> (LedgerAPI)`.
2. **Build the graph** by merging entities across chunks. This is the step that creates
   information: two facts that were in different files are now one edge.
3. **Detect communities** (Leiden or similar) and have the LLM write a summary of each
   cluster, then summaries of clusters of clusters.
4. **Query** in one of two ways: **local** — find the entity, walk its neighbourhood,
   answer from that; **global** — answer from the community summaries and combine.

The community summaries are the whole trick. A global question is answered from a
pre-computed hierarchy of "what is in this part of the corpus", never by retrieving the
corpus.

## For code, the graph is already there

This is the part that changes the calculus, and most GraphRAG writing misses it because
it is written about prose.

A codebase has a real, exact graph that needs **no LLM to extract**: imports, calls,
type references, definitions and their uses. tree-sitter is already parsing every file
in this project — the same pass that produces chunks can produce edges, deterministically
and for free.

That matters because it removes rung 6's main cost. In text GraphRAG the extraction is
an LLM pass over the whole corpus, it costs real money, it is *probabilistic* (the same
document extracted twice gives different entities), and it has to be re-run when the
corpus changes. A call graph is none of those things: it is exact, it is cheap, and it
updates incrementally with the same file-level diff that already drives indexing here.

So for code the sensible order is not "text GraphRAG, applied to code". It is:

**First, the free structural graph.** Symbol → definition, definition → references,
file → imports. Then use it to *expand* retrieval rather than replace it: take the
rung-3 hits, add their one-hop neighbours, hand the agent both. That answers "what calls
this" and "what would break", which is most of the practical demand, and it costs one
extra query per search.

**Only then, and only if needed, the LLM layer** — communities and summaries for the
genuinely global questions ("what are the main subsystems"). That is where the cost and
the staleness live, and it is a much smaller step once the structural graph exists.

## What it costs

| Cost | Text GraphRAG | Code, structural graph |
|---|---|---|
| Build | an LLM pass over every chunk | free, in the parse you already do |
| Rebuild on change | re-extract the changed region, re-cluster | incremental, exact |
| Determinism | probabilistic extraction | exact |
| Query latency | community summaries, multiple LLM calls | one extra lookup |
| Evaluating it | hard — global answers have no single right passage | ordinary Recall@k still applies |

Two of those deserve emphasis.

**Cost.** Published GraphRAG indexing runs are commonly quoted in the tens of dollars
per million tokens of corpus and hours of wall clock. For calibration with something
measured *here*: LLM chunk descriptions — a far simpler per-chunk LLM pass — took about
four hours for a 2.8k-chunk repo on a local 9B. Graph extraction is that shape of cost,
with a clustering pass on top.

**Evaluation.** This is the real reason to be slow about it. A global answer has no
gold passage, so Recall@k does not apply and the honest methods are LLM-as-judge on
comprehensiveness and diversity, or human rating. Both are weaker instruments than the
ones on [rung 4](04-honesty.md). It is entirely possible to build rung 6 and be unable
to demonstrate that it helped — which, by this project's one rule, means you should not
ship it.

## When to climb it

Climb when:

- the questions you are failing are **global** — themes, impact, "what depends on what"
- your corpus has genuine cross-document structure (a codebase, a case file, a wiki
  with real interlinking) rather than a pile of independent articles
- the corpus is stable enough that rebuilding the graph is not a weekly event
- and — for code — you have already taken the free structural graph and it was not enough

Do not climb when the failures you actually see are "it did not find the function" or
"it answered a question it should have refused". Those are rungs 3 and 4, they are far
cheaper, and rung 6 does not fix them.

!!! measured "The honest status of this rung"

    Nothing on this page has been measured on this corpus. The rungs below it have, and
    two of them came out against expectation.

    That is the argument for the ladder, not against rung 6: the reason to know what
    the top rung costs is so you can tell whether you have actually run out of cheaper
    ones. Most systems have not.
