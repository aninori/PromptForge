# PromptForge at Organization Scale — Cost Analysis

**50 developers · GitHub Copilot + self-hosted GPU**

_Prepared 2026-08-04. Every assumption is labelled; replace them with your own
figures and the arithmetic still holds._

---

## Summary

Your team exhausts its Copilot credit allowance halfway through every month and
does **not** buy overage. For the remaining two weeks, 50 developers work without
AI assistance.

A self-hosted PromptForge instance answers a share of questions locally, so the
same credit allowance stretches further.

| | |
|---|---|
| Copilot today | $1,250/month |
| PromptForge GPU | ~$1,400/month |
| **Combined** | **$2,650/month** ($53/developer) |
| What it buys | AI assistance for the whole month instead of half |
| Cost per recovered developer-day | **$3–10** |

**This is not a cost saving.** You don't currently pay for the credits you run
out of, so there is no bill to reduce. It is a **capability purchase**: ~$1,400/month
to stop 50 developers losing AI assistance for two weeks out of every four.

---

## Inputs

| Input | Value | Source |
|---|---|---|
| Developers | 50 | given |
| Credits per developer | 15,000/month | given |
| Copilot price | $300 per 12-member team | given |
| Credits exhausted after | 2 weeks | given |
| Overage purchased | none | given |
| Working days per month | 20 | assumption |
| Credits per user prompt | ~20 (agent mode) | **assumption — verify** |
| Diversion rate | 22% | **estimate — verify** |

---

## Step 1 — Current Copilot spend

```
Teams needed   = 50 ÷ 12       = 4.17
Monthly cost   = 4.17 × $300   = $1,250/month
Per developer  = $1,250 ÷ 50   = $25/developer/month
```

## Step 2 — Credit burn rate

```
Credits available = 50 × 15,000  = 750,000/month
Consumed in       = 2 weeks      = 10 working days
Burn rate         = 750,000 ÷ 10 = 75,000 credits/day
Per developer     = 75,000 ÷ 50  = 1,500 credits/dev/day
```

## Step 3 — Full-month demand

```
Demand at the same pace = 75,000 × 20 = 1,500,000 credits
Available                             =   750,000 credits
Shortfall                             =   750,000 credits  (50%)
```

**The team is funded for exactly half of what it would use.**

## Step 4 — Credits to prompts

Agent mode spends several credits per user prompt — each internal step is a
separate premium request. PromptForge intercepts at the **prompt** level, so
diverting one prompt saves the whole task.

```
Assume ~20 credits per prompt
Prompts/month     = 1,500,000 ÷ 20     = 75,000
Per developer/day = 75,000 ÷ 50 ÷ 20   = 75 prompts/dev/day
```

## Step 5 — GPU load

Every prompt passes through routing; only diverted prompts need full generation.

```
Routing    (all)      : 75,000 × 5s  = 375,000 s
Generation (22%)      : 16,500 × 12s = 198,000 s
Total                                = 573,000 s = 159 GPU-hours/month
```

One GPU supplies 730 hours/month, so raw compute is not the constraint —
**concurrency is:**

```
Average = 75,000 ÷ 20 days ÷ 8 hrs = 469 prompts/hour ≈ 8/min
Peak (×3)                          ≈ 24/min
Concurrent streams required        ≈ 3   →  2× L4 GPU
```

## Step 6 — GPU cost

```
AWS g6.xlarge (L4 24GB)  $0.805/hr × 730 hrs = $587/month
2 GPUs                                        = $1,175
Host, storage, network                        =   $225
Total                                         = $1,400/month
```

Reserved instances or on-prem hardware reduce this materially; treat $1,400 as
the on-demand cloud ceiling.

## Step 7 — Coverage gained

```
Days covered = 20 × 750,000 ÷ (1,500,000 × (1 − f))  =  10 ÷ (1 − f)
```

| Diversion `f` | Days covered | Days gained | Developer-days recovered |
|---|---|---|---|
| 0% (today) | 10 | — | — |
| 22% | 12.8 | 2.8 | 140 |
| 43% | 17.5 | 7.5 | 375 |
| 50% | 20 | 10 | **500** |

At **50% diversion the shortfall disappears entirely** — the allowance covers the
full month.

## Step 8 — Cost per recovered developer-day

| Diversion | Cost per developer-day |
|---|---|
| 22% | $10.00 |
| 43% | $3.73 |
| 50% | $2.80 |

**Break-even:** a loaded developer costs roughly $400/day. Restoring a day of AI
assistance costs $3–10, so the investment pays for itself if AI assistance lifts
output by more than **~2%**. Even the pessimistic case clears that comfortably.

---

## Numbers to verify before committing

Listed in order of how much they move the result.

### 1. Diversion rate (currently assumed 22%)

The single most important figure, and the least certain. It came from a handful
of test queries on one machine, not real usage. Plausible range is 10–45%, which
is a 4× swing in value.

**How to measure:** run a 5-developer pilot on one GPU for two weeks
(~$600). `GET /savings` reports `byType.reasoning` versus
`byType.promptOptimization` — the reasoning share is the diversion rate.

### 2. Credits per prompt (currently assumed 20)

If the real figure is 5, prompt volume is 4× higher and you need ~4 GPUs
(~$2,800/month) rather than 2. Cost per recovered developer-day roughly doubles
but stays well inside the break-even.

**How to measure:** compare premium-request consumption against prompts actually
typed, from the GitHub org billing page.

### 3. Working-day assumption (currently 20)

Minor. Affects burn rate proportionally in both directions and largely cancels.

---

## Blocker — the current code cannot serve 50 people

PromptForge today is built for a single user. This is a design property, not a
bug, and it must be addressed before any multi-developer rollout.

**The index is global — one repository at a time.** Opening a different workspace
wipes the previous index and rebuilds it. With 50 developers on different
projects, each switch would destroy everyone else's index.

Supporting state — brief storage, the semantic cache, the savings counters — is
likewise global, with no concept of who is asking.

### Work required

| Item | Estimate | Why |
|---|---|---|
| Per-repository index | 2 weeks | **The blocker.** Nothing works for a team without it |
| User/team identity and scoping | 1 week | Prevents caches and history leaking between people |
| Request queue and concurrency | 1 week | 50 people, shared GPUs |
| Retrieval quality fixes | 1.5 weeks | See below |
| Authentication | 0.5 weeks | The port is currently open to anyone who can reach it |
| **Total** | **~6 weeks** | roughly $20,000 loaded, one-time |

### Retrieval quality — fix before any pilot

Measured on this codebase: a question about click behaviour returned eight data
and documentation chunks and **zero** components. Both the embedding model and the
cross-encoder favour fluent prose over code, so files that *describe* a feature
outrank the files that *implement* it.

Two mitigations shipped (blended reranking, a documentation quota) and improved
one of two regression queries. The second still fails.

This matters commercially: **if developers ask questions and get poor answers,
they will bypass the tool, and the diversion rate collapses toward zero** — taking
the entire business case with it.

---

## Recommendation

**Run a 5-developer pilot before funding the build-out.**

- One GPU, one shared repository, two weeks — roughly $600
- Measures the diversion rate, the one variable that swings the result 4×
- Confirms whether developers trust the local answers enough to use them
- Produces a measured GPU cost rather than an estimate

If the pilot shows a solid diversion rate and developers find the answers useful,
the ~$20,000 multi-tenancy work is straightforwardly justified. If they route
around it, that is far cheaper to discover with five people than with fifty.

---

## What this analysis deliberately does not claim

- **No cost saving.** You don't buy overage credits, so there is no bill to cut.
  Earlier drafts of this analysis assumed a $30,000/month overage charge; that was
  wrong and has been removed.
- **No productivity multiplier.** The break-even is expressed as the threshold
  (~2%) rather than an asserted gain, because nothing here measures productivity.
- **No token-cost saving.** Copilot bills a flat subscription. Avoided tokens
  reduce credit consumption, not a per-token invoice.
