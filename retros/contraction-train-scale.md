# Training the contraction player at scale — the n = 40 gap mostly closes

`contraction-policy-einsum.md` recorded the honest standing: the
learned policy loses to `opt_einsum`'s **randomised greedy** at n = 40
by ~1.8×, and `contraction-policy-throughput.md` showed the gap is
*quality*, not throughput.  Every other lever had been pulled —
wire-in (structurally impossible), PUCT (worse, lookahead is the wrong
tool), the value head (fixed, didn't move the game).  The one lever
never pulled was the **training distribution**: the policy was trained
on n = 8–12 and tested at n = 20–40.

This retro records the experiment and its diagnosis tool,
`tools/contraction_scale_probe.py`.  The headline is **the first
positive on this thread in a while**:

* **Training at scale narrows the n = 40 gap from ~1.8× to ~1.04×** at
  adequate budget — essentially parity with the best cheap player.
* **The diagnosis answers "generalisation *and* inductive bias":**
  both contribute, and the residual gap is **algorithmic**, not
  trainable.

Three policies, same features, same net, same training budget
(`--iterations 1800`, REINFORCE, DP teacher):

| policy | train scales | trained in |
|---|---|---|
| `base` | 8, 10, 12 | 61 s |
| `scale` | 16, 20, 24 | 347 s |
| `curriculum` | 8, 12, 16, 20, 24 | 205 s |

Reproduce:

    uv sync --group einsum
    .venv/bin/python tools/contraction_scale_probe.py --device cuda \
        --scales 40 --boards 12 --holdout 12 \
        --epochs 400 --iterations 1800 --teacher-budget 1.0

    .venv/bin/python tools/contraction_einsum.py --device cuda \
        --train-scales 16,20,24 --compare-scales 8,12,16,20,24

CUDA used; ~10 min for the three trainings plus the ladders.
Instances are the einsum-valid **bond network** family — the one
`opt_einsum` can price — with each index appearing exactly twice.

## 1. Head-to-head vs `opt_einsum` randomised greedy (n = 40)

Pairwise `learned / oe-rand-greedy` (mean over 3 boards; <1 = learned
wins; equal wall-clock):

| budget | base (8,10,12) | scale (16,20,24) | curriculum |
|---|---|---|---|
| 50 ms | 4.13–5.14 | 5.89 | **1.63** |
| 200 ms | 1.54–2.83 | **1.04** | 1.22 |
| 1000 ms | 1.72–1.78 | 1.31 | 1.49 |

And against the *deterministic* external greedy at n = 40 — the
player the field actually ships single-pass:

| budget | base | scale | curriculum |
|---|---|---|---|
| 200 ms | 1.01–1.68 | **0.69** | **0.76** |
| 1000 ms | 0.86 | **0.68** | **0.73** |

**No regression at small n.**  At n = 20–30 both scale-trained
policies stay ≤ the base's ratios vs `oe-rand-greedy` (0.97–1.28 vs
0.99–1.13) and beat `oe-greedy` everywhere (0.73–0.94).

**The 50 ms caveat is throughput, not quality.**  At n = 40 the
learned player manages only **2.3–2.7 work units** in 50 ms — it is
starved, exactly the per-decision-overhead profile
`contraction-policy-throughput.md` measured.  The pure-scale policy
starves worst; the curriculum (which keeps small boards in training)
degrades gracefully (1.63 vs 5.89).

## 2. The diagnosis — generalisation *and* inductive bias

`contraction_scale_probe.py` measures three things on n = 40 bond
boards (`--scales 40 --boards 12 --holdout 12`):

**Decision agreement.**  Walk `oe-rand-greedy`'s chosen order and ask
each policy which pair it would contract at every state:

| player | agreement with the teacher's order |
|---|---|
| `our-greedy` | 0.130 |
| `base` | 0.267 |
| `scale` | **0.316** |

Training at scale improves agreement modestly — but even the
scale-trained policy reproduces only ~⅓ of the teacher's decisions.

**Feature sufficiency.**  Fit a supervised net — *same features, same
net* — to the teacher's action at n = 40, 12 train / 12 held-out
boards, 400 epochs:

| teacher | train top-1 | held-out top-1 | roll/teacher |
|---|---|---|---|
| `oe-greedy` (deterministic) | 0.989 | **0.596** | **0.821** |
| `oe-rand-greedy` | 0.729 | 0.291 | 2.429 |

Two distinct findings in one table:

* **The features can partially express the strong player's choice** —
  60 % held-out top-1 on the deterministic teacher.  So the residual
  gap is *partly* inductive bias, but not hopeless: **the imitator's
  own greedy rollout *beats* its teacher** (`roll/teacher` = 0.82 —
  the smoothed learned policy prices better than the order it was
  trained to copy).
* **The randomised teacher cannot be imitated single-pass.**  Its
  edge is best-of-N restarts over a staged heuristic — an
  *algorithmic* advantage.  A net fit to it rolls out 2.4× worse than
  the teacher's selected order.

**On-training-board check.**  Each policy's ratio to internal greedy
where it was trained: base 0.413, scale 0.164 — both policies do beat
greedy on their own distribution (and greedy itself is weaker at
larger n, so the ratio is not purely policy strength).

## 3. Verdict

* **The ~1.8× claim is stale.**  With scale training the n = 40 gap is
  **~1.04–1.5×** depending on budget and training mix — down from
  1.54–1.78.  The standing one-liner should now read: *"a single-pass
  learned policy reaches parity-to-1.3× of best-of-N randomised
  greedy at n = 40, and beats the deterministic external greedy by
  ~25–30 %."*
* **Both hypotheses confirmed.**  Part of the old gap was a train/test
  scale mismatch (fixed by training at scale); part is inductive bias
  (features express ~60 % of the teacher's decisions); and the
  remainder is algorithmic — best-of-N restarts is a search procedure,
  not a policy, and cannot be matched by one deterministic pass.
* **Curriculum is the recommended regime** — train on all scales.  It
  costs ~3× the base training time, preserves small-board and
  low-budget behaviour, and loses only marginally to pure-scale at
  200 ms (1.22 vs 1.04).
* **What this does not change:** the learned player still loses to
  `oe-rand-greedy` at starvation budgets (throughput), and no policy
  lever tested beats best-of-N restarts outright.  The honest ceiling
  for a single-pass policy on this game is now measured, not assumed.

## 4. Honesty / limits

* **3 boards per cell, 1 seed.**  The 1.04 vs 1.49 spread between
  scale and curriculum is within run-to-run wobble; the *direction*
  (scale training ≫ base) is what is stable.
* **Bond-network family only** — the einsum-valid instances.  Real
  einsums (attention / MLP, §4 of the ladder output) are unaffected:
  every player reaches `oe-optimal` there.
* **Teacher asymmetry.**  `roll/teacher` = 0.82 shows a *deterministic*
  teacher is beatable by its own student; the randomised teacher's
  advantage is the restart loop, so "beat `oe-rand-greedy`" is
  arguably "beat a search algorithm with one forward pass" — a bar no
  policy on this board has cleared.
* **`packages/` untouched** — all changes are `tools/` experiment code.

## Gates

`tools/`-only change (`contraction_scale_probe.py` new;
`contraction_policy.py` gains DP/teacher-order helpers;
`contraction_einsum.py` gains multi-model comparison):

* `.venv/bin/ruff check` / `ruff format --check` on all touched files — pass
* Training runs: `p1_base.txt` 61 s, `p2_scale.txt` 347 s,
  `p3_curric.txt` 205 s (all CUDA, converge)
* Ladders `e1_base_vs_scale.txt`, `e2_base_vs_curric.txt` — tables above
* Probe `probe.txt` / `probe_fit.txt` — §2 tables
