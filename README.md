<h1 align="center">wetware (Rust) — RSI fork</h1>

<p align="center">
  <strong>A general-purpose recursive self-improvement loop — demonstrated on a real brain.</strong><br />
  Hand the loop any fixed system and a task, and it teaches itself to use that
  system better across generations: search, curricula, gated adaptation, and a
  persistent memory of what worked that each run inherits — the L1–L5 ladder
  from <em>The Last AI Built by Humans</em>, fully observable. The fixed system
  here is the only complete brain wiring ever mapped (a larval fruit fly's; the
  animal is never edited). The first benchmark is handwritten digits. Both are
  incidental — swap either, and the loop is unchanged.
</p>

> **Why these demos?** They're the cheapest stage where the loop's behavior is
> unambiguous: a fixed brain that demonstrably computes (10% baseline → 92.8%)
> with real tuning headroom for a linear readout. The subject of this repo is
> not the task — it's the loop. Anything that fits behind an `Evaluator` works:
> L1–L5, the bank, and the observability are task-agnostic.

A fork of [awdemos/wetware-rs](https://github.com/awdemos/wetware-rs) (itself a
Rust port of the Python `wetware` package).


## What's added in this fork

| piece | file | level |
|---|---|---|
| `optimize <task>` — the improvement loop | `src/rsi/optimize.rs` | L1–L5 |
| `adapt` — online adaptation under drift, naive vs gated | `src/rsi/adapt.rs` | L4 |
| `bank` — the experience bank and its derived policy | `src/rsi/bank.rs` | L5 |
| unified verifier: seeded shuffles, val-only search, one-shot test | `src/rsi/evaluate.rs` | — |
| event stream (JSONL + stderr) for every loop decision | `src/rsi/observe.rs` | — |
| weighted ridge, online RLS readout, retention gate | `src/rsi/learn.rs` | L3/L4 |
| the editable surface (search space) | `src/rsi/config.rs` | — |

### The loop

```console
$ wetware optimize digits --budget 16 --level l5 --log run.jsonl
```

- **L1** executes a seeded random search over the harness knobs (`spectral_radius`,
  `leak`, `ridge`, `input_scale`, `inhibitory_fraction`, `seed`) — the two knobs
  the stock demo never exposed are plumbed through (`tasks/digits.rs` `FeatParams`).
- **L2** takes over *where to search*: each round's distribution is the top
  quartile of everything so far, with diagnostics (generalization gap → ridge up,
  underfit → more dynamics, stagnation → uniform re-exploration) emitted as events.
- **L3** takes over *what to learn from*: the train pool is reweighted toward
  what the learner actually misses (train confusions + per-class validation
  errors), and the best weighting is re-validated before it is retained.
- **L4** rehearses deployment: a drifting stream (digits 0–4, then 5–9) adapts a
  readout online by recursive least squares. The naive readout keeps every
  update; the gated one only persists a change that beats the accepted readout
  on a fixed validation set — rollbacks are events, not silent state loss.
- **L5** is inheritance: every accepted champion is appended to an experience
  bank (`~/.cache/wetware/rsi-bank.jsonl`), and the bank's top quartile defines
  a *policy* — a warm-start distribution that shapes where the next run begins.
  The policy is versioned; runs state which version they built on.

### Verification discipline (fixed infrastructure, not a level)

- Trials are scored on a validation split; the test set is touched **once**,
  at final acceptance. Splits are seeded shuffles — the digits CSV is
  class-grouped, so row-order splits are an exploit surface for a search loop.
- Acceptance is by **median validation over several seeds** (the ±1% RNG wobble
  documented upstream is the seed-cherry-picking hazard in miniature;
  `--verify-seeds` controls it).
- Every decision — trials, diagnostics, curriculum focus, adaptations,
  rollbacks, policy loads/updates — is an event: human line on stderr, JSONL
  with `--log`. The loop's full trajectory is inspectable after the fact.

### Try it

```console
cargo build --release
wetware --sample optimize timeseries --level l5 --budget 12   # offline
wetware optimize digits --level l5 --budget 12                # the real brain
wetware adapt                                                 # L4 drift demo
wetware bank                                                  # what's been retained
```

### Known issues & limits

**Scale.** This loop is built and verified for single-machine runs. Pushing it to
a fleet (or "hundreds of GPUs") changes which parts matter:

- *Trials scale linearly; the loop doesn't.* L1 evaluations are independent and
  embarrassingly parallel, but L2 rounds, L3 rounds, L4 streams, and L5
  generations are dependency chains — wall-clock ends up dominated by the
  sequential tail. The workload is also sequential-latency bound, not FLOP
  bound: one GPU can batch the connectome recurrence across thousands of images,
  but the RLS stream is ~8.7M FLOPs *per sample, in order*, with a ~70MB `P`
  matrix per adapter (×2 for candidate+accepted, plus a full clone per
  rollback). GPUs would mostly idle; a CPU fleet is the honest hardware.
- *The verifier becomes the bottleneck and the target.* Median-over-seeds kills
  RNG cherry-picking, not split-fitting: at 100k+ trials the validation set
  becomes the adaptively-queried surface (the survey's reliable-verification
  problem, one level down from test). Mitigation — rotating/nested validation,
  growing the val set with search depth — is not implemented.
- *The bank and event stream assume one writer.* Append-only JSONL with
  per-event flush is a single-node design: concurrent runs race on version
  bumps and inherit stale policies, and the JSONL volume (GBs per run at fleet
  scale) needs sampling/aggregation, not `flush()` per event.

**Failure modes that only appear at scale** (all currently unhandled):

- *Policy homogenization* — every worker warm-starts from the same top-quartile
  center, so the meta-population clusters in one region and loses diversity.
  Islands/demes with periodic exchange would fix it.
- *Bank pollution* — `champion_fallback` retains flat rounds as anchor-valued
  entries; thousands of flat runs make the derived policy mushy.
- *No retirement* — the bank is append-only with no eviction (the Library Drift
  problem: unbounded accumulation silently degrades retrieval), and curriculum
  judgments come from the readout's own errors, so experience corruption
  compounds across generations.
- *Rule exploitation* — the diagnostics thresholds become part of the
  optimization surface once search pressure is high enough to find them.
- *Unvalidated policy* — champions are gated by median-over-seeds, but the
  policy itself is never validated; A/B policy-warm-start vs uniform under
  matched budgets (autonomy attribution) is future work.

**Ceiling.** A fixed 2952-neuron reservoir with a linear readout has bounded
expressiveness: massive search approaches that asymptote and flatlines. What
wouldn't flatline is the L5 recursion itself — but only with the missing
discipline above (retirement, policy validation, population structure).

New dependencies: `serde`, `serde_json` (structured event/bank records).
Everything else is unchanged from upstream.

---

*Original README follows.*


This is a Rust port of the Python [`wetware`](https://github.com/Roxx0x/wetware) package. Same brain,
same math, same CLI — a library plus a `wetware` binary.

## What this is

In 2023 scientists finished the first synapse-resolution wiring diagram of an entire animal brain — the larval fruit fly. **2952 neurons, ~110,000 connections, every one mapped.** It's public.

`wetware` takes that brain, freezes it, and uses it as a computer.

The trick is reservoir computing: a fixed recurrent network can do real computation if you feed signals in and train a thin linear layer to read the activity back out. So we drive the fly's actual wiring with input, and train a one-layer readout on top. **The brain is never modified — you can't edit an animal's connectome — only the readout learns.**

And it works:

```
$ wetware run digits
{
  "task": "digits",
  "accuracy": 0.93,               # a real fly brain reading handwritten digits
  "baseline_accuracy": 0.0988,
  "test_n": 597,
  "neurons": 2952
}
```

That ~93% is the fly's brain, wired as nature left it, reading handwriting it has never seen — with only a linear readout trained on top. (The exact figure varies by ~1% with the RNG used for the assigned E/I signs, since numpy's PCG64 is not bit-reproducible here.)

## Quickstart

```
git clone https://github.com/Roxx0x/wetware   # the Python original
# this port: build from this directory
cargo install --path .
```

```
wetware download              # fetch + cache the connectome
wetware info                  # neurons, connections, density
wetware run digits            # read handwriting on the real brain
wetware run timeseries        # predict the future on the real brain
wetware --sample run digits   # offline synthetic stand-in, no download
```

As a library:

```rust
use wetware::tasks::{digits, timeseries};

let brain = wetware::load()?;                              // downloads the real fly brain once (~1 MB), then instant
println!("{:?}", digits::demo(&brain, &Default::default()));    // ~93%
println!("{:?}", timeseries::demo(&brain, &Default::default())); // ~2x better than predict-the-mean
# Ok::<(), wetware::WetwareError>(())
```

> [!TIP]
> No connectome download needed to kick the tyres: `--sample` runs everything on a small synthetic network with the same statistics (the `digits` task still fetches its small dataset once). But anything you publish should run on the real thing — `load()` — because the real thing is the entire point.

## How it works

1. **Load** the connectome as a weighted wiring matrix `W` (who connects to whom, by synapse count).
2. **Fix it** — scale it to the echo-state regime and never touch it again. This is the brain.
3. **Drive it** with your input and record how every neuron responds over time.
4. **Train** a single linear readout (ridge regression, closed-form — no backprop) to turn that activity into an answer.

Swap the task, keep the brain. That's what makes it universal: the same fixed brain reads digits, predicts a series, or — with a new readout — whatever you throw at it.

## Layout

| Python | Rust |
|---|---|
| `wetware/data.py` | `src/data.rs` — download/parse/cache the connectome (`.npz` compatible with the Python cache) + synthetic sample |
| `wetware/reservoir.py` | `src/reservoir.rs` — echo-state dynamics on a CSR matrix |
| `wetware/readout.py` | `src/readout.rs` — closed-form ridge regression |
| `wetware/tasks/` | `src/tasks/` — `digits`, `timeseries` |
| `wetware/cli.py` | `src/main.rs` — clap CLI, same subcommands and JSON output |
| `tests/test_wetware.py` | `tests/wetware.rs` — same 7 tests, all offline on the sample |
| `examples/read_digits.py` | `examples/read_digits.rs` |

Differences from the Python, honestly:

- **Digits data**: scikit-learn is Python-only, so the same `digits.csv.gz` sklearn bundles is downloaded once (small, cached) instead of imported.
- **Spectral radius**: found by power iteration (like SciPy's `eigs(k=1)` fast path), with a full eigendecomposition as fallback. The NumPy-only Python path always did the slow dense solve.
- **RNG**: ChaCha (`StdRng`) instead of numpy's PCG64 — same distributions, different draws, hence the ±1% accuracy wobble.

## Performance

Release build, real 2952-neuron connectome, this laptop:

| command | time |
|---|---|
| `wetware run timeseries` | ~1.3 s |
| `wetware run digits` | ~4 s |

The heavy lifting is `faer` (linear algebra) with a hand-rolled CSR matvec for the recurrence.

## Tasks

| task | what the brain does | result on the real connectome |
|---|---|---|
| `digits` | read 8x8 handwritten digits | **~93%** accuracy (baseline 10%) |
| `timeseries` | predict a nonlinear series (NARMA-style) | **~2x** better than predict-the-mean |

Write your own: build `(inputs, targets)`, run them through a `Reservoir`, fit a `Readout`. Two objects, one closed-form solve.

## The science

This isn't a metaphor. Every piece is real and cited.

- **The brain:** the complete larval *Drosophila* connectome — [Winding et al., *Science* 2023](https://www.science.org/doi/10.1126/science.add9330), the first whole-brain synaptic wiring diagram of any animal.
- **The method:** connectome-as-reservoir / "Biological Processing Unit" — a fixed connectome core with a trained readout. See [Biological Processing Units (arXiv 2507.10951)](https://arxiv.org/pdf/2507.10951), [The Connectome of a Fly as a Computational Reservoir (ESA ACT)](https://www.esa.int/gsp/ACT/projects/fly_connectome/), and [the Drosophila connectome for time-series prediction (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12109256/).
- **The recurrence:** echo-state networks — a fixed recurrent reservoir plus a linear readout is enough to compute, as long as the spectral radius keeps it in the echo-state regime.

More in [docs/the-science.md](docs/the-science.md).

## Honest about what it is

- **It's the larval brain, not the 139k-neuron adult.** The larval connectome (2952 neurons) is the only *complete* brain wiring of an animal, and it runs on a laptop.
- **The E/I signs are assigned, not measured.** The published all-to-all matrix is unsigned synapse *counts*; a fraction of neurons is assigned an inhibitory sign to give the reservoir usable dynamics. The wiring is the real measured connectome; the sign pattern is a modelling choice, disclosed here.
- **This is reservoir computing, not "the fly thinking".** We use the brain's *wiring* as a fixed dynamical system and read it with an artificial linear layer. The claim is narrower and true: a real connectome, used as a reservoir, computes — and computes well.
- **Weights are synapse counts.** Connection strength = number of synapses between two neurons, straight from the connectome.

## Install and test

```
git clone https://github.com/awdemos/wetware-rs && cd wetware-rs
cargo build --release
cargo test            # offline, on the synthetic sample
cargo run --release -- --sample run digits
cargo run --release --example read_digits
```

## License

MIT. It's a brain in a box — go build something strange.
