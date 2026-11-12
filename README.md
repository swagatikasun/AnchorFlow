# AnchorFlow

Privacy-preserving fund tracing across blockchains and custodial ledgers via column ownership.

AnchorFlow lets custodians (exchanges, swap services, custodial bridges) and an investigator compute haircut exposure over public chains plus their private ledgers. Each custodian applies its own columns of the flow matrix to a masked state; one masked opening per iteration does both the fixed-point truncation and the unmasking.

- Tier H hides even a custodian's own per-hop exposure; its mask corrections are prepared offline with threshold Paillier encryption.
- Tier V lets a custodian see the per-hop exposure of its own accounts and needs no encryption or preprocessing; it reveals which custodian owns each custodial vertex.
- Outputs reach each custodian through masked delivery, so tier H does not reveal which custodian owns which vertex.
- Every party commits to its seed and ledger columns; `protocol.audit` replays a party's messages and reports deviations from the committed inputs. It cannot tell whether a committed ledger is true or complete.
- Use a fresh `seed` for every query: masks and tier-H corrections must never be reused.
- After a ledger update, tier-H corrections can be refreshed column by column (`dynamic.py`).

## Layout

```
anchorflow/
  ring128.py        Z_2^128 vectors (four 32-bit limbs) and overflow-safe sparse products
  paillier.py       Paillier with n-out-of-n threshold decryption (uses gmpy2 if installed)
  gen.py            synthetic multi-chain ecosystem with planted laundering paths
  protocol.py       AnchorFlow online phase (tiers H and V), rounding offset, masked output delivery, commitments and replay audit
  he.py             homomorphic offline phase, HE-propagation baseline, exact operation counts
  dynamic.py        ledger updates and exact refresh of tier-H corrections
  variants.py       uncertain bridge matching and omitted custodial transfers
  dataio.py         CSV import and export of vertices, transfers and labels
  meter.py          bytes, rounds, CPU and modelled LAN/WAN time
  graphiti/         Graphiti-style (2+1) baseline: shuffles, insecure sort, message passing
tests/              unit and end-to-end tests
experiments/run.py  experiment batches (each batch has at most 5 configurations)
experiments/analyze.py  figures, tables and paper/numbers.tex from results/
results/            JSONL results of our runs
paper/              output of analyze.py: figures, tables and numbers.tex (created on first run)
```

## Install

Python 3.10 or newer.

```
git clone <repository-url> anchorflow
cd anchorflow
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install gmpy2
```

`gmpy2` is optional. With it, Paillier operations use GMP and the homomorphic batches run much faster; the reported homomorphic costs come from the unit costs measured on your machine, so re-run the `he-units-*` batches after installing it.

## Tests

```
python -m pytest -q
```

All 21 tests should pass in under a minute.

## Experiments

Each command below is one batch. Run them one at a time, check the printed lines, then go to the next one. Each batch appends to `results/<batch>.jsonl`, so delete a batch's file before re-running it.

To start from a clean state:

```
rm -f results/*.jsonl
```

| Batch | What it measures | Time on a 2-core cloud VM | Memory |
| --- | --- | --- | --- |
| `he-units-2048` | Cost of each Paillier operation, 2048-bit | 10 s | small |
| `he-units-3072` | Same, 3072-bit | 30 s | small |
| `accuracy` | Error and wrong alerts with and without the rounding offset, n = 3 to 16 | 2.5 min | 1 GB |
| `scale-a` | 10k, 30k, 100k vertices, AnchorFlow and Graphiti-style | 35 s | 1 GB |
| `scale-b` | 300k vertices | 1.5 min | 2 GB |
| `scale-c` | 1M vertices | 5 min | 4 GB |
| `scale-d` | 3M vertices, AnchorFlow only | 4 min | 5 GB |
| `parties` | n = 3, 5, 10, 15, 25 at 100k vertices | 4 min | 1 GB |
| `density` | average degree 2 to 12 at 100k vertices | 2.5 min | 2 GB |
| `forensic` | recall of planted endpoints, five graphs | 30 s | 1 GB |
| `uncertain` | best-match vs confidence-weighted bridge matching, five noise levels | 2 min | 1 GB |
| `omission` | random and targeted omission of custodial transfers | 1 min | 1 GB |
| `dynamic` | ledger updates of 100, 1k, 10k transactions | 8 min | 2 GB |
| `audit` | replay audit with and without injected deviations | 10 s | small |
| `anchors-missing` | 0, 5, 10, 20% of links between custodians broken | 1 min | 1 GB |
| `anchors-wrong` | 5, 10, 20% of bridge links matched to a wrong address | 1 min | 1 GB |
| `mixing` | laundering paths split among random addresses | 1 min | 1 GB |
| `timeexp` | size of the time-expanded graph for 1 to 256 windows | 30 s | 1 GB |
| `hops` | fixed-point error for 2 to 20 hops | 4 min | 1 GB |
| `he-validate` | full homomorphic runs vs predicted cost | 10 min (pure Python) | small |

```
python experiments/run.py he-units-2048
python experiments/run.py he-units-3072
python experiments/run.py accuracy
python experiments/run.py scale-a
python experiments/run.py scale-b
python experiments/run.py scale-c
python experiments/run.py scale-d
python experiments/run.py parties
python experiments/run.py density
python experiments/run.py forensic
python experiments/run.py uncertain
python experiments/run.py omission
python experiments/run.py dynamic
python experiments/run.py audit
python experiments/run.py anchors-missing
python experiments/run.py anchors-wrong
python experiments/run.py mixing
python experiments/run.py timeexp
python experiments/run.py hops
python experiments/run.py he-validate
```

Then build figures, tables and the numbers used in the text:

```
python experiments/analyze.py
```

This writes `paper/figs/*.pdf`, `paper/tables/*.tex`, `paper/numbers.tex` and `paper/summary.json`. Pass another directory as the first argument to write elsewhere.

## Using your own data

`anchorflow/dataio.py` reads a folder with three CSV files:

- `vertices.csv`: `id,chain,owner,kind` with ids 0..n-1, `owner` = -1 for public addresses or the custodian index 0..K-1, `kind` 0 public, 1 deposit address, 2 account, 3 swap deposit, 4 virtual anchor
- `transfers.csv`: `src,dst,amount,value_kept,bridge` with amounts in one unit of value, `value_kept` = 1 minus the fee, `bridge` = 1 for matched bridge links
- `labels.csv`: `id,label` with `seed` for theft addresses and `endpoint` for known cash-out addresses (optional)

```
python -c "from anchorflow import dataio, protocol; eco = dataio.load('path/to/incident'); print(protocol.run(eco, 10)['exposure'][:10])"
```

## Notes on the baselines

- The Graphiti-style baseline is our reimplementation of Graphiti's message passing in its cheapest setting (two servers and a helper, 64-bit ring). Its trust model is weaker than AnchorFlow's.
- The HE-propagation baseline follows the design of van Egmond et al. (FC 2024). Its cost for large graphs is projected from exact operation counts and measured unit costs; `he-validate` checks the counts against full runs.
