"""Paired AI procurement trials with a named supplier as the decision."""
import argparse
import csv
import gzip
import itertools
import json
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PREV = HERE.parent / "71_Rank_Trade_Benefit_Experiment_20261111"
if not (HERE / "market.py").exists():
    sys.path.insert(0, str(PREV))
import market as src
from rank_market import allocate, selling

DRAWS = 384
PERIODS = 3
FEE = .005
SLACK = .10
SHORTFALL = .20
ARMS = ("CC","CA","AC","AA","AO","AP")
CONTRASTS = {"joint": (0, 3), "rank_at_arctic": (2, 3),
             "rank_at_conventional": (0, 1), "routes_at_old_rank": (0, 2),
             "own_rank_at_arctic": (2,4), "partner_rank_at_arctic": (2,5)}


def menus(x, route, budget=.20):
    b = x["base"].T
    cap = 2*x["states"][route]["reach"].T
    q = np.zeros((len(b), 4, b.shape[1]))
    ids = np.full((len(b), 4), -1)
    lot = np.zeros((len(b), 4))
    for j, (hist, limits) in enumerate(zip(b, cap, strict=True)):
        d = hist.sum()
        q[j, 0] = src.cap_allocate(hist, limits, d)
        indices = np.argsort(-hist, kind="stable")
        k = 1
        for i in indices[1:]:
            spare = limits[i]-q[j, 0, i]
            others = q[j, 0].sum()-q[j, 0, i]
            shift = min(budget*d, spare, others)
            if shift < .02*d:
                continue
            alt = q[j, 0]*(1-shift/others)
            alt[i] = q[j, 0, i]+shift
            np.testing.assert_allclose(alt.sum(), q[j, 0].sum(), atol=1e-8, rtol=0)
            assert np.all(alt <= limits+1e-8) and np.all(alt[hist == 0] == 0)
            q[j, k], ids[j, k], lot[j, k] = alt, i, shift
            k += 1
            if k == 4:
                break
    return q, ids, lot


def dashboard(x, state, indices, j):
    def r(iso, updated):
        old = x["r"]["C"].get(iso)
        new = x["r"]["A" if updated else "C"].get(iso)
        if old is None or new is None:
            return "not reported"
        direction = f"up {old-new}" if old > new else f"down {new-old}" if old < new else "unchanged"
        return f"{old} to {new} ({direction})"
    return f"Buyer {r(x['bs'][j], state in 'AO')}; " + "; ".join(
        f"S{i+1} {r(x['ss'][i], state in 'AP')}" for i in indices)


def prompt(x, arm, j, variant, order, menu, candidates, lot):
    b = x["base"][:, j]
    st = x["states"][arm[0]]
    historic = int(b.argmax())
    indices = [historic]+[int(i) for i in candidates[1:] if i >= 0]
    role = ("Select a procurement contract for an anonymous importing economy." if variant == 0 else
            "As an anonymous buyer, decide whether to shift part of your sourcing contract.")
    lines = [role, f"Commodity: {x['meta']['label']}. All suppliers listed are existing trade partners.",
             "Balance delivered imports, shipment cost and supplier concentration. Exporters may ration orders.",
             "Moving a contract costs 0.5% of the amount reassigned once, even if orders are not filled.",
             "Coastal transit rank measures passing traffic (1 is highest); it does not certify quality or access.",
             "Rank dashboard: "+dashboard(x, arm[1], indices, j)]
    for i in indices:
        lines.append(f"S{i+1}: historical order share {100*b[i]/b.sum():.1f}%; "
                     f"reachable {100*st['reach'][i,j]/b[i]:.1f}% of historic trade; "
                     + (f"extra shipment cost {100*st['cost'][i,j]:.2f}%." if st["reach"][i,j] > 0 else
                        "no feasible deliveries."))
    for letter, k in zip(src.LETTERS, order, strict=False):
        q = menu[k]
        sh = q/q.sum()
        name = "Keep proportional orders" if k == 0 else f"Shift {100*lot[k]/b.sum():.1f}% of demand to S{candidates[k]+1}"
        lines.append(f"{letter}. {name}; intended coverage {100*q.sum()/b.sum():.1f}%, "
                     f"mean cost wedge {100*np.dot(sh, st['cost'][:,j]):.2f}%, "
                     f"largest supplier S{int(q.argmax())+1} ({100*sh.max():.1f}%).")
    return "\n".join(lines+["Answer with one letter only."])


def elicit(model, checkpoint, batch):
    score, provenance = src.model_scorer(checkpoint, batch, model)
    expected = json.loads((src.HERE/f"policy_provenance_{model}.json").read_text())
    assert all(provenance[k] == expected[k] for k in ("model","revision","files")), "Checkpoint drift"
    jobs_total, audits = 0, 0
    output = HERE/f"buyer_logits_{model}.jsonl.gz"
    tmp = output.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        for ship, x in src.mk_inputs().items():
            for route, v in itertools.product("CA", range(2)):
                q, ids, lot = menus(x, route)
                jobs = {2: [], 3: [], 4: []}
                for j in range(len(ids)):
                    n = 1+np.count_nonzero(ids[j] >= 0)
                    if n == 1:
                        continue
                    order = src.perm(f"direct-contract/{ship}/{x['bs'][j]}", v, n)
                    states = "CAOP" if route=="A" else "CA"
                    prompts = [prompt(x, route+s, j, v, order, q[j], ids[j], lot[j]) for s in states]
                    assert len(set(re.sub(r"(?m)^Rank dashboard: .*", "", p) for p in prompts)) == 1
                    audits += len(states)-1
                    jobs[n].extend((route+s,j,order,p) for s,p in zip(states,prompts,strict=True))
                for n, group in jobs.items():
                    for (arm,j,order,p), (_, logits, mass) in zip(group, score([g[-1] for g in group], n), strict=True):
                        f.write(json.dumps(dict(ship=ship, variant=v, arm=arm, buyer=x["bs"][j],
                                                order=order, prompt=p, logits=logits, mass=mass))+"\n")
                        jobs_total += 1
            print(f"{model} {ship}: {jobs_total} scored prompts", flush=True)
    tmp.replace(output)
    provenance.update(decisions=jobs_total, rank_pairs_audited=audits, log_sha=src.sha(output))
    (HERE/f"buyer_provenance_{model}.json").write_text(json.dumps(provenance,indent=2)+"\n")


def clear(x, route, q, lot, buy, sell, slack=SLACK, fee=FEE):
    b = x["base"]
    cap = (2*x["states"][route]["reach"]).T[None]
    planned = q[np.arange(len(q))[None,:],buy]
    charge = fee*lot[np.arange(len(q))[None,:],buy]/PERIODS
    prior = np.ones((len(buy),1,len(b)))
    mean = np.zeros_like(planned)
    for _ in range(PERIODS):
        hold = planned*(.5+.5*prior)
        residual = planned.sum(-1,keepdims=True)-hold.sum(-1,keepdims=True)
        additional = allocate(b.T[None]*prior**2, np.maximum(cap-hold,0), residual)
        orders = hold+additional
        sold = selling(b[None], orders.transpose(0,2,1), sell[...,None], slack)
        receipts = sold.transpose(0,2,1)
        np.testing.assert_allclose(sold.sum(-1), np.minimum(orders.sum(1),(1+slack)*b.sum(1)), atol=1e-6, rtol=0)
        assert np.all(receipts >= 0) and np.all(receipts <= orders+1e-7)
        assert np.all(orders <= cap+1e-7) and np.all(receipts[:,b.T==0] == 0)
        assert np.all(orders.sum(-1)<=b.sum(0)+1e-7)
        orig_orders = orders.sum(1,keepdims=True)
        prior = np.divide(receipts.sum(1,keepdims=True),orig_orders,
                          out=np.ones_like(prior),where=orig_orders>1e-9)
        mean += receipts/PERIODS
    return mean, charge


def write_csv(name, rows):
    with (HERE/name).open("w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=src.SPECS, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--batch", type=int, default=20)
    a = p.parse_args()
    elicit(a.model, a.model_dir, a.batch)
