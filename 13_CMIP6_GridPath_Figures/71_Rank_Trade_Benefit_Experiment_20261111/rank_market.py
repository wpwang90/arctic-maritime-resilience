"""Matched-rank information experiment with interacting procurement policies."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import re
import time

import numpy as np

import market as src

HERE = Path(__file__).resolve().parent
DRAWS = 384
PERIODS = 3
ARMS = ("CC", "CA", "AC", "AA")
SLACKS = (0.10, 0.00, 0.25)
PENALTIES = (0.20, 0.10, 0.40)
SEED = 24731


def policies(role):
    if role == "B":
        return ("Keep historical sourcing shares", "Prefer lower disruption costs",
                "Diversify across existing suppliers", "Prefer accessible, historically reliable suppliers")
    return ("Ration proportionally to submitted orders", "Prioritise historic customers",
            "Distribute across more existing customers")


def rank_line(ids, x, state, prefix):
    parts = []
    for j, iso in ids:
        old = x["r"]["C"].get(iso)
        new = x["r"][state].get(iso)
        if old is None or new is None:
            status = "not reported" if old is None else f"{old} to not reported"
        else:
            step = old-new
            action = f"up {step}" if step > 0 else f"down {-step}" if step < 0 else "unchanged"
            status = f"{old} to {new} ({action})"
        parts.append(f"{prefix}{j}: {status}")
    return "; ".join(parts)


def prompt(x, actor, arm, role, order, variant, preview):
    base, st = x["base"], x["states"][arm[0]]
    buyer = role == "B"
    own = x["bs" if buyer else "ss"][actor]
    ids = x["ss" if buyer else "bs"]
    hist = base[:, actor] if buyer else base[actor]
    indices = [i for i in np.argsort(-hist, kind="stable")[:4] if hist[i] > 0]
    prefix = "S" if buyer else "B"
    role_text = (("You choose a procurement policy for an anonymous economy.",
                  "As the buyer's decision agent, select a standing import plan.")[variant]
                 if buyer else ("You choose a rationing policy for an anonymous exporting economy.",
                                "As the seller's decision agent, select a standing allocation plan.")[variant])
    goal = ("Weigh demand fulfilled, additional logistics costs and dependence on a single supplier."
            if buyer else "Weigh shipment volume, retaining historic customers and spreading shortages.")
    partner_text = []
    for i in indices:
        if buyer:
            reachable, wedge = st["reach"][i, actor], st["cost"][i, actor]
        else:
            reachable, wedge = st["reach"][actor, i], st["cost"][actor, i]
        partner_text.append(f"{prefix}{i+1}: historic share {100*hist[i]/hist.sum():.1f}%, "
                            f"reachable {100*reachable/hist[i]:.1f}%, extra cost {100*wedge:.2f}%"
                            if reachable > 0 else f"{prefix}{i+1}: historic share "
                            f"{100*hist[i]/hist.sum():.1f}%, unreachable")
    rank = rank_line([("", own)], x, arm[1], "own")+ "; "+rank_line(
        [(i+1, ids[i]) for i in indices], x, arm[1], prefix)
    lines = [role_text, f"Commodity: {x['meta']['label']}. {goal}",
             "The policy applies for three trading periods. The market clears subject to existing trade ties and capacity.",
             "Partner records: " + "; ".join(partner_text),
             "Coastal transit rank measures traffic passing near a country's coast; 1 is the highest rank.",
             "Rank dashboard (reference to displayed): " + rank,
             "Illustrative first-period portfolios before full market clearing:" if buyer else
             "Illustrative plans under the same hypothetical demand: historic orders multiplied by 1.25, subject to route limits."]
    for letter, k in zip(src.LETTERS, order):
        q = preview[k]
        share = q/max(q.sum(), src.EPS)
        total = 100*q.sum()/hist.sum()
        hhi = np.dot(share, share)
        if buyer:
            cost = 100*np.dot(share, st["cost"][:, actor])
            outcome = f"orders/reference demand {total:.1f}%, extra cost {cost:.2f}%, concentration {hhi:.2f}"
        else:
            gap = 50*abs(share-hist/hist.sum()).sum()
            outcome = f"shipments/reference exports {total:.1f}%, historic share departure {gap:.1f} pp, concentration {hhi:.2f}"
        lines.append(f"{letter}. {policies(role)[k]}. {outcome}.")
    return "\n".join(lines+["Answer with one letter only."])


def allocate(weight, cap, target):
    """Batched proportional rationing; shape (draw, seller, buyer)."""
    desired = np.minimum(target, cap.sum(axis=-1, keepdims=True))
    result = np.zeros_like(cap)
    for _ in range(cap.shape[-1]+2):
        left = desired-result.sum(axis=-1, keepdims=True)
        if left.max() <= 1e-9:
            break
        spare = np.maximum(cap-result, 0)
        active = spare > 1e-12
        w = np.where(active, weight, 0)
        w = np.where(w.sum(axis=-1, keepdims=True) > 1e-12, w, active)
        result += np.minimum(spare, np.maximum(left, 0)*w/np.maximum(1e-30, w.sum(axis=-1, keepdims=True)))
    assert np.max(abs(result.sum(axis=-1, keepdims=True)-desired)) < 1e-6
    return result


def buying(base, reach, cost, prior, choice):
    # Inputs here are transposed: (draw, buyer, seller).
    cap = 2*reach
    hold = np.minimum(.70*base, cap)
    remaining = np.maximum(cap-hold, 0)
    decay = np.exp(-cost/.02)
    availability = np.divide(reach, base, out=np.zeros_like(reach), where=base > 0)
    weight = np.broadcast_to(base, np.broadcast_shapes(base.shape, prior.shape)).copy()
    weight *= np.where(choice == 1, decay, 1)
    weight = np.where(choice == 2, np.sqrt(base), weight)
    weight *= np.where(choice == 3, availability*(.10+prior)**4, 1)
    return hold+allocate(weight, np.broadcast_to(remaining, weight.shape),
                         base.sum(axis=-1, keepdims=True)-hold.sum(axis=-1, keepdims=True))


def selling(base, order, choice, slack):
    weight = np.where(choice == 0, order, np.where(choice == 1, base, np.sqrt(base)))
    return allocate(weight, order, (1+slack)*base.sum(axis=-1, keepdims=True))


def make_jobs(x, ship, arm, variant, role):
    n = len(policies(role))
    buyer = role == "B"
    b, s = x["base"], x["states"][arm[0]]
    for a, iso in enumerate(x["bs" if buyer else "ss"]):
        if buyer:
            # A single existing supplier leaves the four strategies identical.
            if np.count_nonzero(b[:, a]) <= 1:
                continue
            args = tuple(t[:, a][None, None, :] for t in (b, s["reach"], s["cost"]))
            preview = [buying(*args, np.ones_like(args[0]), np.array([[[k]]]))[0, 0] for k in range(n)]
        else:
            if np.count_nonzero(b[a]) <= 1:
                continue
            order = np.minimum(1.25*b[a], 2*s["reach"][a])[None, None, :]
            preview = [selling(b[a][None, None, :], order, np.array([[[k]]]), .10)[0, 0] for k in range(n)]
        labels = src.perm(f"S24b/{ship}/{role}/{iso}", variant, n)
        yield a, labels, prompt(x, a, arm, role, labels, variant, preview)


def collect(model, model_dir, batch):
    out = {}
    score, provenance = src.model_scorer(model_dir, batch, model)
    markets = src.mk_inputs()
    started = time.monotonic()
    count = 0
    output = HERE/f"policy_logits_{model}.jsonl.gz"
    tmp = output.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        for ship, x in markets.items():
            for variant in range(2):
                for role in ("B", "S"):
                    n = len(policies(role))
                    jobs = [(arm, i, order, p) for arm in ARMS
                            for i, order, p in make_jobs(x, ship, arm, variant, role)]
                    for (arm, i, order, p), (_, logits, mass) in zip(jobs, score([j[3] for j in jobs], n), strict=True):
                        key = ship, variant, arm, role, i
                        out[key] = (p, order)
                        f.write(json.dumps(dict(key=key, prompt=p, order=order, logits=logits, mass=mass))+"\n")
                    count += len(jobs)
                print(f"{model} {ship} presentation {variant+1}/2: {count} policies; "
                      f"{time.monotonic()-started:.1f}s", flush=True)
    checked = 0
    for (ship, v, arm, role, i), (p, order) in out.items():
        if arm[1] == "C":
            other, other_order = out[(ship, v, arm[0]+"A", role, i)]
            assert order == other_order
            assert re.sub(r"(?m)^Rank dashboard .*$", "", p) == re.sub(r"(?m)^Rank dashboard .*$", "", other)
            checked += 1
    tmp.replace(output)
    provenance.update(decisions=count, rank_only_prompt_pairs_audited=checked, log_sha=src.sha(output))
    (HERE/f"policy_provenance_{model}.json").write_text(json.dumps(provenance, indent=2)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=src.SPECS, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--batch", type=int, default=20)
    a = p.parse_args()
    collect(a.model, a.model_dir, a.batch)
