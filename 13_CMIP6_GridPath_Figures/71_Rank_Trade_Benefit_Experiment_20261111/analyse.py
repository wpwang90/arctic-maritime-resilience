"""Clear paired simulated markets and estimate country-level incidence."""
import argparse
import csv
import gzip
import itertools
import json

import numpy as np

from rank_market import ARMS, DRAWS, HERE, PENALTIES, PERIODS, SEED, SLACKS, buying, selling, src

MODELS = ("qwen3", "qwen25")
CONTRASTS = ("joint", "rank_at_arctic", "rank_at_conventional", "route_component", "rank_component")
EFFECT = np.array([[-1, 0, 0, 1], [0, 0, -1, 1], [-1, 1, 0, 0],
                   [-.5, -.5, .5, .5], [-.5, .5, -.5, .5]])
TOL_PP = .10
METRICS = ("import_change_pp", "export_change_pp", "logistics_saving_pp", "shortfall_reduction_pp",
           "loss_avoided_pp", "loss_avoided_m", "loss_reduction_pct", "supplier_displacement_pct")
FOCUS = ("USA", "RUS", "CHN", "GBR", "DEU")


def load_policy(ship, model, v, arm):
    return [POLICIES[(model, ship, v, arm, role)] for role in ("B", "S")]


def read_logits(markets, model):
    data = {}
    fn = HERE/f"policy_logits_{model}.jsonl.gz"
    provenance = json.loads((HERE/f"policy_provenance_{model}.json").read_text())
    assert src.sha(fn) == provenance["log_sha"]
    with gzip.open(fn, "rt") as f:
        for line in f:
            z = json.loads(line)
            ship, v, arm, role, i = z["key"]
            x = markets[ship]
            n = len(x["bs"] if role == "B" else x["ss"])
            k = model, ship, v, arm, role
            if k not in data:
                data[k] = np.full((n, 4 if role == "B" else 3), -np.inf)
                data[k][:, 0] = 0  # inactive actors have only the default policy
            data[k][i, z["order"]] = z["logits"]
    return data


def select(scores, gumbel, greedy):
    return (scores + (0 if greedy else gumbel)).argmax(axis=-1)[..., None]


def episode(x, arm, buy, sell, slack):
    st = x["states"][arm[0]]
    base, reach, cost = (a[None] for a in (x["base"], st["reach"], st["cost"]))
    prior = np.ones((len(buy), 1, base.shape[1]))
    mean = np.zeros((len(buy), *base.shape[1:]))
    for _ in range(PERIODS):
        orders = buying(base.transpose(0, 2, 1), reach.transpose(0, 2, 1),
                        cost.transpose(0, 2, 1), prior, buy).transpose(0, 2, 1)
        delivered = selling(base, orders, sell, slack)
        np.testing.assert_allclose(delivered.sum(axis=-1), np.minimum(orders.sum(axis=-1),
                                  (1+slack)*base.sum(axis=-1)), atol=1e-6, rtol=0)
        assert (orders <= 2*reach+1e-7).all() and (delivered <= orders+1e-7).all()
        assert (orders.sum(axis=1) <= base.sum(axis=1)+1e-7).all()
        assert (delivered >= 0).all() and (delivered[:, base[0] == 0] == 0).all()
        oo = orders.sum(axis=-1)[:, None, :]
        prior = np.divide(delivered.sum(axis=-1)[:, None, :], oo,
                          out=np.ones_like(prior), where=oo > 1e-9)
        mean += delivered / PERIODS
    return mean


def totals(x, q, arm, codes):
    ref = x["base"]
    d = np.zeros(len(codes))
    e = d.copy()
    dst = [codes.index(k) for k in x["bs"]]
    orig = [codes.index(k) for k in x["ss"]]
    d[dst], e[orig] = ref.sum(0), ref.sum(1)
    m = np.zeros((len(q), len(codes)))
    xport, logistics = m.copy(), m.copy()
    m[:, dst] = q.sum(1)
    xport[:, orig] = q.sum(2)
    logistics[:, dst] = (q*x["states"][arm[0]]["cost"]).sum(1)
    return np.stack((m, xport, logistics), axis=-1), d, e


def normalized(num, den):
    return np.divide(100*num, den, out=np.full_like(num, np.nan), where=den > 1e-9)


def effects(level, d, e, penalty):
    m, x, log = level.transpose(3, 0, 1, 2)
    loss = log+penalty*np.maximum(d-m, 0)
    value = -np.einsum("ka,ard->krd", EFFECT, loss)
    delta_m = np.einsum("ka,ard->krd", EFFECT, m)
    delta_x = np.einsum("ka,ard->krd", EFFECT, x)
    delta_log = -np.einsum("ka,ard->krd", EFFECT, log)
    # Relative loss reduction has a unique baseline for the three simple contrasts only.
    pct = np.full_like(value, np.nan)
    pct[:3] = normalized(value[:3], loss[[0, 2, 0]])
    np.testing.assert_allclose(value[0], value[3]+value[4], rtol=0, atol=1e-7)
    return np.stack((normalized(delta_m, d), normalized(delta_x, e),
                     normalized(delta_log, d), normalized(delta_m, d),
                     normalized(value, d), value, pct), axis=-1)


def qtile(x, p):
    return float(np.quantile(x, p)) if np.isfinite(x).all() else None


def cell_stats(x):
    return dict(mean=float(x.mean()) if np.isfinite(x).all() else None,
                low=qtile(x, .025), high=qtile(x, .975),
                mcse=float(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 and np.isfinite(x).all() else None)


def market_table(x, q, d):
    rows = []
    denom = d.sum()
    for c, a, b in (("joint", 0, 3), ("rank_at_arctic", 2, 3), ("rank_at_conventional", 0, 1),
                    ("routes_at_old_rank", 0, 2)):
        diff = q[b]-q[a]
        # Unmet demand is an explicit outside option for each buyer.
        reallocation = (abs(diff).sum(axis=(1,2))+abs(diff.sum(axis=1)).sum(axis=1))*50/denom
        fills = 100*diff.sum(axis=(1,2))/denom
        rows.append((c, reallocation, fills))
    return rows


def sourcing(x, q, codes, d):
    # Treat unserved demand as an outside option to keep the distribution closed.
    dst = [codes.index(k) for k in x["bs"]]
    result = np.full((len(CONTRASTS), len(q[0]), len(codes)), np.nan)
    for k, (a, b) in enumerate(((0,3), (2,3), (0,1))):
        diff = q[b]-q[a]
        result[k][:, dst] = normalized((abs(diff).sum(axis=1)+abs(diff.sum(axis=1)))/2, d[dst])
    return result


def classify(by_config, tol=TOL_PP):
    if not np.isfinite(by_config).all():
        return "not_covered"
    lo, hi = np.quantile(by_config, (.025, .975), axis=1)
    if min(lo) > tol:
        return "robust_benefit"
    if max(hi) < -tol:
        return "robust_loss"
    return "no_robust_difference"


def simulate(markets, greedy):
    raw, market_rows, setting_rows, comparison, flows = {}, [], [], [], []
    n_draws = 1 if greedy else DRAWS
    for ix, (ship, x) in enumerate(markets.items()):
        codes = sorted(set(x["bs"]+x["ss"]))
        rng = np.random.default_rng(SEED+ix)
        gb = rng.gumbel(size=(n_draws, len(x["bs"]), 4))
        gs = rng.gumbel(size=(n_draws, len(x["ss"]), 3))
        default = (np.ones((1, len(x["bs"]), 1), dtype=int), np.zeros((1, len(x["ss"]), 1), dtype=int))
        noc = [episode(x, a, *default, .10) for a in ARMS]
        np.testing.assert_array_equal(noc[0], noc[1])
        np.testing.assert_array_equal(noc[2], noc[3])
        raw[f"{ship}_benchmark"] = np.stack(noc)
        configs = list(itertools.product(MODELS, range(2)))
        primary, stress, baseline_losses = [], [], []
        for model, v in configs:
            choices = [tuple(select(s, g, greedy) for s, g in zip(load_policy(ship, model, v, a),
                       (gb, gs), strict=True)) for a in ARMS]
            for sl in SLACKS:
                q = [episode(x, a, *ch, sl) for a, ch in zip(ARMS, choices, strict=True)]
                ll = [totals(x, z, a, codes) for a, z in zip(ARMS, q, strict=True)]
                level = np.stack([z[0] for z in ll])
                d, e = ll[0][1:]
                raw[f"{ship}_{model}_{v}_{sl:g}"] = level
                evaluated = [effects(level,d,e,p) for p in PENALTIES]
                stress.append(np.stack([a[...,4].mean(axis=1) for a in evaluated]))
                if sl != SLACKS[0]:
                    continue
                eff = np.concatenate((evaluated[0], sourcing(x,q,codes,d)[..., None]), axis=-1)
                primary.append(eff)
                baseline_losses.append(level[..., 2]+PENALTIES[0]*np.maximum(d-level[..., 0],0))
                for c, rd, fd in market_table(x, q, d):
                    market_rows.append(dict(ship=ship,model=model,variant=v,contrast=c,reference_m=d.sum(),
                                            **{f"displacement_{k}":t for k,t in cell_stats(rd).items()},
                                            **{f"fulfilment_{k}":t for k,t in cell_stats(fd).items()}))
                for i, j in zip(*np.nonzero(x["base"]), strict=True):
                    row = dict(ship=ship,model=model,variant=v,seller=x["ss"][i],buyer=x["bs"][j],
                               reference_m=x["base"][i,j])
                    row.update({a: z[:,i,j].mean() for a,z in zip(ARMS,q,strict=True)})
                    flows.append(row)
                for k,c in enumerate(CONTRASTS):
                    for j,iso in enumerate(codes):
                        row = dict(ship=ship,iso3=iso,model=model,variant=v,contrast=c)
                        for m,name in enumerate(METRICS):
                            row.update({f"{name}_{s}":t for s,t in cell_stats(eff[k,:,j,m]).items()})
                        setting_rows.append(row)
            print(f"{ship} {model} form {v+1}: all capacity settings cleared", flush=True)
        raw[f"{ship}_codes"] = np.array(codes)
        raw[f"{ship}_reference"] = np.stack((d,e))
        raw[f"{ship}_primary_effects"] = np.stack(primary)
        np.testing.assert_allclose(d.sum(), e.sum(), atol=1e-7, rtol=0)
        blocks = np.stack(primary)
        sens = np.stack(stress)
        mean_losses = np.stack(baseline_losses).mean(axis=(0,2))
        for k, contrast in enumerate(CONTRASTS):
            for j, iso in enumerate(codes):
                old, new = x["r"]["C"].get(iso), x["r"]["A"].get(iso)
                row = dict(ship=ship,month=x["meta"]["month"],commodity=x["meta"]["label"],iso3=iso,
                           contrast=contrast,rank_before=old,rank_after=new,
                           rank_gain=old-new if old is not None and new is not None else None,
                           import_reference_m=d[j],export_reference_m=e[j],
                           incidence=classify(blocks[:,k,:,j,4]))
                for m, name in enumerate(METRICS):
                    z = blocks[:,k,:,j,m]
                    row.update({f"{name}_{s}":t for s,t in cell_stats(z.ravel()).items() if s!="mcse"})
                    row[f"{name}_min_config_low"] = min(qtile(a,.025) for a in z) if np.isfinite(z).all() else None
                    row[f"{name}_max_config_high"] = max(qtile(a,.975) for a in z) if np.isfinite(z).all() else None
                    row[f"{name}_mcse"] = float(z.mean(axis=0).std(ddof=1)/np.sqrt(n_draws)) if n_draws>1 and np.isfinite(z).all() else None
                sweep = sens[:,:,k,j]
                row["stress_min_mean_pp"] = float(sweep.min()) if np.isfinite(sweep).all() else None
                row["stress_max_mean_pp"] = float(sweep.max()) if np.isfinite(sweep).all() else None
                row["baseline_loss_m"] = row["comparison_loss_m"] = row["ratio_of_mean_loss_reduction_pct"] = None
                if k < 3 and d[j] > 0:
                    a, b = ((0,3), (2,3), (0,1))[k]
                    old_loss, new_loss = mean_losses[a,j], mean_losses[b,j]
                    row["baseline_loss_m"], row["comparison_loss_m"] = old_loss, new_loss
                    if old_loss > 1e-9:
                        row["ratio_of_mean_loss_reduction_pct"] = 100*(old_loss-new_loss)/old_loss
                comparison.append(row)
    return raw, market_rows, setting_rows, comparison, flows


def output(filename, rows):
    with (HERE/filename).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--greedy", action="store_true", help="Compare argmax policies without sampling")
    a = p.parse_args()
    markets = src.mk_inputs()
    POLICIES = {}
    for model in MODELS:
        POLICIES.update(read_logits(markets,model))
    t, m, s, c, f = simulate(markets,a.greedy)
    suffix = "_argmax" if a.greedy else ""
    np.savez_compressed(HERE/f"paired_country_draws{suffix}.npz", **t)
    for name, rows in zip(("market_effects", "country_by_configuration", "country_benefits", "mean_bilateral_trade"),
                          (m,s,c,f), strict=True):
        output(f"{name}{suffix}.csv", rows)
