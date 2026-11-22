"""Anonymous buyer/seller agents in a quota-constrained procurement market."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time

import numpy as np

HERE = Path(__file__).resolve().parent
EPS = 1e-9
LETTERS = "ABCD"
PERIODS = 3
ARMS = ("CC", "CA", "AC", "AA")
SPECS = {
    "qwen3": ("mlx-community/Qwen3-4B-Instruct-2507-4bit", "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"),
    "qwen25": ("mlx-community/Qwen2.5-1.5B-Instruct-4bit", "8b403126fc14f14cfc99bb4cfa72ecbc129ea677"),
}


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def records(name):
    with (HERE / name).open(newline="") as f:
        return list(csv.DictReader(f))


def cap_allocate(w, cap, target):
    """Allocate proportionally, redistributing residual after reaching caps."""
    out = np.zeros_like(cap)
    left = min(float(target), float(cap.sum()))
    while left > EPS:
        active = cap-out > EPS
        if not active.any():
            break
        p = np.where(active, w, 0.)
        if p.sum() <= EPS:
            p = active.astype(float)
        move = np.minimum(cap-out, left*p/p.sum())
        out += move
        used = float(move.sum())
        if used <= EPS:
            break
        left -= used
    assert np.all(out >= -EPS) and np.all(out <= cap+EPS)
    assert abs(out.sum()-min(target, cap.sum())) <= 1e-6
    return out


def mk_inputs():
    src, ranks = records("market_inputs.csv"), records("rank_inputs.csv")
    meta = json.loads((HERE / "input_provenance.json").read_text())
    assert meta["market_inputs_sha256"] == sha(HERE / "market_inputs.csv")
    assert meta["rank_inputs_sha256"] == sha(HERE / "rank_inputs.csv")
    out = {}
    for ship in ("container", "bulk", "tanker", "lng"):
        rows = [r for r in src if r["shiptype"] == ship]
        ss = sorted({r["seller"] for r in rows})
        bs = sorted({r["buyer"] for r in rows})
        base = np.zeros((len(ss), len(bs)))
        state = {k: dict(reach=base.copy(), cost=base.copy()) for k in "CA"}
        for r in rows:
            i, j = ss.index(r["seller"]), bs.index(r["buyer"])
            ref, reach = float(r["reference"]), float(r["reachable"])
            assert 0 <= reach <= ref+1e-7 and ref > 0
            assert base[i, j] == 0 or abs(base[i, j]-ref) < 1e-7
            base[i, j] = ref
            a = "C" if r["state"] == "without_arctic" else "A"
            state[a]["reach"][i, j] = reach
            state[a]["cost"][i, j] = float(r["wedge_weighted"])/reach if reach > 0 else 0
        for x in state.values():
            assert np.isfinite(x["cost"]).all() and (x["cost"]>=0).all()
        rr = [r for r in ranks if r["shiptype"] == ship]
        assert len(rr) == 50 and {int(r["month"]) for r in rr} == {meta["markets"][ship]["month"]}
        assert all(r["reference"] == "same_closure" and r["radius_km"] == "100" and r["step"] == "5" for r in rr)
        out[ship] = dict(meta=meta["markets"][ship], ss=ss, bs=bs, base=base, states=state,
                         r={"C": {r["iso3"]: int(r["conventional_rank"]) for r in rr},
                            "A": {r["iso3"]: int(r["arctic_rank"]) for r in rr}})
    return out


def buy_menu(base, cap, cost, fill):
    target = base.sum()
    incumbent = base/target
    low_cost = np.exp(-cost/.02)
    hold = np.minimum(.70*base, cap)
    remaining = cap-hold
    return [hold+cap_allocate(w, remaining, target-hold.sum()) for w in (
        incumbent, incumbent*low_cost, np.sqrt(incumbent),
        incumbent*low_cost*(.10+fill)**4)]


def sell_menu(orders, base, capacity):
    return [cap_allocate(w, orders, capacity) for w in (orders, base, np.sqrt(base))]


def perm(k, variant, n):
    order = list(range(n))
    random.Random("S24-20261111:"+k).shuffle(order)
    return order if variant == 0 else order[::-1]


def node_info(ids, ranks, shares, prefix):
    names = []
    take = [int(j) for j in np.argsort(-shares, kind="stable")[:3] if shares[j] > EPS]
    for j in take:
        names.append(f"{prefix}{j+1} {shares[j]*100:.0f}% (rank {ranks.get(ids[j], 'NA')})")
    rest = float(shares.sum())-sum(shares[j] for j in take)
    return ", ".join(names)+(f", others {rest*100:.0f}%" if rest > .005 else "") if names else "no feasible order"


def buy_prompt(x, j, choices, order, wedge, prior, ranks, period, variant):
    d = x["base"][:, j].sum()
    descriptions = ("historical sourcing", "lower freight and delay cost", "more diversified sourcing", "more reliable supply")
    role = ("Select one import portfolio for an anonymous economy." if variant == 0 else
            "You manage procurement for an anonymous coastal economy; choose one portfolio.")
    info = [f"{role} Period {period}. Commodity group: {x['meta']['label']}. "
            "Prioritise filling demand, then weigh lower cost, diversification and established partners. "
            "Fulfilment estimates use last period's supplier fills (initially full), not a guarantee. "
            "Coastal transit rank 1 is highest; rank is not supplier quality or capacity. NA is unavailable. "
            f"Your coastal transit rank: {ranks.get(x['bs'][j], 'NA')}. One letter only."]
    for label, k in zip(LETTERS, order):
        y = choices[k]
        p = y/max(EPS, y.sum())
        forecast = np.dot(y, prior)/d*100
        cost = np.dot(y, wedge)/max(EPS, y.sum())*100
        info.append(f"{label}. {descriptions[k]}: expected fill {forecast:.1f}%, cost wedge {cost:.2f}%, "
                    f"concentration {np.dot(p,p):.2f}; order shares: {node_info(x['ss'], ranks, y/d, 'S')}.")
    return "\n".join(info)


def sell_prompt(x, i, choices, order, orders, ranks, period, variant):
    desc = ("proportional to current orders", "protect historic buyer shares", "flatten historic buyer shares")
    role = ("Choose one shipment allocation for an anonymous exporter." if variant == 0 else
            "You allocate exports for an anonymous coastal economy. Pick one plan.")
    info = [f"{role} Period {period}. All plans deliver the same total volume and revenue. "
            "Weigh current customer fill, retention of existing buyers and diversification. "
            "Coastal transit rank 1 is highest; it is not buyer quality or capacity. NA is unavailable. "
            f"Your coastal transit rank: {ranks.get(x['ss'][i], 'NA')}. One letter only."]
    base = x["base"][i]
    for label, k in zip(LETTERS, order):
        q = choices[k]
        p = q/max(EPS, q.sum())
        gap = abs(p-base/base.sum()).sum()/2
        fill = q[orders > EPS]/orders[orders > EPS]
        info.append(f"{label}. {desc[k]}: lowest customer fill {100*fill.min():.1f}%, "
                    f"historical share deviation {100*gap:.1f}%, concentration {np.dot(p,p):.2f}; "
                    f"delivered shares: {node_info(x['bs'], ranks, p, 'B')}.")
    return "\n".join(info)


def model_scorer(path, batch_size, model_key):
    import mlx.core as mx
    from mlx_lm import load
    model, tok = load(str(path))
    token = [tok.encode(s, add_special_tokens=False) for s in LETTERS]
    assert all(len(t) == 1 for t in token)
    ids = mx.array([t[0] for t in token])
    cache = {}

    def score(prompts, n):
        missing = list(dict.fromkeys(p for p in prompts if (p, n) not in cache))
        for k in range(0, len(missing), batch_size):
            pp = missing[k:k+batch_size]
            enc = []
            for p in pp:
                full = tok.apply_chat_template([{"role":"user", "content":p}],
                                               tokenize=False, add_generation_prompt=True)
                t = tok.encode(full, add_special_tokens=False)
                assert all(tok.encode(full+s, add_special_tokens=False)==t+l for s,l in zip(LETTERS[:n],token[:n]))
                enc.append(t)
            lens = [len(t) for t in enc]
            padded = mx.array([t+[0]*(max(lens)-len(t)) for t in enc])
            hidden = model.model(padded)[mx.arange(len(enc)), mx.array(lens)-1]
            logits = (model.model.embed_tokens.as_linear(hidden) if model.args.tie_word_embeddings
                      else model.lm_head(hidden)).astype(mx.float32)
            allowed = logits[:, ids[:n]]
            mass = mx.exp(mx.logsumexp(allowed, axis=-1)-mx.logsumexp(logits, axis=-1))
            mx.eval(allowed, mass)
            for p, a, m in zip(pp, np.asarray(allowed), np.asarray(mass)):
                cache[(p, n)] = (int(a.argmax()), a.tolist(), float(m))
        return [cache[(p,n)] for p in prompts]

    provenance = {"model": SPECS[model_key][0], "revision": SPECS[model_key][1],
                  "files": {p.name: sha(p) for p in sorted(path.iterdir()) if p.is_file()},
                  "versions": {n: importlib.metadata.version(n) for n in ("mlx","mlx-lm","transformers","numpy")}}
    return score, provenance


def fixtures(score):
    """Balanced labels for checks that have a unique correct answer."""
    pp, labels = [], []
    for i, (a,b) in enumerate([(29,2),(1,7),(5,5),(62,15),(8,47),(31,31),
                               (44,20),(15,34),(7,7),(30,5),(5,50),(16,16)]):
        options = ("improved","worsened","unchanged")
        order = perm(f"check{i}", i%2, 3)
        label = order.index(0 if b<a else 1 if b>a else 2)
        pp.append(f"Rank 1 is best. A country's rank goes from {a} to {b}. How did its rank change? "
                  +" ".join(f"{l}. {options[k]}." for l,k in zip(LETTERS,order))+" Answer with one letter.")
        labels.append(label)
    for i in range(12):
        options = ("fill 97%, cost 1%","fill 90%, cost 2%","fill 95%, cost 4%")
        order = perm(f"dominated{i}", i%2, 3)
        pp.append("Choose the portfolio with both the highest demand fill and the lowest cost. "
                  +" ".join(f"{l}. {options[k]}." for l,k in zip(LETTERS,order))+" Answer with one letter.")
        labels.append(order.index(0))
    return [dict(test="rank" if i<12 else "dominance", prompt=p, correct=LETTERS[l],
                 chosen=LETTERS[o[0]], logits=o[1], allowed_mass=o[2])
            for i,(p,l,o) in enumerate(zip(pp,labels,score(pp,3)))]


def run_arm(x, ship, arm, variant, score, writer, slack):
    base = x["base"]
    s = x["states"][arm[0]]
    rank = x["r"][arm[1]]
    k = 2*s["reach"]
    cap = (1+slack)*base.sum(axis=1)
    prior = np.ones(len(cap))
    out = []
    counters = defaultdict(int)
    for period in range(1, PERIODS+1):
        orders = np.zeros_like(base)
        executions = np.zeros_like(base)
        buyer_jobs = []
        for j, buyer in enumerate(x["bs"]):
            menu = buy_menu(base[:,j], k[:,j], s["cost"][:,j], prior)
            if all(np.max(abs(m-menu[0]))<EPS for m in menu[1:]):
                orders[:,j] = menu[0]
                continue
            if score is None:
                def objective(y):
                    p = y/max(EPS,y.sum())
                    demand = base[:,j].sum()
                    return (np.dot(y,s["cost"][:,j])+.20*(demand-np.dot(y,prior)))/demand+.01*np.dot(p,p)
                orders[:,j] = min(menu, key=objective)
                continue
            order = perm(f"{ship}/buyer/{buyer}",variant,4)
            brief = buy_prompt(x,j,menu,order,s["cost"][:,j],prior,rank,period,variant)
            buyer_jobs.append((j,menu,order,brief))
        if buyer_jobs:
            for (j,menu,order,brief),(ix,logits,mass) in zip(buyer_jobs,score([q[-1] for q in buyer_jobs],4)):
                orders[:,j] = menu[order[ix]]
                writer.write(json.dumps(dict(ship=ship,arm=arm,variant=variant,period=period,
                    actor=x["bs"][j], role="buyer", prompt=brief, option_order=order,
                    letter=LETTERS[ix], logits=logits, allowed_mass=mass))+"\n")
            counters["buyer"] += len(buyer_jobs)
        seller_jobs = []
        for i,seller in enumerate(x["ss"]):
            if orders[i].sum() <= cap[i]+EPS:
                executions[i] = orders[i]
                continue
            menu = sell_menu(orders[i],base[i],cap[i])
            if score is None or all(np.max(abs(m-menu[0]))<EPS for m in menu[1:]):
                executions[i] = menu[0]
                continue
            order = perm(f"{ship}/seller/{seller}",variant,3)
            brief = sell_prompt(x,i,menu,order,orders[i],rank,period,variant)
            seller_jobs.append((i,menu,order,brief))
        if seller_jobs:
            for (i,menu,order,brief),(ix,logits,mass) in zip(seller_jobs,score([q[-1] for q in seller_jobs],3)):
                executions[i] = menu[order[ix]]
                writer.write(json.dumps(dict(ship=ship,arm=arm,variant=variant,period=period,
                    actor=x["ss"][i],role="seller",prompt=brief,option_order=order,
                    letter=LETTERS[ix],logits=logits,allowed_mass=mass))+"\n")
            counters["seller"] += len(seller_jobs)
        np.testing.assert_allclose(executions.sum(1),np.minimum(cap,orders.sum(1)),rtol=0,atol=1e-6)
        assert np.all(orders>=-EPS) and np.all(executions>=-EPS)
        assert np.all(orders <= k+EPS) and np.all(executions<=orders+EPS)
        assert np.all(orders.sum(0)<=base.sum(0)+1e-6)
        prior = np.divide(executions.sum(1),orders.sum(1),out=np.ones_like(prior),where=orders.sum(1)>EPS)
        out.append((orders,executions))
    return out,counters


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",choices=(*SPECS,"rule"),required=True)
    p.add_argument("--model-dir",type=Path)
    p.add_argument("--batch",type=int,default=16)
    p.add_argument("--slack",type=float,default=.10)
    args=p.parse_args()
    markets=mk_inputs()
    score,provenance=None,{"model":"rank-blind rule"}
    if args.model!="rule":
        assert args.model_dir and args.model_dir.is_dir()
        score,provenance=model_scorer(args.model_dir,args.batch,args.model)
        provenance["fixtures"]=fixtures(score)
        print("Fixtures:",{k:sum(t["correct"]==t["chosen"] for t in provenance["fixtures"] if t["test"]==k)
                           for k in ("rank","dominance")},flush=True)
    suffix=args.model if args.slack==.10 else f"{args.model}_{args.slack:g}"
    start=time.monotonic()
    result={}
    counts=defaultdict(int)
    with gzip.open(HERE/f"decisions_{suffix}.jsonl.gz","wt",encoding="utf-8") as writer:
        for ship,x in markets.items():
            for variant in range(2 if score else 1):
                for arm in ARMS:
                    series,counter=run_arm(x,ship,arm,variant,score,writer,args.slack)
                    key=f"{ship}_{variant}_{arm}"
                    result[key+"_orders"]=np.stack([z[0] for z in series])
                    result[key+"_executions"]=np.stack([z[1] for z in series])
                    for k,v in counter.items():
                        counts[k]+=v
                print(f"{ship} variant {variant}: {dict(counts)}, {time.monotonic()-start:.1f}s",flush=True)
    np.savez_compressed(HERE/f"outcomes_{suffix}.npz",**result)
    provenance.update(status="COMPLETE",slack=args.slack,periods=PERIODS,arms=ARMS,counts=dict(counts),
                      input_sha=sha(HERE/"market_inputs.csv"),rank_sha=sha(HERE/"rank_inputs.csv"),
                      log_sha=sha(HERE/f"decisions_{suffix}.jsonl.gz"),
                      outcomes_sha=sha(HERE/f"outcomes_{suffix}.npz"),seconds=time.monotonic()-start)
    (HERE/f"run_{suffix}.json").write_text(json.dumps(provenance,indent=2))


if __name__=="__main__":
    main()
