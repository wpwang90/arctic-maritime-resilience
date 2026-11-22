"""Measure contract changes, realised substitution and country incidence."""
from collections import Counter
import gzip
import itertools
import json

import numpy as np

from switch_market import (ARMS, CONTRASTS, DRAWS, FEE, HERE, PERIODS, SHORTFALL, SLACK,
                           clear, menus, src, write_csv)

MODELS = ("qwen3","qwen25")
CONFIGS = list(itertools.product(MODELS,range(2)))
SETTINGS = {"primary": (SLACK,FEE,SHORTFALL), "no_export_spare": (0,FEE,SHORTFALL),
            "high_export_spare": (.25,FEE,SHORTFALL), "lower_shortfall": (SLACK,FEE,.10),
            "higher_shortfall": (SLACK,FEE,.40), "high_fee": (SLACK,.02,SHORTFALL),
            "modal": (SLACK,FEE,SHORTFALL)}
TOL = .10


def logits(file, manifest):
    assert src.sha(file) == json.loads(manifest.read_text())["log_sha"]
    with gzip.open(file,"rt") as f:
        return [json.loads(l) for l in f]


def load(x, ship, model, v, arm, menus_, buyers, sellers):
    q, ids, lot = menus_[arm[0]]
    b = np.full((len(q),4),-np.inf)
    b[:,0] = 0
    s = np.full((len(x["ss"]),3),-np.inf)
    s[:,0] = 0
    found = set()
    for z in buyers[model]:
        if (z["ship"],z["variant"],z["arm"]) == (ship,v,arm):
            j = x["bs"].index(z["buyer"])
            n = np.count_nonzero(ids[j]>=0)+1
            assert j not in found and set(z["order"]) == set(range(n))
            b[j,z["order"]] = z["logits"]
            found.add(j)
    assert found == set(np.nonzero((ids>=0).any(axis=1))[0])
    found = set()
    for z in sellers[model]:
        k = z["key"]
        # Exporters respond to orders but do not receive the buyer's rank treatment.
        if k[:4] == [ship,v,arm[0]+"C","S"]:
            j = k[4]
            assert j not in found
            s[j,z["order"]] = z["logits"]
            found.add(j)
    assert found == set(np.nonzero((x["base"]>0).sum(axis=1)>1)[0])
    return b,s


def evaluate(x, a, b, qa, qb, ca, cb, contract_a, contract_b, penalty):
    ref = x["base"]
    d, e = ref.sum(0), ref.sum(1)
    gain = qb-qa
    volume = np.minimum(np.maximum(gain,0).sum(-1),np.maximum(-gain,0).sum(-1))
    lead_a, lead_b = qa.argmax(-1), qb.argmax(-1)
    new_lead = np.take_along_axis(qb,lead_b[...,None],axis=-1)[...,0]
    old_lead = np.take_along_axis(qb,lead_a[...,None],axis=-1)[...,0]
    switched = ((lead_a!=lead_b) & (qa.sum(-1)>1e-9) & (qb.sum(-1)>1e-9)
                & (new_lead-old_lead>=.01*d))
    fill_a, fill_b = qa.sum(-1), qb.sum(-1)
    log_a = (qa*x["states"][ARMS[a][0]]["cost"].T).sum(-1)
    log_b = (qb*x["states"][ARMS[b][0]]["cost"].T).sum(-1)
    loss_a = log_a+ca+penalty*np.maximum(d-fill_a,0)
    loss_b = log_b+cb+penalty*np.maximum(d-fill_b,0)
    return dict(volume=volume, switched=switched, lead_a=lead_a, lead_b=lead_b,
                proposal=contract_a!=contract_b, fill=fill_b-fill_a, exports=gain.sum(1),
                loss_a=loss_a, avoided=loss_a-loss_b, logistics=log_a-log_b, fee=cb-ca,
                redistribution=(abs(gain).sum(-1)+abs(fill_b-fill_a))/2)


def interval(m):
    q = np.quantile(m,(.025,.975),axis=-1)
    return m.mean(),q[0].min(),q[1].max()


def simulate():
    models = {m:logits(HERE/f"buyer_logits_{m}.jsonl.gz",HERE/f"buyer_provenance_{m}.json") for m in MODELS}
    suppliers = {m:logits(src.HERE/f"policy_logits_{m}.jsonl.gz",src.HERE/f"policy_provenance_{m}.json") for m in MODELS}
    market_rows,countries,transitions,flow_rows,sensitivity = [],[],[],[],[]
    for ix,(ship,x) in enumerate(src.mk_inputs().items()):
        d,e = x["base"].sum(0),x["base"].sum(1)
        D = d.sum()
        menus_ = {r: menus(x,r) for r in "CA"}
        rng = np.random.default_rng(80421+ix)
        gb = rng.gumbel(size=(DRAWS,len(d),4))
        gs = rng.gumbel(size=(DRAWS,len(e),3))
        sets = {k:{c:[] for c in CONTRASTS} for k in SETTINGS}
        # A rank-blind benchmark leaves both fixed-route rank contrasts exactly zero.
        for r,(q,ids,lot) in menus_.items():
            costs = (q*x["states"][r]["cost"].T[:,None,:]).sum(-1)+FEE/PERIODS*lot
            default = np.where((ids>=0) | (np.arange(4)==0),costs,np.inf).argmin(-1)[None]
            zeros = np.zeros((1,len(e)),dtype=int)
            q1,c1 = clear(x,r,q,lot,default,zeros)
            q2,c2 = clear(x,r,q,lot,default,zeros)
            np.testing.assert_array_equal(q1,q2)
            np.testing.assert_array_equal(c1,c2)
        for model,v in CONFIGS:
            inputs = [load(x,ship,model,v,arm,menus_,models,suppliers) for arm in ARMS]
            np.testing.assert_array_equal(inputs[0][1],inputs[1][1])
            for k in (3,4,5):
                np.testing.assert_array_equal(inputs[2][1],inputs[k][1])
            cache = {}
            for setting,(slack,fee,penalty) in SETTINGS.items():
                key = (slack,fee,setting=="modal")
                # Accounting sensitivity uses identical decisions and physical clearing.
                if key not in cache:
                    flows,charges,contracts = [],[],[]
                    for arm,(B,S) in zip(ARMS,inputs,strict=True):
                        buy = (B[None] if setting=="modal" else B+gb).argmax(-1)
                        sell = (S[None] if setting=="modal" else S+gs).argmax(-1)
                        q,ids,lot = menus_[arm[0]]
                        Q,C = clear(x,arm[0],q,lot,buy,sell,slack,fee)
                        flows.append(Q)
                        charges.append(C)
                        contracts.append(ids[np.arange(len(d))[None,:],buy])
                    cache[key] = flows,charges,contracts
                flows,charges,contracts = cache[key]
                if setting == "primary":
                    for i,j in zip(*np.nonzero(x["base"]),strict=True):
                        flow_rows.append(dict(ship=ship,model=model,presentation=v,buyer=x["bs"][j],
                            supplier=x["ss"][i],reference_m=x["base"][i,j],
                            **{arm:Q[:,j,i].mean() for arm,Q in zip(ARMS,flows,strict=True)}))
                for c,(a,b) in CONTRASTS.items():
                    m = evaluate(x,a,b,flows[a],flows[b],charges[a],charges[b],
                                 contracts[a],contracts[b],penalty)
                    market = dict(lead_switch_pct=m["switched"].mean(1)*100,
                                  proposal_change_pct=m["proposal"].mean(1)*100,
                                  transferred_pct=m["volume"].sum(1)*100/D,
                                  demand_redistribution_pct=m["redistribution"].sum(1)*100/D,
                                  fill_change_pp=m["fill"].sum(1)*100/D,
                                  import_avoided_m=m["avoided"].sum(1))
                    row = dict(ship=ship,market=x["meta"]["label"],model=model,presentation=v,
                               setting=setting,contrast=c,buyers=len(d),
                               buyers_with_contract_option_C=int((menus_["C"][1]>=0).any(1).sum()),
                               buyers_with_contract_option_A=int((menus_["A"][1]>=0).any(1).sum()))
                    for k,z in market.items():
                        row.update({f"{k}_mean":z.mean(),f"{k}_low":np.quantile(z,.025),f"{k}_high":np.quantile(z,.975)})
                    market_rows.append(row)
                    sets[setting][c].append(m["avoided"].mean(0)*100/d)
                    if setting != "primary":
                        continue
                    for j,iso in enumerate(x["bs"]):
                        flag = m["switched"][:,j]
                        counts = Counter(zip(m["lead_a"][flag,j],m["lead_b"][flag,j],strict=True))
                        for (i,k),n in sorted(counts.items()):
                            transitions.append(dict(ship=ship,contrast=c,model=model,presentation=v,
                                buyer=iso,from_supplier=x["ss"][i],to_supplier=x["ss"][k],count=n,draws=DRAWS))
                    # Save only the sufficient statistics from the primary runs.
                    sets.setdefault("details",{}).setdefault(c,[]).append({k:z for k,z in m.items()
                        if k not in ("lead_a","lead_b","redistribution")})
            print(f"{ship} {model} presentation {v+1}: settlement audited",flush=True)
        for setting in SETTINGS:
            for c in CONTRASTS:
                for (model,v),effect in zip(CONFIGS,sets[setting][c],strict=True):
                    sensitivity.extend(dict(ship=ship,contrast=c,model=model,presentation=v,setting=setting,
                                            iso3=iso,avoided_pp=effect[j]) for j,iso in enumerate(x["bs"]))
        for c in CONTRASTS:
            block = {k:np.stack([z[k] for z in sets["details"][c]]) for k in sets["details"][c][0]}
            for iso in sorted(set(x["bs"]+x["ss"])):
                old,new = (x["r"][r].get(iso) for r in "CA")
                row = dict(ship=ship,commodity=x["meta"]["label"],month=x["meta"]["month"],contrast=c,
                           iso3=iso,rank_before=old,rank_after=new)
                if iso in x["bs"]:
                    j = x["bs"].index(iso)
                    z = block["avoided"][:,:,j]
                    mean,lo,hi = interval(100*z/d[j])
                    baseline = block["loss_a"][:,:,j].mean()
                    row.update(import_reference_m=d[j],baseline_loss_m=baseline,avoided_m=z.mean(),
                               avoided_pp=mean,avoided_pp_lo=lo,avoided_pp_hi=hi,
                               loss_reduction_pct=100*z.mean()/baseline if baseline>1e-9 else None,
                               incidence="robust_benefit" if lo>TOL else "robust_loss" if hi<-TOL else "not_robust",
                               fill_change_pp=100*block["fill"][:,:,j].mean()/d[j],
                               logistics_avoided_m=block["logistics"][:,:,j].mean(),
                               contract_cost_change_m=block["fee"][:,:,j].mean(),
                               contract_change_pct=100*block["proposal"][:,:,j].mean(),
                               lead_switch_pct=100*block["switched"][:,:,j].mean(),
                               shifted_between_suppliers_pct=100*block["volume"][:,:,j].mean()/d[j],
                               benefit_positive_pct=100*(z>1e-9).mean(),worst_setting_pp=min(
                                   s[j] for setting in SETTINGS for s in sets[setting][c]),
                               best_setting_pp=max(s[j] for setting in SETTINGS for s in sets[setting][c]))
                if iso in x["ss"]:
                    i = x["ss"].index(iso)
                    z = block["exports"][:,:,i]
                    row.update(export_reference_m=e[i],export_change_m=z.mean(),export_change_pp=100*z.mean()/e[i])
                countries.append(row)
    fields = list(dict.fromkeys(k for r in countries for k in r))
    countries = [{k:r.get(k) for k in fields} for r in countries]
    for fn,rows in (("country_switch_benefits",countries),("market_switch_effects",market_rows),
                    ("leading_supplier_transitions",transitions),("mean_bilateral_deliveries",flow_rows),
                    ("country_benefit_sensitivity",sensitivity)):
        write_csv(fn+".csv",rows)


if __name__ == "__main__":
    simulate()
