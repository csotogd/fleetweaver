"""Frozen held-out benchmark. Every returned route is checked independently."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from fleetweaver.core import make_policy,generate,instance,predict,actions_to_routes,validate_routes,seed_all
from fleetweaver.baselines import greedy,two_opt,ortools

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--instances",type=int,default=32)
    parser.add_argument("--ortools-seconds",type=float,default=1.0)
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--only",default="")
    parser.add_argument("--models",nargs="+",default=["mixed-s42","mixed-s43","mixed-s44","uniform-s42"])
    args=parser.parse_args()
    torch.set_num_threads(args.threads);seed_all(20261013)
    root=Path(__file__).resolve().parents[1]
    study=json.loads((root/"configs/study.json").read_text())
    models={}; validation={}
    for name in args.models:
        ck=torch.load(root/f"checkpoints/{name}-best.pt",map_location="cpu",weights_only=False)
        p=make_policy();p.load_state_dict(ck["policy"]);p.eval();models[name]=p
        validation[name]=ck["validation"]
    representative=min((k for k in models if k.startswith("mixed")),key=lambda k:validation[k])
    seed_all(42)
    models["untrained"]=make_policy().eval()
    metadata=dict(args=vars(args),representative=representative,validation=validation,
        selection="minimum validation score, before examining test results",test_seed_base=900000000,
        torch_version=torch.__version__,threads=args.threads,device="cpu")
    (root/"results/benchmark-metadata.json").write_text(json.dumps(metadata,indent=2))
    rows=[]; examples=[]
    output=root/"results/benchmark.csv"
    # A resumed benchmark keeps existing fully completed scenarios.
    if output.exists():rows=pd.read_csv(output).to_dict("records")
    example_file=root/"results/routes.json"
    if example_file.exists():examples=json.loads(example_file.read_text())
    completed={r["scenario"] for r in rows}
    for si,(name,n,m,dist) in enumerate(study["scenarios"]):
        if args.only and name not in args.only.split(","):continue
        if name in completed:continue
        td=generate(n,m,args.instances,900000000+si,dist)
        scenario_rows=[]; scenario_examples=[]
        def add(i,method,routes,seconds,timing_mode="single",expected=None,fallback=False):
            value,times=validate_routes(instance(td[i]),routes)
            if expected is not None and not np.isclose(value,expected,rtol=2e-5,atol=2e-5):
                raise AssertionError(f"Environment reward mismatch {name} {method} {i}: {value} != {expected}")
            scenario_rows.append(dict(scenario=name,n=n,m=m,distribution=dist,instance=i,
                method=method,makespan=value,seconds=seconds,timing_mode=timing_mode,
                valid=True,solver_fallback=fallback))
            if i==0:
                obj={k:v.tolist() for k,v in instance(td[i]).items()}
                scenario_examples.append(dict(scenario=name,method=method,instance=obj,routes=routes,
                    makespan=value,vehicle_times=times))
        # Warm each shape outside measured intervals. Batch timing measures throughput,
        # while the dedicated latency table below uses single instances.
        for model_name,policy in models.items():
            predict(policy,td[:1])
            t=time.perf_counter();out=predict(policy,td);elapsed=time.perf_counter()-t
            for i in range(args.instances):
                add(i,model_name,actions_to_routes(out["actions"][i],m),elapsed/args.instances,
                    "batch_amortized",float(-out["reward"][i]))
        for samples in [16]:
            policy=models[representative]
            seed_all(1000000000+si)
            for start in range(0,args.instances,8):
                batch=td[start:start+8]
                t=time.perf_counter();out=predict(policy,batch,samples=samples);elapsed=time.perf_counter()-t
                for j in range(len(batch)):
                    add(start+j,"mixed-sample16",actions_to_routes(out["actions"][j],m),elapsed/len(batch),
                        "batch_amortized",float(-out["reward"][j]))
        for i in range(args.instances):
            inst=instance(td[i])
            t=time.perf_counter();routes=greedy(inst,False);elapsed=time.perf_counter()-t
            add(i,"nearest",routes,elapsed)
            t=time.perf_counter();routes=greedy(inst,True);elapsed=time.perf_counter()-t
            add(i,"balanced-greedy",routes,elapsed)
            t=time.perf_counter();routes=two_opt(inst,greedy(inst,True));elapsed=time.perf_counter()-t
            add(i,"balanced-2opt",routes,elapsed)
            t=time.perf_counter();opt_routes=ortools(inst,args.ortools_seconds);elapsed=time.perf_counter()-t
            fallback=opt_routes is None
            if fallback:opt_routes=routes
            add(i,"ortools-1s",opt_routes,elapsed,fallback=fallback)
            if i%8==0:print(f"{name}: classical {i+1}/{args.instances}",flush=True)
        rows.extend(scenario_rows);examples.extend(scenario_examples)
        pd.DataFrame(rows).to_csv(output,index=False)
        example_file.write_text(json.dumps(examples))
        print(f"Completed {name}: {len(scenario_rows)} feasible solutions",flush=True)
    # Dedicated latency measurements: five instances, three repetitions, after warmup.
    latency=[]
    for n,m in [(20,3),(40,5),(100,10)]:
        td=generate(n,m,5,1100000000+n)
        policy=models[representative]
        for samples in [1,16]:
            predict(policy,td[:1],samples=samples)
            for i in range(5):
                for rep in range(3):
                    t=time.perf_counter();out=predict(policy,td[i:i+1],samples=samples);elapsed=time.perf_counter()-t
                    validate_routes(instance(td[i]),actions_to_routes(out["actions"][0],m))
                    latency.append(dict(n=n,m=m,method="mixed-greedy" if samples==1 else "mixed-sample16",
                        instance=i,repetition=rep,seconds=elapsed))
    pd.DataFrame(latency).to_csv(root/"results/latency.csv",index=False)

if __name__=="__main__":main()
