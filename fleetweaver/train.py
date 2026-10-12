"""Original PARCO symmetry loss with a smaller policy and explicit experiment logs."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torchrl.modules.models import MLP
from rl4co.data.transforms import StateAugmentation
from rl4co.models.zoo.symnco.losses import invariance_loss, solution_symmetricity_loss
from rl4co.utils.ops import unbatchify
from fleetweaver.core import make_policy, generate, seed_all, predict, POLICY_CONFIG
from parco.envs import HCVRPEnv

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--name",default="mixed-s42")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--steps",type=int,default=1800)
    p.add_argument("--batch",type=int,default=32)
    p.add_argument("--augment",type=int,default=4)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--device",default="cpu")
    p.add_argument("--regime",choices=["mixed","uniform"],default="mixed")
    p.add_argument("--validate-every",type=int,default=100)
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    if args.augment<2: raise ValueError("Symmetry baseline requires at least 2 augmentations")
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    root=Path(__file__).resolve().parents[1]
    checkpoints=root/"checkpoints"; checkpoints.mkdir(exist_ok=True)
    results=root/"results"; results.mkdir(exist_ok=True)
    policy=make_policy().to(args.device)
    projection=MLP(64,64,1,64,torch.nn.ReLU).to(args.device)
    parameters=list(policy.parameters())+list(projection.parameters())
    optimizer=torch.optim.Adam(parameters,lr=1e-4)
    augment=StateAugmentation(num_augment=args.augment)
    env=HCVRPEnv()
    rng=np.random.default_rng(args.seed)
    validation=[generate(n,m,64,800000+i,d) for i,(n,m,d) in enumerate(
        [(20,3,"uniform"),(40,5,"uniform"),(30,4,"clustered"),(30,4,"corner")])]
    best=float("inf"); start=0; elapsed_before=0.
    last=checkpoints/f"{args.name}-last.pt"
    if args.resume and last.exists():
        ck=torch.load(last,map_location=args.device,weights_only=False)
        policy.load_state_dict(ck["policy"]); projection.load_state_dict(ck["projection"])
        optimizer.load_state_dict(ck["optimizer"])
        start=ck["step"]; best=ck["best"]; elapsed_before=ck["seconds"]
        rng.bit_generator.state=ck["numpy_rng"]
        torch.set_rng_state(ck["torch_rng"].cpu())
    else:
        torch.save({"policy":policy.state_dict(),"config":POLICY_CONFIG},checkpoints/f"{args.name}-initial.pt")
    start_time=time.perf_counter()
    log_path=results/f"train-{args.name}.jsonl"
    if not args.resume: log_path.write_text("")
    def evaluate(step):
        nonlocal best
        with torch.random.fork_rng():
            scores=[float(-predict(policy,td,args.device)["reward"].mean()) for td in validation]
        score=float(np.mean(scores))
        record=dict(step=step,validation=score,validation_scenarios=scores,
            seconds=elapsed_before+time.perf_counter()-start_time)
        with log_path.open("a") as f: f.write(json.dumps(record)+"\n")
        print(json.dumps(record),flush=True)
        if score<best:
            best=score
            torch.save(dict(policy=policy.state_dict(),config=POLICY_CONFIG,step=step,
                validation=score,args=vars(args)),checkpoints/f"{args.name}-best.pt")
    if start==0: evaluate(0)
    for step in range(start+1,args.steps+1):
        policy.train(); projection.train()
        n=int(rng.integers(20,41)); m=int(rng.integers(3,6))
        dist="uniform" if args.regime=="uniform" else rng.choice(
            ["uniform","clustered","corner"],p=[0.6,0.25,0.15])
        td=generate(n,m,args.batch,int(rng.integers(1,500000000)),dist).to(args.device)
        out=policy(augment(env.reset(td)),env,phase="train",max_steps=10*n+100)
        reward=unbatchify(out["reward"],(0,args.augment))
        ll=unbatchify(out["log_likelihood"],(0,args.augment))
        loss_ss=solution_symmetricity_loss(reward[...,None],ll,dim=1)
        loss_inv=invariance_loss(projection(out["init_embeds"]),args.augment)
        loss=loss_ss+0.2*loss_inv
        if not torch.isfinite(loss): raise FloatingPointError("Nonfinite training loss")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        grad=float(torch.nn.utils.clip_grad_norm_(parameters,1.0))
        if not np.isfinite(grad): raise FloatingPointError("Nonfinite gradient")
        optimizer.step()
        if step%20==0:
            row=dict(step=step,loss=float(loss.detach()),reward=float(out["reward"].mean().detach()),
                grad_norm=grad,n=n,m=m,distribution=dist,seconds=elapsed_before+time.perf_counter()-start_time)
            with log_path.open("a") as f:f.write(json.dumps(row)+"\n")
            print(json.dumps(row),flush=True)
        if step%args.validate_every==0 or step==args.steps:
            evaluate(step)
            torch.save(dict(policy=policy.state_dict(),projection=projection.state_dict(),
                optimizer=optimizer.state_dict(),step=step,best=best,args=vars(args),
                numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                seconds=elapsed_before+time.perf_counter()-start_time),last)
    print(f"Finished {args.name}; best validation {best:.6f}",flush=True)

if __name__=="__main__":main()
