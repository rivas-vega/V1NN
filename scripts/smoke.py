"""Check optimization plumbing. NOT a topology efficacy experiment."""
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from v1nn.graph import Graph
from v1nn.model import ConnectomeRNN,FixedRetina
from v1nn.losses import flow_loss,future_sensor_loss
from v1nn.synthetic import moving_textures


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--steps",type=int,default=20)
    parser.add_argument("--seed",type=int,default=7)
    parser.add_argument("--graph",type=Path)
    parser.add_argument("--objective",choices=["motion","sensory_ssl"],default="motion")
    parser.add_argument("--output",type=Path,default=Path("outputs/smoke.json"))
    args=parser.parse_args()
    if args.steps<1:parser.error("--steps must be positive")
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    if args.graph:
        data=np.load(args.graph,allow_pickle=False)
        graph=Graph(len(data["signs"]),data["src"],data["dst"],data["signs"])
        provenance="user_supplied_npz; inspect adjacent provenance JSON"
    else:
        rng=np.random.default_rng(args.seed);n=64
        edges=[(i,j) for i in range(n) for j in range(n) if i!=j and rng.random()<0.08]
        src,dst=np.array(edges,dtype=np.int64).T
        graph=Graph(n,src,dst,np.where(np.arange(n)%5==0,-1,1))
        provenance="synthetic graph; NOT MICrONS"
    retina=FixedRetina(grid=4)
    output_dim=2 if args.objective=="motion" else 32
    model=ConnectomeRNN(graph,32,output_dim,routing_seed=args.seed)
    optimizer=torch.optim.Adam(model.parameters(),lr=0.003)
    start=time.perf_counter();losses=[]
    target_sum=torch.zeros(32);target_count=0
    for step in range(args.steps):
        observed,clean,velocity=moving_textures(batch=12,steps=8,seed=args.seed+1000+step)
        x,q=retina(observed),retina(clean)
        target_sum+=q.sum(dim=(0,1));target_count+=q.shape[0]*q.shape[1]
        prediction,_=model(x)
        loss=(flow_loss(prediction[:,-1],velocity) if args.objective=="motion" else
              future_sensor_loss(prediction,q,horizon=1))
        optimizer.zero_grad();loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("Nonfinite gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step();losses.append(float(loss.detach()))
    with torch.no_grad():
        observed,clean,velocity=moving_textures(batch=32,steps=8,seed=args.seed+90000)
        prediction,_=model(retina(observed))
        heldout=(flow_loss(prediction[:,-1],velocity) if args.objective=="motion" else
                 future_sensor_loss(prediction,retina(clean),horizon=1))
        if args.objective=="motion":
            baselines={"zero_velocity_loss":float(flow_loss(torch.zeros_like(velocity),velocity))}
        else:
            target=retina(clean)
            constant=(target_sum/target_count).reshape(1,1,-1).expand_as(target)
            baselines={"persistence_loss":float(future_sensor_loss(target,target)),
                       "training_mean_loss":float(future_sensor_loss(constant,target))}
    result={"purpose":"finite-gradient integration check, not scientific evidence",
            "graph_provenance":provenance,"nodes":graph.n,"edges":len(graph.src),
            "objective":args.objective,"steps":args.steps,"seed":args.seed,
            "first_training_loss":losses[0],"last_training_loss":losses[-1],
            "heldout_loss":float(heldout),"baselines":baselines,
            "seconds":time.perf_counter()-start,"torch":torch.__version__,
            "python":platform.python_version(),"device":"cpu","scientific_claim":None}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
