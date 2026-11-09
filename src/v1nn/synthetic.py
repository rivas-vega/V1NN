"""Periodic translations: single-frame marginal distribution has no velocity cue."""
import torch


def moving_textures(batch=16,steps=8,size=16,seed=0,blackout=0):
    if min(batch,steps,size)<1 or not 0<=blackout<steps:
        raise ValueError("Invalid sequence dimensions or blackout")
    generator = torch.Generator().manual_seed(seed)
    texture = torch.rand(batch,1,size,size,generator=generator)
    directions = torch.tensor([[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]])
    ids = torch.randint(len(directions),(batch,),generator=generator)
    velocity = directions[ids]
    clips=[]
    for t in range(steps):
        clips.append(torch.stack([torch.roll(texture[b],
            shifts=(int(velocity[b,1])*t,int(velocity[b,0])*t),dims=(-2,-1))
            for b in range(batch)]))
    clean = torch.stack(clips,dim=1)
    observed = clean.clone()
    if blackout:
        observed[:,-blackout:]=0
    return observed,clean,velocity.float()
