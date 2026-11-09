"""Losses with explicit target provenance, causal alignment and missingness."""
import torch
from torch.nn import functional as F


def masked_mean(values,mask=None):
    if mask is None:
        mask = torch.ones_like(values,dtype=torch.bool)
    mask = torch.broadcast_to(mask.to(device=values.device,dtype=torch.bool),values.shape)
    if not bool(mask.any()):
        raise ValueError("No observed target; do not report a zero loss")
    if not bool(torch.isfinite(values[mask]).all()):
        raise ValueError("Nonfinite observed loss")
    return torch.where(mask,values,0).sum()/mask.sum()


def flow_loss(prediction,target,mask=None):
    """Supervised global 2D velocity squared error, not latent-teacher targets."""
    if prediction.shape != target.shape or prediction.shape[-1] != 2:
        raise ValueError("Expected matching [...,2] velocities")
    observed = torch.ones_like(target[...,0],dtype=torch.bool) if mask is None else torch.broadcast_to(mask.to(device=target.device,dtype=torch.bool),target.shape[:-1])
    # Mask BEFORE arithmetic: masking a NaN loss afterwards can leave NaN gradients.
    safe_prediction = torch.where(observed.unsqueeze(-1),prediction,0)
    safe_target = torch.where(observed.unsqueeze(-1),target,0)
    return masked_mean((safe_prediction-safe_target).square().sum(-1),observed)


def future_sensor_loss(prediction,sensor_target,horizon=1):
    """Teacher-free SSL: prediction at t targets only fixed q(x[t+h])."""
    if prediction.shape != sensor_target.shape or prediction.ndim != 3:
        raise ValueError("Expected identical [B,T,D] shapes")
    if not 1 <= horizon < prediction.shape[1]:
        raise ValueError("horizon must be in [1,T-1]")
    target = sensor_target[:,horizon:].detach()
    error = F.smooth_l1_loss(prediction[:,:-horizon],target,beta=0.1,reduction="none")
    return masked_mean(error)


def fluorescence_loss(prediction,target,observed,scale):
    """Gaussian NLL up to constants; TRAIN-ONLY per-neuron fixed scales.

    Apply after an observation model, not directly to raw RNN rates. Each
    observed neuron has equal weight irrespective of recording length.
    """
    if prediction.shape != target.shape or observed.shape != target.shape:
        raise ValueError("Expected identical [B,T,N] arrays")
    if scale.shape != (target.shape[-1],) or not bool((scale>0).all()):
        raise ValueError("Positive training-derived scales required")
    observed = observed.bool()
    safe_target = torch.where(observed,target,prediction.detach())
    error = 0.5*((prediction-safe_target)/scale).square()
    if not bool(torch.isfinite(error[observed]).all()):
        raise ValueError("Nonfinite observed fluorescence")
    count = observed.sum(dim=(0,1)); active = count>0
    if not bool(active.any()):
        raise ValueError("No observed neuron")
    total = torch.where(observed,error,0).sum(dim=(0,1))
    return (total[active]/count[active]).mean()


def poisson_loss(rate,count,observed=None,allow_fractional=False):
    """Counts only; fractional targets need an explicitly named quasi-likelihood."""
    if rate.shape != count.shape:
        raise ValueError("Rate and count shapes differ")
    mask = torch.ones_like(count,dtype=torch.bool) if observed is None else observed.bool()
    mask = torch.broadcast_to(mask,count.shape)
    if not bool(mask.any()):
        raise ValueError("No observed counts")
    y,r = count[mask],rate[mask]
    if not bool(torch.isfinite(y).all() and torch.isfinite(r).all()) or bool((y<0).any() or (r<=0).any()):
        raise ValueError("Counts must be nonnegative and rates strictly positive")
    if not allow_fractional and not bool(torch.allclose(y,y.round())):
        raise ValueError("Fractional deconvolved activity is not a spike count")
    return (r-y*r.log()).mean()


def calcium_filter(rates,dt,tau):
    """Causal normalized exponential kernel; gain/offset fitted separately."""
    if rates.ndim != 3 or dt<=0 or tau<=0:
        raise ValueError("Expected [B,T,N], dt>0 and tau>0")
    decay = torch.exp(rates.new_tensor(-dt/tau))
    calcium = torch.zeros_like(rates[:,0]); result=[]
    for rate in rates.unbind(1):
        calcium = decay*calcium+(1-decay)*rate
        result.append(calcium)
    return torch.stack(result,1)
