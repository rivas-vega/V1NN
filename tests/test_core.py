import numpy as np
import pytest
import torch
from v1nn.graph import Graph,rewire
from v1nn.model import ConnectomeRNN,FixedRetina
from v1nn.losses import (masked_mean,flow_loss,future_sensor_loss,
                         fluorescence_loss,poisson_loss,calcium_filter)
from v1nn.synthetic import moving_textures


def small():return Graph(3,np.array([0,1]),np.array([1,2]),np.array([1,-1,1]))


def test_orientation_and_dale():
    model=ConnectomeRNN(small(),4,2);state=torch.tensor([[2.,3.,0.]])
    weights=model.effective_weights();current=model.recurrent_current(state)
    assert current[0,0]==0
    assert torch.allclose(current[0,1],2*weights[0])
    assert torch.allclose(current[0,2],3*weights[1])
    assert weights[0]>0 and weights[1]<0


def test_edge_parameters_and_gradients():
    model=ConnectomeRNN(small(),4,2);out,_=model(torch.ones(2,5,4));out.square().mean().backward()
    assert model.theta.shape==(2,)
    assert model.theta.grad is not None and torch.isfinite(model.theta.grad).all()


def test_contractivity_cap():
    model=ConnectomeRNN(small(),4,2)
    with torch.no_grad():model.theta.fill_(100)
    mass=torch.zeros(3).index_add(0,model.dst,model.effective_weights().abs())
    assert bool((mass<=0.950001).all())


def test_causal_prefix():
    torch.manual_seed(5);model=ConnectomeRNN(small(),4,2)
    x=torch.rand(2,6,4);z=x.clone();z[:,3:]+=100
    a,_=model(x);b,_=model(z)
    assert torch.equal(a[:,:3],b[:,:3])


def test_noedge_still_has_memory():
    graph=Graph(2,np.array([],dtype=int),np.array([],dtype=int),np.ones(2))
    model=ConnectomeRNN(graph,4,2);x=torch.zeros(1,3,4);x[:,0]=1
    _,persistent=model(x);_,reset=model(x,reset_each_frame=True)
    assert persistent[:,-1].sum()>0 and reset[:,-1].sum()==0


def test_input_route_independent_of_graph():
    g=small();other=Graph(3,np.array([1,2]),np.array([0,1]),g.signs)
    a=ConnectomeRNN(g,8,2,routing_seed=91);b=ConnectomeRNN(other,8,2,routing_seed=91)
    assert torch.equal(a.routing,b.routing)


def test_retina_stateless():
    retina=FixedRetina(4);x=torch.rand(2,5,1,8,8);y=x.clone();y[:,3:]=0
    assert not list(retina.parameters())
    assert torch.equal(retina(x)[:,:3],retina(y)[:,:3])
    assert retina(x).shape==(2,5,32)


def test_future_alignment_and_detach():
    target=torch.arange(5.).reshape(1,5,1).requires_grad_()
    pred=torch.tensor([[[1.],[2.],[3.],[4.],[9.]]],requires_grad=True)
    loss=future_sensor_loss(pred,target,1);loss.backward()
    assert loss==0 and target.grad is None


def test_missing_values_are_not_zeros():
    assert masked_mean(torch.tensor([2.,float("nan")]),torch.tensor([True,False]))==2
    with pytest.raises(ValueError):masked_mean(torch.ones(2),torch.zeros(2,dtype=bool))


def test_fluorescence_missing_gradient():
    p=torch.ones(1,2,2,requires_grad=True)
    y=torch.tensor([[[2.,float("nan")],[2.,float("nan")]]]);mask=torch.isfinite(y)
    loss=fluorescence_loss(p,y,mask,torch.ones(2));loss.backward()
    assert torch.isfinite(p.grad).all() and (p.grad[:,:,1]==0).all()


def test_poisson_not_calcium():
    with pytest.raises(ValueError):poisson_loss(torch.ones(2),torch.tensor([-1.,2.]))
    with pytest.raises(ValueError):poisson_loss(torch.ones(2),torch.tensor([0.2,0.5]))
    assert torch.isfinite(poisson_loss(torch.ones(2),torch.tensor([0.2,0.5]),allow_fractional=True))


def test_calcium_causal():
    r=torch.zeros(1,4,1);r[:,1]=1;c=calcium_filter(r,0.1,0.4)
    assert c[0,0,0]==0 and c[0,1,0]>c[0,2,0]>0


def test_bad_graph_rejected():
    with pytest.raises(ValueError):Graph(2,np.array([0,0]),np.array([1,1]),np.ones(2))
    with pytest.raises(ValueError):Graph(2,np.array([0.5]),np.array([1]),np.ones(2))
    g=Graph(2,np.array([0]),np.array([1]),np.array([0,1]))
    with pytest.raises(ValueError):ConnectomeRNN(g,2,1)


def test_rewiring_degrees_groups_and_geometry():
    n=20;rng=np.random.default_rng(10)
    pairs=np.array([(a,b) for a in range(n) for b in range(n) if a!=b and rng.random()<0.2])
    g=Graph(n,pairs[:,0],pairs[:,1],np.ones(n));groups=np.arange(n)%2
    positions=np.column_stack([np.arange(n),np.zeros(n)])
    null,report=rewire(g,40,seed=5,groups=groups,positions=positions,distance_bins=[5,10,15],max_attempts=50000)
    for before,after in zip(g.degrees(),null.degrees()):assert np.array_equal(before,after)
    for node in range(n):
        for group in [0,1]:
            assert ((g.src==node)&(groups[g.dst]==group)).sum()==((null.src==node)&(groups[null.dst]==group)).sum()
            assert ((g.dst==node)&(groups[g.src]==group)).sum()==((null.dst==node)&(groups[null.src]==group)).sum()
    old=np.searchsorted([5,10,15],np.abs(g.src-g.dst));new=np.searchsorted([5,10,15],np.abs(null.src-null.dst))
    assert np.array_equal(old,new) and report["edge_change_fraction"]>0


def test_failed_null_does_not_silently_pass():
    g=Graph(2,np.array([0,1]),np.array([1,0]),np.ones(2))
    with pytest.raises(RuntimeError):rewire(g,1,max_attempts=10)


def test_video_target_separation():
    seen,clean,v=moving_textures(seed=3,blackout=2)
    assert (seen[:,-2:]==0).all() and clean[:,-2:].sum()>0
    assert seen.data_ptr()!=clean.data_ptr() and v.shape==(16,2)


def test_flow_shape_and_loss():
    assert flow_loss(torch.zeros(1,2),torch.ones(1,2))==2
    with pytest.raises(ValueError):flow_loss(torch.zeros(2),torch.zeros(3))


def test_missing_flow_has_finite_gradients():
    prediction=torch.tensor([[[0.,0.],[2.,1.]]],requires_grad=True)
    target=torch.tensor([[[float("nan"),float("nan")],[1.,1.]]]);mask=torch.tensor([[False,True]])
    value=flow_loss(prediction,target,mask);value.backward()
    assert torch.isfinite(prediction.grad).all()
    assert torch.equal(prediction.grad[0,0],torch.zeros(2))
    assert value.item()==1.0
