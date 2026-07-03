#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tier-1 control experiments for the flowMC ensemble paper.
=========================================================
目的：分离 "ensemble/diversity 的贡献" 与 "样本数变多的贡献"。

跑两个对照（都用 equal-per-member 下 ensemble 的总样本数 = 10000 做基准）：
  A) single chain @ 10000 particles        —— 和 ensemble 总样本数一致的单链
  B) homogeneous ensemble (5 个相同成员)    —— 隔离 diversity（vs 原来的 diverse ensemble）

【重要】本脚本只写入 ./control_outputs/，绝不覆盖你已有的 ./paper_outputs/。
如果当前目录下能找到 ./paper_outputs/results_equal_per_member.csv，
脚本结尾会自动把新结果和你已有的 diverse-ensemble 结果做配对对比并打印结论。

直接运行：python run_controls.py   （Colab 里 !python run_controls.py 或整段粘贴进 cell）
依赖：和你原脚本完全一样（flowMC, jax, numpy, scipy, sklearn, pandas）。
"""

import os, json, time, zlib, warnings
warnings.filterwarnings("ignore")
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from scipy.special import kl_div
from scipy.stats import ttest_rel
from sklearn.cluster import KMeans

from flowMC.Sampler import Sampler
from flowMC.resource_strategy_bundle.RQSpline_MALA import RQSpline_MALA_Bundle

# =============================================================================
# Section 1: Target potentials  (与原脚本完全一致)
# =============================================================================
def standard_gaussian_potential(x): return jnp.squeeze(0.5*jnp.sum(x**2, axis=-1))

def banana_potential(x, a=1.0, b=100.0):
    t1 = 0.5*x[...,0]**2
    t2 = 0.5*(x[...,1]-a*x[...,0]**2+b)**2
    if x.ndim>1 and x.shape[-1]>2:
        return jnp.squeeze(t1+t2+0.5*jnp.sum(x[...,2:]**2, axis=-1))
    return jnp.squeeze(t1+t2)

def bimodal_gaussian_potential(x, mode_distance=6.0):
    d=x.shape[-1]; off=mode_distance/2
    mu1=jnp.zeros(d).at[0].set(-off); mu2=jnp.zeros(d).at[0].set(+off)
    d1=jnp.sum((x-mu1)**2, axis=-1); d2=jnp.sum((x-mu2)**2, axis=-1)
    m=jnp.maximum(-0.5*d1, -0.5*d2)
    return jnp.squeeze(-(m+jnp.log(jnp.exp(-0.5*d1-m)+jnp.exp(-0.5*d2-m))))

def funnel_potential(x, sigma_v=0.3):
    v=x[...,0]; pot_v=0.5*(v/sigma_v)**2
    rest=jnp.sum(x[...,1:]**2, axis=-1)
    pot_x=0.5*rest/jnp.exp(v)+0.5*(x.shape[-1]-1)*v
    return jnp.squeeze(pot_v+pot_x)

def correlated_gaussian_potential(x, rho=0.95):
    d=x.shape[-1]; x1,x2=x[...,0],x[...,1]
    Z=(x1**2-2*rho*x1*x2+x2**2)/(1-rho**2)
    rest=jnp.sum(x[...,2:]**2, axis=-1) if d>2 else 0.0
    return jnp.squeeze(0.5*(Z+rest)+0.5*jnp.log(1-rho**2))

def rosenbrock_potential(x, a=1.0, b=100.0):
    d=x.shape[-1]; u=(a-x[...,0])**2+b*(x[...,1]-x[...,0]**2)**2
    if d>2: u=u+jnp.sum(x[...,2:]**2, axis=-1)
    return jnp.squeeze(0.5*u)

def gaussian_ring_potential(x, radius=5.0, K=8, std=0.7):
    if x.ndim==1: x=x[None,:]; sq=True
    else: sq=False
    theta=jnp.linspace(0,2*jnp.pi,K,endpoint=False)
    centers=jnp.stack([radius*jnp.cos(theta), radius*jnp.sin(theta)], axis=1)
    x2=x[...,:2]; dists=jnp.sum((x2[:,None,:]-centers[None,:,:])**2, axis=-1)
    loglik2=jax.scipy.special.logsumexp(-0.5*dists/(std**2), axis=1)-jnp.log(K)
    d=x.shape[-1]; rest=0.5*jnp.sum(x[:,2:]**2, axis=-1) if d>2 else 0.0
    res=-loglik2+rest
    return jnp.squeeze(res) if sq else res

def log_post_from_cfg(potential_fn, **p):
    pots={"standard_gaussian":lambda x:-standard_gaussian_potential(x),
          "banana":lambda x:-banana_potential(x,**p),
          "bimodal_gaussian":lambda x:-bimodal_gaussian_potential(x,**p),
          "funnel":lambda x:-funnel_potential(x,**p),
          "correlated_gaussian":lambda x:-correlated_gaussian_potential(x,**p),
          "rosenbrock":lambda x:-rosenbrock_potential(x,**p),
          "gaussian_ring":lambda x:-gaussian_ring_potential(x,**p)}
    return lambda x, data=None: pots[potential_fn](x)

# =============================================================================
# Section 2: Reference samples  (与原脚本完全一致)
# =============================================================================
def generate_reference_samples(potential_fn, n_dims, n_samples=20000, **p):
    key=jax.random.PRNGKey(0)
    if potential_fn=="standard_gaussian":
        return jax.random.normal(key,(n_samples,n_dims))
    elif potential_fn=="banana":
        a,b=p.get('a',1.0),p.get('b',100.0)
        key,k1,k2=jax.random.split(key,3)
        x1=jax.random.normal(k1,(n_samples,)); x2=jax.random.normal(k2,(n_samples,))+a*x1**2-b
        if n_dims>2:
            key,kr=jax.random.split(key)
            return jnp.column_stack((x1,x2,jax.random.normal(kr,(n_samples,n_dims-2))))
        return jnp.column_stack((x1,x2))
    elif potential_fn=="bimodal_gaussian":
        md=p.get('mode_distance',6.0); off=md/2
        mu1=jnp.zeros(n_dims).at[0].set(-off); mu2=jnp.zeros(n_dims).at[0].set(+off)
        key,k1,k2=jax.random.split(key,3)
        mask=jax.random.bernoulli(k1,0.5,(n_samples,))
        s=jnp.zeros((n_samples,n_dims)); n1=mask.sum().item()
        if n1>0:
            key,k1r=jax.random.split(key); s=s.at[mask].set(jax.random.normal(k1r,(n1,n_dims))+mu1)
        if n1<n_samples:
            key,k2r=jax.random.split(key); s=s.at[~mask].set(jax.random.normal(k2r,(n_samples-n1,n_dims))+mu2)
        return s
    elif potential_fn=="funnel":
        sv=p.get('sigma_v',0.3); key,k1=jax.random.split(key)
        v=jax.random.normal(k1,(n_samples,))*sv
        s=jnp.zeros((n_samples,n_dims)).at[:,0].set(v)
        for i in range(n_samples):
            key,kr=jax.random.split(key)
            s=s.at[i,1:].set(jax.random.normal(kr,(n_dims-1,))*jnp.exp(v[i]/2))
        return s
    elif potential_fn=="correlated_gaussian":
        rho=p.get("rho",0.95); L=jnp.linalg.cholesky(jnp.array([[1.0,rho],[rho,1.0]]))
        key,k1=jax.random.split(key); xy=jax.random.normal(k1,(n_samples,2))@L.T
        if n_dims>2:
            key,kr=jax.random.split(key)
            return jnp.column_stack((xy,jax.random.normal(kr,(n_samples,n_dims-2))))
        return xy
    elif potential_fn=="rosenbrock":
        a,b=p.get("a",1.0),p.get("b",100.0); key,k1,k2=jax.random.split(key,3)
        x0=jax.random.normal(k1,(n_samples,))*2.0+a; y0=x0**2+jax.random.normal(k2,(n_samples,))*0.5
        if n_dims>2:
            key,kr=jax.random.split(key)
            return jnp.column_stack((x0,y0,jax.random.normal(kr,(n_samples,n_dims-2))))
        return jnp.column_stack((x0,y0))
    elif potential_fn=="gaussian_ring":
        R=p.get("radius",5.0); std=p.get("std",0.7)
        theta=jax.random.uniform(key,(n_samples,))*2*jnp.pi; key,k1=jax.random.split(key)
        r=R+jax.random.normal(k1,(n_samples,))*std; x=r*jnp.cos(theta); y=r*jnp.sin(theta)
        if n_dims>2:
            key,kr=jax.random.split(key)
            return jnp.column_stack((x,y,jax.random.normal(kr,(n_samples,n_dims-2))))
        return jnp.column_stack((x,y))
    raise ValueError(potential_fn)

# =============================================================================
# Section 3: Metrics  (与原脚本完全一致)
# =============================================================================
def _ess_1d(x):
    n=len(x); mu=x.mean()
    ac=np.correlate(x-mu,x-mu,mode='full'); v0=float(ac[n-1])
    if not np.isfinite(v0) or abs(v0)<1e-12: return float(n)
    ac=ac[n-1:]/max(v0,1e-12); idx=np.where(ac<0)[0]
    tau=max(0.5+np.sum(ac[1:(int(idx[0]) if idx.size else None)]),1e-9)
    return float(n/(2*tau))

def compute_ess_d_dim(s):
    s=np.asarray(s); return float(np.mean([_ess_1d(s[:,i]) for i in range(s.shape[1])]))

def compute_marginal_js_distance(samples, reference, nbins=100, pad=0.1):
    s=np.asarray(samples); r=np.asarray(reference); js=[]
    for i in range(s.shape[1]):
        lo=min(np.percentile(s[:,i],0.5),np.percentile(r[:,i],0.5))
        hi=max(np.percentile(s[:,i],99.5),np.percentile(r[:,i],99.5))
        rng=hi-lo; lo-=pad*rng; hi+=pad*rng
        bins=np.linspace(lo,hi,nbins+1)
        h1,_=np.histogram(s[:,i],bins=bins,density=True)
        h2,_=np.histogram(r[:,i],bins=bins,density=True)
        eps=1e-12; h1=(h1+eps)/(h1.sum()+eps*len(h1)); h2=(h2+eps)/(h2.sum()+eps*len(h2))
        m=0.5*(h1+h2); js.append(np.sqrt(0.5*np.sum(kl_div(h1,m))+0.5*np.sum(kl_div(h2,m))))
    return float(np.mean(js))

# =============================================================================
# Section 4: flowMC single + ensemble  (与原脚本完全一致)
# =============================================================================
class SamplerWithReturn(Sampler):
    def sample(self, initial_position, data=None):
        last=initial_position; key=self.rng_key
        for name in self.strategy_order:
            key,self.resources,last=self.strategies[name](key,self.resources,last,data)
        return last

def flowmc_inference(rng_key, n_particles, n_dims, n_local_steps=10, n_global_steps=10,
                     n_training_loops=2, n_production_loops=2, n_epochs=5,
                     hidden_units=(64,64), n_bins=8, n_layers=3,
                     potential_fn="standard_gaussian", log_post_override=None, **p):
    log_post = log_post_override if log_post_override is not None else log_post_from_cfg(potential_fn,**p)
    rng_key,sub=jax.random.split(rng_key); init=jax.random.normal(sub,(n_particles,n_dims))
    rng_key,sub=jax.random.split(rng_key)
    bundle=RQSpline_MALA_Bundle(sub,n_particles,n_dims,log_post,n_local_steps,n_global_steps,
                                n_training_loops,n_production_loops,n_epochs,
                                rq_spline_hidden_units=list(hidden_units),rq_spline_n_bins=n_bins,
                                rq_spline_n_layers=n_layers,verbose=False)
    sampler=SamplerWithReturn(n_dims,n_particles,rng_key,resource_strategy_bundles=bundle)
    t0=time.time(); final=sampler.sample(init,{})
    return np.array(final), time.time()-t0

def _cluster_weights(X,K=8,rare_boost=0.35,seed=0):
    K=max(2,min(K,max(2,X.shape[0]//50)))
    km=KMeans(n_clusters=K,n_init=5,random_state=seed).fit(X)
    cid=km.labels_; cnt=np.bincount(cid); inv=1.0/np.maximum(cnt,1); w=inv[cid]
    w=w*(1.0+rare_boost*(w/np.max(w)-1.0)); w=np.maximum(w,0.0); return w/np.sum(w)

def enhanced_flowmc_ensemble(rng_key, n_particles_single, n_dims, potential_fn,
                             n_members=5, budget="equal_per_member",
                             beta=1.0, gamma=2.0, lam=0.05, alpha_temp=0.35,
                             K_clusters=8, rare_boost=0.35, adaptive_temp=True,
                             init_temp=1.0, target_ess_ratio=0.5,
                             use_member_weight=True, use_cov_balance=True,
                             palette=None, **p):
    if palette is None:
        palette=[dict(hidden_units=(64,64),n_layers=3,n_bins=8,n_local_steps=10,n_global_steps=10),
                 dict(hidden_units=(96,96),n_layers=3,n_bins=12,n_local_steps=10,n_global_steps=10),
                 dict(hidden_units=(128,64),n_layers=4,n_bins=8,n_local_steps=15,n_global_steps=8),
                 dict(hidden_units=(64,128),n_layers=4,n_bins=10,n_local_steps=8,n_global_steps=15),
                 dict(hidden_units=(32,64,32),n_layers=3,n_bins=8,n_local_steps=12,n_global_steps=12)]
    base_log_post=log_post_from_cfg(potential_fn,**p)
    ref=np.array(generate_reference_samples(potential_fn,n_dims,20000,**p))
    n_per = max(1,int(np.floor(n_particles_single/n_members))) if budget=="equal_total" else int(n_particles_single)
    member_samples=[]; ess_l=[]; js_l=[]; slp_l=[]; total_time=0.0; temp=float(init_temp)
    for m in range(n_members):
        cfg=palette[m%len(palette)]; rng_key,sub=jax.random.split(rng_key)
        log_post_t=(lambda x,d=None: base_log_post(x,d)/temp) if adaptive_temp else None
        s_m,t_m=flowmc_inference(sub,n_per,n_dims,
                                 n_local_steps=cfg.get("n_local_steps",10),
                                 n_global_steps=cfg.get("n_global_steps",10),
                                 n_training_loops=2,n_production_loops=2,n_epochs=5,
                                 hidden_units=cfg.get("hidden_units",(64,64)),
                                 n_bins=cfg.get("n_bins",8),n_layers=cfg.get("n_layers",3),
                                 potential_fn=potential_fn,log_post_override=log_post_t,**p)
        total_time+=t_m; X=np.asarray(s_m); member_samples.append(X)
        ess_i=compute_ess_d_dim(X); js_i=compute_marginal_js_distance(X,ref)
        lp=np.asarray(jax.vmap(log_post_t if adaptive_temp else base_log_post)(jnp.asarray(X)))
        ess_l.append(ess_i); js_l.append(js_i); slp_l.append(float(np.std(lp)))
        if adaptive_temp:
            r=ess_i/max(1.0,n_per)
            temp=min(2.0,temp*1.1) if r<target_ess_ratio else max(0.1,temp*0.95)
    allX=np.vstack(member_samples)
    if use_member_weight:
        ess=np.maximum(np.asarray(ess_l,float),1e-6); js=np.maximum(np.asarray(js_l,float),1e-6)
        slp=np.maximum(np.asarray(slp_l,float),1e-6)
        score=(ess**beta)*np.exp(-gamma*js)*np.exp(-lam*slp)
        z=(score-score.mean())/(score.std()+1e-8); wa=np.exp(alpha_temp*z); wa/=wa.sum()
    else:
        wa=np.ones(n_members)/n_members
    memb_id=np.concatenate([[i]*len(member_samples[i]) for i in range(n_members)]); w_m=wa[memb_id]
    if use_cov_balance:
        cd=min(2,allX.shape[1]); cov=_cluster_weights(allX[:,:cd],K=K_clusters,rare_boost=rare_boost,seed=0)
        wf=w_m*cov; wf/=wf.sum()
    else:
        wf=w_m/np.sum(w_m)
    N=n_per*n_members; rng=np.random.default_rng(2025)
    idx=rng.choice(len(allX),size=N,replace=True,p=wf)
    return allX[idx], float(total_time), dict(N_total=int(N))

# =============================================================================
# Section 5: 实验配置 (与原脚本完全一致的 11 个 config + 同样的 seed 逻辑)
# =============================================================================
CONFIGS=[dict(potential_fn="banana",n_dims=20,params=dict(a=1.0,b=100.0)),
         dict(potential_fn="bimodal_gaussian",n_dims=20,params=dict(mode_distance=6.0)),
         dict(potential_fn="funnel",n_dims=20,params=dict(sigma_v=0.3)),
         dict(potential_fn="correlated_gaussian",n_dims=20,params=dict(rho=0.95)),
         dict(potential_fn="rosenbrock",n_dims=20,params=dict(a=1.0,b=100.0)),
         dict(potential_fn="gaussian_ring",n_dims=20,params=dict(radius=5.0,K=8,std=0.7)),
         dict(potential_fn="banana",n_dims=50,params=dict(a=1.0,b=100.0)),
         dict(potential_fn="bimodal_gaussian",n_dims=50,params=dict(mode_distance=6.0)),
         dict(potential_fn="funnel",n_dims=50,params=dict(sigma_v=0.3)),
         dict(potential_fn="correlated_gaussian",n_dims=50,params=dict(rho=0.95)),
         dict(potential_fn="rosenbrock",n_dims=50,params=dict(a=1.0,b=100.0))]
N_REPEATS=5

def seed_for(fn, rep):
    return 10_000 + 97*rep + (zlib.crc32(fn.encode("utf-8")) % 997)

# =============================================================================
# Section 6: 跑两个对照
# =============================================================================
def run_single_at(n_particles, label):
    rows=[]
    for cfg in CONFIGS:
        fn=cfg["potential_fn"]; d=cfg["n_dims"]; params=cfg.get("params",{})
        ref=generate_reference_samples(fn,d,20000,**params)
        print(f"[{label}] {fn} d={d} ...", flush=True)
        for rep in range(N_REPEATS):
            master=jax.random.PRNGKey(np.uint32(seed_for(fn,rep)))
            master,sk=jax.random.split(master)
            Xs,t=flowmc_inference(sk,n_particles,d,potential_fn=fn,**params)
            ess=compute_ess_d_dim(Xs); js=compute_marginal_js_distance(Xs,np.array(ref))
            rows.append(dict(potential_fn=fn,dimension=d,repeat=rep,method=label,
                             ESS=ess,runtime_sec=t,JS_distance=js,n_particles=n_particles))
    return pd.DataFrame(rows)

def run_homogeneous_ensemble(label):
    # 5 个完全相同的成员 = baseline config，各 2000 粒子，其余设置(含三个机制)与原 diverse ensemble 一致
    same=dict(hidden_units=(64,64),n_layers=3,n_bins=8,n_local_steps=10,n_global_steps=10)
    palette=[dict(same) for _ in range(5)]
    rows=[]
    for cfg in CONFIGS:
        fn=cfg["potential_fn"]; d=cfg["n_dims"]; params=cfg.get("params",{})
        ref=generate_reference_samples(fn,d,20000,**params)
        print(f"[{label}] {fn} d={d} ...", flush=True)
        for rep in range(N_REPEATS):
            master=jax.random.PRNGKey(np.uint32(seed_for(fn,rep)))
            master,_=jax.random.split(master); master,sk=jax.random.split(master)
            Xe,t,_=enhanced_flowmc_ensemble(sk,2000,d,fn,n_members=5,budget="equal_per_member",
                                            palette=palette,**params)
            ess=compute_ess_d_dim(Xe); js=compute_marginal_js_distance(Xe,np.array(ref))
            rows.append(dict(potential_fn=fn,dimension=d,repeat=rep,method=label,
                             ESS=ess,runtime_sec=t,JS_distance=js,n_particles=10000))
    return pd.DataFrame(rows)

def paired_summary(df_a, name_a, df_b, name_b):
    """df_a, df_b 各含 potential_fn,dimension,repeat,JS_distance；按 (fn,dim,rep) 配对"""
    out=[]
    for (fn,d),ga in df_a.groupby(["potential_fn","dimension"]):
        gb=df_b[(df_b.potential_fn==fn)&(df_b.dimension==d)]
        if len(gb)==0: continue
        ga=ga.sort_values("repeat"); gb=gb.sort_values("repeat")
        a=ga["JS_distance"].values; b=gb["JS_distance"].values
        n=min(len(a),len(b)); a,b=a[:n],b[:n]
        if n<2: continue
        t,pv=ttest_rel(a,b)
        out.append(dict(target=f"{fn}-{d}D", JS_A=a.mean(), JS_B=b.mean(),
                        diff_pct=(a.mean()-b.mean())/a.mean()*100, p=pv))
    s=pd.DataFrame(out)
    print(f"\n==== {name_a}  vs  {name_b}  (JS, lower better) ====")
    print(f"{'target':<18}{name_a[:10]:>12}{name_b[:10]:>12}{'(A-B)/A%':>10}{'p':>10}")
    for _,r in s.iterrows():
        print(f"{r.target:<18}{r.JS_A:>12.4f}{r.JS_B:>12.4f}{r.diff_pct:>10.1f}{r.p:>10.2e}")
    print(f"  MEAN JS  {name_a}={s.JS_A.mean():.4f}   {name_b}={s.JS_B.mean():.4f}")
    return s

def main():
    OUT="control_outputs"; os.makedirs(OUT,exist_ok=True)
    print("="*70); print("Tier-1 controls — 写入 ./control_outputs/ (不动 paper_outputs)"); print("="*70)

    # ---- 对照A：single @ 10000 ----
    df_single10k = run_single_at(10000, "single@10000")
    df_single10k.to_csv(os.path.join(OUT,"results_single_10000.csv"), index=False)

    # ---- 对照B：同质 ensemble ----
    df_homog = run_homogeneous_ensemble("homogeneous_ensemble@5x2000")
    df_homog.to_csv(os.path.join(OUT,"results_homogeneous_ensemble.csv"), index=False)

    print("\n[OK] 原始结果已保存到 control_outputs/。请把这两个 csv 发回。")

    # ---- 若能找到已有的 diverse ensemble 结果，自动对比给出初步结论 ----
    existing="paper_outputs/results_equal_per_member.csv"
    if os.path.exists(existing):
        de=pd.read_csv(existing)
        # 取 diverse ensemble 行（method 里含 'ensemble'），single@2000 行（含 'single'）
        de_ens=de[de["method"].str.contains("ensemble",case=False)].copy()
        de_sgl=de[de["method"].str.contains("single",case=False)].copy()
        print("\n找到已有 diverse-ensemble 结果，做初步配对对比：")
        paired_summary(df_single10k,"single@10000", de_ens,"diverse_ens")
        paired_summary(df_homog,"homog_ens", de_ens,"diverse_ens")
        print("\n判读：")
        print(" * 若 single@10000 的 JS ≈ 或 < diverse_ens → 'ensemble有用'主要是样本数效应，主线需改写。")
        print(" * 若 diverse_ens 明显 < single@10000 → ensemble确实带来超出样本数的增益，主线成立。")
        print(" * 若 homog_ens ≈ diverse_ens → 'diversity'一词无支撑，应改为'multi-member pooling'。")
        print(" * 若 diverse_ens < homog_ens → diversity 这个具体说法才站得住。")
    else:
        print("\n（没在 ./paper_outputs/ 找到原 diverse-ensemble 结果，跳过自动对比；把新csv发我即可。）")

if __name__=="__main__":
    main()
