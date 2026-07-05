"""A-priori truncation test of the FV viscous operator L_visc(u) = mu[ Σ_f o_f(u_i-u_nbr) - Σ_a W_a g_a ]
vs the exact ∫_fluidCV(-mu ∇²u) dV, for the manufactured field u = sdf (u=0 on the sphere, ∇u=n̂,
∇²u = 2/r). The wall gradient g_a = n_a is EXACT for u=sdf, so this isolates the FACE-flux term.
Compares (A) face-CENTRE two-point flux vs (B) face flux evaluated at the open-area centroid."""
import numpy as np
R=0.3102; C0=np.array([0.013,-0.007,0.004]); MU=0.1
def sdf(x,y,z): return np.sqrt((x-C0[0])**2+(y-C0[1])**2+(z-C0[2])**2)-R
def lap(x,y,z): return 2.0/np.sqrt((x-C0[0])**2+(y-C0[1])**2+(z-C0[2])**2)  # ∇²(sdf)=2/r
def face_open(N,axis,h):
    a1,a2=(axis+1)%3,(axis+2)%3; w=np.arange(N+1)*h-0.5; t1lo=np.arange(N)*h-0.5; t2lo=t1lo
    rho2=R*R-(w-C0[axis])**2; O=np.ones((N+1,N,N)); c1,c2=C0[a1],C0[a2]
    for k in np.nonzero(rho2>0)[0]:
        rho=np.sqrt(rho2[k]); j1=np.nonzero(np.abs(t1lo+h/2-c1)<rho+h)[0]; j2=np.nonzero(np.abs(t2lo+h/2-c2)<rho+h)[0]
        if len(j1)==0 or len(j2)==0: continue
        J1,J2=np.meshgrid(j1,j2,indexing="ij"); lo1=t1lo[J1.ravel()]; lo2=t2lo[J2.ravel()]
        t=lo1[:,None]+(np.arange(256)[None,:]+0.5)*(h/256); s2=rho*rho-(t-c1)**2; half=np.sqrt(np.maximum(s2,0))
        zlo=np.maximum(c2-half,lo2[:,None]); zhi=np.minimum(c2+half,lo2[:,None]+h)
        O[k,J1.ravel(),J2.ravel()]=1.0-np.clip(zhi-zlo,0,None).mean(axis=1)/h
    return O
def run(N):
    h=1.0/N; c=(np.arange(N)+0.5)*h-0.5; X,Y,Z=np.meshgrid(c,c,c,indexing="ij")
    S=sdf(X,Y,Z); fl=S>=0; U=np.where(fl,S,0.0)  # u=sdf in fluid, 0 in solid
    o=[]; W=[]
    for a in range(3):
        O=face_open(N,a,h); om=O[:-1]; op=O[1:]
        om=np.moveaxis(om,[0,1,2],[a,(a+1)%3,(a+2)%3]); op=np.moveaxis(op,[0,1,2],[a,(a+1)%3,(a+2)%3])
        o.append((om,op)); W.append(h*h*(om-op))
    # discrete FV viscous: mu[ Σ o_f h (u_i-u_nbr) - Σ W_a g_a ], g_a = n_a (exact for u=sdf)
    r=np.sqrt((X-C0[0])**2+(Y-C0[1])**2+(Z-C0[2])**2); nx=(X-C0[0])/r; ny=(Y-C0[1])/r; nz=(Z-C0[2])/r
    nvec=[nx,ny,nz]
    faces=np.zeros((N,N,N))
    for a in range(3):
        om,op=o[a]; um=np.roll(U,+1,a); up=np.roll(U,-1,a)  # u_{i-a}, u_{i+a}
        faces+= h*(om*(U-um)+op*(U-up))    # o_f * h * (u_i - u_nbr), h from area h^2 / spacing h
    wall=sum(W[a]*nvec[a] for a in range(3))  # Σ W_a n_a  (h^2 folded into W)
    Lvisc=MU*(faces - wall)
    # exact ∫_fluidCV(-mu ∇²u) dV via 6^3 subsample
    ns=6; off=(np.arange(ns)+0.5)/ns-0.5
    ex=np.zeros((N,N,N))
    cut=(np.abs(W[0])+np.abs(W[1])+np.abs(W[2]))>1e-9
    ii,jj,kk=np.nonzero(cut)
    for I,J,K in zip(ii,jj,kk):
        xs=c[I]+off*h; ys=c[J]+off*h; zs=c[K]+off*h
        XX,YY,ZZ=np.meshgrid(xs,ys,zs,indexing="ij"); fld=sdf(XX,YY,ZZ)>=0
        ex[I,J,K]=-MU*np.sum(np.where(fld,lap(XX,YY,ZZ),0.0))*(h/ns)**3
    # interior (non-cut fluid): exact = -mu*lap*h^3 ; discrete faces (o=1) = mu*(6u_i-Σnbr) approx
    res=(Lvisc-ex)[cut]
    return np.sqrt(np.mean(res**2)), np.max(np.abs(res)), cut.sum()
prev=None
print(f"{'N':>4} | {'rms resid':>11} {'ord':>5} | {'max resid':>11} {'ord':>5} | Ncut")
for N in (32,64,128):
    rms,mx,nc=run(N)
    o=np.log2(prev[0]/rms) if prev else float('nan'); om=np.log2(prev[1]/mx) if prev else float('nan')
    print(f"{N:>4} | {rms:>11.3e} {o:>5.2f} | {mx:>11.3e} {om:>5.2f} | {nc}")
    prev=(rms,mx)
