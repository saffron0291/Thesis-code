#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def metropolis(target, initial, proposal, iterations=100_000):
    samples = [initial()]

    for _ in range(iterations):
        current = samples[-1]
        proposed = proposal(current)
        if np.random.random() < target(proposed) / target(current):
            samples.append(proposed)
        else:
            samples.append(current)

    return samples


# In[3]:


def sigmoid(arr):
    return 1 / (1 + np.exp(-arr))
alpha = -1.5
beta = np.array([1.5, -1.6])

data = np.random.multivariate_normal(
    np.zeros(2), np.array([[4, -2], [-2, 4]]), size=10
)
labels = np.random.binomial(1, p=sigmoid(data @ beta + alpha))


# In[5]:


class log_pdf:
    def __init__(self, x, y, prior_scale=5):
        self.x = x
        self.y = y
        self.prior_scale = prior_scale

    @staticmethod
    def likelihood_func(x, y, alpha, beta):
        eta = 1 / (1 + np.exp(-(alpha + np.dot(beta, x))))
        if y == 1:
            return eta
        return (1 - eta)


    def call(self, sample):
        alpha, beta = sample
        prior = np.exp(
            -(alpha ** 2 + np.linalg.norm(beta) ** 2) / (2 * self.prior_scale ** 2)
        )

        likelihood = 1
        for x, y in zip(self.x, self.y):
            likelihood *= self.likelihood_func(x, y, alpha, beta)

        return prior * likelihood


# In[13]:


class norm_prop_log:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        alpha, beta = sample
        alpha_jump = np.random.normal(
            scale=self.scale, size=alpha.shape
        )
        beta_jump = np.random.normal(
            scale=self.scale, size=beta.shape
        )
        return alpha + alpha_jump, beta + beta_jump


# In[14]:


target = log_pdf(data, labels)
proposal = norm_prop_log(0.1)
samples = metropolis(target, lambda: (np.array(0), np.array([0, 0])), proposal)


# In[8]:


fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

axes[0].scatter(  data[labels == 0, 0], data[labels == 0, 1], c="tab:orange",
    edgecolors="white", linewidths=1.8, s=90, marker="o", label="Class 0"
)

axes[0].scatter( data[labels == 1, 0], data[labels == 1, 1],  c="tab:blue", edgecolors="white",
    linewidths=1.8, s=90, marker="o", label="Class 1"
)

axes[0].set_title("Observed data")
axes[0].set_xlabel("x1")
axes[0].set_ylabel("x2")
axes[0].legend()

mesh = axes[1].pcolormesh( x1_grid, x2_grid, pred_probs, shading="auto",
    cmap="RdBu", vmin=0, vmax=1)

axes[1].scatter( data[labels == 0, 0], data[labels == 0, 1], c="tab:orange", edgecolors="white",  linewidths=1.8,
    s=90, marker="o", label="Class 0"
)

axes[1].scatter( data[labels == 1, 0], data[labels == 1, 1], c="tab:blue", edgecolors="white",
    linewidths=1.8, s=90, marker="o" label="Class 1"
)

axes[1].set_title("Posterior predictive probability")
axes[1].set_xlabel("x1")
axes[1].set_ylabel("x2")
axes[1].legend()


for ax in axes:
    ax.set_aspect("equal")
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)

cbar = fig.colorbar(mesh, ax=axes[1])
cbar.set_label("Predicted probability")

plt.tight_layout()
plt.show()


# Posterior sampling for the following model
# \begin{align*}
#     p(x) &\propto \exp(-x_1^2/10 - x_2^2/10 - 2 (x_2 - x_1^2)^2), \\
#     p(y | x) &= \mathcal{N}(y; H x, 0.1)
# \end{align*}
# where $H = [0, 1]$.

# In[9]:


y = np.array([2.0])
sig_lik = 0.1
H = np.array([0, 1])
rng = np.random.default_rng(36)


# In[10]:


def prior(x):
    return np.exp(-x[0]**2/10 - x[1]**2/10 - 2 * (x[1] - x[0]**2)**2)

def log_prior(x): 
    return -x[0]**2/10 - x[1]**2/10 - 2 * (x[1] - x[0]**2)**2

def log_likelihood(y, x, sig_lik): 
    return -(y - H@x)**2/(2 * sig_lik) - np.log((sig_lik * 2 * np.pi)**(-0.5))


# In[11]:


N = 10000 #random walk not included 
samples_RW = np.zeros((2, N))

# initial values
samples_RW[:, 0] = np.array([0, 0])


# parameters

sigma_RW = 0.1 
burnin = 50


for n in range(1, N):
    # random walk
    x_s = samples_RW[:, n-1] + np.sqrt(sigma_RW) * rng.normal(0, 1, 2) 
    # metropolis
    u = rng.uniform(0, 1)

    if np.log(u) < (log_prior(x_s) - log_prior(samples_RW[:, n-1]) + log_likelihood(y, x_s, sig_lik) - 
                    log_likelihood(y, samples_RW[:, n-1], sig_lik)):
        samples_RW[:, n] = x_s #acceptance ratio condition, if holds keep x_s, else use previous sample.

    else:
        samples_RW[:, n] = samples_RW[:, n-1] #not accepted
        
x_bb = np.linspace(-4, 4, 100)
y_bb = np.linspace(-2, 6, 100)
X_bb , Y_bb = np.meshgrid(x_bb , y_bb) 
Z_bb = np.zeros((100 , 100))
for i in range(100):
    for j in range(100):
        Z_bb[i, j] = prior([X_bb[i, j], Y_bb[i, j]])
plt.contourf(X_bb , Y_bb , Z_bb , 100 , cmap='RdBu') 
plt.scatter(samples_RW[0, burnin:], samples_RW[1, burnin:], s=10 , c='white')
plt.show()


# In[12]:


#mala
N = 10000 
gamma = 0.05 
burnin = 1000
sigma_langevin = np.sqrt(2 * gamma)
samples_langevin = np.zeros((2, N))
samples_langevin[:, 0] = np.array([0, 0])

def grad_log_prior(x): #array of partial derivatives
    return np.array([-x[0]/5 + 8 * (x[1] - x[0]**2) * x[0], -x[1]/5 - 4 * (x[1] - x[0]**2)]) 

def grad_log_likelihood(y, x, sig_lik): 
    return ((y - H@x)/sig_lik) * H 

def log_MALA_kernel(x_s, x, gamma, grad_log_prior): #the proposal modified by the gradient of the log prior and lkelihood
    return -(1/(4*gamma)) * np.linalg.norm(x_s - x - gamma * (grad_log_prior(x)
+ grad_log_likelihood(y, x, sig_lik)))**2 - np.log(4 * np.pi * gamma)

for n in range(1, N): # langevin
    x_s_l = (samples_langevin[:, n-1] + gamma * (grad_log_prior(samples_langevin[:, n-1]) + 
                                                 grad_log_likelihood(y, samples_langevin[:, n-1], sig_lik))+ 
                                                  sigma_langevin * rng.normal(0, 1, 2))
    # metropolis

    u = rng.normal(0, 1)

    ratio = (log_prior(x_s_l) - log_prior(samples_langevin[:, n-1])+ 
             log_MALA_kernel(samples_langevin[:, n-1], x_s_l, gamma, grad_log_prior)- 
             log_MALA_kernel(x_s_l, samples_langevin[:, n-1], gamma, grad_log_prior)+ 
             log_likelihood(y, x_s_l, sig_lik) - log_likelihood(y, samples_langevin[:, n-1], sig_lik))
    
    
    if np.log(u) < ratio:
        samples_langevin[:, n] = x_s_l
    else:
        samples_langevin[:, n] = samples_langevin[:, n-1]

x_bb = np.linspace(-4, 4, 100)
y_bb = np.linspace(-2, 6, 100)
X_bb , Y_bb = np.meshgrid(x_bb , y_bb) 
Z_bb = np.zeros((100 , 100))
for i in range(100):
    for j in range(100):
        Z_bb[i, j] = prior([X_bb[i, j], Y_bb[i, j]])
plt.contourf(X_bb , Y_bb , Z_bb , 100 , cmap='RdBu')
plt.scatter(samples_langevin[0, burnin:], samples_langevin[1, burnin:], s=10 , c='white') 
plt.title("MALA")
plt.show()


# In[13]:


#ula
N = 10000 
gamma = 0.01 
burnin = 50

sigma_langevin2 = np.sqrt(2 * gamma) 
samples_langevin2 = np.zeros((2, N))
samples_langevin2[:, 0] = np.array([0, 0])
for n in range(1, N): # langevin
    x_s_l = (samples_langevin2[:, n-1] + gamma * (grad_log_prior(samples_langevin2[:, n-1]) 
                                                  + grad_log_likelihood(y, samples_langevin2[:, n-1], sig_lik)) 
             + sigma_langevin2 * rng.normal(0, 1, 2)) #standard normal sampling 
    
    samples_langevin2[:, n] = x_s_l
    
x_bb = np.linspace(-4, 4, 100)
y_bb = np.linspace(-2, 6, 100)
X_bb , Y_bb = np.meshgrid(x_bb , y_bb) 
Z_bb = np.zeros((100 , 100))
for i in range(100):
    for j in range(100):
        Z_bb[i, j] = prior([X_bb[i, j], Y_bb[i, j]])
plt.contourf(X_bb , Y_bb , Z_bb , 100 , cmap='RdBu')
plt.scatter(samples_langevin2[0, burnin:], samples_langevin[1, burnin:], s=10 , c='white') 
plt.title("ULA")
plt.show()


# In[14]:


x_bb = np.linspace(-4, 4, 100)
y_bb = np.linspace(-2, 6, 100)

X_bb, Y_bb = np.meshgrid(x_bb, y_bb)
Z_bb = np.zeros((100, 100))

for i in range(100):
    for j in range(100):
        Z_bb[i, j] = prior([X_bb[i, j], Y_bb[i, j]])

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)

axes[0].contourf(X_bb, Y_bb, Z_bb, 100, cmap="RdBu")
axes[0].scatter(
    samples_langevin[0, burnin:],
    samples_langevin[1, burnin:],
    s=10,
    c="white"
)
axes[0].set_title("MALA")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")



axes[1].contourf(X_bb, Y_bb, Z_bb, 100, cmap="RdBu")
axes[1].scatter(
    samples_langevin2[0, burnin:],      # change this if your ULA samples have another name
    samples_langevin2[1, burnin:],
    s=10,
    c="white"
)
axes[1].set_title("ULA")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")

plt.tight_layout()
plt.show()


# In[16]:


y = np.array([2.0])
sig_lik = 0.1
h_mat = np.array([0, 1])
rng = np.random.default_rng(36)

def log_posterior(y, x, sig_lik):
    return log_prior(x) + log_likelihood(y, x, sig_lik).item()

def U(y, x, sig_lik):
    return -log_posterior(y, x, sig_lik)

def log_likelihood(y, x, sig_lik): 
    return -(y - h_mat@x)**2/(2 * sig_lik) - np.log((sig_lik * 2 * np.pi)**(-0.5))


def log_prior(x):
    return -x[0]**2/10 - x[1]**2/10 - 2 * (x[1] - x[0]**2)**2

def grad_log_prior(x):
    x1 = x[0]
    x2 = x[1]

    dlogprior_dx1 = -x1 / 5 + 8 * x1 * (x2 - x1**2)
    dlogprior_dx2 = -x2 / 5 - 4 * (x2 - x1**2)

    return np.array([dlogprior_dx1, dlogprior_dx2])

def grad_log_likelihood(y, x, sig_lik):
    return ((y - h_mat @ x) / sig_lik * h_mat).reshape(2)

def grad_log_posterior(y, x, sig_lik):
    return grad_log_prior(x) + grad_log_likelihood(y, x, sig_lik)


def grad_U(y, x, sig_lik):
    return -grad_log_posterior(y, x, sig_lik)



# In[34]:


N = 10000
samples_HMC = np.zeros((2, N))

# initial values
samples_HMC[:, 0] = np.array([0, 0])

# HMC parameters
epsilon = 0.05
L = 20
burnin = 50

accept_count = 0

for n in range(1, N):

    current_x = samples_HMC[:, n-1].copy() #  setting current position

    p = rng.normal(0, 1, 2) # sample momentum
    current_p = p.copy()

    x_s = current_x.copy() # proposal position initialised at current position

    # leapfrog integration
    p = p - 0.5 * epsilon * grad_U(y, x_s, sig_lik) # leapfrog integration

    for i in range(L):
        x_s = x_s + epsilon * p

        if i != L - 1:
            p = p - epsilon * grad_U(y, x_s, sig_lik)

    p = p - 0.5 * epsilon * grad_U(y, x_s, sig_lik)

    # negate momentum
    p = -p

    # hamiltonian equations
    current_U = U(y, current_x, sig_lik)
    current_K = 0.5 * np.sum(current_p**2)

    proposed_U = U(y, x_s, sig_lik)
    proposed_K = 0.5 * np.sum(p**2)

    # metropolis acceptance step
    u = rng.uniform(0, 1)

    if np.log(u) < current_U - proposed_U + current_K - proposed_K:
        samples_HMC[:, n] = x_s
        accept_count += 1
    else:
        samples_HMC[:, n] = current_x

print("HMC acceptance rate:", accept_count / (N - 1))


# In[36]:


x_bb = np.linspace(-4, 4, 100)
y_bb = np.linspace(-2, 6, 100)
X_bb , Y_bb = np.meshgrid(x_bb , y_bb) 
Z_bb = np.zeros((100 , 100))
for i in range(100):
    for j in range(100):
        Z_bb[i, j] = prior([X_bb[i, j], Y_bb[i, j]])
plt.contourf(X_bb , Y_bb , Z_bb , 100 , cmap='RdBu')
plt.scatter(samples_HMC[0, burnin:], samples_HMC[1, burnin:], s=10 , c='white') 
plt.title("HMC")
plt.grid(False)
plt.show()


# In[22]:


# Figure 9 Gaussian HMC trajectory

def hmc(n=6000,burn=1000,eps=0.075,L=18):
    x=np.array([0.0,1.0]); out=[]; acc=0
    for i in range(n):
        p=rng.normal(size=2); x0=x.copy(); p0=p.copy()
        p=p-0.5*eps*grad_U_banana(x)
        for j in range(L):
            x=x+eps*p
            if j!=L-1:
                p=p-eps*grad_U_banana(x)
        p=p-0.5*eps*grad_U_banana(x)
        p=-p
        H0=U_banana(x0)+0.5*np.dot(p0,p0); H1=U_banana(x)+0.5*np.dot(p,p)
        if np.log(rng.random()) < min(0,-H1+H0):
            acc+=1
        else:
            x=x0
        if i>=burn: out.append(x.copy())
    return np.array(out),acc/n



Sigma=np.array([[1,0.95],[0.95,1]]); Prec=np.linalg.inv(Sigma)

def U_gauss(q): 
    return 0.5*q@Prec@q

def grad_U_gauss(q): 
    return Prec@q

q=np.array([-1.5,-1.55]); p=np.array([-1.0,1.0]); eps=0.25; Lsteps=25

qs=[q.copy()]; ps=[p.copy()]; Hs=[U_gauss(q)+0.5*p@p]

p=p-0.5*eps*grad_U_gauss(q)

for j in range(Lsteps):
    q=q+eps*p
    if j!=Lsteps-1:
        p=p-eps*grad_U_gauss(q)
    qs.append(q.copy()); ps.append(p.copy()); Hs.append(U_gauss(q)+0.5*p@p)
    
p=p-0.5*eps*grad_U_gauss(q)

qs=np.array(qs); ps=np.array(ps); Hs=np.array(Hs)

#plottng 
fig,axs=plt.subplots(1,3,figsize=(12,3.5))
xx=np.linspace(-3,3,200); yy=np.linspace(-3,3,200); XX,YY=np.meshgrid(xx,yy); P=np.exp(-0.5*(Prec[0,0]*XX**2+2*Prec[0,1]*XX*YY+Prec[1,1]*YY**2)); P/=P.max()

axs[0].contour(XX,YY,P,levels=10,cmap='Greys')
axs[0].plot(qs[:,0],qs[:,1],'o-',markersize=2,color=COL['hmc'])
axs[0].set_title('(a) Position trajectory'); axs[0].set_xlabel('$q_1$'); axs[0].set_ylabel('$q_2$'); axs[0].set_aspect('equal',adjustable='box')

axs[1].plot(ps[:,0],ps[:,1],'o-',markersize=2,color=COL['hb'])
axs[1].set_title('(b) Momentum trajectory'); axs[1].set_xlabel('$p_1$'); axs[1].set_ylabel('$p_2$'); axs[1].set_aspect('equal',adjustable='box')

axs[2].plot(np.arange(len(Hs)),Hs-Hs[0],'o-',markersize=3,color=COL['nag'])
axs[2].axhline(0,color='black',lw=0.8)
axs[2].set_title('(c) Hamiltonian error'); axs[2].set_xlabel('leapfrog step'); axs[2].set_ylabel('$H_l-H_0$')


# In[25]:


def hb_sigma2_cal(h, lam, q):
    return (1.0 - lam) * (2.0 * (1.0 + lam) - h * q) / (h * (1.0 + lam))

def hb_iact(h, lam, q):
    return (1.0 - lam) * (2.0 / (h * q) - 1.0 / (1.0 + lam))

def gd_iact(h, q):
    return (2.0 - h * q) / (h * q)

def stationary_hb_pair(rng, R, q, h, lam):
    # Var(u_n)=1/q and Cov(u_n,u_{n-1})=rho1/q for the calibrated chain.
    rho1 = (1.0 + lam - h * q) / (1.0 + lam)
    cov = np.array([[1.0/q, rho1/q], [rho1/q, 1.0/q]])
    z = rng.multivariate_normal([0.0, 0.0], cov, size=R)
    return z[:, 0], z[:, 1]  # u_{-1}, u_0

def empirical_iact_hb(h, lam, q, N=6000, R=1500, seed=0):
    rng = np.random.default_rng(seed)
    u_prev, u = stationary_hb_pair(rng, R, q, h, lam)
    sigma2 = hb_sigma2_cal(h, lam, q)
    update_sd = h * np.sqrt(sigma2)
    means = np.zeros(R)
    for _ in range(N):
        noise = rng.normal(size=R)
        u_next = u + lam * (u - u_prev) - h * q * u + update_sd * noise
        u_prev, u = u, u_next
        means += u
    means /= N
    return N * np.var(means, ddof=1) * q

def empirical_iact_gd(h, q, N=6000, R=1500, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(scale=1.0 / np.sqrt(q), size=R)  # start in stationarity
    update_var = h * (2.0 - h * q)
    update_sd = np.sqrt(update_var)
    means = np.zeros(R)
    a = 1.0 - h * q
    for _ in range(N):
        x = a * x + update_sd * rng.normal(size=R)
        means += x
    means /= N
    return N * np.var(means, ddof=1) * q

qA = 1.0
hA = 0.5
lam_grid = np.linspace(0.0, 0.95, 300)
iact_A = hb_iact(hA, lam_grid, qA)
lam_pts = np.array([0.0, 0.2, 0.4, 0.6, 0.75, 0.85, 0.92])
emp_A = np.array([empirical_iact_hb(hA, lam, qA, N=5000, R=1600, seed=300+i)
                  for i, lam in enumerate(lam_pts)])

#part b
kappas = np.array([10, 30, 100, 300, 1000, 3000], dtype=float)

# analytical worst-direction vals
iact_gd_exact = kappas
iact_hb_exact = 2.0 * kappas**1.5 / (kappas + 1.0)
emp_gd = []
emp_hb = []
for i, kappa in enumerate(kappas):
    m = 1.0
    L = kappa
    h_gd = 2.0 / (L + m)
    sqrtL = np.sqrt(L)
    h_hb = 4.0 / (sqrtL + 1.0)**2
    lam_hb = ((sqrtL - 1.0) / (sqrtL + 1.0))**2
    # Use longer chains for larger kappa; started exactly in stationarity.
    N_gd = int(max(30000, min(300000, 80 * kappa)))
    N_hb = int(max(30000, min(120000, 300 * iact_hb_exact[i])))
    R = 600
    emp_gd.append(empirical_iact_gd(h_gd, m, N=N_gd, R=R, seed=500+i))
    emp_hb.append(empirical_iact_hb(h_hb, lam_hb, m, N=N_hb, R=R, seed=600+i))
emp_gd = np.array(emp_gd)
emp_hb = np.array(emp_hb)

# part c, IACT for kappa=1000.
kC = 1000.0
m = 1.0
L = kC
h_gd_C = 2.0 / (L + m)
sqrtL = np.sqrt(L)
h_hb_C = 4.0 / (sqrtL + 1.0)**2
lam_hb_C = ((sqrtL - 1.0) / (sqrtL + 1.0))**2
q_grid = np.logspace(np.log10(m), np.log10(L), 400)
iact_gd_dir = gd_iact(h_gd_C, q_grid)
iact_hb_dir = hb_iact(h_hb_C, lam_hb_C, q_grid)

# fig  d: relaxation scale via spectral radius.
rho_gd = (kappas - 1.0) / (kappas + 1.0)
rho_hb = (np.sqrt(kappas) - 1.0) / (np.sqrt(kappas) + 1.0)
relax_gd = 1.0 / (1.0 - rho_gd)
relax_hb = 1.0 / (1.0 - rho_hb)

fig, axs = plt.subplots(2, 2, figsize=(10, 7.4))

ax = axs[0, 0]
ax.plot(lam_grid, iact_A, label='exact IACT')
ax.scatter(lam_pts, emp_A, s=24, marker='o', label='stationary simulation', zorder=3)
ax.set_xlabel(r'momentum $\lambda$')
ax.set_ylabel(r'$\tau_{\mathrm{int}}$')
ax.set_title(r'(a) Scalar calibrated HB IACT, $hq=0.5$')
ax.legend()

ax = axs[0, 1]
ax.loglog(kappas, iact_gd_exact, label=r'GD exact: $O(\kappa)$')
ax.loglog(kappas, iact_hb_exact, label=r'HB exact: $O(\sqrt{\kappa})$')
ax.scatter(kappas, emp_gd, s=24, marker='o', label='GD simulation', zorder=3)
ax.scatter(kappas, emp_hb, s=24, marker='s', label='HB simulation', zorder=3)
ax.set_xlabel(r'condition number $\kappa$')
ax.set_ylabel(r'worst-coordinate $\tau_{\mathrm{int}}$')
ax.set_title('(b) Analytical IACT scaling, simulation check')
ax.legend(ncol=1)

ax = axs[1, 0]
ax.semilogx(q_grid, iact_gd_dir, label='GD')
ax.semilogx(q_grid, iact_hb_dir, label='HB')
ax.axvline(m, linestyle='--', linewidth=1.0, color='0.4')
ax.axvline(L, linestyle=':', linewidth=1.0, color='0.4')
ax.set_xlabel(r'eigenvalue $q$')
ax.set_ylabel(r'$\tau_{\mathrm{int}}(q)$')
ax.set_title(r'(c) Smallest eigenvalue is bottleneck, $\kappa=1000$')
ax.legend()

ax = axs[1, 1]
ax.loglog(kappas, relax_gd, label=r'GD $1/(1-\rho)$')
ax.loglog(kappas, relax_hb, label=r'HB $1/(1-\rho)$')
ax.set_xlabel(r'condition number $\kappa$')
ax.set_ylabel('relaxation scale')
ax.set_title('(d) Spectral-radius relaxation scale')
ax.legend()

fig.suptitle('Autocorrelation acceleration for Gaussian-calibrated Heavy Ball', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
#print('Saved', FIGDIR / 'fig11_iact_scaling.pdf')
print('Panel A exact', iact_A[[0,-1]], 'emp', emp_A)
print('Panel B empirical GD', emp_gd)
print('Panel B empirical HB', emp_hb)


# In[30]:


#discretisation ssytems as seen in aper refernces in hmc part of paper by Neal
def euler(q, p, eps, n):
    Q = [q]
    P = [p]
    for _ in range(n):
        q_old, p_old = q, p
        p = p_old - eps * q_old
        q = q_old + eps * p_old
        Q.append(q)
        P.append(p)
    return np.array(Q), np.array(P)

def modified_euler(q, p, eps, n):
    Q = [q]
    P = [p]
    for _ in range(n):
        p = p - eps * q
        q = q + eps * p
        Q.append(q)
        P.append(p)
    return np.array(Q), np.array(P)

def leapfrog_harm(q, p, eps, n):
    Q = [q]
    P = [p]
    for _ in range(n):
        p = p - 0.5 * eps * q
        q = q + eps * p
        p = p - 0.5 * eps * q
        Q.append(q)
        P.append(p)
    return np.array(Q), np.array(P)

settings = [
    (r'Explicit Euler, $\epsilon=0.3$', euler, 0.3),
    (r'Modified Euler, $\epsilon=0.3$', modified_euler, 0.3),
    (r'Leapfrog, $\epsilon=0.3$', leapfrog_harm, 0.3),
    (r'Leapfrog, $\epsilon=1.2$', leapfrog_harm, 1.2),
]

theta = np.linspace(0, 2 * np.pi, 400)

fig, axs = plt.subplots(
    2, 2,
    figsize=(8.5, 7.2),
    constrained_layout=True
)

for ax, (title, fn, eps) in zip(axs.ravel(), settings):
    q, p = fn(0.0, 1.0, eps, 34)

    ax.plot(np.cos(theta), np.sin(theta), color='gray', lw=1, alpha=0.8)
    ax.plot(q, p, 'o-', color='black', markersize=2, lw=0.8)

    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title, fontsize=10, pad=8)
    ax.tick_params(labelsize=8)

    # Make sure labels do not crowd neighbouring panels
    ax.set_xlabel(r'$q$', labelpad=4)
    ax.set_ylabel(r'$p$', labelpad=4)

    # Optional: keep all panels on comparable limits
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)

fig.suptitle('Hamiltonian harmonic oscillator discretisations', fontsize=12)

plt.savefig(
    'harmonic_integrators.pdf',
    bbox_inches='tight',
    pad_inches=0.08
)

plt.show()


# In[29]:


def hb_var(h, lam, q, sigma2):
    return h * sigma2 * (1 + lam) / ((1 - lam) * q * (2 * (1 + lam) - h * q))

def gd_var(h, q, sigma2):
    return h * sigma2 / (q * (2 - h * q))

def hb_cal_sigma2(h, lam, q):
    return (1 - lam) * (2 * (1 + lam) - h * q) / (h * (1 + lam))

def simulate_hb_scalar(h, lam, q, sigma2, n=120000, burn=10000, seed=0):
    rng = np.random.default_rng(seed)
    u_prev = 0.0
    u = 0.0
    vals = []
    sigma = np.sqrt(sigma2)
    for k in range(n + burn):
        noise = rng.normal()
        u_next = u + lam * (u - u_prev) - h * q * u + h * sigma * noise
        u_prev, u = u, u_next
        if k >= burn:
            vals.append(u)
    return np.var(np.asarray(vals), ddof=1)

def simulate_hb_diag(h, lam, q_vec, sigma2_vec, n=180000, burn=15000, seed=1):
    rng = np.random.default_rng(seed)
    q_vec = np.asarray(q_vec, dtype=float)
    sigma = np.sqrt(np.asarray(sigma2_vec, dtype=float))
    d = len(q_vec)
    u_prev = np.zeros(d)
    u = np.zeros(d)
    vals = []
    for k in range(n + burn):
        noise = rng.normal(size=d)
        u_next = u + lam * (u - u_prev) - h * q_vec * u + h * sigma * noise
        u_prev, u = u, u_next
        if k >= burn:
            vals.append(u.copy())
    vals = np.asarray(vals)
    return np.cov(vals, rowvar=False)

# scalar stationary variance vs lambda
q = 8.0
h = 0.04
sigma2 = 1.0
lam_grid = np.linspace(0.0, 0.94, 200)
var_theory = hb_var(h, lam_grid, q, sigma2)
lam_pts = np.linspace(0.0, 0.9, 10)
var_emp = np.array([simulate_hb_scalar(h, lam, q, sigma2, seed=100+i) for i, lam in enumerate(lam_pts)])

# HB/GD ratio
ratio_exact = hb_var(h, lam_grid, q, sigma2) / gd_var(h, q, sigma2)
ratio_asym = 1.0 / (1.0 - lam_grid)

# normal calibration errors
q_vec = np.array([1.0, 20.0])
h2 = 0.02
lam2 = 0.88
true_cov = np.diag(1.0 / q_vec)
uncal_sigma2 = np.ones_like(q_vec)
cal_sigma2 = np.array([hb_cal_sigma2(h2, lam2, qi) for qi in q_vec])
emp_uncal = simulate_hb_diag(h2, lam2, q_vec, uncal_sigma2, seed=7)
emp_cal = simulate_hb_diag(h2, lam2, q_vec, cal_sigma2, seed=8)
th_uncal = np.diag([hb_var(h2, lam2, qi, si2) for qi, si2 in zip(q_vec, uncal_sigma2)])
th_cal = np.diag([hb_var(h2, lam2, qi, si2) for qi, si2 in zip(q_vec, cal_sigma2)])
errors_emp = [np.linalg.norm(emp_uncal - true_cov, ord='fro'), np.linalg.norm(emp_cal - true_cov, ord='fro')]
errors_th = [np.linalg.norm(th_uncal - true_cov, ord='fro'), np.linalg.norm(th_cal - true_cov, ord='fro')]

# Panel D: collapse vs h for fixed noise and calibrated noise
qD = 5.0
lamD = 0.8
h_grid = np.logspace(-3, np.log10(0.25), 180)
fixed_var = hb_var(h_grid, lamD, qD, sigma2=1.0)
cal_var = np.array([hb_var(hh, lamD, qD, hb_cal_sigma2(hh, lamD, qD)) for hh in h_grid])
true_var = 1.0 / qD

fig, axs = plt.subplots(2, 2, figsize=(10, 7.4))
ax = axs[0,0]
ax.plot(lam_grid, var_theory, label='closed form')
ax.scatter(lam_pts, var_emp, s=22, marker='o', label='simulation', zorder=3)
ax.set_xlabel(r'momentum $\lambda$')
ax.set_ylabel(r'$\operatorname{Var}(u_\infty)$')
ax.set_title('(a) Stationary variance of noisy HB')
ax.legend()

ax = axs[0,1]
ax.plot(lam_grid, ratio_exact, label='exact HB/GD ratio')
ax.plot(lam_grid, ratio_asym, linestyle='--', label=r'$(1-\lambda)^{-1}$')
ax.set_xlabel(r'momentum $\lambda$')
ax.set_ylabel('variance amplification')
ax.set_title('(b) Momentum amplifies the noise floor')
ax.set_ylim(0.8, min(20, ratio_exact.max()*1.05))
ax.legend()

ax = axs[1,0]
x = np.arange(2)
width = 0.36
ax.bar(x - width/2, errors_th, width, label='theory')
ax.bar(x + width/2, errors_emp, width, label='simulation')
ax.set_xticks(x)
ax.set_xticklabels(['uncalibrated', 'calibrated'])
ax.set_ylabel(r'$\|\widehat\Sigma - Q^{-1}\|_F$')
ax.set_yscale('log')
ax.set_ylim(1e-4, 2)
ax.set_title('(c) Gaussian posterior calibration')
for xi, val in zip(x + width/2, errors_emp):
    ax.text(xi, max(val, 1e-6) + max(errors_emp+errors_th)*0.04, f'{val:.3g}', ha='center', va='bottom', fontsize=7)
ax.legend()

ax = axs[1,1]
ax.loglog(h_grid, fixed_var, label='fixed gradient-noise variance')
ax.loglog(h_grid, cal_var, label='calibrated noise')
ax.axhline(true_var, linestyle='--', linewidth=1.2, label='posterior variance')
ax.set_xlabel(r'step size $h$')
ax.set_ylabel(r'$\operatorname{Var}(u_\infty)$')
ax.set_title(r'(d) Fixed noise collapses as $h\to 0$')
ax.legend()

fig.suptitle('Noise amplification and Gaussian calibration for Heavy Ball', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
print('Panel C errors empirical', errors_emp, 'theory', errors_th)



# In[ ]:




