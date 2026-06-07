#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#initially when writing his code it was done in separte files,
#therefore there are some repeats in functions created, which i will keep for consistency


# In[16]:


#figure 1
#reproduced from kovachki paper
#numerical experiments were recreated directly from the exact algorithmic definitions and parameters as described 
#in the paper using NumPy and Matplotlib, as utilising the library descriptions 
#This hold for figures 1,2 and 3 as 2 has all its algorithm described in th paper and the plots in 3 are the same
#but with noise added to the gradient.

LAM = 0.9
U0 = np.array([1.0, 1.0], dtype=float)
HS = [2**-8, 2**-9, 2**-10]   
KAPPAS = [5, 10, 20]
T = 2.0

#firstly we define all the basic functiosn to be inline iwth the paper exerts we wish to recreate:

def Q_mat(kappa: float) -> np.ndarray:
    #positie definite matrix required as stated in the paper allows for convex onjective function with no elongated valley etc 
    return np.diag([1.0, float(kappa)])

def f(u: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return -Q @ u # gradient flow created as defined for quadratic objective in paper 

def grad_flow_sol(ts: np.ndarray, u0: np.ndarray, Q: np.ndarray, lam: float) -> np.ndarray:
    q = np.diag(Q)
    g = np.exp(-np.outer(ts, q) / (1.0 - lam)) 
    return g * u0 #ode separates into two variabnles and this finds solution to grad flow equation

def momentum_traj(u0: np.ndarray,h: float,T: float,Q: np.ndarray,lam: float,a: float,)-> tuple[np.ndarray, np.ndarray]:
#momentum function with iterative steps as defined in kovachki and stuart paper

    n_steps = int(np.floor(T / h)) 
    ts = np.arange(n_steps + 1) * h

    us = np.zeros((n_steps + 1, len(u0)), dtype=float)
    us[0] = u0 #using initial starting value for itertaive scheme 

    if n_steps >= 1:
        us[1] = u0 + h * f(u0, Q) 

    for n in range(1, n_steps):
        delta = us[n] - us[n - 1] #consecutive updates as seen in chapter 3 of paper
        us[n + 1] = us[n] + lam * delta + h * f(us[n] + a * delta, Q) #update step as seen in kavachki

    return ts, us

def plot_fig(fig, row_idx: int, method_name: str, a: float) -> None:

    for col_idx, kappa in enumerate(KAPPAS):
        ax = fig.add_subplot(2, 3, row_idx * 3 + col_idx + 1)
        Q = Q_mat(kappa)

        t_exact = np.linspace(0.0, T, 2000)
        u_exact = grad_flow_sol(t_exact, U0, Q, LAM)
        ax.plot(u_exact[:, 1], u_exact[:, 0], "k--", linewidth=2, label="GF")

        for h in HS:
            _, u_num = momentum_traj(U0, h, T, Q, LAM, a)
            ax.plot(u_num[:, 1], u_num[:, 0], linewidth=1.8, label=f"h={h:g}")

        ax.set_title(f"{method_name}: kappa={kappa}")
        ax.legend(fontsize=9)
        ax.grid(False)
        ax.set_xlabel("u_2")
        ax.set_ylabel("u_1")
        ax.set_aspect("auto")
#obviously comparing u_1 and u_2 as is seen in paper as we have a coupled system
fig = plt.figure(figsize=(14, 9))
plot_fig(fig, row_idx=0, method_name="HB", a=0.0) #HB using a = 0 so momentume step matches polyak
plot_fig(fig, row_idx=1, method_name="NAG", a=LAM)#NAG using a = lambda, however not time dependent in this part

fig.tight_layout()
plt.show()


# In[8]:


#figure 2
#reproduced from kovachki paper
#numerical experiments were recreated directly from the exact algorithmic definitions and parameters as described 
#in the paper using NumPy and Matplotlib, as utilising the library descriptions 
LAM = 0.9
U0 = np.array([1.0, 1.0], dtype=float)

HS = [2**-5, 2**-6, 2**-7]

def ham_traj(ts: np.ndarray,u0: np.ndarray,Q: np.ndarray,lam: float,a: float,h: float,
) -> np.ndarray:

    alpha = 0.5 * (1.0 + lam - 2.0 * a * (1.0 - lam))
    q = np.diag(Q).astype(float)
    v0 = ((1.0 - 2.0 * alpha) / (2.0 * alpha - lam + 1.0)) * f(u0, Q)

    us = np.zeros((len(ts), len(u0)), dtype=float)

    for j, qj in enumerate(q):
        A = h * alpha
        B = 1.0 - lam
        C = qj

        disc = B * B - 4.0 * A * C

        if disc > 0:
            s = np.sqrt(disc)
            r1 = (-B + s) / (2.0 * A)
            r2 = (-B - s) / (2.0 * A)

            denom = r1 - r2
            c1 = (v0[j] - r2 * u0[j]) / denom
            c2 = (r1 * u0[j] - v0[j]) / denom

            us[:, j] = c1 * np.exp(r1 * ts) + c2 * np.exp(r2 * ts)

        elif np.isclose(disc, 0.0): #solutions to the hamiltonian ode as described
            r = -B / (2.0 * A)

            # u_j(t) = (c1 + c2 t)e^{rt}
            c1 = u0[j]
            c2 = v0[j] - r * c1

            us[:, j] = (c1 + c2 * ts) * np.exp(r * ts)

        else:
            s = np.sqrt(-disc)
            real = -B / (2.0 * A)
            imag = s / (2.0 * A)

            c1 = u0[j]
            c2 = (v0[j] - real * c1) / imag

            us[:, j] = np.exp(real * ts) * (
                c1 * np.cos(imag * ts) + c2 * np.sin(imag * ts)
            )

    return us


def momentum_traj(u0: np.ndarray,h: float,T: float,Q: np.ndarray,lam: float,a: float,
) -> tuple[np.ndarray, np.ndarray]:

    n_steps = int(np.floor(T / h))
    ts = np.arange(n_steps + 1) * h

    us = np.zeros((n_steps + 1, len(u0)), dtype=float)
    us[0] = u0

    if n_steps >= 1:
        us[1] = u0 + h * f(u0, Q)

    for n in range(1, n_steps):
        delta = us[n] - us[n - 1]
        us[n + 1] = us[n] + lam * delta + h * f(us[n] + a * delta, Q)

    return ts, us

def plot_fig(fig, row_idx: int, method_name: str, a: float) -> tuple:
    handles, labels = None, None

    for col_idx, kappa in enumerate(KAPPAS):
        ax = fig.add_subplot(2, 3, row_idx * 3 + col_idx + 1)
        Q = make_Q(kappa)

        for h in HS:
            t_exact = np.linspace(0.0, T, 4000)
            u_exact = ham_traj(t_exact, U0, Q, LAM, a, h)
            ax.plot(
                u_exact[:, 1],
                u_exact[:, 0],
                "--",
                linewidth=1.6,
                label=f"HAM(h={h:g})",
            )

            _, u_num = momentum_traj(U0, h, T, Q, LAM, a)
            ax.plot(
                u_num[:, 1],
                u_num[:, 0],
                linewidth=1.8,
                label=f"MM(h={h:g})",
            )

        ax.set_title(f"{method_name}: kappa={kappa}")
        ax.grid(False)
        ax.set_xlabel(r"$u_2$")
        ax.set_ylabel(r"$u_1$")
        ax.set_aspect("auto")

        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    return handles, labels


fig = plt.figure(figsize=(14, 9))

handles, labels = plot_method_row(fig, row_idx=0, method_name="HB", a=0.0)
plot_fig(fig, row_idx=1, method_name="NAG", a=LAM)

fig.legend(handles, labels, loc="center right",  fontsize=9, frameon=True, title="Trajectories",
)

fig.tight_layout(rect=[0.0, 0.0, 0.86, 1.0])
plt.show()


# In[17]:


#figure 3 is just reproducing previous figure but with additive noise in gradient step and taking mean 
#adapted from kovachki paper
#numerical experiments were recreated directly from the exact algorithmic definitions and parameters as described 
#in the paper using NumPy and Matplotlib, as utilising the library descriptions 
LAM = 0.9
U0 = np.array([1.0, 1.0], dtype=float)
HS = [2**-8, 2**-9, 3**-10]
KAPPAS = [5, 10, 20]
T = 2.0
#adapted from kovachki paper

def Q_mat(kappa: float) -> np.ndarray:
    return np.diag([1.0, float(kappa)]) #same as above in fig 1


def reg_grad(u: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return Q @ u #same as above in fig 1


def stoch_grad(u: np.ndarray,Q: np.ndarray,rng: np.random.Generator,noise_scale: float = 0.0) -> np.ndarray:
    noise = noise_scale * rng.standard_normal(size=u.shape)
    return Q @ u + noise #very similar to figure one exact with random added noise 


def grad_flow_sol(ts: np.ndarray,u0: np.ndarray,Q: np.ndarray,lam: float,) -> np.ndarray:
    q = np.diag(Q)
    g = np.exp(-np.outer(ts, q) / (1.0 - lam))
    return g * u0 #gradient flow as stated in paper and same as figure 1


def momentum_traj(u0: np.ndarray,h: float,T: float,Q: np.ndarray,lam: float,a: float,stochastic: bool = False,
    noise_scale: float = 0.0, rng: np.random.Generator | None = None,) -> tuple[np.ndarray, np.ndarray]: 
    #same as above above but we can choose to add noise element

    n_steps = int(np.floor(T / h)) #number of iteration we want to do
    ts = np.arange(n_steps + 1) * h

    us = np.zeros((n_steps + 1, len(u0)), dtype=float)
    us[0] = u0

    if stochastic and rng is None: #default random no. generator but cam make specifc choise if you want 
        rng = np.random.default_rng() #easier reproducibility

    if n_steps >= 1:
        if stochastic:
            g0 = stoch_grad(u0, Q, rng=rng, noise_scale=noise_scale)
        else:
            g0 = reg_grad(u0, Q)
        us[1] = u0 - h * g0

    for n in range(1, n_steps):
        delta = us[n] - us[n - 1]
        y = us[n] + a * delta  

        if stochastic:
            g = stoch_grad(y, Q, rng=rng, noise_scale=noise_scale)
        else:
            g = reg_grad(y, Q)

        us[n + 1] = us[n] + lam*delta - h*g

    return ts, us


def plot_fig(fig,row_idx: int,method_name: str,a: float,stochastic: bool = False,noise_scale: float = 0.0,
    n_paths: int = 20,seed: int = 0,) -> None:
    
    for col_idx, kappa in enumerate(KAPPAS): #creating plots for multiple condition numbers which are determined by
        ax = fig.add_subplot(2, 3, row_idx * 3 + col_idx + 1) #our choice of Q
        Q = Q_mat(kappa)

        t_exact = np.linspace(0.0, T, 2000)
        u_exact = grad_flow_sol(t_exact, U0, Q, LAM)
        ax.plot(u_exact[:, 1], u_exact[:, 0], "k--", linewidth=2, label="GF")

        for h in HS:
            if stochastic:# several stochastic paths lightly and one mean path
                paths = []
                for j in range(n_paths):
                    rng = np.random.default_rng(seed + 1000 * col_idx + 100 * j)
                    _, u_num = momentum_traj(
                        U0, h, T, Q, LAM, a,
                        stochastic=True,
                        noise_scale=noise_scale,
                        rng=rng,
                    )
                    paths.append(u_num)
                    ax.plot(u_num[:, 1], u_num[:, 0], alpha=0.18, linewidth=1.0)

                mean_path = np.mean(np.stack(paths, axis=0), axis=0)
                ax.plot(
                    mean_path[:, 1],
                    mean_path[:, 0],
                    linewidth=2.2,
                    label=f"h={h:g} (mean)"
                )
            else:
                _, u_num = momentum_traj(
                    U0, h, T, Q, LAM, a,
                    stochastic=False,
                )
                ax.plot(u_num[:, 1], u_num[:, 0], linewidth=1.8, label=f"h={h:g}")

        title = f"{method_name}: kappa={kappa}"
        if stochastic:
            title += f", SGD noise={noise_scale:g}"
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(False)
        ax.set_xlabel("u_2")
        ax.set_ylabel("u_1")
        ax.set_aspect("auto")


USE_STOCHASTIC = True
NOISE_SCALE = 0.25   # tried different vals, this one gives clearest results 
N_PATHS = 25

fig = plt.figure(figsize=(14, 9))

plot_fig(fig,row_idx=0, method_name="HB", a=0.0, stochastic=USE_STOCHASTIC, noise_scale=NOISE_SCALE,
        n_paths=N_PATHS, seed=123,)

plot_fig(fig,row_idx=1, method_name="NAG", a=LAM, stochastic=USE_STOCHASTIC, noise_scale=NOISE_SCALE,
        n_paths=N_PATHS, seed=456, ) #adding differnet noise to hb and nag to be more representative

fig.tight_layout()
plt.show()


# In[9]:


#figure 3 also
#adapted from kovachki paper and publically available code sources
LAM = 0.9
U0 = np.array([1.0, 1.0], dtype=float)

HS = [2**-5, 2**-6, 2**-7]
KAPPAS = [5, 10, 20]
T = 2.0


def Q_mat(kappa: float) -> np.ndarray:
    return np.diag([1.0, float(kappa)])


def f(u: np.ndarray, Q: np.ndarray) -> np.ndarray:
    # deterministic drift used in the modified equation / Hamiltonian model
    return -Q @ u


def grad_flow(u: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return Q @ u


def stoch_grad(u: np.ndarray, Q: np.ndarray, rng: np.random.Generator,  noise_scale: float = 0.0,
) -> np.ndarray:
    noise = noise_scale * rng.standard_normal(size=u.shape)
    return Q @ u + noise


def ham_traj( ts: np.ndarray, u0: np.ndarray,  Q: np.ndarray, lam: float, a: float, h: float,
) -> np.ndarray:
    alpha = 0.5 * (1.0 + lam - 2.0 * a * (1.0 - lam))
    q = np.diag(Q).astype(float)

    v0 = ((1.0 - 2.0 * alpha) / (2.0 * alpha - lam + 1.0)) * f(u0, Q)

    us = np.zeros((len(ts), len(u0)), dtype=float)

    for j, qj in enumerate(q):
        A = h * alpha
        B = 1.0 - lam
        C = qj

        disc = B * B - 4.0 * A * C

        if disc > 0:
            s = np.sqrt(disc)
            r1 = (-B + s) / (2.0 * A)
            r2 = (-B - s) / (2.0 * A)

            denom = r1 - r2
            c1 = (v0[j] - r2 * u0[j]) / denom
            c2 = (r1 * u0[j] - v0[j]) / denom

            us[:, j] = c1 * np.exp(r1 * ts) + c2 * np.exp(r2 * ts)

        elif np.isclose(disc, 0.0):
            r = -B / (2.0 * A)

            c1 = u0[j]
            c2 = v0[j] - r * c1

            us[:, j] = (c1 + c2 * ts) * np.exp(r * ts)

        else:
            s = np.sqrt(-disc)
            real = -B / (2.0 * A)
            imag = s / (2.0 * A)

            c1 = u0[j]
            c2 = (v0[j] - real * c1) / imag

            us[:, j] = np.exp(real * ts) * (
                c1 * np.cos(imag * ts) + c2 * np.sin(imag * ts)
            )

    return us


def momentum_traj(u0: np.ndarray, h: float, T: float, Q: np.ndarray, lam: float, a: float, stochastic: bool = False,
    noise_scale: float = 0.0, rng: np.random.Generator | None = None, ) -> tuple[np.ndarray, np.ndarray]:
    n_steps = int(np.floor(T / h))
    ts = np.arange(n_steps + 1) * h

    us = np.zeros((n_steps + 1, len(u0)), dtype=float)
    us[0] = u0

    if stochastic and rng is None:
        rng = np.random.default_rng()

    # first step
    if n_steps >= 1:
        if stochastic:
            g0 = stochastic_gradient(u0, Q, rng=rng, noise_scale=noise_scale)
        else:
            g0 = full_gradient(u0, Q)

        us[1] = u0 - h * g0

    for n in range(1, n_steps):
        delta = us[n] - us[n - 1]
        y = us[n] + a * delta

        if stochastic:
            g = stochastic_gradient(y, Q, rng=rng, noise_scale=noise_scale)
        else:
            g = full_gradient(y, Q)

        us[n + 1] = us[n] + lam * delta - h * g

    return ts, us


def plot_fig( fig, row_idx: int, method_name: str,  a: float, stochastic: bool = False, noise_scale: float = 0.0,
    n_paths: int = 20, seed: int = 0,
) -> tuple[list, list]:
    legend_handles = None
    legend_labels = None

    for col_idx, kappa in enumerate(KAPPAS):
        ax = fig.add_subplot(2, 3, row_idx * 3 + col_idx + 1)
        Q = Q_mat(kappa)

        for h_idx, h in enumerate(HS):
            t_ham = np.linspace(0.0, T, 4000)
            u_ham = ham_traj(t_ham, U0, Q, LAM, a, h)
            ax.plot( u_ham[:, 1], u_ham[:, 0],  "--",
                linewidth=1.6, label=f"HAM(h={h:g})",
            )

            if stochastic:
                paths = []
                for j in range(n_paths):
                    rng = np.random.default_rng(
                        seed + 10000 * col_idx + 1000 * h_idx + j
                    )
                    _, u_num = momentum_traj( U0, h,  T, Q, LAM, a,
                        stochastic=True, noise_scale=noise_scale, rng=rng,
                    )
                    paths.append(u_num)

                    # No label here, to avoid cluttering the legend
                    ax.plot(u_num[:, 1], u_num[:, 0], alpha=0.18,linewidth=1.0, )

                mean_path = np.mean(np.stack(paths, axis=0), axis=0)
                ax.plot(
                    mean_path[:, 1],
                    mean_path[:, 0],
                    linewidth=2.2,
                    label=f"SMM(h={h:g}, mean)",
                )
            else:
                _, u_num = momentum_traj( U0,  h, T, Q, LAM, a,
                    stochastic=False,)
                ax.plot( u_num[:, 1], u_num[:, 0], linewidth=1.8, label=f"MM(h={h:g})",)

        title = f"{method_name}: kappa={kappa}"
        if stochastic:
            title += f", SGD noise={noise_scale:g}"

        ax.set_title(title)
        ax.grid(False)
        ax.set_xlabel(r"$u_2$")
        ax.set_ylabel(r"$u_1$")
        ax.set_aspect("auto")

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    return legend_handles, legend_labels

USE_STOCHASTIC = True
NOISE_SCALE = 0.25
N_PATHS = 25

fig = plt.figure(figsize=(16, 9))

handles, labels = plot_method_row( fig, row_idx=0,  method_name="HB", a=0.0, stochastic=USE_STOCHASTIC, noise_scale=NOISE_SCALE,
    n_paths=N_PATHS, seed=123,)

plot_method_row(fig,  row_idx=1, method_name="NAG",  a=LAM, stochastic=USE_STOCHASTIC, noise_scale=NOISE_SCALE,
    n_paths=N_PATHS,  seed=456,)

fig.legend( handles, labels, loc="center right", fontsize=9, frameon=True,
    title="Trajectories",)

fig.tight_layout(rect=[0.0, 0.0, 0.86, 1.0])
plt.show()


# In[11]:


#figure 4
#reproduced from Su et al.
#we are looking at a classical nesterov example, many public resources available explaining
#the numpy functions etc, the code explicitly follows the formulas algorithms and parameters set out in the paper
#the plots just zooms in  on later times to show oscillatory behaviour as time goes on.
a= 0.02

b= 0.005
x0 = np.array([1.0, 1.0])

def f(x):
    return a*x[0]**2 + b*x[1]**2

def grad(x):
    return np.array([2*a*x[0], 2*b*x[1]])

def nesterov(s, K):
    xs = np.zeros((K+1, 2))
    ys = np.zeros((K+1, 2))
    xs[0] = x0
    ys[0] = x0

    for k in range(1, K+1):
        xs[k] = ys[k-1] - s * grad(ys[k-1])
        ys[k] = xs[k] + ((k-1)/(k+2)) * (xs[k] - xs[k-1])

    return xs

def ode_rhs(t, z):
    x = z[:2]
    v = z[2:]
    damping = 3/t if t > 1e-8 else 0
    return np.r_[v, -damping*v - grad(x)]

z0 = np.r_[x0, [0.0, 0.0]]#initial cond
t_eval = np.linspace(1e-6, 300, 3000)

sol = solve_ivp(
    ode_rhs, 
    (1e-6, 300), 
    z0, 
    t_eval=t_eval, 
    rtol=1e-9, 
    atol=1e-11
)

ode_x = sol.y[:2].T
ode_f = np.array([f(x) for x in ode_x])

runs = {
    "s = 1": nesterov(1.0, 300),
    "s = 0.25": nesterov(0.25, 600),
    "s = 0.05": nesterov(0.05, 1400),
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
for label, xs in runs.items():
    if label in ["s = 1", "s = 0.25"]:
        ax.plot(xs[:, 0], xs[:, 1], label=label)
ax.plot(ode_x[:, 0], ode_x[:, 1], "k-", label="ODE")
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_title("Trajectories")
ax.legend()
ax.axis("equal")

ax = axes[1]
for label, xs in runs.items():
    if label in ["s = 0.25", "s = 0.05"]:
        ax.plot(xs[:, 0], xs[:, 1], label=label)
ax.plot(ode_x[:, 0], ode_x[:, 1], "k-", label="ODE")
ax.set_xlim([-0.12, 0.14])
ax.set_ylim([-0.2, 0.2])
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_title("Zoomed trajectories")
ax.legend()

ax = axes[2]
for label, xs in runs.items():
    s = float(label.split("=")[1])
    k = np.arange(len(xs))
    t = k * np.sqrt(s)
    vals = np.array([f(x) for x in xs])
    ax.semilogy(t, vals, label=label)

ax.semilogy(t_eval, ode_f, "k-", label="ODE")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$f - f^*$")
ax.set_title("Errors")
ax.legend()

plt.tight_layout()
plt.show()


# In[13]:


#figure 5
#adapted from candes et al. 

def momentum_scheme(Q, u0, h, lam, a, T):
    f = lambda x: -Q @ x
    nsteps = int(np.round(T/h))
    U = np.zeros((nsteps+1, len(u0)))
    U[0] = u0
    if nsteps >= 1:
        U[1] = U[0] + h*f(U[0])
    for n in range(1, nsteps):
        U[n+1] = U[n] + lam*(U[n] - U[n-1]) + h*f(U[n] + a*(U[n] - U[n-1]))
    t = np.arange(nsteps+1)*h
    return t, U

def rescaled_flow(Q, u0, t, lam):
    # Q assumed diagonal for this plotting code
    vals = np.diag(Q)
    return u0 * np.exp(-np.outer(t/(1-lam), vals))

def objective(Q, U):
    return 0.5*np.einsum('...i,ij,...j->...', U, Q, U)


COL = { 
    'ode': '#222222',
    'nag': '#DE8F05',
}

Qn=np.diag([1.0,10.0]); L=10.0; s=1.0/L; N=450; x0=np.array([1.4,1.0])
x=np.zeros((N+1,2)); y=np.zeros((N+1,2)); x[0]=x0; y[0]=x0


for k in range(1,N+1):
    
    x[k]=y[k-1]-s*(Qn@y[k-1])
    beta=(k-1)/(k+2)
    y[k]=x[k]+beta*(x[k]-x[k-1])
    
    
t_nag=np.arange(N+1)*np.sqrt(s)

t0=np.sqrt(s)
def rhs_nag(t,z):
    u=z[:2]; v=z[2:]
    return np.concatenate([v, -(3/max(t,1e-8))*v - Qn@u])
sol=solve_ivp(rhs_nag,(t0,t_nag[-1]),np.concatenate([x0,np.zeros(2)]),t_eval=t_nag[1:],rtol=1e-8,atol=1e-10)


fig,axs=plt.subplots(1,3,figsize=(12,3.5))

axs[0].plot(x[:,0],x[:,1],color=COL['nag'],label='discrete NAG')
axs[0].plot(sol.y[0],sol.y[1],'--',color=COL['ode'],label='ODE')
axs[0].scatter([0],[0],color='black',s=15)
axs[0].set_xlabel('$x_1$'); axs[0].set_ylabel('$x_2$'); axs[0].set_title('(a) Trajectories'); axs[0].legend()
axs[1].plot(x[-120:,0],x[-120:,1],color=COL['nag'],label='discrete')
axs[1].plot(sol.y[0,-120:],sol.y[1,-120:],'--',color=COL['ode'],label='ODE')
axs[1].set_xlabel('$x_1$'); axs[1].set_ylabel('$x_2$'); axs[1].set_title('(b) Late-time oscillations')


err_disc=objective(Qn,x)
err_ode=objective(Qn,sol.y[:2].T)

axs[2].semilogy(t_nag,err_disc+1e-16,color=COL['nag'],label='discrete NAG')
axs[2].semilogy(t_nag[1:],err_ode+1e-16,'--',color=COL['ode'],label='ODE')
axs[2].set_xlabel('$t=k\\sqrt{s}$'); axs[2].set_ylabel('$f(x)-f^\\ast$'); axs[2].set_title('(c) Objective error'); axs[2].legend()


# In[14]:


# figure 6
#reproduced from candes except with varying r values
def rhs_r(t,y,r):
    x,v=y[0],y[1]
    return [v, -(r/max(t,1e-5))*v - x]

fig,ax=plt.subplots(figsize=(7.2,4.2))
t_eval=np.linspace(0.01,60,5000)

for r in [1,2,3,4,5]:
    sol=solve_ivp(rhs_r,(0.01,60),[1.0,0.0],args=(r,),t_eval=t_eval,rtol=1e-9,atol=1e-11)
    err=0.5*sol.y[0]**2
    ax.plot(t_eval,t_eval**2*err,label=f'$r={r}$')
    
ax.set_xlabel('$t$'); ax.set_ylabel('$t^2(f(X(t))-f^*)$')
ax.set_title('Scaled error for $\\ddot X+(r/t)\\dot X+\\nabla f(X)=0$')
ax.set_yscale('symlog',linthresh=1e-2)
ax.legend(ncol=3)


# In[ ]:





# In[ ]:





# In[ ]:




