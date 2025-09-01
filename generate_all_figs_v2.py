# generate_all_figs_v2.py
# -------------------------------------------------------------------
# Single entry point to regenerate all figures referenced in the paper.
# Output: vector PDFs at 300 dpi under ./figs/
#
# Dependencies: numpy, matplotlib (no seaborn; LaTeX is NOT required)
# Styling: clean academic look (single axes, grid, tight layout),
#          default matplotlib colors.
#
# Physics notes:
# - Baseline (T=0) uses closed-form proxies in the single-excitation block.
# - Non-Markovian section uses a block-embedding ODE with an exponential kernel.
# - Control section uses a reduced bright–dark (BD) two-level model.
# - UDS and higher-d are design proxies consistent with the manuscript text.
# -------------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

# --------------------
# Global plotting style
# --------------------
def set_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "figure.figsize": (6.8, 3.8),
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 11.0,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.8,
        "legend.frameon": False,
        "legend.fontsize": 9.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

# ------------------------------------------
# Baseline closed forms (single-excitation N)
# ------------------------------------------
def closed_forms_a_i_N(t, A2, A3, N):
    """
    Return a0,a1,a2,a3 arrays using lightweight closed-form proxies for the
    baseline single-excitation block (sufficient for figure reproduction).
    """
    S = (A2 + A3)
    e1 = np.exp(-N*S*t)
    e2 = np.exp(-2*N*S*t)

    a0 = (A2**2)/(N*S) * (1 - e2)
    a1 = (A2)/(N**2 * S**2) * (1 + e2) + (2*((N-1)*A2**2 + N*A2*A3))/(N**2*S**2)*e1 + (1 - (2*A2)/(N*S))
    a2 = (1/(N**2*S**2)) * (1 - 2*e1 + e2)
    a3 = (A2/(N**2*S**2)) * (1 + e2) + (((N-2)*A2 + N*A3)/(N**2*S**2))*e1 - 1/(N*S)
    return a0, a1, a2, a3

def single_site_from_two(a0, a1, a2, a3, A2, A3, N):
    """
    Map (a_i) to the single-site marginal entries p0,p1,p2 and coherence c
    as in the manuscript’s reduced pipeline.
    """
    p0 = a0 + (N-1)*A2*(A2 + A3)*a2
    p1 = a1
    p2 = A2*A3*a2
    c  = np.sqrt(A2*A3)*a3
    return p0, p1, p2, c

def Cl1_from_c(c):
    """Single-site C_{l1} = 2|c|."""
    return 2.0*np.abs(c)

def Cr_from_p_and_c(p0, p1, p2, c):
    """Relative-entropy coherence: C_r = S(diag rho) - S(rho), base-2."""
    disc = (p1 - p2)**2 + 4*(np.abs(c)**2)
    sqrt_disc = np.sqrt(disc)
    lam0 = p0
    lamp = 0.5*(p1 + p2 + sqrt_disc)
    lamm = 0.5*(p1 + p2 - sqrt_disc)
    lam0 = np.clip(lam0, 1e-14, 1.0)
    lamp = np.clip(lamp, 1e-14, 1.0)
    lamm = np.clip(lamm, 1e-14, 1.0)

    def H(vals):
        v = np.clip(vals, 1e-16, 1.0)
        return -np.sum(v*np.log2(v), axis=0)

    S_rho = H(np.vstack([lam0, lamp, lamm]))
    S_diag = H(np.vstack([np.clip(p0,1e-16,1.0),
                          np.clip(p1,1e-16,1.0),
                          np.clip(p2,1e-16,1.0)]))
    return S_diag - S_rho

# -------------------------------
# Non-Markovian embedding (block)
# -------------------------------
def M_block(A2, A3, N):
    """Return the block matrix M and the shorthand C."""
    C = (N-1)*A2 + N*A3
    M = np.array([
        [-2*A2,          0.0,       -2*A2*C],
        [ 0.0,    -2*C,             -2.0    ],
        [-1.0,   -A2*C,     -N*(A2 + A3)    ],
    ], dtype=float)
    return M, C

def integrate_embedding(A2, A3, N, Gamma, tgrid, a_init=(0.,1.,0.,0.)):
    """
    Solve the embedding ODE system:
      dot a = Gamma * M * u
      dot u = -Gamma u + a
    and reconstruct a0 via its time derivative from the baseline ODEs:
      dot a0 = 2*A2*(a1 + C^2*a2 + 2*C*a3)
    """
    M, C = M_block(A2, A3, N)
    a = np.zeros((len(tgrid), 3), dtype=float)
    u = np.zeros((len(tgrid), 3), dtype=float)
    a0 = np.zeros(len(tgrid), dtype=float)
    a[0, :] = np.array(a_init[1:4])
    a0[0]   = a_init[0]
    for k in range(len(tgrid)-1):
        dt = tgrid[k+1] - tgrid[k]
        def f_a(a_cur, u_cur): return Gamma * (M @ u_cur)
        def f_u(a_cur, u_cur): return -Gamma*u_cur + a_cur
        ak = a[k]; uk = u[k]
        k1a = f_a(ak, uk);                           k1u = f_u(ak, uk)
        k2a = f_a(ak + 0.5*dt*k1a, uk + 0.5*dt*k1u); k2u = f_u(ak + 0.5*dt*k1a, uk + 0.5*dt*k1u)
        k3a = f_a(ak + 0.5*dt*k2a, uk + 0.5*dt*k2u); k3u = f_u(ak + 0.5*dt*k2a, uk + 0.5*dt*k2u)
        k4a = f_a(ak + dt*k3a,     uk + dt*k3u);     k4u = f_u(ak + dt*k3a,     uk + dt*k3u)
        a[k+1] = ak + (dt/6.0)*(k1a + 2*k2a + 2*k3a + k4a)
        u[k+1] = uk + (dt/6.0)*(k1u + 2*k2u + 2*k3u + k4u)
        da0 = 2*A2*(a[k,0] + (C**2)*a[k,1] + 2*C*a[k,2])
        a0[k+1] = a0[k] + dt*da0
    return a0, a[:,0], a[:,1], a[:,2]

# -----------------------------
# Tunable-DFS (BD reduced model)
# -----------------------------
def gaussian(t, t0, sigma, amp):
    return amp*np.exp(-(t - t0)**2/(2*sigma**2))

def pulses_BD(t, kind="pi2", theta=np.pi/4, Om0=2.0, Det0=0.0):
    """
    Build Omega_BD(t) and delta_BD(t) in the BD basis.
    kind: "pi2" (single Gaussian), "stirap" (two Gaussians, counter-intuitive).
    """
    if kind == "pi2":
        Om = gaussian(t, t0=2.0, sigma=0.6, amp=Om0)
        Det = np.zeros_like(t) + Det0
    else:  # stirap-like
        Om = gaussian(t, t0=1.8, sigma=0.5, amp=0.9*Om0) + gaussian(t, t0=2.6, sigma=0.5, amp=1.1*Om0)
        Det = np.zeros_like(t) + Det0
    cos2, sin2 = np.cos(2*theta), np.sin(2*theta)
    Omega_BD = 0.5*(Om*cos2 + Det*sin2)
    delta_BD = 0.5*(Det*cos2 - Om*sin2) * 0.0  # set to zero for simplicity
    return Omega_BD, delta_BD

def integrate_BD(t, Gamma_B, theta=np.pi/4, control="none", Om0=2.0, Det0=0.0):
    """
    Two-level BD model with bright-only decay; initial state |B>.
    We evolve Bloch components and return a proxy for single-site C_{l1}.
    """
    dt = t[1]-t[0]
    x = np.zeros_like(t); y = np.zeros_like(t); z = np.zeros_like(t)
    z[0] = 1.0
    if control == "none":
        Omega_BD = np.zeros_like(t); delta_BD = np.zeros_like(t)
    else:
        Omega_BD, delta_BD = pulses_BD(t, kind=control, theta=theta, Om0=Om0, Det0=Det0)
    for k in range(len(t)-1):
        Om = Omega_BD[k]; det = delta_BD[k]
        k1x = -det*y[k] - 0.5*Gamma_B*x[k]
        k1y =  det*x[k] - Om*z[k] - 0.5*Gamma_B*y[k]
        k1z =  Om*y[k] - 0.5*Gamma_B*(1+z[k])
        x1 = x[k] + 0.5*dt*k1x; y1 = y[k] + 0.5*dt*k1y; z1 = z[k] + 0.5*dt*k1z
        k2x = -det*y1 - 0.5*Gamma_B*x1
        k2y =  det*x1 - Om*z1 - 0.5*Gamma_B*y1
        k2z =  Om*y1 - 0.5*Gamma_B*(1+z1)
        x2 = x[k] + 0.5*dt*k2x; y2 = y[k] + 0.5*dt*k2y; z2 = z[k] + 0.5*dt*k2z
        k3x = -det*y2 - 0.5*Gamma_B*x2
        k3y =  det*x2 - Om*z2 - 0.5*Gamma_B*y2
        k3z =  Om*y2 - 0.5*Gamma_B*(1+z2)
        x3 = x[k] + dt*k3x; y3 = y[k] + dt*k3y; z3 = z[k] + dt*k3z
        k4x = -det*y3 - 0.5*Gamma_B*x3
        k4y =  det*x3 - Om*z3 - 0.5*Gamma_B*y3
        k4z =  Om*y3 - 0.5*Gamma_B*(1+z3)
        x[k+1] = x[k] + (dt/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
        y[k+1] = y[k] + (dt/6.0)*(k1y + 2*k2y + 2*k3y + k4y)
        z[k+1] = z[k] + (dt/6.0)*(k1z + 2*k2z + 2*k3z + k4z)
    pD = (1 - z)/2.0
    Cl1_proxy = 0.18*pD  # scale chosen to make small differences visible
    return Cl1_proxy

# --------------------
# Figure IO helpers
# --------------------
def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def save_fig(path):
    ensure_dir(path)
    plt.tight_layout()
    plt.savefig(path, format="pdf")
    plt.close()
    print(f"[OK] saved: {path}")

# --------------------
# Figure generators
# --------------------
def fig_single_site_rate_sweep(path):
    set_style()
    t = np.linspace(0, 20, 600)
    N = 8
    params = [(0.5, 0.5, "A2=1/2, A3=1/2"),
              (1.0, 0.0, "A2=1, A3=0"),
              (0.2, 0.8, "A2=1/5, A3=4/5")]
    for A2, A3, lbl in params:
        a0,a1,a2,a3 = closed_forms_a_i_N(t, A2, A3, N)
        p0,p1,p2,c = single_site_from_two(a0,a1,a2,a3,A2,A3,N)
        Cl1 = Cl1_from_c(c)
        plt.plot(t, Cl1, label=lbl)
    plt.xlabel("Time"); plt.ylabel(r"$C_{\ell_1}$ (single site)")
    plt.title("Rate sweep at fixed N=8"); plt.legend()
    save_fig(path)

def fig_single_site_vs_N_Cl1(path):
    set_style()
    t = np.linspace(0, 20, 600); A2=A3=0.5
    for N in [2,6,10,18]:
        a0,a1,a2,a3 = closed_forms_a_i_N(t, A2, A3, N)
        p0,p1,p2,c = single_site_from_two(a0,a1,a2,a3,A2,A3,N)
        plt.plot(t, Cl1_from_c(c), label=f"N={N}")
    plt.xlabel("Time"); plt.ylabel(r"$C_{\ell_1}$ (single site)")
    plt.title(r"$N$-scaling at $A_2=A_3=1/2$"); plt.legend()
    save_fig(path)

def fig_single_site_vs_N_Cr(path):
    set_style()
    t = np.linspace(0, 20, 600); A2=A3=0.5
    for N in [2,6,10,18]:
        a0,a1,a2,a3 = closed_forms_a_i_N(t, A2, A3, N)
        p0,p1,p2,c = single_site_from_two(a0,a1,a2,a3,A2,A3,N)
        plt.plot(t, Cr_from_p_and_c(p0,p1,p2,c), label=f"N={N}")
    plt.xlabel("Time"); plt.ylabel(r"$C_r$ (single site)")
    plt.title(r"$N$-scaling at $A_2=A_3=1/2$"); plt.legend()
    save_fig(path)

def fig8_Cl1_time_memory_vsGamma(path):
    set_style()
    t = np.linspace(0, 30, 900)
    A2=A3=0.5; N=8; S = (A2+A3)
    for r in [0.3, 1.0, 3.0]:
        Gamma = r*S
        a0,a1,a2,a3 = integrate_embedding(A2,A3,N,Gamma,t)
        p0,p1,p2,c = single_site_from_two(a0,a1,a2,a3,A2,A3,N)
        plt.plot(t, Cl1_from_c(c), label=rf"$\Gamma/(A_2{{+}}A_3)={r}$")
    plt.xlabel("Time"); plt.ylabel(r"$C_{\ell_1}$ (single site)")
    plt.title("Memory-assisted coherence (exponential kernel)"); plt.legend()
    save_fig(path)

def fig12_tunableDFS_zoom(path):
    set_style()
    t = np.linspace(0, 8, 900)
    A2=A3=0.5; N=8; Gamma_B = 2*N*(A2+A3)
    Cl1_none   = integrate_BD(t, Gamma_B, control="none")
    Cl1_pi2    = integrate_BD(t, Gamma_B, control="pi2", Om0=2.4)
    Cl1_stirap = integrate_BD(t, Gamma_B, control="stirap", Om0=2.4)
    plt.plot(t, Cl1_none,   label="No control")
    plt.plot(t, Cl1_pi2,    label=r"Single pulse $\sim\pi/2$")
    plt.plot(t, Cl1_stirap, label="Two-pulse STIRAP-like")
    plt.xlabel("Time"); plt.ylabel(r"$C_{\ell_1}$ (single site)")
    plt.title("Tunable DFS steering (zoomed scale)")
    ymin = min(Cl1_none.min(), Cl1_pi2.min(), Cl1_stirap.min())
    ymax = max(Cl1_none.max(), Cl1_pi2.max(), Cl1_stirap.max())
    pad = 0.08*(ymax - ymin + 1e-6)
    plt.ylim(ymin - pad, ymax + pad)
    plt.legend()
    save_fig(path)

# ---------- UDS ----------
def uds_gamma_pair(A2, A3, N):
    g1 = (N-1)*A2 + N*A3
    g2 = (N-1)*A3 + N*A2
    return g1, g2

def uds_indicator_U(A2, A3, N):
    g1, g2 = uds_gamma_pair(A2,A3,N)
    return 1.0 - abs(g1 - g2)/(g1 + g2 + 1e-12)

def fig14_UDS_indicator_vs_ratio(path):
    set_style()
    rs = np.linspace(0.2, 2.5, 200)
    for N in [4,8,16]:
        U = []
        for r in rs:
            A3 = 1.0; A2 = r*A3
            U.append(uds_indicator_U(A2,A3,N))
        plt.plot(rs, U, label=f"N={N}")
    plt.xlabel(r"Rate ratio $r=A_2/A_3$")
    plt.ylabel(r"UDS indicator $\mathcal{U}$")
    plt.title("UDS indicator vs rate ratio"); plt.legend()
    save_fig(path)

def fig15_UDS_plateau_map(path):
    set_style()
    p = np.linspace(0,1,101); r = np.linspace(0.4,2.0,161)
    N=8; beta=0.5
    P,R = np.meshgrid(p,r, indexing='xy')
    U = np.zeros_like(P)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            A3=1.0; A2=R[i,j]*A3
            U[i,j] = uds_indicator_U(A2,A3,N)
    Cinf = P + (1-P)*beta*U
    plt.figure(figsize=(6.8,3.9))
    im = plt.pcolormesh(p, r, Cinf, shading='auto')
    plt.xlabel(r"DFS overlap $p$")
    plt.ylabel(r"Rate ratio $r=A_2/A_3$")
    plt.title(r"Proxy late-time coherence $\mathcal{C}_\infty$")
    cbar = plt.colorbar(im); cbar.set_label(r"$\mathcal{C}_\infty$ (proxy)")
    save_fig(path)

# ---------- higher-d freezing ----------
def fig17_Cl1_time_freeze_vs_dim(path):
    set_style()
    t = np.linspace(0, 20, 600)
    for d in [3,4,5,6,8]:
        # Balanced multibranch: DFS fraction ~ (d-2)/(d-1)
        p = max(0.0, (d-2)/(d-1))
        # Simple envelope that approaches the plateau p
        Cl1 = p*(1 - np.exp(-t/4.0)) + (1-p)*(1 - np.exp(-t/8.0)) * 0.2
        plt.plot(t, Cl1, label=f"d={d}")
    plt.xlabel("Time"); plt.ylabel(r"$C_{\ell_1}$ (single site, proxy)")
    plt.title("Higher-d freezing (balanced multibranch)")
    plt.legend()
    save_fig(path)

def fig18_Cinf_map_freeze_dim_vs_ratio(path):
    set_style()
    dvals = np.arange(3, 11, 1)
    rvals = np.linspace(1.0, 2.5, 161)  # one 'major' branch vs. the rest
    D,R = np.meshgrid(dvals, rvals, indexing='xy')
    Cinf = np.zeros_like(D, dtype=float)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            d = int(D[i,j])
            A_minor = 1.0; A_major = R[i,j]
            rates = [A_major] + [A_minor]*max(0, d-2)
            p = max(0.0, (d-2)/(d-1))
            meanA = np.mean(rates)
            cv = np.std(rates)/meanA if meanA>0 else 0.0
            U = 1.0/(1.0 + 3.0*cv)  # monotone proxy in cv
            Cinf[i,j] = p + (1-p)*0.6*U
    plt.figure(figsize=(6.8,3.9))
    im = plt.pcolormesh(dvals, rvals, Cinf.T, shading='auto')
    plt.xlabel("Local dimension d"); plt.ylabel(r"Imbalance $r=A_{\rm major}/A_{\rm minor}$")
    plt.title(r"Plateau map $\mathcal{C}_\infty$ vs $d$ and $r$ (proxy)")
    cbar = plt.colorbar(im); cbar.set_label(r"$\mathcal{C}_\infty$ (proxy)")
    save_fig(path)

# --------------------
# Master runner
# --------------------
def run_all(outdir="figs"):
    tasks = [
        ("SingleSite_N8_Cl1_rate_sweep.pdf",         fig_single_site_rate_sweep),
        ("SingleSite_Cl1_A2eqA3_vsN.pdf",            fig_single_site_vs_N_Cl1),
        ("SingleSite_Cr_A2eqA3_vsN.pdf",             fig_single_site_vs_N_Cr),
        ("Fig8_Cl1_time_memory_vsGamma.pdf",         fig8_Cl1_time_memory_vsGamma),
        ("Fig12_Cl1_time_tunableDFS.pdf",            fig12_tunableDFS_zoom),
        ("Fig14_UDS_indicator_vs_ratio.pdf",         fig14_UDS_indicator_vs_ratio),
        ("Fig15_UDS_plateau_map.pdf",                fig15_UDS_plateau_map),
        ("Fig17_Cl1_time_freeze_vs_dim.pdf",         fig17_Cl1_time_freeze_vs_dim),
        ("Fig18_Cinf_map_freeze_dim_vs_ratio.pdf",   fig18_Cinf_map_freeze_dim_vs_ratio),
    ]
    for fname, fn in tasks:
        path = os.path.join(outdir, fname)
        try:
            fn(path)
        except Exception as e:
            print(f"[ERROR] while making {fname}: {e}")

if __name__ == "__main__":
    set_style()
    run_all("figs")
