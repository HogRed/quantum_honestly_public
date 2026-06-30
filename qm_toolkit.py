"""
qm_toolkit -- the shared toolkit for the "Quantum, Honestly" YouTube course.

Design contract (from the course outline):
  * ONE canonical state object: a plain NumPy complex array. Transparent and
    copy-paste-safe across every episode. (QuTiP/matplotlib are only for drawing.)
  * ONE measurement function:  measure(state, axis) -> (outcome, collapsed_state)
    plus born_probs(...) and born_sample(...). Episodes never rebuild projectors.
  * Helpers are deposited where they are first taught:
        normalize()  -- E2 (first use)
        spread()/variance() -- E4 (qubit standard deviation)
        spread_x()/expect_p()/spread_p()/uncertainty_product() -- E4/E7 (Heisenberg, on the wavefunction)
        solve_schrodinger() -- E8 (the "one solver to rule them all")
        split_step()        -- E7 (the wavepacket propagator), reused in E10
        annihilation()/creation()/number_op()/oscillator_H() -- E9 (ladder operators, pure NumPy)
        barrier_transmission() -- E10 (the analytic square-barrier T, Tong eq. 2.54)
        two_slit_intensity()/sample_hits() -- E1, reused in the wave block
  * The names in the course-outline.md code blocks (E7-E11) are kept resolvable by
    thin aliases at the bottom of PART C: propagate=split_step, solve_well=solve_schrodinger,
    fd_eigensolver(x,V), gaussian_packet(width=), T_analytic=barrier_transmission. The
    canonical name is the source of truth; test_qm_toolkit.py's Part D guards this contract.

Everything runs on numpy + scipy. matplotlib is only needed for the plotting
helpers; importing the rest of the module never requires it. QuTiP is NOT required
anywhere (the E9 ladder operators are pure NumPy); to_qutip() is an optional Bloch-only adapter.

Units: throughout the wave-mechanics section we use hbar = m = 1 unless you pass
your own values. This keeps the on-screen numbers clean.
"""

from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------------
# 0.  Reproducible randomness (so on-camera demos are repeatable)
# ---------------------------------------------------------------------------
_rng = np.random.default_rng()

def seed(s: int) -> None:
    """Reseed the toolkit's random generator so a demo replays identically."""
    global _rng
    _rng = np.random.default_rng(s)


# ===========================================================================
# PART A -- THE QUBIT  (Episodes 2-5)
# ===========================================================================
# A qubit state is just two complex numbers in a column: [a, b].
# |a|^2 is the chance of measuring "up", |b|^2 the chance of "down".

def ket(a, b) -> np.ndarray:
    """Build a qubit state from two (complex) numbers and normalize it."""
    return normalize(np.array([a, b], dtype=complex))

def normalize(state: np.ndarray) -> np.ndarray:
    """Scale a state so the squared lengths add to 1 ('the particle is somewhere')."""
    state = np.asarray(state, dtype=complex)
    norm = np.sqrt(np.vdot(state, state).real)
    if norm == 0:
        raise ValueError("Cannot normalize the zero vector.")
    return state / norm

# The computational / Z basis -- "up" and "down".
UP   = np.array([1, 0], dtype=complex)   # |0>, spin up,   Z = +1
DOWN = np.array([0, 1], dtype=complex)   # |1>, spin down, Z = -1

# The other four cardinal states (equator of the Bloch sphere).
PLUS_X  = ket(1,  1)    # (1/sqrt2)[1, 1]
MINUS_X = ket(1, -1)    # (1/sqrt2)[1,-1]
PLUS_Y  = ket(1,  1j)   # (1/sqrt2)[1, i]
MINUS_Y = ket(1, -1j)   # (1/sqrt2)[1,-i]

def bloch_state(theta: float, phi: float) -> np.ndarray:
    """The state at polar angle theta, azimuth phi on the Bloch sphere:
       a = cos(theta/2),  b = e^{i phi} sin(theta/2)."""
    return np.array([np.cos(theta / 2),
                     np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex)

# --- the Pauli matrices: the three measurement directions, as 2x2 grids ----
I2 = np.array([[1, 0], [0, 1]], dtype=complex)
X  = np.array([[0, 1], [1, 0]], dtype=complex)
Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z  = np.array([[1, 0], [0, -1]], dtype=complex)

def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """[A, B] = AB - BA. Zero means 'order doesn't matter' (compatible)."""
    return A @ B - B @ A

def overlap(phi: np.ndarray, psi: np.ndarray) -> complex:
    """The inner product <phi|psi>: conjugate the first, multiply componentwise, add.
    Its squared magnitude is the chance of finding |psi> to be |phi>."""
    return complex(np.vdot(phi, psi))     # vdot already conjugates the first arg

def _axis_operator(axis) -> np.ndarray:
    """Turn an axis label ('X','Y','Z') or a Bloch direction (nx,ny,nz)
    into the 2x2 observable n . sigma with outcomes +-1."""
    if isinstance(axis, str):
        return {"X": X, "Y": Y, "Z": Z}[axis.upper()]
    n = np.asarray(axis, dtype=float)
    nn = np.linalg.norm(n)
    if nn == 0:
        raise ValueError("axis direction cannot be the zero vector")
    n = n / nn
    return n[0] * X + n[1] * Y + n[2] * Z

def _eigenstates(axis):
    """Return {+1: |+>, -1: |->} -- the two states the detector can snap to."""
    op = _axis_operator(axis)
    vals, vecs = np.linalg.eigh(op)          # vals ascending: [-1, +1]
    return {int(round(vals[1])): vecs[:, 1],
            int(round(vals[0])): vecs[:, 0]}

def born_probs(state: np.ndarray, axis="Z") -> dict:
    """The odds of each outcome along an axis: P(+1)=|<+|psi>|^2, P(-1)=|<-|psi>|^2."""
    state = normalize(state)          # be forgiving: odds are defined for any nonzero state
    es = _eigenstates(axis)
    return {ev: abs(overlap(vec, state)) ** 2 for ev, vec in es.items()}

def measure(state: np.ndarray, axis="Z", rng=None):
    """Measure once. Returns (outcome, collapsed_state): the detector reads +1 or -1
    with the Born-rule odds, and the state SNAPS to the matching eigenstate."""
    rng = rng or _rng
    state = normalize(state)
    es = _eigenstates(axis)
    p_plus = abs(overlap(es[+1], state)) ** 2
    if rng.random() < p_plus:
        return +1, normalize(es[+1])
    return -1, normalize(es[-1])

def born_sample(state: np.ndarray, axis="Z", shots=1000, rng=None) -> np.ndarray:
    """Fire the same prepared state through the detector `shots` times; return the +-1 readings."""
    rng = rng or _rng
    p = born_probs(state, axis)
    return rng.choice([+1, -1], size=shots, p=[p[+1], p[-1]])

def expectation(state: np.ndarray, axis="Z") -> float:
    """The average reading <O> = <psi| O |psi> (a number between -1 and +1)."""
    state = normalize(state)
    op = _axis_operator(axis)
    return float(np.real(np.vdot(state, op @ state)))

def variance(state: np.ndarray, axis="Z") -> float:
    """Spread^2 of the readings. For a +-1 observable: 1 - <O>^2."""
    return 1.0 - expectation(state, axis) ** 2

def spread(state: np.ndarray, axis="Z") -> float:
    """Standard deviation of the measurement outcomes (introduced in E4)."""
    return float(np.sqrt(max(variance(state, axis), 0.0)))

def bloch_vector(state: np.ndarray):
    """Where the state's pin sits on the globe: (<X>, <Y>, <Z>)."""
    return (expectation(state, "X"), expectation(state, "Y"), expectation(state, "Z"))


# ===========================================================================
# PART B -- THE DOUBLE SLIT  (Episode 1, reused in the wave block)
# ===========================================================================

def two_slit_intensity(screen, slit_sep=20.0, wavelength=1.0, dist=1000.0,
                       slit_width=4.0, both=True):
    """Brightness along a screen for the two-slit experiment.
    `both=True`  -> amplitudes add THEN square (|psi1+psi2|^2): real fringes.
    `both=False` -> the incoherent (add-the-intensities) result: a single fringe-free
                    diffraction hump. (Note: NOT the two-pile *ballistic* baseline, which is
                    two separated Gaussians -- see E1_demo.)
    Returns an array normalized to a peak of 1 (shape only; peak-normalization drops the
    physical factor of 2 between the incoherent sum and a single slit)."""
    screen = np.asarray(screen, dtype=float)
    theta = np.arctan2(screen, dist)
    k = 2 * np.pi / wavelength
    # single-slit diffraction envelope (each slit alone)
    beta = k * slit_width * np.sin(theta) / 2
    envelope = np.sinc(beta / np.pi) ** 2          # (sin x / x)^2
    if both:
        delta = k * slit_sep * np.sin(theta) / 2   # half the path difference
        pattern = np.cos(delta) ** 2 * envelope    # amplitudes add, then squared
    else:
        pattern = envelope                          # incoherent sum: one diffraction hump (peak-normalized)
    return pattern / pattern.max()

def sample_hits(screen, intensity, shots=2000, rng=None) -> np.ndarray:
    """Drop `shots` electrons one at a time, landing them with probability ~ intensity.
    Returns the screen positions where they hit (build the pattern dot-by-dot)."""
    rng = rng or _rng
    p = np.asarray(intensity, dtype=float)
    s = p.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError("intensity must be non-negative with a positive total")
    p = p / s
    idx = rng.choice(len(screen), size=shots, p=p)
    return np.asarray(screen)[idx]


# ===========================================================================
# PART C -- THE WAVEFUNCTION  (Episodes 6-11)
# ===========================================================================
# A wavefunction is the qubit idea taken to the limit: instead of 2 numbers,
# it's one complex number at every point in space -- a vector with infinitely
# many components, sampled here onto a grid.

def grid(x_min=-10.0, x_max=10.0, n=1000, N=None, L=None):
    """A 1-D position grid. Returns (x, dx). Two calling conventions:
        grid(x_min, x_max, n)     -- explicit endpoints (the E8 style)
        grid(N=points, L=length)  -- N points over the centered domain [-L/2, L/2] (the E10 style)
    """
    if N is not None or L is not None:
        if N is None or L is None:
            raise ValueError("grid(N=, L=) needs BOTH N (points) and L (domain length)")
        x = np.linspace(-L / 2.0, L / 2.0, int(N))
        return x, x[1] - x[0]
    x = np.linspace(x_min, x_max, n)
    return x, x[1] - x[0]

def solve_schrodinger(V, x, n_states=6, hbar=1.0, mass=1.0):
    """THE ONE SOLVER (E8). Turn the time-independent Schrodinger equation into a
    matrix and ask the computer for its eigenvalues/eigenvectors.

    V : array of the potential at each grid point (or a callable V(x)).
    Returns (energies, states) with `n_states` lowest bound states,
    each wavefunction normalized so sum(|psi|^2) dx = 1."""
    x = np.asarray(x, dtype=float)
    dx = x[1] - x[0]
    Vv = V(x) if callable(V) else np.asarray(V, dtype=float)
    n = len(x)
    # kinetic energy: -hbar^2/2m * d^2/dx^2  via the tridiagonal second difference
    main = hbar**2 / (mass * dx**2) * np.ones(n) + Vv
    off  = -hbar**2 / (2 * mass * dx**2) * np.ones(n - 1)
    from scipy.linalg import eigh_tridiagonal
    energies, vecs = eigh_tridiagonal(main, off, select="i",
                                      select_range=(0, n_states - 1))
    states = vecs.T
    states = np.array([s / np.sqrt(np.sum(np.abs(s)**2) * dx) for s in states])
    # pin a sign convention (largest-magnitude lobe positive) so plots don't flip run-to-run
    states = np.array([s * np.sign(s[np.argmax(np.abs(s))]) for s in states])
    return energies, states

def gaussian_wavepacket(x, x0=0.0, k0=0.0, sigma=1.0):
    """A localized moving 'lump': a Gaussian bump of width sigma at x0,
    given momentum by the phase factor e^{i k0 x}. Normalized."""
    x = np.asarray(x, dtype=float)
    psi = np.exp(1j * k0 * x) * np.exp(-(x - x0)**2 / (2 * sigma**2))
    dx = x[1] - x[0]
    return psi / np.sqrt(np.sum(np.abs(psi)**2) * dx)

def split_step(psi, x, V, dt, steps=1, hbar=1.0, mass=1.0):
    """THE PROPAGATOR (E7). Evolve psi forward in time with the split-step Fourier
    method: half a potential kick, a full free 'drift' in momentum space, half a kick.
    Reused for tunnelling in E10. Returns the evolved wavefunction.

    Note: the FFT makes the boundaries PERIODIC, so a packet that reaches an edge wraps
    around. Keep the grid wide enough that |psi|^2 stays ~0 at both ends for the whole run
    (E10 will add an absorbing edge mask)."""
    x = np.asarray(x, dtype=float)
    dx = x[1] - x[0]
    n = len(x)
    k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    Vv = V(x) if callable(V) else np.asarray(V, dtype=float)
    expV = np.exp(-1j * Vv * dt / (2 * hbar))            # half potential step
    expK = np.exp(-1j * hbar * k**2 * dt / (2 * mass))   # full kinetic step
    psi = np.asarray(psi, dtype=complex).copy()
    for _ in range(steps):
        psi = expV * psi
        psi = np.fft.ifft(expK * np.fft.fft(psi))
        psi = expV * psi
    return psi

def prob_density(psi, x):
    """|psi|^2 normalized so it integrates to 1 -- the probability heatmap."""
    psi = np.asarray(psi, dtype=complex)
    dx = x[1] - x[0]
    p = np.abs(psi)**2
    return p / (p.sum() * dx)

def expect_x(psi, x):
    """Average position <x> = integral x |psi|^2 dx."""
    return float(np.sum(np.asarray(x) * prob_density(psi, x)) * (x[1] - x[0]))

def spread_x(psi, x):
    """Position spread Dx = sqrt(<x^2> - <x>^2) of |psi|^2 (deposited in E4, reused E7)."""
    p = prob_density(psi, x)
    dx = x[1] - x[0]
    mean = np.sum(np.asarray(x) * p) * dx
    var = np.sum(np.asarray(x)**2 * p) * dx - mean**2
    return float(np.sqrt(max(var, 0.0)))

def _momentum_density(psi, x):
    """(k grid, |phi(k)|^2 normalized to sum 1) -- the momentum-space picture of psi, via FFT."""
    psi = np.asarray(psi, dtype=complex)
    n = len(x)
    dx = x[1] - x[0]
    k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    phi = np.fft.fft(psi)
    pk = np.abs(phi)**2
    s = pk.sum()
    if s <= 0:
        raise ValueError("psi has zero norm; cannot form a momentum distribution")
    return k, pk / s

def expect_p(psi, x, hbar=1.0):
    """Average momentum <p> = hbar <k>, read straight off the FFT of psi (E7).
    A plane wave e^{i k0 x} reports <p> = hbar*k0."""
    k, pk = _momentum_density(psi, x)
    return float(hbar * np.sum(k * pk))

def spread_p(psi, x, hbar=1.0):
    """Momentum spread Dp = hbar * sqrt(<k^2> - <k>^2) (E4/E7)."""
    k, pk = _momentum_density(psi, x)
    mean = np.sum(k * pk)
    var = np.sum(k**2 * pk) - mean**2
    return float(hbar * np.sqrt(max(var, 0.0)))

def uncertainty_product(psi, x, hbar=1.0):
    """The Heisenberg product Dx*Dp. Never dips below hbar/2 (= 0.5 in natural units);
    a Gaussian packet and the oscillator ground state sit right at the floor."""
    return spread_x(psi, x) * spread_p(psi, x, hbar)


# ---------------------------------------------------------------------------
# Harmonic-oscillator ladder operators (E9) -- pure NumPy, no QuTiP required.
# A finite N-level (Fock) truncation; exact except for the very top rung.
# ---------------------------------------------------------------------------

def annihilation(N):
    """The lowering operator a on an N-level truncation: a|n> = sqrt(n)|n-1>."""
    return np.diag(np.sqrt(np.arange(1, N)), 1).astype(complex)

def creation(N):
    """The raising operator a-dagger: a-dagger|n> = sqrt(n+1)|n+1> (= annihilation(N) conjugate-transposed)."""
    return annihilation(N).conj().T

def number_op(N):
    """The number operator a-dagger a; its eigenvalues are the rung labels 0,1,2,...,N-1."""
    return creation(N) @ annihilation(N)

def fock(N, n):
    """The n-th rung |n> as a length-N column (an energy eigenstate of the oscillator)."""
    v = np.zeros(N, dtype=complex)
    v[n] = 1.0
    return v

def oscillator_H(N, hbar=1.0, omega=1.0):
    """The oscillator Hamiltonian H = hbar*omega*(a-dagger a + 1/2): an EVENLY spaced ladder
    n + 1/2. The calculus-free route of E9 -- agrees with solve_schrodinger(0.5*x**2, x)
    on the low rungs, which is E9's 'verify it two independent ways' beat."""
    return hbar * omega * (number_op(N) + 0.5 * np.eye(N, dtype=complex))


def barrier_transmission(E, U, a, hbar=1.0, mass=1.0):
    """Exact transmission probability T for a rectangular barrier of height U and HALF-width a
    (the wall spans |x| < a, full width 2a), hit by a particle of energy E > 0. Tong eq. 2.54;
    R + T = 1. Handles E < U (tunnelling, the sinh form) and E > U (over-the-barrier, the sin form)."""
    if E <= 0:
        raise ValueError("energy E must be positive")
    w = 2.0 * a
    if E < U:
        kappa = np.sqrt(2 * mass * (U - E)) / hbar
        return float(1.0 / (1.0 + (U**2 * np.sinh(kappa * w)**2) / (4 * E * (U - E))))
    if E > U:
        kp = np.sqrt(2 * mass * (E - U)) / hbar
        return float(1.0 / (1.0 + (U**2 * np.sin(kp * w)**2) / (4 * E * (E - U))))
    return float(1.0 / (1.0 + (mass * U * w**2) / (2 * hbar**2)))   # E == U limit


# ---------------------------------------------------------------------------
# Thin, documented aliases so the NAMES used in the course-outline.md code
# blocks (E7-E11) all resolve to the canonical functions above. The canonical
# name is the source of truth; these only rename/re-order to match the bible.
# ---------------------------------------------------------------------------
propagate = split_step                 # E7 prose: "packaged as qm_toolkit.propagate"
solve_well = solve_schrodinger         # E8 demo: solve_well(V, x)
T_analytic = barrier_transmission      # E10 demo: T_analytic(E, U, a)

def fd_eigensolver(x, V, **kwargs):
    """E9 alias for solve_schrodinger, with the (x, V) argument order the E9 demo uses.
    Canonical call is solve_schrodinger(V, x)."""
    return solve_schrodinger(V, x, **kwargs)

def gaussian_packet(x, x0=0.0, k0=0.0, width=1.0, **kwargs):
    """E10 alias for gaussian_wavepacket; the outline calls the width 'width', the toolkit 'sigma'."""
    return gaussian_wavepacket(x, x0=x0, k0=k0, sigma=width, **kwargs)


# ===========================================================================
# PART D -- DRAWING  (optional; needs matplotlib. Imported lazily.)
# ===========================================================================

def plot_bloch(states, labels=None, ax=None, show=True):
    """Draw qubit states as pins on the Bloch sphere using pure matplotlib
    (no QuTiP needed). `states` is one state or a list of them."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    if isinstance(states, np.ndarray) and states.ndim == 1:
        states = [states]
    labels = labels or [None] * len(states)
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    ax.plot_wireframe(np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v),
                      color="lightgray", linewidth=0.4)
    for a in (("x", [1,0,0]), ("y", [0,1,0]), ("z (up)", [0,0,1])):
        ax.plot([0, a[1][0]], [0, a[1][1]], [0, a[1][2]], color="gray", lw=1)
    for st, lab in zip(states, labels):
        bx, by, bz = bloch_vector(st)
        ax.quiver(0, 0, 0, bx, by, bz, length=1.0, normalize=False, lw=2)
        if lab:
            ax.text(bx*1.15, by*1.15, bz*1.15, lab, fontsize=11)
    ax.set_box_aspect([1, 1, 1]); ax.set_axis_off()
    if show:
        plt.show()
    return ax


def two_slit_schematic(ax=None, show=False):
    """A labeled teaching schematic of the two-slit experiment (E1): an electron source ->
    a barrier with two slits -> a screen showing interference fringes. The brightness on the
    screen is the REAL pattern from two_slit_intensity(), so the diagram stays math-honest."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Arc
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5.6))
    ax.set_xlim(0, 10.6); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_aspect("equal")
    blue, wall = "#2b8cbe", "#4a4a4a"

    # electron source (left) + rays toward the two slits
    ax.add_patch(plt.Circle((0.7, 3.0), 0.22, color=blue, zorder=3))
    ax.text(0.7, 2.30, "electron\nsource", ha="center", va="top", fontsize=11)
    for sy in (2.7, 3.3):
        ax.annotate("", xy=(3.78, sy), xytext=(0.95, 3.0),
                    arrowprops=dict(arrowstyle="-", color=blue, lw=1.0, alpha=0.5))

    # barrier with two slits (gaps at y=2.7 and y=3.3)
    bx, bw = 3.8, 0.18
    for (y0, y1) in [(0.3, 2.55), (2.85, 3.15), (3.45, 5.7)]:
        ax.add_patch(Rectangle((bx, y0), bw, y1 - y0, color=wall, zorder=2))
    ax.text(bx + bw / 2, 0.0, "two slits", ha="center", fontsize=11)

    # circular wavefronts spreading from each slit
    for sy in (2.7, 3.3):
        for r in (0.55, 1.1, 1.65, 2.2, 2.75, 3.3):
            ax.add_patch(Arc((bx + bw, sy), 2 * r, 2 * r, theta1=-62, theta2=62,
                             color=blue, lw=0.8, alpha=0.4, zorder=1))

    # screen (right) painted with the real fringe brightness
    sx = 9.7
    ax.add_patch(Rectangle((sx, 0.3), 0.12, 5.4, color="#333", zorder=3))
    ax.text(sx + 0.06, 0.0, "screen", ha="center", fontsize=11)
    ys = np.linspace(0.4, 5.6, 500)
    inten = two_slit_intensity((ys - 3.0) * 110.0, slit_sep=20.0, wavelength=1.0)
    for yy, I in zip(ys, inten):
        if I > 0.02:
            ax.plot([sx - 0.04 - 0.9 * I, sx - 0.04], [yy, yy], color=blue,
                    lw=1.5, alpha=min(1.0, 0.2 + 0.8 * I), zorder=2)
    ax.text(sx - 0.55, 5.95, "interference\nfringes", ha="center", va="bottom",
            fontsize=10, color=blue)

    ax.set_title("The two-slit experiment", fontsize=15, pad=4)
    if show:
        plt.show()
    return ax


def bloch_schematic(show=False):
    """A labeled teaching globe for the qubit (E2): north pole = up, south = down, the
    equator = 50/50 on Z, and longitude = the phase. |+x> and |+y> sit on the equator 90
    degrees apart in LONGITUDE -- same latitude (identical Z odds), different longitude
    (so a detector turned to X can tell them apart). This is the whole E2 thesis, drawn."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Ellipse
    fig, ax = plt.subplots(figsize=(8, 7.2))
    ax.set_xlim(-1.9, 1.9); ax.set_ylim(-1.8, 1.7); ax.set_aspect("equal"); ax.axis("off")
    blue, orange = "#2b8cbe", "#cb4b16"
    ax.add_patch(Circle((0, 0), 1.0, fill=False, color="#9bb3c4", lw=1.6))
    ax.add_patch(Ellipse((0, 0), 2.0, 0.55, fill=False, color="#9bb3c4", lw=1.1, ls="--"))   # equator
    # poles
    ax.plot(0, 1.0, "o", color="#333", ms=7); ax.text(0, 1.16, r"$|{\uparrow}\rangle$  up  (north = 100% up on Z)", ha="center", fontsize=12)
    ax.plot(0, -1.0, "o", color="#333", ms=7); ax.text(0, -1.18, r"$|{\downarrow}\rangle$  down  (south)", ha="center", va="top", fontsize=12)
    ax.text(1.06, 0.16, "equator\n= 50/50 on Z", ha="left", va="center", fontsize=11, color="#222")
    # the two equatorial pins, 90 deg apart in longitude (|+x> at front, |+y> at right)
    px, py = (0.0, -0.27), (1.0, 0.0)
    ax.annotate("", xy=px, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=blue, lw=3))
    ax.text(px[0], px[1] - 0.16, r"$|{+}x\rangle$", color=blue, ha="center", va="top", fontsize=14)
    ax.annotate("", xy=py, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=orange, lw=3))
    ax.text(py[0] + 0.1, py[1], r"$|{+}y\rangle$", color=orange, ha="left", va="center", fontsize=14)
    ax.text(0, -1.62, "longitude = the phase  →  this is what a detector turned to X can see", ha="center", fontsize=10.5, color="#222")
    ax.set_title("The qubit globe (Bloch sphere)", fontsize=15, pad=2)
    if show:
        plt.show()
    return fig


def which_path_schematic(show=False):
    """The which-path effect (E1): with no detector the electrons interfere (fringes); the
    instant a detector reveals which slit each took, the interference is ERASED and only the
    two classical humps remain. Left = quantum fringes; right = 'we looked' -> two humps."""
    import matplotlib.pyplot as plt
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    blue, orange = "#2b8cbe", "#cb4b16"
    s = np.linspace(-300, 300, 1201)
    fr = two_slit_intensity(s, both=True)
    g = np.exp(-((s + 70)**2) / (2 * 45**2)) + np.exp(-((s - 70)**2) / (2 * 45**2))
    g /= g.max()
    axL.plot(s, fr, color=blue, lw=2); axL.fill_between(s, fr, color=blue, alpha=0.15)
    axL.set_title("No detector\n→ interference fringes", fontsize=12)
    axR.plot(s, g, color=orange, lw=2); axR.fill_between(s, g, color=orange, alpha=0.15)
    axR.set_title("Detector ON — we know the path\n→ fringes gone, just two humps", fontsize=12)
    for ax in (axL, axR):
        ax.set_xlabel("position on screen"); ax.set_ylim(0, 1.08); ax.set_yticks([])
    fig.suptitle("Looking erases the interference", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    if show:
        plt.show()
    return fig


def interference_schematic(show=False):
    """Why there are fringes (E1): two waves arriving IN PHASE add to a bigger wave (bright);
    a half-wavelength OUT OF PHASE they cancel (dark). Real sine waves added point-by-point."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    x = np.linspace(0, 4 * np.pi, 600)
    blue, orange, green = "#2b8cbe", "#cb4b16", "#859900"
    for ax, shift, title, tag in [
        (axes[0], 0.0, "In phase  →  BRIGHT", "crest meets crest: they reinforce"),
        (axes[1], np.pi, "Out of phase  →  DARK", "crest meets trough: they cancel"),
    ]:
        w1, w2 = np.sin(x), np.sin(x + shift)
        ax.plot(x, w1 + 3.2, color=blue, lw=2)
        ax.plot(x, w2 + 0.0, color=orange, lw=2)
        ax.plot(x, (w1 + w2) - 3.8, color=green, lw=2.8)
        ax.axhline(-3.8, color="gray", lw=0.6, ls=":")
        for y, lab, col in [(3.2, "from slit 1", "#1c6a91"), (0.0, "from slit 2", "#a73a10"),
                            (-3.8, "add them", "#4f5c00")]:
            ax.text(-0.4, y, lab, ha="right", va="center", fontsize=9.5, color=col, fontweight="bold")
        ax.set_title(title, fontsize=13)
        ax.text(0.5, -0.02, tag, transform=ax.transAxes, ha="center", va="top",
                fontsize=10, color="#333")
        ax.set_xlim(-5.0, 4 * np.pi + 0.3); ax.set_ylim(-6.4, 5); ax.axis("off")
    fig.suptitle("Adding two waves: same phase brightens, opposite phase cancels", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    if show:
        plt.show()
    return fig


def amplitude_addition_schematic(show=False):
    """The E1 rule made visual: amplitudes are little ARROWS that add tip-to-tail, and only
    THEN do we square the total to get the brightness. Aligned -> long sum -> bright;
    opposed -> they cancel -> dark. (Qualitative: E1 keeps zero complex-number formalism.)"""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    blue, orange, green = "#2b8cbe", "#cb4b16", "#859900"
    for ax, a2, title, res, payoff in [
        (axes[0], 1.0, "Arrows aligned", 2.0, "then square → BIG → bright"),
        (axes[1], -1.0, "Arrows opposed", 0.0, "then square → 0 → dark"),
    ]:
        y1, y2 = 0.95, 0.62                                    # slit-2 on its own row so tip-to-tail is clear
        ax.annotate("", xy=(1, y1), xytext=(0, y1), arrowprops=dict(arrowstyle="-|>", color=blue, lw=3))
        ax.text(0.5, y1 + 0.14, "slit 1", color="#1c6a91", ha="center", fontsize=10, fontweight="bold")
        ax.annotate("", xy=(1 + a2, y2), xytext=(1, y2), arrowprops=dict(arrowstyle="-|>", color=orange, lw=3))
        ax.text(1 + a2 / 2.0, y2 - 0.17, "slit 2", color="#a73a10", ha="center", fontsize=10, fontweight="bold")
        if res > 0:
            ax.annotate("", xy=(res, 0.2), xytext=(0, 0.2), arrowprops=dict(arrowstyle="-|>", color=green, lw=4))
            ax.text(res / 2.0, -0.02, "sum", color="#4f5c00", ha="center", fontsize=11, fontweight="bold")
        else:
            ax.plot(0, 0.2, "o", color=green, ms=12)
            ax.text(0.16, 0.2, "sum = 0", color="#4f5c00", ha="left", va="center", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=13)
        ax.text(0.5, -0.04, payoff, transform=ax.transAxes, ha="center", va="top", fontsize=12.5, color="black")
        ax.set_xlim(-0.7, 2.7); ax.set_ylim(-0.25, 1.6); ax.axis("off")
    fig.suptitle("Amplitudes add tip-to-tail, THEN we square", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    if show:
        plt.show()
    return fig


def to_qutip(state: np.ndarray):
    """Optional adapter: wrap the canonical NumPy state as a QuTiP Qobj, ONLY for
    QuTiP's nicer Bloch rendering. Raises a friendly error if QuTiP isn't installed."""
    try:
        import qutip
    except ImportError as e:
        raise ImportError("QuTiP is optional and not installed. The NumPy state is the "
                          "source of truth; use plot_bloch() for a no-dependency globe.") from e
    return qutip.Qobj(np.asarray(state, dtype=complex).reshape(2, 1))
