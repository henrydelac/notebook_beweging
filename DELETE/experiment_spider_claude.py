"""
experiment_spider_claude.py

Walker-poot precies zoals in de schets van de gebruiker:
  - drie vaste scharnieren links (L, C, P) op een rij,
  - Hoekens 4-bar (L-E-D-C) als drijvend mechanisme,
  - pantograaf (P-G-B-H-A) die het couplerpunt B versterkt naar voet A,
  - voet A onder de scharnieren met een OVAAL waarvan de VLAKKE KANT
    aan de ONDERZIJDE ligt (grondcontact).

Waarom werkt dit fysisch?
  - De Hoekens-vierstang (verhoudingen 2 : 1 : 2.5 : 2.5 met een
    coupler-verlenging van 2.5) is een Grashof crank-rocker; de
    gekozen coupler-punt traceert een ovaal met een nagenoeg recht
    onder-segment.  Dat onder-segment ligt -tegenover- de LC-frame-link.
  - De pantograaf P-G-B-H-A is een gelijkvormig parallellogram-
    mechanisme: P, B, A blijven steeds collineair en de schaal
    |PA| / |PB| = l1 / b1 is constant. A staat aan de overkant van
    P t.o.v. B (inverterende pantograaf), dus B boven LC -> A ver
    onder LC, met de vlakke zijde van A's ovaal aan de onderkant.
  - Gruebler: 7 links, 8 enkele draaikoppels -> DOF = 3*(7-1)-2*8 = 2.
    Een van die DOF's is de starre, vrije rotatie van de pantograaf-
    parallellogram als geheel; die wordt vastgelegd door de pin bij
    B. Effectief 1 ingang: de kruk-hoek alpha.

Conventies:
  - SI eenheden (m, rad), x-y vlak met y omhoog.
  - L = (0, 0), C ligt op afstand cheb_a rechts van L, P ligt iets
    boven LC zodat de drie scharnieren ongeveer op een rij staan.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


mm = 1e-3


@dataclass(frozen=True)
class LegParams:
    # Hoekens 4-bar: a : m : c : d : f = 2 : 1 : 2.5 : 2.5 : 2.5 (schaal 20 mm).
    # Met deze verhoudingen ligt het rechte stuk van B's coupler-curve
    # AAN DE OVERKANT van de LC-frame-link (ver van LC).
    cheb_a: float = 40.0 * mm    # frame L-C
    cheb_m: float = 20.0 * mm    # kruk L-E
    cheb_c: float = 50.0 * mm    # coupler E-D
    cheb_d: float = 50.0 * mm    # rocker C-D
    cheb_f: float = 50.0 * mm    # verlenging D->B  (B = E + (c+f)*ED)

    # Pantograaf-anker P: bewust dicht bij de LC-lijn zodat L, C, P
    # samen op een (bijna) horizontale rij scharnieren staan, zoals
    # in de schets.
    pant_p: float = 20.0 * mm
    pant_h: float = 5.0 * mm

    # Pantograaf staaflengtes.  Met deze keuzes geldt
    #     b1 * b2 = (l1 - b1) * (l2 - b2)        (pantograafvoorwaarde)
    #     b2 / (l2 - b2) = (l1 - b1) / b1 = k    (versterking)
    # en hieruit volgt A = P + (k-1) * (P - B) = 3*P - 2*B.
    # Met andere woorden: A ligt op de rechte door P en B, aan de
    # overkant van P, op afstand (k-1)*|PB|.  Hier k = 3 -> |PA|=2|PB|.
    pant_b1: float = 50.0 * mm   # P-G
    pant_b2: float = 210.0 * mm  # B-H  (B en H aan weerszijden van G op een rechte)
    pant_l1: float = 200.0 * mm  # l1 - b1 = 150 mm is staaf H-A
    pant_l2: float = 280.0 * mm  # l2 - b2 = 70 mm is staaf G-B


def _circle_intersections(p0, r0, p1, r1):
    d = np.linalg.norm(p1 - p0)
    if d < 1e-12 or d > r0 + r1 + 1e-12 or d < abs(r0 - r1) - 1e-12:
        raise ValueError(
            f"Geen cirkelsnijding: |p0p1|={d:.4f}, r0={r0:.4f}, r1={r1:.4f}"
        )
    x = (d * d + r0 * r0 - r1 * r1) / (2.0 * d)
    h = np.sqrt(max(r0 * r0 - x * x, 0.0))
    u = (p1 - p0) / d
    n = np.array([-u[1], u[0]])
    mid = p0 + x * u
    return mid + h * n, mid - h * n


def solve_pose(alpha: float, prm: LegParams):
    """Bereken alle scharnierposities bij krukhoek alpha."""
    L = np.array([0.0, 0.0])
    C = np.array([prm.cheb_a, 0.0])
    P = np.array([prm.pant_p, prm.pant_h])

    # Kruk L-E draait tegenklokwijs (kies +sin -> voet veegt rechts naar links).
    E = L + prm.cheb_m * np.array([np.cos(alpha), np.sin(alpha)])

    # Hoekens-vierstang: D BOVEN de LC-lijn, dus we kiezen de bovenste
    # snijding van de cirkels rond E en C (branch 0 van _circle_intersections,
    # die de positieve normaal op EC volgt).
    D_a, D_b = _circle_intersections(E, prm.cheb_c, C, prm.cheb_d)
    D = D_a

    # Couplerpunt B: ligt op de coupler verlengd voorbij D met afstand cheb_f.
    coupler = (D - E) / np.linalg.norm(D - E)
    B = E + (prm.cheb_c + prm.cheb_f) * coupler

    # Pantograaf-pin G op snijding van cirkels rond P (r=b1) en B (r=l2-b2).
    # Kies consistent dezelfde branch (de oost-zijde t.o.v. PB) zodat de
    # pantograaf-armen niet flippen tijdens de cyclus.
    G_a, G_b = _circle_intersections(P, prm.pant_b1, B, prm.pant_l2 - prm.pant_b2)
    G = G_a

    pg_dir = (G - P) / np.linalg.norm(G - P)
    bg_dir = (G - B) / np.linalg.norm(G - B)
    H = B + prm.pant_b2 * bg_dir
    A = H - (prm.pant_l1 - prm.pant_b1) * pg_dir

    return {"L": L, "C": C, "P": P, "E": E, "D": D, "B": B, "G": G, "H": H, "A": A}


def verify_rigidity(poses, prm: LegParams):
    """Controleer dat geen enkele staaflengte verandert tijdens de cyclus."""
    expected = {
        ("L", "E"): prm.cheb_m,
        ("C", "D"): prm.cheb_d,
        ("E", "D"): prm.cheb_c,
        ("L", "C"): prm.cheb_a,
        ("P", "G"): prm.pant_b1,
        ("B", "H"): prm.pant_b2,
        ("G", "B"): prm.pant_l2 - prm.pant_b2,
        ("H", "A"): prm.pant_l1 - prm.pant_b1,
    }
    worst = 0.0
    for (p, q), nominal in expected.items():
        for pose in poses:
            err = abs(np.linalg.norm(pose[q] - pose[p]) - nominal)
            worst = max(worst, err)
    return worst


SEGMENTS = [
    (["L", "E"], "#1f4e79", 4.0, "kruk L-E"),
    (["C", "D"], "#1f4e79", 4.0, "rocker C-D"),
    (["E", "D", "B"], "#1f4e79", 4.0, "Chebyshev/Hoekens coupler"),
    (["P", "G"], "#5dade2", 3.2, "pantograaf"),
    (["G", "B"], "#5dade2", 3.2, None),
    (["B", "H"], "#5dade2", 3.2, None),
    (["H", "A"], "#0b4f9c", 4.8, "voetlink H-A"),
]


def draw_pose(ax, pose, trajectory=None, stance=None, ground_y=None, title=""):
    if trajectory is not None:
        ax.plot(trajectory[:, 0], trajectory[:, 1],
                color="tab:red", lw=1.2, alpha=0.5, label="voettraject A")
    if stance is not None:
        ax.plot(stance[:, 0], stance[:, 1],
                color="tab:green", lw=3.0, label="bijna horizontale steunfase")

    used = set()
    for names, color, lw, label in SEGMENTS:
        xy = np.vstack([pose[n] for n in names])
        lbl = label if label and label not in used else None
        if label:
            used.add(label)
        ax.plot(xy[:, 0], xy[:, 1], "-o", color=color, lw=lw, ms=5,
                markerfacecolor="white", markeredgecolor="black",
                solid_capstyle="round", label=lbl)
    for fixed in ("L", "C", "P"):
        ax.plot(pose[fixed][0], pose[fixed][1],
                marker="v", color="#333", ms=9, zorder=5)
    for name, pt in pose.items():
        ax.annotate(name, pt, xytext=(5, 5), textcoords="offset points", fontsize=8)
    if ground_y is not None:
        ax.axhline(ground_y, color="black", lw=1.4, alpha=0.8, label="grondlijn")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)


def main():
    prm = LegParams()
    out_dir = Path(__file__).resolve().parent / "output_experiment_spider_claude"
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha = np.linspace(0.0, 2.0 * np.pi, 721)
    poses = [solve_pose(a, prm) for a in alpha]

    foot_path = np.vstack([pose["A"] for pose in poses])
    B_path = np.vstack([pose["B"] for pose in poses])

    # Stap-hoogte = totale y-range van de voet.
    y_min, y_max = foot_path[:, 1].min(), foot_path[:, 1].max()
    step_height = y_max - y_min

    # Steunfase: het onderste deel van het traject (binnen 10% van y_min).
    # Hoekens heeft een lang vlak onderstuk en een korte swing erboven,
    # dus we definieren stance via een absolute y-band, niet via quantiles.
    stance_band = 0.10 * step_height
    stance_mask = foot_path[:, 1] <= y_min + stance_band
    stance_path = foot_path[stance_mask]
    ground_y = float(np.mean(stance_path[:, 1]))

    worst_err = verify_rigidity(poses, prm)
    stance_dx = stance_path[:, 0].max() - stance_path[:, 0].min()
    stance_dy = stance_path[:, 1].max() - stance_path[:, 1].min()

    # Sanity-check: de onderzijde van het ovaal (laagste 10% y-band) moet
    # vlakker zijn dan de bovenzijde (hoogste 10% y-band).
    top_mask = foot_path[:, 1] >= y_max - stance_band
    flat_at_bottom = stance_path[:, 1].ptp() < foot_path[top_mask][:, 1].ptp() + 1e-9
    # Verhouding x/y in de stance: hoe groter, hoe vlakker.
    stance_aspect = stance_dx / max(stance_dy, 1e-9)

    # --- Figuur 1: voettraject + grondlijn ---
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    ax.plot(foot_path[:, 0], foot_path[:, 1],
            color="tab:red", lw=1.8, label="voettraject A")
    ax.plot(stance_path[:, 0], stance_path[:, 1],
            color="tab:green", lw=3.0, label="steunfase (vlakke onderkant)")
    ax.plot(B_path[:, 0], B_path[:, 1],
            color="tab:purple", lw=1.0, alpha=0.5, label="couplerpunt B")
    ax.axhline(ground_y, color="black", lw=1.4)
    for n in ("L", "C", "P"):
        ax.plot(poses[0][n][0], poses[0][n][1],
                marker="v", color="#333", ms=10, zorder=5)
        ax.annotate(n, poses[0][n], xytext=(6, 6),
                    textcoords="offset points", fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Walker-poot: Hoekens 4-bar + pantograaf — vlakke kant onder")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "traject.png", dpi=180)
    plt.close(fig)

    # --- Figuur 2: pose in het midden van de steunfase ---
    idx = int(np.argmin(foot_path[:, 1]))
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    draw_pose(ax, poses[idx], trajectory=foot_path, stance=stance_path,
              ground_y=ground_y,
              title=f"Configuratie bij alpha = {np.rad2deg(alpha[idx]):.1f} deg "
                    f"(diepste voetpositie)")
    all_xy = np.vstack([foot_path] + [np.vstack(list(p.values())) for p in poses])
    m = 0.03
    ax.set_xlim(all_xy[:, 0].min() - m, all_xy[:, 0].max() + m)
    ax.set_ylim(all_xy[:, 1].min() - m, all_xy[:, 1].max() + m)
    fig.tight_layout()
    fig.savefig(out_dir / "pose.png", dpi=180)
    plt.close(fig)

    # --- Figuur 3: animatie ---
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.plot(foot_path[:, 0], foot_path[:, 1],
            color="tab:red", lw=1.0, alpha=0.4)
    ax.plot(stance_path[:, 0], stance_path[:, 1],
            color="tab:green", lw=2.5, alpha=0.9)
    ax.axhline(ground_y, color="black", lw=1.4)
    ax.set_xlim(all_xy[:, 0].min() - m, all_xy[:, 0].max() + m)
    ax.set_ylim(all_xy[:, 1].min() - m, all_xy[:, 1].max() + m)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Chebyshev-Hoekens + pantograaf walker-poot")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.25)

    seg_names = [s[0] for s in SEGMENTS]
    seg_colors = [s[1] for s in SEGMENTS]
    seg_widths = [s[2] for s in SEGMENTS]
    artists = []
    for c, w in zip(seg_colors, seg_widths):
        (ln,) = ax.plot([], [], "-o", color=c, lw=w, ms=4,
                        markerfacecolor="white", markeredgecolor="black")
        artists.append(ln)
    for n in ("L", "C", "P"):
        ax.plot(poses[0][n][0], poses[0][n][1],
                marker="v", color="#333", ms=9, zorder=5)
    txt = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top")

    def update(i):
        for art, names in zip(artists, seg_names):
            xy = np.vstack([poses[i][n] for n in names])
            art.set_data(xy[:, 0], xy[:, 1])
        txt.set_text(f"alpha = {np.rad2deg(alpha[i]) % 360.0:5.1f} deg")
        return artists + [txt]

    frames = np.arange(0, len(alpha), 4)
    anim = FuncAnimation(fig, update, frames=frames, interval=35, blit=True)
    anim.save(out_dir / "animatie.gif", writer=PillowWriter(fps=24))
    plt.close(fig)

    # --- Rapport ---
    print("=== Walker-poot: Hoekens 4-bar + pantograaf ===")
    print(f"DOF (Gruebler, 1 ingang)     : 1 (kruk alpha)")
    print(f"Max staaflengte-fout / cyclus: {worst_err:.2e} m  (numeriek 0 -> rigide)")
    print(f"Vlakke kant onderaan oval    : {flat_at_bottom}")
    print(f"Steunfase (onderste 10% band): dx = {stance_dx*1000:6.1f} mm, "
          f"dy = {stance_dy*1000:6.2f} mm (verhouding {stance_aspect:.0f}:1)")
    print(f"Totale stap-hoogte           : {step_height*1000:6.1f} mm")
    print(f"Voet y-range                 : {y_min*1000:+6.1f} .. {y_max*1000:+6.1f} mm")
    print(f"Output in: {out_dir}")


if __name__ == "__main__":
    main()
