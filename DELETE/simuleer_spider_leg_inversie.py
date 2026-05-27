import os
from pathlib import Path

import numpy as np

mpl_config_dir = Path("/private/tmp/mplconfig")
mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# Experimentele simulatie van het prof-mechanisme:
# Chebyshev-Pantograph leg mechanism uit de paper.
#
# Paperlabels:
# - L, C en P zijn vaste punten op het frame.
# - L-E is de aangedreven kruk met volledige 360 graden rotatie.
# - L-E-D-C is de Chebyshev-vierstang.
# - B is het couplerpunt dat een ovoid traject maakt.
# - P-G-B-H-A is de pantograaf die het traject versterkt.
# - A is het voetpunt.
#
# Parameters uit Table 1 van de paper, omgezet van mm naar m.
mm = 1e-3
cheb_a = 50.0 * mm
cheb_m = 25.0 * mm
cheb_c = 62.5 * mm
cheb_d = 62.5 * mm
cheb_f = 62.5 * mm

pant_p = 30.0 * mm
pant_h = 30.0 * mm
pant_l1 = 300.0 * mm
pant_l2 = 200.0 * mm
pant_b1 = 75.0 * mm
pant_b2 = 150.0 * mm

L = np.array([0.0, 0.0])
C = np.array([cheb_a, 0.0])
P = np.array([pant_p, pant_h])


def circle_intersections(p0, r0, p1, r1):
    delta = p1 - p0
    distance = np.linalg.norm(delta)
    if distance < 1e-12:
        raise ValueError("Concentrische cirkels geven geen unieke sluiting.")
    if distance > r0 + r1 or distance < abs(r0 - r1):
        raise ValueError("Geen cirkelsnijding voor deze configuratie.")

    x = (distance**2 - r1**2 + r0**2) / (2.0 * distance)
    h_sq = r0**2 - x**2
    h = np.sqrt(max(h_sq, 0.0))
    unit = delta / distance
    normal = np.array([-unit[1], unit[0]])
    base = p0 + x * unit
    return base + h * normal, base - h * normal


def raw_pose(alpha):
    # Chebyshev-vierstang, met dezelfde tekenconventie als de paperformules:
    # E = (m cos alpha, -m sin alpha).
    E = L + cheb_m * np.array([np.cos(alpha), -np.sin(alpha)])

    D_options = circle_intersections(E, cheb_c, C, cheb_d)
    D = D_options[0]
    coupler_dir = (D - E) / np.linalg.norm(D - E)
    B = E + (cheb_c + cheb_f) * coupler_dir

    # Pantograaf: G is het snijpunt van cirkels rond P en B.
    G_options = circle_intersections(P, pant_b1, B, pant_l2 - pant_b2)
    G = G_options[0]
    pg_dir = (G - P) / np.linalg.norm(G - P)
    bg_dir = (G - B) / np.linalg.norm(G - B)

    H = B + pant_b2 * bg_dir
    A = H - (pant_l1 - pant_b1) * pg_dir

    return {
        "L": L,
        "C": C,
        "P": P,
        "E": E,
        "D": D,
        "B": B,
        "G": G,
        "H": H,
        "A": A,
    }


def transform_pose(points, y_reference):
    # Geen verticale spiegeling: de bijna rechte fase blijft bovenaan,
    # zoals in de oorspronkelijke Chebyshev-Pantograph-oriëntatie.
    return {
        name: np.array([point[0], point[1] - y_reference])
        for name, point in points.items()
    }


def path_stats(path, stance_mask):
    stance = path[stance_mask]
    return {
        "x_min": path[:, 0].min(),
        "x_max": path[:, 0].max(),
        "y_min": path[:, 1].min(),
        "y_max": path[:, 1].max(),
        "stance_x_span": stance[:, 0].max() - stance[:, 0].min(),
        "stance_y_span": stance[:, 1].max() - stance[:, 1].min(),
        "stance_fraction": stance_mask.mean(),
    }


def draw_pose(ax, points, title, trajectory=None, stance_path=None, contact_y=None):
    if trajectory is not None:
        ax.plot(trajectory[:, 0], trajectory[:, 1], color="tab:red", lw=1.2, alpha=0.45, label="voettraject A")
    if stance_path is not None:
        ax.plot(stance_path[:, 0], stance_path[:, 1], color="tab:green", lw=3.0, label="bijna horizontale steunfase")

    dark = "#1f77b4"
    light = "#8ecae6"
    foot = "#0b4f9c"
    fixed = "#333333"

    segments = [
        (["L", "E"], dark, 4.2, "kruk L-E"),
        (["C", "D"], dark, 4.2, "rocker C-D"),
        (["E", "D", "B"], dark, 4.2, "Chebyshev coupler"),
        (["P", "G"], light, 3.5, "pantograaf"),
        (["G", "B"], light, 3.5, None),
        (["B", "H"], light, 3.5, None),
        (["H", "A"], foot, 4.8, "voetlink H-A"),
    ]

    used_labels = set()
    for names, color, lw, label in segments:
        xy = np.vstack([points[name] for name in names])
        draw_label = label if label and label not in used_labels else None
        if label:
            used_labels.add(label)
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            "-o",
            color=color,
            lw=lw,
            ms=5,
            markerfacecolor="white",
            markeredgecolor="black",
            solid_capstyle="round",
            label=draw_label,
        )

    for name in ["L", "C", "P"]:
        ax.plot(points[name][0], points[name][1], marker="v", color=fixed, ms=8)

    for name, point in points.items():
        ax.annotate(name, point, xytext=(5, 5), textcoords="offset points", fontsize=8)

    if contact_y is not None:
        ax.axhline(contact_y, color="black", lw=1.8, label="vlakke lijn boven")
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)


def main():
    out_dir = Path("stangen/output_spider_leg_inversie")
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha = np.linspace(0.0, 2.0 * np.pi, 721)
    raw_poses = [raw_pose(a) for a in alpha]
    raw_path = np.vstack([pose["A"] for pose in raw_poses])

    # In de ruwe paper-orientatie ligt de rechte fase aan de bovenzijde van het
    # ovoid. We behouden die orientatie en verschuiven alleen de figuur.
    y_reference = raw_path[:, 1].min()
    poses = [transform_pose(pose, y_reference) for pose in raw_poses]
    foot_path = np.vstack([pose["A"] for pose in poses])

    stance_mask = foot_path[:, 1] >= np.quantile(foot_path[:, 1], 0.50)
    stance_path = foot_path[stance_mask]
    contact_y = float(np.mean(stance_path[:, 1]))
    stats = path_stats(foot_path, stance_mask)

    idx_pose = int(np.argmax(foot_path[:, 1]))
    points_pose = poses[idx_pose]

    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    ax.plot(foot_path[:, 0], foot_path[:, 1], color="tab:red", lw=1.8, label="voettraject A bij 360 graden krukrotatie")
    ax.plot(stance_path[:, 0], stance_path[:, 1], color="tab:green", lw=3.0, label="bijna horizontale steunfase")
    ax.axhline(contact_y, color="black", lw=1.8, label="vlakke lijn boven")
    ax.set_title("Chebyshev-Pantograph poot: vlakke beweging bovenaan")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "trajecten.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    draw_pose(
        ax,
        points_pose,
        f"Prof-topologie: Chebyshev-Pantograph, alpha = {np.rad2deg(alpha[idx_pose]):.1f} deg",
        trajectory=foot_path,
        stance_path=stance_path,
        contact_y=contact_y,
    )
    all_xy = np.vstack([foot_path, np.vstack([np.vstack(list(pose.values())) for pose in poses])])
    margin = 0.04
    ax.set_xlim(all_xy[:, 0].min() - margin, all_xy[:, 0].max() + margin)
    ax.set_ylim(all_xy[:, 1].min() - margin, all_xy[:, 1].max() + margin)
    fig.tight_layout()
    fig.savefig(out_dir / "spider_leg_pose.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    ax.plot(foot_path[:, 0], foot_path[:, 1], color="tab:red", lw=1.2, alpha=0.4)
    ax.plot(stance_path[:, 0], stance_path[:, 1], color="tab:green", lw=2.6, alpha=0.95)
    ax.axhline(contact_y, color="black", lw=1.8)
    ax.set_xlim(all_xy[:, 0].min() - margin, all_xy[:, 0].max() + margin)
    ax.set_ylim(all_xy[:, 1].min() - margin, all_xy[:, 1].max() + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_title("Chebyshev-Pantograph: vlakke beweging bovenaan")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    animation_segments = [
        ["L", "E"],
        ["C", "D"],
        ["E", "D", "B"],
        ["P", "G"],
        ["G", "B"],
        ["B", "H"],
        ["H", "A"],
    ]
    colors = ["#1f77b4", "#1f77b4", "#1f77b4", "#8ecae6", "#8ecae6", "#8ecae6", "#0b4f9c"]
    widths = [4.0, 4.0, 4.0, 3.2, 3.2, 3.2, 4.8]
    artists = []
    for color, width in zip(colors, widths):
        (line,) = ax.plot([], [], "-o", color=color, lw=width, ms=4, markerfacecolor="white", markeredgecolor="black")
        artists.append(line)

    for name in ["L", "C", "P"]:
        ax.plot(points_pose[name][0], points_pose[name][1], marker="v", color="#333333", ms=8)

    crank_angle_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top")

    def update(frame_idx):
        points = poses[frame_idx]
        for artist, names in zip(artists, animation_segments):
            xy = np.vstack([points[name] for name in names])
            artist.set_data(xy[:, 0], xy[:, 1])
        crank_angle_text.set_text(f"alpha = {np.rad2deg(alpha[frame_idx]) % 360.0:5.1f} deg")
        return artists + [crank_angle_text]

    frame_indices = np.arange(0, len(alpha), 4)
    anim = FuncAnimation(fig, update, frames=frame_indices, interval=35, blit=True)
    anim.save(out_dir / "spider_leg_animatie.gif", writer=PillowWriter(fps=24))
    plt.close(fig)

    print("Prof-mechanisme gesimuleerd in de experimentele file.")
    print("Topologie: Chebyshev-vierstang + pantograaf, met voetpunt A.")
    print("Kruk: L-E draait volledig 0..360 graden.")
    print("Parameters uit Table 1 van de paper:")
    print(f"- a = {cheb_a:.4f} m, m = {cheb_m:.4f} m, c = {cheb_c:.4f} m, d = {cheb_d:.4f} m, f = {cheb_f:.4f} m")
    print(f"- p = {pant_p:.4f} m, h = {pant_h:.4f} m")
    print(f"- l1 = {pant_l1:.4f} m, l2 = {pant_l2:.4f} m, b1 = {pant_b1:.4f} m, b2 = {pant_b2:.4f} m")
    print(f"Voetpad x-range: {stats['x_min']:.4f} .. {stats['x_max']:.4f} m")
    print(f"Voetpad y-range: {stats['y_min']:.5f} .. {stats['y_max']:.5f} m")
    print(
        "Bijna horizontale steunfase "
        f"({100.0 * stats['stance_fraction']:.0f}% van de cyclus): "
        f"dx = {stats['stance_x_span']:.4f} m, dy = {stats['stance_y_span']:.6f} m"
    )
    print("Bestanden:")
    print(f"- {out_dir / 'trajecten.png'}")
    print(f"- {out_dir / 'spider_leg_pose.png'}")
    print(f"- {out_dir / 'spider_leg_animatie.gif'}")


if __name__ == "__main__":
    main()
