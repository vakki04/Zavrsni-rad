"""
Matplotlib graph visualization for Q-Learning training progress.
Runs in a separate process to avoid blocking the main rendering loop.
"""

import multiprocessing
import numpy as np
from config import GRAPH_UPDATE_INTERVAL, GRID_H, GRID_W


# Global handles for graph process and queue
_graph_proc  = None
_graph_queue = None


def _make_snapshot(agent):
    """Create a snapshot of agent training data."""
    return {
        "rewards": list(agent.rewards_history),
        "steps":   list(agent.steps_history),
        "epsilon": list(agent.epsilon_history),
        "success": list(agent.success_history),
        "avg100":  list(agent.avg100),
        "Q":       agent.Q.flatten(),
    }


def _graphs_worker(queue, grid, grid_h, grid_w):
    """
    Worker process that handles matplotlib visualization.
    Waits for data from Queue and updates graphs in real-time.
    'STOP' message closes the window.
    """
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.animation as animation
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig = plt.figure(figsize=(14, 9), facecolor="#12121c")
    fig.suptitle("Q-Learning — Analiza učenja | Warehouse RL  [LIVE]",
                 color="#b4c8ff", fontsize=13, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    line_kw  = dict(linewidth=1.0, alpha=0.6)
    avg_kw   = dict(linewidth=2.0, alpha=0.95)
    label_kw = dict(color="#8090c0", fontsize=9)

    def style_ax(ax, title):
        ax.set_facecolor("#1a1a2e")
        ax.set_title(title, color="#b4c8ff", fontsize=10, pad=6)
        ax.tick_params(colors="#5060a0", labelsize=8)
        for sp in ax.spines.values(): sp.set_color("#2a2a4a")
        ax.grid(alpha=0.18, color="#3a3a6a")

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    # Colorbar placeholder for ax5
    cb_container = [None]

    def redraw(data):
        rewards  = data["rewards"]
        steps    = data["steps"]
        epsilon  = data["epsilon"]
        success  = data["success"]
        avg100   = data["avg100"]
        Q        = data["Q"].reshape(-1, 4)
        n        = len(rewards)
        if n < 2:
            return
        ep = np.arange(1, n + 1)

        # ── 1. Rewards ──
        ax1.cla()
        ax1.plot(ep, rewards, color="#4080c0", **line_kw, label="Nagrada")
        ax1.plot(ep, avg100,  color="#50e8a0", **avg_kw,  label="Prosjek 100")
        ax1.set_xlabel("Epizoda", **label_kw)
        ax1.set_ylabel("Ukupna nagrada", **label_kw)
        ax1.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="#b4c8ff", framealpha=0.7)
        style_ax(ax1, "Nagrada po epizodi")

        # ── 2. Steps ──
        ax2.cla()
        ax2.plot(ep, steps, color="#c06040", **line_kw, label="Koraci")
        win2 = min(20, n)
        if win2 > 1:
            sm = np.convolve(steps, np.ones(win2)/win2, mode="valid")
            ax2.plot(ep[win2-1:win2-1+len(sm)], sm, color="#ff9060", **avg_kw, label=f"Prosjek {win2}")
        ax2.set_xlabel("Epizoda", **label_kw)
        ax2.set_ylabel("Broj koraka", **label_kw)
        ax2.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="#b4c8ff", framealpha=0.7)
        style_ax(ax2, "Koraci do cilja")

        # ── 3. Epsilon ──
        ax3.cla()
        ax3.fill_between(ep, epsilon, alpha=0.3, color="#e0a030")
        ax3.plot(ep, epsilon, color="#ffc040", linewidth=1.5)
        ax3.set_xlabel("Epizoda", **label_kw)
        ax3.set_ylabel("ε", **label_kw)
        ax3.set_ylim(0, 1.05)
        style_ax(ax3, "Epsilon — istraživanje vs. iskorištavanje")

        # ── 4. Success rate ──
        ax4.cla()
        win = min(50, max(2, n))
        suc = np.array(success, dtype=float)
        if win > 1:
            sa = np.convolve(suc, np.ones(win)/win, mode="valid")
            ep_s = ep[win-1:win-1+len(sa)]
            ax4.fill_between(ep_s, sa, alpha=0.4, color="#40e080")
            ax4.plot(ep_s, sa, color="#60ff90", linewidth=1.8, label=f"Prosjek {win}")
        ax4.set_xlabel("Epizoda", **label_kw)
        ax4.set_ylabel("Stopa uspjeha", **label_kw)
        ax4.set_ylim(0, 1.05)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
        style_ax(ax4, "Stopa uspjeha")

        # ── 5. Q heatmap ──
        ax5.cla()
        if cb_container[0] is not None:
            try: cb_container[0].remove()
            except Exception: pass
        q_all = np.max(Q, axis=1)
        q_pos = np.maximum(q_all[0::2], q_all[1::2])
        q_max = q_pos.reshape(grid_h, grid_w)
        q_max[grid == 1] = np.nan
        im = ax5.imshow(q_max, cmap="plasma", interpolation="nearest", extent=[0, grid_w, grid_h, 0])
        ax5.set_aspect('equal', adjustable='box')
        ax5.set_xlim(0, grid_w)
        ax5.set_ylim(grid_h, 0)
        try:
            ax5.set_anchor('C')
        except Exception:
            pass
        divider = make_axes_locatable(ax5)
        cax = divider.append_axes("right", size="5%", pad=0.04)
        cb_container[0] = fig.colorbar(im, cax=cax)
        cb_container[0].ax.tick_params(colors="#8090c0")
        ax5.set_title("Max Q-vrijednost (heatmapa)", color="#b4c8ff", fontsize=10, pad=6)
        ax5.tick_params(colors="#5060a0", labelsize=7)
        for sp in ax5.spines.values(): sp.set_color("#2a2a4a")

        # ── 6. Reward distribution ──
        ax6.cla()
        last_n = rewards[-200:]
        ax6.hist(last_n, bins=min(30, max(5, len(last_n)//2)),
                 color="#5080d0", edgecolor="#2a3060", alpha=0.8)
        ax6.axvline(np.mean(last_n), color="#50e8a0", linewidth=1.8,
                    linestyle="--", label="Prosjek")
        ax6.set_xlabel("Nagrada", **label_kw)
        ax6.set_ylabel("Frekvencija", **label_kw)
        ax6.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="#b4c8ff", framealpha=0.7)
        style_ax(ax6, f"Raspodjela nagrađivanja (zadnjih {len(last_n)})")

        fig.suptitle(f"Q-Learning — Analiza učenja | Warehouse RL  [LIVE]  ep. {n}",
                     color="#b4c8ff", fontsize=13, fontweight="bold", y=0.98)

        fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.08,
                            hspace=0.45, wspace=0.38)

    # Latest received snapshot
    latest = [None]

    def animate(_frame):
        # Empty queue - get only the newest snapshot
        while not queue.empty():
            msg = queue.get_nowait()
            if msg == "STOP":
                plt.close(fig)
                return
            latest[0] = msg
        if latest[0] is not None:
            try:
                redraw(latest[0])
            except Exception as e:
                print(f"[graf] greška pri crtanju: {e}")

    ani = animation.FuncAnimation(
        fig, animate,
        interval=GRAPH_UPDATE_INTERVAL,
        cache_frame_data=False
    )

    plt.show()


def toggle_graphs(agent):
    """
    Open live graph window if not already open, otherwise close it.
    
    Returns:
        True if graphs are now open, False if closed
    """
    global _graph_proc, _graph_queue

    # Close if already exists
    if _graph_proc is not None and _graph_proc.is_alive():
        try:
            _graph_queue.put("STOP")
        except Exception:
            pass
        _graph_proc.join(timeout=2)
        _graph_proc  = None
        _graph_queue = None
        return False

    if agent is None or len(agent.rewards_history) < 2:
        return False

    _graph_queue = multiprocessing.Queue()
    _graph_queue.put(_make_snapshot(agent))
    _graph_proc  = multiprocessing.Process(
        target=_graphs_worker,
        args=(_graph_queue, np.zeros((GRID_H, GRID_W)), GRID_H, GRID_W),
        daemon=True
    )
    _graph_proc.start()
    return True


def push_graph_update(agent):
    """Send fresh data to graph if window is open."""
    global _graph_proc, _graph_queue
    if _graph_proc is not None and _graph_proc.is_alive():
        try:
            # Empty old snapshots to always send the newest
            while not _graph_queue.empty():
                _graph_queue.get_nowait()
            _graph_queue.put_nowait(_make_snapshot(agent))
        except Exception:
            pass


def close_graphs():
    """Close graph process on exit."""
    global _graph_proc, _graph_queue
    if _graph_proc is not None and _graph_proc.is_alive():
        try:
            _graph_queue.put("STOP")
        except Exception:
            pass
        _graph_proc.join(timeout=3)
