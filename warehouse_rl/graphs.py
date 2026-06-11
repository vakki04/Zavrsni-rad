import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import config

def show_graphs(agent, grid=None):
    # provjeri ima li uopce podataka za crtanje
    if agent is None or len(agent.rewards_history) < 2:
        print("[graphs] nema dovoljno podataka.")
        return

    total_episodes = len(agent.rewards_history)
    episodes_list = range(1, total_episodes + 1)

    # napravi prozor sa tamnom pozadinom
    window = plt.figure(figsize=(14, 8), facecolor="#12121c")
    window.suptitle(
        f"Warehouse Q-Learning (Total episodes: {total_episodes})",
        color="#b4c8ff",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # mreza za raspored (2 reda, 6 stupaca) kako bi centrirali donja 2 grafa
    gs = window.add_gridspec(2, 6)
    
    # gornji red (3 grafa, svaki zauzima 2 stupca)
    graph_rewards = window.add_subplot(gs[0, 0:2])
    graph_steps = window.add_subplot(gs[0, 2:4])
    graph_epsilon = window.add_subplot(gs[0, 4:6])
    
    # donji red (2 grafa, centrirana u sredini)
    graph_success = window.add_subplot(gs[1, 1:3])
    graph_heatmap = window.add_subplot(gs[1, 3:5])

    # pomocna funkcija za tamni stil grafa
    def apply_dark_style(graph, title):
        graph.set_facecolor("#1a1a2e")
        graph.set_title(title, color="#b4c8ff", fontsize=10, pad=6)
        graph.tick_params(colors="#5060a0", labelsize=8)
        
        for edge in graph.spines.values():
            edge.set_color("#2a2a4a")
            
        graph.grid(alpha=0.18, color="#3a3a6a")
        graph.xaxis.label.set_color("#8090c0")
        graph.yaxis.label.set_color("#8090c0")

    # graf prosjecne nagrade
    graph_rewards.plot(episodes_list, agent.rewards_history, color="#4080c0", alpha=0.6, label="Reward")
    
    # racunanje prosjeka zadnjih 100 epizoda
    average_rewards = []
    for i in range(total_episodes):
        start = max(0, i - 100)
        subset = agent.rewards_history[start : i+1]
        average = sum(subset) / len(subset)
        average_rewards.append(average)

    graph_rewards.plot(episodes_list, average_rewards, color="#50e8a0", linewidth=2, label="Avg 100")
    graph_rewards.set_xlabel("Episode")
    graph_rewards.set_ylabel("Total reward")
    graph_rewards.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="#b4c8ff", framealpha=0.7)
    apply_dark_style(graph_rewards, "Reward per episode")

    # graf prosjecnog broja koraka
    graph_steps.plot(episodes_list, agent.steps_history, color="#c06040", alpha=0.6, label="Steps")
    
    # racunanje prosjeka zadnjih 20 epizoda
    average_steps = []
    for i in range(total_episodes):
        start = max(0, i - 20)
        subset = agent.steps_history[start : i+1]
        average = sum(subset) / len(subset)
        average_steps.append(average)
        
    graph_steps.plot(episodes_list, average_steps, color="#ff9060", linewidth=2, label="Avg 20")
    graph_steps.set_xlabel("Episode")
    graph_steps.set_ylabel("Steps count")
    graph_steps.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="#b4c8ff", framealpha=0.7)
    apply_dark_style(graph_steps, "Steps to goal")

    # epsilon decay graf
    graph_epsilon.fill_between(episodes_list, agent.epsilon_history, alpha=0.3, color="#e0a030")
    graph_epsilon.plot(episodes_list, agent.epsilon_history, color="#ffc040", linewidth=1.5)
    graph_epsilon.set_xlabel("Episode")
    graph_epsilon.set_ylabel("Epsilon value")
    graph_epsilon.set_ylim(0, 1.05)
    apply_dark_style(graph_epsilon, "Epsilon (exploration vs exploitation)")

    # graf stope uspjesnosti
    average_success = []
    for i in range(total_episodes):
        start = max(0, i - 50)
        subset = agent.success_history[start : i+1]
        average = sum(subset) / len(subset)
        average_success.append(average)
        
    graph_success.fill_between(episodes_list, average_success, alpha=0.4, color="#40e080")
    graph_success.plot(episodes_list, average_success, color="#60ff90", linewidth=2)
    graph_success.set_xlabel("Episode")
    graph_success.set_ylabel("Success rate")
    graph_success.set_ylim(0, 1.05)
    
    # funkcija za pretvaranje u postotak
    def format_percentage(value, position):
        return f"{value * 100:.0f}%"
    
    # koristimo ispravan modul ticker za funcformatter
    graph_success.yaxis.set_major_formatter(ticker.FuncFormatter(format_percentage))
    apply_dark_style(graph_success, "Success rate (avg 50 ep)")

    # heatmapa
    rows = config.GRID_H
    cols = config.GRID_W
    heatmap_data = np.zeros((rows, cols))

    for r in range(rows):
        for c in range(cols):
            # racunanje indeksa stanja
            cell_index = r * cols + c
            state_no_item = cell_index * 2
            state_with_item = cell_index * 2 + 1
            
            # provjeri je li polje zid
            if grid is not None and grid[r][c] == 1:
                heatmap_data[r][c] = np.nan # np.nan ostavlja prazno za zid
            else:
                # provjeri postoji li stanje u q tablici
                if state_with_item < len(agent.Q):
                    max_q_without = max(agent.Q[state_no_item])
                    max_q_with = max(agent.Q[state_with_item])
                    heatmap_data[r][c] = max(max_q_without, max_q_with)
                else:
                    heatmap_data[r][c] = 0

    image = graph_heatmap.imshow(heatmap_data, cmap="plasma", interpolation="nearest")
    
    # dodaj colorbar sa strane
    color_bar = window.colorbar(image, ax=graph_heatmap, fraction=0.046, pad=0.04)
    color_bar.ax.tick_params(colors="#8090c0")
    apply_dark_style(graph_heatmap, "Max q-value (heatmap)")

    # podesi margine da nema preklapanja
    window.subplots_adjust(left=0.06, right=0.96, top=0.9, bottom=0.08, hspace=0.45, wspace=0.35)

    # prikazi prozor dok se ne zatvori
    plt.show()

def close_graphs():
    plt.close('all')