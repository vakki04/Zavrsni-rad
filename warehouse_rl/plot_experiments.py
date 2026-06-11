import csv
import os
import matplotlib.pyplot as plt

RESULTS_DIR = "results"

def moving_average(data, window):
    result = []

    for i in range(len(data)):
        start = max(0, i - window + 1)
        subset = data[start:i + 1]
        result.append(sum(subset) / len(subset))

    return result


def load_experiments():
    experiments = {}
    for filename in os.listdir(RESULTS_DIR):

        if not filename.endswith(".csv"):
            continue

        path = os.path.join(RESULTS_DIR, filename)

        episodes = []
        rewards = []
        success = []
        steps = []
        epsilon = []
        avg100 = []

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                episodes.append(int(row["episode"]))
                rewards.append(float(row["reward"]))
                success.append(float(row["success"]))
                steps.append(float(row["steps"]))
                epsilon.append(float(row["epsilon"]))
                avg100.append(float(row["avg100"]))
        experiments[filename[:-4]] = {"episode": episodes, "reward": rewards, "success": success, "steps": steps, "epsilon": epsilon, "avg100": avg100}

    return experiments


def plot_rewards(experiments):
    plt.figure(figsize=(10, 6))

    for name, data in experiments.items():
        plt.plot(data["episode"], data["avg100"], linewidth=2, label=name)

    plt.title("Prosječna nagrada tijekom učenja")
    plt.xlabel("Epizoda")
    plt.ylabel("Avg100")
    plt.grid(False)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_success(experiments):
    plt.figure(figsize=(10, 6))

    for name, data in experiments.items():
        success_rate = moving_average(data["success"], 50)
        success_rate = [x * 100 for x in success_rate]
        plt.plot(data["episode"], success_rate, linewidth=2, label=name)

    plt.title("Stopa uspješnosti")
    plt.xlabel("Epizoda")
    plt.ylabel("Uspješnost [%]")
    plt.grid(False)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_steps(experiments):
    plt.figure(figsize=(10, 6))
    for name, data in experiments.items():
        avg_steps = moving_average(data["steps"], 50)
        plt.plot(data["episode"], avg_steps, linewidth=2, label=name)

    plt.title("Prosječan broj koraka")
    plt.xlabel("Epizoda")
    plt.ylabel("Koraci")
    plt.grid(False)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    experiments = load_experiments()

    plot_rewards(experiments)
    plot_success(experiments)
    plot_steps(experiments)