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

        experiments[filename[:-4]] = {
            "episode": episodes,
            "reward": rewards,
            "success": success,
            "steps": steps,
            "epsilon": epsilon,
            "avg100": avg100
        }

    return experiments


def plot_rewards(experiments):

    plt.figure(figsize=(10, 6))

    for name, data in experiments.items():

        plt.plot(
            data["episode"],
            data["avg100"],
            linewidth=2,
            label=name
        )

    plt.title("Prosječna nagrada tijekom učenja")
    plt.xlabel("Epizoda")
    plt.ylabel("Avg100")
    plt.grid(False)
    plt.legend()

    plt.tight_layout()
    plt.savefig("reward_comparison.png", dpi=300)
    plt.show()


def plot_success(experiments):

    plt.figure(figsize=(10, 6))

    for name, data in experiments.items():

        success_rate = moving_average(
            data["success"],
            50
        )

        success_rate = [
            x * 100 for x in success_rate
        ]

        plt.plot(
            data["episode"],
            success_rate,
            linewidth=2,
            label=name
        )

    plt.title("Stopa uspješnosti")
    plt.xlabel("Epizoda")
    plt.ylabel("Uspješnost [%]")
    plt.grid(False)
    plt.legend()

    plt.tight_layout()
    plt.savefig("success_comparison.png", dpi=300)
    plt.show()


def plot_steps(experiments):

    plt.figure(figsize=(10, 6))

    for name, data in experiments.items():

        avg_steps = moving_average(
            data["steps"],
            50
        )

        plt.plot(
            data["episode"],
            avg_steps,
            linewidth=2,
            label=name
        )

    plt.title("Prosječan broj koraka")
    plt.xlabel("Epizoda")
    plt.ylabel("Koraci")
    plt.grid(False)
    plt.legend()

    plt.tight_layout()
    plt.savefig("steps_comparison.png", dpi=300)
    plt.show()


def print_summary(experiments):

    print("\n=== REZULTATI ===\n")

    print(
        f"{'Eksperiment':20}"
        f"{'Avg100':>12}"
        f"{'Success %':>12}"
        f"{'Steps':>12}"
    )

    print("-" * 60)

    for name, data in experiments.items():

        avg_reward = data["avg100"][-1]

        success = (
            sum(data["success"][-100:])
            / len(data["success"][-100:])
            * 100
        )

        avg_steps = (
            sum(data["steps"][-100:])
            / len(data["steps"][-100:])
        )

        print(
            f"{name:20}"
            f"{avg_reward:12.2f}"
            f"{success:12.2f}"
            f"{avg_steps:12.2f}"
        )


if __name__ == "__main__":

    experiments = load_experiments()

    plot_rewards(experiments)
    plot_success(experiments)
    plot_steps(experiments)

    print_summary(experiments)