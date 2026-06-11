import csv
import os


def save_experiment(
    experiment_name,
    rewards,
    success,
    steps,
    epsilon,
    avg100,
    folder="results"
):
    os.makedirs(folder, exist_ok=True)

    filename = os.path.join(folder, f"{experiment_name}.csv")

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "episode",
            "reward",
            "success",
            "steps",
            "epsilon",
            "avg100"
        ])

        for i in range(len(rewards)):
            writer.writerow([
                i + 1,
                rewards[i],
                success[i],
                steps[i],
                epsilon[i],
                avg100[i]
            ])

    print(f"[INFO] Saved experiment -> {filename}")