import csv
import os

def append_budget_log_csv(log_dir, iteration, vis_count, base_budget, target_budget, allocated_budget, alloc_1, alloc_2, alloc_3):
    csv_path = os.path.join(log_dir, "budget_log.csv")
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "iteration",
                "vis",
                "base_budget",
                "target_budget",
                "allocated",
                "alloc1",
                "alloc2",
                "alloc3",
            ])
        writer.writerow([
            iteration,
            vis_count,
            base_budget,
            target_budget,
            allocated_budget,
            alloc_1,
            alloc_2,
            alloc_3,
        ])