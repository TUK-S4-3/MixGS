# python tools/inspect_visibility_pt.py \
#   --pt debug/visibility_cam0_iter100000.pt \
#   --top_k 20   #상위 20개 글로벌의 가시성 점수와 위치 정보 출력

#.pt 전용 확인 스크립트

import torch
import csv
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument("--pt", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--csv_out", type=str, default="")
    args = parser.parse_args()

    x = torch.load(args.pt)

    print("keys:", list(x.keys()))
    print("image_name:", x.get("image_name", "N/A"))
    print("camera_idx:", x.get("camera_idx", "N/A"))
    print("split:", x.get("split", "N/A"))

    global_ids = x["global_ids"]
    scores = x["scores"]
    xyz = x["xyz"]

    print("\n[summary]")
    print("num_visible =", scores.numel())
    print("score min =", scores.min().item())
    print("score max =", scores.max().item())
    print("score mean =", scores.mean().item())
    print("score std =", scores.std().item())

    topk = min(args.top_k, scores.numel())
    vals, idx = torch.topk(scores, k=topk)

    print("\n[top-k]")
    for rank in range(topk):
        print(
            f"rank={rank+1:02d} "
            f"global_id={global_ids[idx[rank]].item()} "
            f"score={vals[rank].item():.6f} "
            f"xyz={xyz[idx[rank]].tolist()}"
        )

    if args.csv_out:
        with open(args.csv_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "global_id", "score", "x", "y", "z"])
            for rank in range(topk):
                p = xyz[idx[rank]].tolist()
                writer.writerow([
                    rank + 1,
                    global_ids[idx[rank]].item(),
                    vals[rank].item(),
                    p[0], p[1], p[2]
                ])
        print(f"\nsaved csv: {args.csv_out}")

if __name__ == "__main__":
    main()
