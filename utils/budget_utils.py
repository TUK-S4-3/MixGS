#   
# K_i= allocate_detail_budget(v_i, B)
#전체 budget B 안에서 detail 개수를 나눔
#

import torch

def allocate_detail_budget(
    vis_score,
    total_budget,
    max_details,
    min_details=1,
    temperature=1.0,
):
    if vis_score.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=vis_score.device)

    score = torch.clamp(vis_score.float(), min=0)
    if score.sum() <= 0:
        score = torch.ones_like(score)

    weights = torch.softmax(score / max(temperature, 1e-6), dim=0)

    min_details = min(min_details, max_details)
    counts = torch.full_like(score, fill_value=min_details, dtype=torch.long)

    remaining = max(int(total_budget) - int(min_details * score.numel()), 0)
    if remaining == 0:
        return counts.clamp(max=max_details)

    cap = max_details - min_details
    fractional = weights * remaining
    extra = torch.floor(fractional).long().clamp(max=cap)
    counts = counts + extra

    leftover = remaining - int(extra.sum().item())
    if leftover > 0:
        frac = fractional - extra.float()
        room = cap - extra
        order = torch.argsort(frac, descending=True)
        for idx in order.tolist():
            if leftover == 0:
                break
            if room[idx] > 0:
                counts[idx] += 1
                leftover -= 1

    return counts.clamp(max=max_details)