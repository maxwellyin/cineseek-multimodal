from __future__ import annotations

import math


def ranking_metrics(ranked_ids: list[list[int]], positive_ids: list[set[int]], k_values=(10, 50, 100)) -> dict[str, float]:
    total = len(ranked_ids)
    if total == 0:
        return {f"recall@{k}": 0.0 for k in k_values} | {"mrr": 0.0, "ndcg": 0.0}

    hit_counts = {k: 0 for k in k_values}
    reciprocal_ranks = []
    ndcgs = []
    for ranked, positives in zip(ranked_ids, positive_ids):
        for k in k_values:
            if any(item_id in positives for item_id in ranked[:k]):
                hit_counts[k] += 1
        matching_ranks = [rank + 1 for rank, item_id in enumerate(ranked) if item_id in positives]
        if matching_ranks:
            best_rank = min(matching_ranks)
            reciprocal_ranks.append(1.0 / best_rank)
            ndcgs.append(1.0 / math.log2(best_rank + 1))
        else:
            reciprocal_ranks.append(0.0)
            ndcgs.append(0.0)

    metrics = {f"recall@{k}": hit_counts[k] / total for k in k_values}
    metrics["mrr"] = sum(reciprocal_ranks) / total
    metrics["ndcg"] = sum(ndcgs) / total
    return metrics
