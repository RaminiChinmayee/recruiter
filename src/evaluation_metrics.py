import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error
)

from scipy.stats import spearmanr


# =========================================================
# 1. RESUME EXTRACTION EVALUATION
# =========================================================

def extraction_precision_recall_f1(
    true_values,
    predicted_values
):
    """
    Evaluate extracted resume fields such as Skills.

    Example:

    True:
        ["Python", "SQL", "RAG"]

    Predicted:
        ["Python", "SQL"]

    Returns:
        Precision
        Recall
        F1
    """

    true_values = set(
        str(x).strip().lower()
        for x in true_values
        if x
    )

    predicted_values = set(
        str(x).strip().lower()
        for x in predicted_values
        if x
    )

    if not true_values and not predicted_values:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0
        }

    if not predicted_values:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }

    true_positive = len(
        true_values.intersection(predicted_values)
    )

    precision = (
        true_positive /
        len(predicted_values)
    )

    recall = (
        true_positive /
        len(true_values)
    )

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# =========================================================
# 2. CLASSIFICATION METRICS
# =========================================================

def classification_metrics(
    true_labels,
    predicted_labels
):
    """
    Calculate Precision, Recall and F1
    for binary candidate relevance labels.

    Example:

    1 = Relevant
    0 = Not Relevant
    """

    precision = precision_score(
        true_labels,
        predicted_labels,
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        predicted_labels,
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        predicted_labels,
        zero_division=0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


# =========================================================
# 3. PRECISION@K
# =========================================================

def precision_at_k(
    retrieved_candidates,
    relevant_candidates,
    k=5
):
    """
    Measures how many of the top-K retrieved
    candidates are actually relevant.

    Formula:

        Precision@K =
        Relevant candidates in top K / K
    """

    if k <= 0:
        return 0.0

    retrieved = retrieved_candidates[:k]

    if not retrieved:
        return 0.0

    relevant = set(relevant_candidates)

    hits = sum(
        1
        for candidate in retrieved
        if candidate in relevant
    )

    return hits / len(retrieved)


# =========================================================
# 4. RECALL@K
# =========================================================

def recall_at_k(
    retrieved_candidates,
    relevant_candidates,
    k=5
):
    """
    Measures how many of all relevant candidates
    were found in the top-K results.
    """

    if k <= 0:
        return 0.0

    relevant = set(relevant_candidates)

    if not relevant:
        return 0.0

    retrieved = retrieved_candidates[:k]

    hits = len(
        set(retrieved).intersection(relevant)
    )

    return hits / len(relevant)


# =========================================================
# 5. MRR@K
# =========================================================

def mrr_at_k(
    retrieved_candidates,
    relevant_candidates,
    k=5
):
    """
    Mean Reciprocal Rank for a single query/JD.

    If the first relevant candidate is at:

        Rank 1 -> 1.0
        Rank 2 -> 0.5
        Rank 3 -> 0.333
        Rank 4 -> 0.25
    """

    if k <= 0:
        return 0.0

    retrieved = retrieved_candidates[:k]

    relevant = set(relevant_candidates)

    for rank, candidate in enumerate(
        retrieved,
        start=1
    ):

        if candidate in relevant:
            return 1.0 / rank

    return 0.0


# =========================================================
# 6. NDCG@K
# =========================================================

def ndcg_at_k(
    relevance_scores,
    k=5
):
    """
    Calculate NDCG@K.

    relevance_scores must be in the
    ranking order produced by the system.

    Example:

        [5, 3, 1, 0, 0]

    where:

        5 = Excellent
        4 = Strong
        3 = Good
        2 = Weak
        1 = Poor
        0 = Irrelevant
    """

    if k <= 0:
        return 0.0

    scores = np.asarray(
        relevance_scores,
        dtype=float
    )[:k]

    if len(scores) == 0:
        return 0.0

    positions = np.arange(
        1,
        len(scores) + 1
    )

    discounts = np.log2(
        positions + 1
    )

    dcg = np.sum(
        (2 ** scores - 1)
        / discounts
    )

    ideal_scores = np.sort(
        scores
    )[::-1]

    ideal_dcg = np.sum(
        (2 ** ideal_scores - 1)
        / discounts
    )

    if ideal_dcg == 0:
        return 0.0

    return float(
        dcg / ideal_dcg
    )


# =========================================================
# 7. NDCG FROM CANDIDATE RANKING
# =========================================================

def candidate_ndcg_at_k(
    candidates,
    relevance_dict,
    k=5
):
    """
    Calculate NDCG@K when you have:

        candidates = system ranking

        relevance_dict = human relevance scores

    Example:

        candidates =
            ["C1", "C2", "C3", "C4"]

        relevance_dict =
            {
                "C1": 5,
                "C2": 2,
                "C3": 4,
                "C4": 0
            }
    """

    ranked_scores = [
        relevance_dict.get(
            candidate,
            0
        )
        for candidate in candidates[:k]
    ]

    return ndcg_at_k(
        ranked_scores,
        k
    )


# =========================================================
# 8. SPEARMAN RANK CORRELATION
# =========================================================

def spearman_correlation(
    human_scores,
    system_scores
):
    """
    Measures how similar the human ranking
    and system ranking are.

    Value:

        +1 = perfect ranking agreement
         0 = no correlation
        -1 = opposite ranking
    """

    if len(human_scores) != len(system_scores):
        raise ValueError(
            "Human and system scores must have "
            "the same length."
        )

    if len(human_scores) < 2:
        return 0.0

    correlation, _ = spearmanr(
        human_scores,
        system_scores
    )

    if np.isnan(correlation):
        return 0.0

    return float(correlation)


# =========================================================
# 9. MAE
# =========================================================

def score_mae(
    human_scores,
    system_scores
):
    """
    Mean Absolute Error between
    human scores and system scores.
    """

    if len(human_scores) != len(system_scores):
        raise ValueError(
            "Human and system scores must have "
            "the same length."
        )

    return float(
        mean_absolute_error(
            human_scores,
            system_scores
        )
    )


# =========================================================
# 10. ATS SCORE EVALUATION
# =========================================================

def evaluate_ats_scores(
    human_scores,
    system_scores
):
    """
    Evaluate your ATS scores against
    human recruiter scores.
    """

    correlation = spearman_correlation(
        human_scores,
        system_scores
    )

    mae = score_mae(
        human_scores,
        system_scores
    )

    return {
        "spearman_correlation": correlation,
        "mae": mae
    }


# =========================================================
# 11. RETRIEVAL EVALUATION
# =========================================================

def evaluate_retrieval(
    retrieved_candidates,
    relevant_candidates,
    k=5
):
    """
    Evaluate FAISS/BM25/Hybrid retrieval.
    """

    precision = precision_at_k(
        retrieved_candidates,
        relevant_candidates,
        k
    )

    recall = recall_at_k(
        retrieved_candidates,
        relevant_candidates,
        k
    )

    mrr = mrr_at_k(
        retrieved_candidates,
        relevant_candidates,
        k
    )

    return {
        f"precision@{k}": precision,
        f"recall@{k}": recall,
        f"mrr@{k}": mrr
    }


# =========================================================
# 12. CROSSENCODER EVALUATION
# =========================================================

def evaluate_reranking(
    before_candidates,
    after_candidates,
    relevant_candidates,
    k=5
):
    """
    Compare candidate ranking before
    and after CrossEncoder reranking.
    """

    before_mrr = mrr_at_k(
        before_candidates,
        relevant_candidates,
        k
    )

    after_mrr = mrr_at_k(
        after_candidates,
        relevant_candidates,
        k
    )

    before_precision = precision_at_k(
        before_candidates,
        relevant_candidates,
        k
    )

    after_precision = precision_at_k(
        after_candidates,
        relevant_candidates,
        k
    )

    before_recall = recall_at_k(
        before_candidates,
        relevant_candidates,
        k
    )

    after_recall = recall_at_k(
        after_candidates,
        relevant_candidates,
        k
    )

    return {
        f"before_precision@{k}":
            before_precision,

        f"after_precision@{k}":
            after_precision,

        f"before_recall@{k}":
            before_recall,

        f"after_recall@{k}":
            after_recall,

        f"before_mrr@{k}":
            before_mrr,

        f"after_mrr@{k}":
            after_mrr
    }


# =========================================================
# 13. TOP-K ACCURACY
# =========================================================

def top_k_accuracy(
    ranked_candidates,
    relevant_candidates,
    k=5
):
    """
    Returns 1 if at least one relevant candidate
    appears in the top K, otherwise 0.
    """

    relevant = set(relevant_candidates)

    top_candidates = ranked_candidates[:k]

    return int(
        any(
            candidate in relevant
            for candidate in top_candidates
        )
    )


# =========================================================
# 14. HIT RATE@K
# =========================================================

def hit_rate_at_k(
    ranked_candidates,
    relevant_candidates,
    k=5
):
    """
    Similar to top-K accuracy.

    1 = at least one relevant candidate
        appears in top K

    0 = no relevant candidate appears.
    """

    return top_k_accuracy(
        ranked_candidates,
        relevant_candidates,
        k
    )


# =========================================================
# 15. FULL RETRIEVAL REPORT
# =========================================================

def retrieval_report(
    retrieved_candidates,
    relevant_candidates,
    relevance_dict=None,
    k=5
):
    """
    Generate a complete retrieval evaluation report.
    """

    report = evaluate_retrieval(
        retrieved_candidates,
        relevant_candidates,
        k
    )

    report[f"hit_rate@{k}"] = hit_rate_at_k(
        retrieved_candidates,
        relevant_candidates,
        k
    )

    if relevance_dict is not None:

        report[f"ndcg@{k}"] = (
            candidate_ndcg_at_k(
                retrieved_candidates,
                relevance_dict,
                k
            )
        )

    return report


# =========================================================
# 16. COMPLETE RECRUITMENT EVALUATION
# =========================================================

def complete_evaluation(
    retrieved_candidates,
    relevant_candidates,
    human_scores=None,
    system_scores=None,
    relevance_dict=None,
    k=5
):
    """
    Run the main evaluation metrics
    for the recruitment system.

    This combines:

        Precision@K
        Recall@K
        MRR@K
        Hit Rate@K
        NDCG@K
        Spearman Correlation
        MAE
    """

    report = {}

    # ---------------------------------------------
    # Retrieval
    # ---------------------------------------------

    retrieval = retrieval_report(
        retrieved_candidates,
        relevant_candidates,
        relevance_dict,
        k
    )

    report.update(retrieval)

    # ---------------------------------------------
    # ATS / Human Evaluation
    # ---------------------------------------------

    if (
        human_scores is not None
        and system_scores is not None
    ):

        ats = evaluate_ats_scores(
            human_scores,
            system_scores
        )

        report.update(ats)

    return report


# =========================================================
# 17. PRINT EVALUATION REPORT
# =========================================================

def print_evaluation_report(report):
    """
    Print evaluation metrics in a readable format.
    """

    print("\n")
    print("=" * 55)
    print("        RECRUITMENT SYSTEM EVALUATION")
    print("=" * 55)

    for metric, value in report.items():

        if isinstance(value, float):

            print(
                f"{metric:<30}: "
                f"{value:.4f}"
            )

        else:

            print(
                f"{metric:<30}: "
                f"{value}"
            )

    print("=" * 55)