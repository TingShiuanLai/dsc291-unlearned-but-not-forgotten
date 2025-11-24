"""
ROUGE-L and A-ESR metrics
"""

from typing import List, Dict


class RougeL:
    """
    ROUGE-L (Longest Common Subsequence) metric implementation.

    ROUGE-L measures the longest common subsequence between candidate and reference,
    providing a measure of content overlap and order preservation.
    """

    @staticmethod
    def _lcs_length(x: List[str], y: List[str]) -> int:
        """
        Compute length of longest common subsequence.

        Args:
            x: First sequence (list of tokens)
            y: Second sequence (list of tokens)

        Returns:
            Length of LCS
        """
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    @staticmethod
    def compute(candidate: str, reference: str) -> Dict[str, float]:
        """
        Compute ROUGE-L scores.

        Args:
            candidate: Generated text
            reference: Reference/ground truth text

        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        # Tokenize (simple whitespace tokenization)
        candidate_tokens = candidate.lower().split()
        reference_tokens = reference.lower().split()

        if not candidate_tokens or not reference_tokens:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        # Compute LCS length
        lcs_len = RougeL._lcs_length(candidate_tokens, reference_tokens)

        # Compute precision, recall, F1
        precision = lcs_len / len(candidate_tokens) if candidate_tokens else 0.0
        recall = lcs_len / len(reference_tokens) if reference_tokens else 0.0

        if precision + recall > 0:
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


class ExtractionSuccessRate:
    """
    A-ESR (Approximate Extraction Success Rate) metric.

    Measures the proportion of samples where ROUGE-L F1 score
    exceeds a given threshold θ.

    Common thresholds:
    - θ = 0.9: High similarity (90% match)
    - θ = 1.0: Exact match (100% match)
    """

    @staticmethod
    def compute(rouge_scores: List[float], threshold: float) -> Dict[str, float]:
        """
        Compute A-ESR at given threshold.

        Args:
            rouge_scores: List of ROUGE-L F1 scores
            threshold: Similarity threshold (0.0 to 1.0)

        Returns:
            Dictionary with A-ESR score and success count
        """
        if not rouge_scores:
            return {'a_esr': 0.0, 'success_count': 0, 'total_count': 0}

        successes = sum(1 for score in rouge_scores if score >= threshold)
        a_esr = successes / len(rouge_scores)

        return {
            'a_esr': a_esr,
            'success_count': successes,
            'total_count': len(rouge_scores),
            'threshold': threshold
        }