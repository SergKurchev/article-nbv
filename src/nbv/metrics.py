"""NBV evaluation metrics — aligned with Wang et al. and Korbach et al. protocols.

Mathematical definitions:

Objects Found (fraction):
    F_found = (1/K) * sum_{k=1}^{K} 1[obj_k detected >= 1 time]

Frames-to-Find:
    T_find = min{t : all objects detected}, or T_max + 1 if never achieved

Frames-to-Classify:
    T_class = min{t : all pred_scores_k >= theta}, or T_max + 1

Success Rate (over N episodes):
    SR = (1/N) * sum_n 1[success_n]

Wang macro metrics (per-class one-vs-all, averaged over classes in support):
    precision_c = TP_c / (TP_c + FP_c)
    recall_c    = TP_c / (TP_c + FN_c)
    F1_c        = 2 * prec * rec / (prec + rec)
    macro_* = mean over support classes

Korbach confidence difference (over T views):
    conf_diff_t = top1_confidence_t - top2_confidence_t
    final_confidence_diff = conf_diff_{T}
    best_confidence_diff  = max_t conf_diff_t

ODIN uncertainty:
    unc_t = mean epistemic uncertainty at step t
    uncertainty_reduction = unc_0 - unc_T
    uncertainty_auc = (1/T) * sum_t unc_t
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Sequence


@dataclass
class EpisodeMetrics:
    policy: str
    seed: int
    steps: int
    total_reward: float
    final_margin: float          # top1_conf - top2_conf at episode end
    final_unc: float             # mean uncertainty at episode end
    initial_unc: float           # mean uncertainty at episode start
    correct: bool                # all objects correctly classified
    objects_correct: int = 0     # number of correctly classified objects
    objects_found: int = 0       # number of objects detected >= 1 time
    objects_in_scene: int = 0    # total ground-truth objects
    target_correct: bool = False  # primary target correctly classified
    initial_confidence: float = 0.0
    final_confidence: float = 0.0
    initial_confidence_diff: float = 0.0
    first_step_confidence_diff: float = 0.0
    final_confidence_diff: float = 0.0
    best_confidence_diff: float = 0.0
    steps_to_best_confidence_diff: int = 0
    wang_macro_accuracy: float = 0.0
    wang_macro_precision: float = 0.0
    wang_macro_recall: float = 0.0
    wang_macro_f1: float = 0.0
    steps_to_all_objects: int = 0   # T_find
    steps_to_all_correct: int = 0   # T_class
    geometry_visible_classes: int = 0
    geometry_mean_coverage: float = 0.0
    collision_steps: int = 0
    out_of_reach_steps: int = 0
    p_hidden_initial: float = 1.0
    p_hidden_final: float = 1.0
    uncertainty_curve: list[float] = field(default_factory=list)
    confidence_diff_curve: list[float] = field(default_factory=list)
    reward_curve: list[float] = field(default_factory=list)

    @property
    def uncertainty_reduction(self) -> float:
        if not self.uncertainty_curve:
            return 0.0
        return float(self.initial_unc - self.uncertainty_curve[-1])

    @property
    def uncertainty_auc(self) -> float:
        if not self.uncertainty_curve:
            return 0.0
        return float(mean(float(u) for u in self.uncertainty_curve))

    @property
    def steps_to_confidence(self) -> int:
        """First step at which mean uncertainty drops below 0.3."""
        for i, u in enumerate(self.uncertainty_curve):
            if u <= 0.3:
                return i
        return len(self.uncertainty_curve)

    @property
    def objects_found_fraction(self) -> float:
        if self.objects_in_scene == 0:
            return 0.0
        return self.objects_found / self.objects_in_scene

    def to_dict(self) -> dict:
        data = asdict(self)
        data["uncertainty_reduction"] = self.uncertainty_reduction
        data["uncertainty_auc"] = self.uncertainty_auc
        data["steps_to_confidence"] = self.steps_to_confidence
        data["objects_found_fraction"] = self.objects_found_fraction
        return data


def aggregate_metrics(logs: Sequence[EpisodeMetrics]) -> dict:
    """Group per-policy and compute mean of every metric."""
    by_policy: dict[str, list[EpisodeMetrics]] = {}
    for log in logs:
        by_policy.setdefault(log.policy, []).append(log)

    out: dict[str, dict] = {}
    for name, ls in by_policy.items():
        out[name] = {
            "target_accuracy": mean(float(log.target_correct) for log in ls),
            "scene_accuracy": mean(float(log.correct) for log in ls),
            "wang_macro_accuracy": mean(log.wang_macro_accuracy for log in ls),
            "wang_macro_precision": mean(log.wang_macro_precision for log in ls),
            "wang_macro_recall": mean(log.wang_macro_recall for log in ls),
            "wang_macro_f1": mean(log.wang_macro_f1 for log in ls),
            "korbach_initial_confidence": mean(log.initial_confidence for log in ls),
            "korbach_final_confidence": mean(log.final_confidence for log in ls),
            "korbach_first_step_confidence_diff": mean(log.first_step_confidence_diff for log in ls),
            "korbach_final_confidence_diff": mean(log.final_confidence_diff for log in ls),
            "korbach_best_confidence_diff": mean(log.best_confidence_diff for log in ls),
            "korbach_steps_to_best": mean(log.steps_to_best_confidence_diff for log in ls),
            "odin_uncertainty_reduction": mean(log.uncertainty_reduction for log in ls),
            "odin_steps_to_confidence": mean(log.steps_to_confidence for log in ls),
            "odin_uncertainty_auc": mean(log.uncertainty_auc for log in ls),
            "mean_reward": mean(log.total_reward for log in ls),
            "mean_final_unc": mean(log.final_unc for log in ls),
            "mean_final_margin": mean(log.final_margin for log in ls),
            "objects_correct": mean(log.objects_correct for log in ls),
            "objects_found": mean(log.objects_found for log in ls),
            "objects_found_fraction": mean(log.objects_found_fraction for log in ls),
            "objects_in_scene": mean(log.objects_in_scene for log in ls),
            "steps_to_all_objects": mean(log.steps_to_all_objects for log in ls),
            "steps_to_all_correct": mean(log.steps_to_all_correct for log in ls),
            "p_hidden_initial": mean(log.p_hidden_initial for log in ls),
            "p_hidden_final": mean(log.p_hidden_final for log in ls),
            "geometry_visible_classes": mean(log.geometry_visible_classes for log in ls),
            "geometry_mean_coverage": mean(log.geometry_mean_coverage for log in ls),
            "collision_steps": mean(log.collision_steps for log in ls),
            "out_of_reach_steps": mean(log.out_of_reach_steps for log in ls),
            "episodes": len(ls),
        }
    return out


def print_metrics_table(agg: dict) -> None:
    cols = [
        "scene_accuracy", "wang_macro_f1", "wang_macro_precision", "wang_macro_recall",
        "korbach_final_confidence_diff", "korbach_best_confidence_diff",
        "odin_uncertainty_reduction", "odin_steps_to_confidence",
        "mean_reward", "objects_found_fraction", "objects_correct",
        "p_hidden_final", "collision_steps", "out_of_reach_steps",
    ]
    w = 22
    header = f"{'policy':<18} " + " ".join(c[:w].rjust(w) for c in cols)
    print(header)
    print("-" * len(header))
    for name, m in agg.items():
        row = f"{name:<18} " + " ".join(f"{m[c]:>{w}.4f}" for c in cols)
        print(row)


def macro_classification_metrics(
    predicted_classes: list[int],
    gt_classes: list[int],
    num_classes: int,
) -> dict[str, float]:
    """Macro one-vs-all metrics for unordered multi-object classification.

    Averages precision/recall/F1 only over classes in the support set
    (those appearing in either prediction or ground truth for this episode).
    Accuracy is averaged over all num_classes (counts true-negatives).

    Args:
        predicted_classes: List of predicted class ids (from ODIN)
        gt_classes: List of ground-truth class ids
        num_classes: Total number of classes in the taxonomy

    Returns:
        dict with keys: accuracy, precision, recall, f1, support
    """
    pred = Counter(int(x) for x in predicted_classes if int(x) >= 0)
    gt = Counter(int(x) for x in gt_classes if int(x) >= 0)
    total = max(sum(gt.values()), sum(pred.values()), 1)
    support = sorted(set(pred.keys()) | set(gt.keys()))

    precision_list, recall_list, f1_list, accuracy_list = [], [], [], []
    for cls in range(num_classes):
        tp = min(pred[cls], gt[cls])
        fp = max(0, pred[cls] - gt[cls])
        fn = max(0, gt[cls] - pred[cls])
        tn = max(0, total - tp - fp - fn)
        accuracy_list.append((tp + tn) / max(1, tp + fp + fn + tn))
        if cls in support:
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            precision_list.append(p)
            recall_list.append(r)
            f1_list.append((2.0 * p * r / (p + r)) if (p + r) else 0.0)

    return {
        "accuracy": float(mean(accuracy_list)),
        "precision": float(mean(precision_list)) if precision_list else 0.0,
        "recall": float(mean(recall_list)) if recall_list else 0.0,
        "f1": float(mean(f1_list)) if f1_list else 0.0,
        "support": int(len(support)),
    }


class EpisodeMetricsTracker:
    """Stateful per-episode metrics accumulator — call update() at each step."""

    def __init__(self, policy: str, seed: int, num_classes: int, max_steps: int):
        self.policy = policy
        self.seed = seed
        self.num_classes = num_classes
        self.max_steps = max_steps

        self._step = 0
        self._total_reward = 0.0
        self._collision_steps = 0
        self._oob_steps = 0

        self._gt_classes: list[int] = []
        self._objects_in_scene = 0
        self._objects_found_ids: set[int] = set()
        self._step_all_found: int = max_steps + 1

        self._pred_classes_final: list[int] = []
        self._pred_scores_final: list[float] = []
        self._step_all_correct: int = max_steps + 1

        self._unc_curve: list[float] = []
        self._conf_diff_curve: list[float] = []
        self._reward_curve: list[float] = []
        self._p_hidden_initial: float = 1.0
        self._p_hidden_final: float = 1.0
        self._p_hidden_set = False

        self._initial_unc: float = 1.0
        self._initial_conf: float = 0.0
        self._initial_conf_diff: float = 0.0

    def set_scene(self, gt_classes: list[int]) -> None:
        self._gt_classes = list(gt_classes)
        self._objects_in_scene = len(gt_classes)

    def update(
        self,
        *,
        reward: float,
        pred_classes: list[int],
        pred_scores: list[float],
        detected_ids: list[int],
        p_hidden: float,
        uncertainty: float,
        is_collision: bool = False,
        is_oob: bool = False,
    ) -> None:
        self._step += 1
        self._total_reward += reward
        self._reward_curve.append(reward)
        self._unc_curve.append(uncertainty)
        self._p_hidden_final = p_hidden

        if not self._p_hidden_set:
            self._p_hidden_initial = p_hidden
            self._p_hidden_set = True

        if is_collision:
            self._collision_steps += 1
        if is_oob:
            self._oob_steps += 1

        if self._step == 1:
            self._initial_unc = uncertainty
            if pred_scores:
                sorted_s = sorted(pred_scores, reverse=True)
                self._initial_conf = sorted_s[0]
                self._initial_conf_diff = sorted_s[0] - (sorted_s[1] if len(sorted_s) > 1 else 0.0)

        # Track objects found (by id, not class, since same class may appear multiple times)
        for did in detected_ids:
            self._objects_found_ids.add(did)
        if len(self._objects_found_ids) >= self._objects_in_scene and self._step_all_found > self.max_steps:
            self._step_all_found = self._step

        # Confidence difference curve
        conf_diff = 0.0
        if pred_scores:
            sorted_s = sorted(pred_scores, reverse=True)
            conf_diff = sorted_s[0] - (sorted_s[1] if len(sorted_s) > 1 else 0.0)
        self._conf_diff_curve.append(conf_diff)

        self._pred_classes_final = list(pred_classes)
        self._pred_scores_final = list(pred_scores)

    def finalize(self, success: bool = False) -> EpisodeMetrics:
        wang = macro_classification_metrics(
            self._pred_classes_final, self._gt_classes, self.num_classes
        )

        final_conf = max(self._pred_scores_final) if self._pred_scores_final else 0.0
        sorted_final = sorted(self._pred_scores_final, reverse=True)
        final_conf_diff = sorted_final[0] - (sorted_final[1] if len(sorted_final) > 1 else 0.0) if sorted_final else 0.0
        best_conf_diff = max(self._conf_diff_curve) if self._conf_diff_curve else 0.0
        steps_to_best = (self._conf_diff_curve.index(best_conf_diff) + 1) if self._conf_diff_curve else 0

        objects_correct = sum(
            min(Counter(self._pred_classes_final)[c], Counter(self._gt_classes)[c])
            for c in set(self._gt_classes)
        )
        correct = (success and objects_correct == self._objects_in_scene)

        return EpisodeMetrics(
            policy=self.policy,
            seed=self.seed,
            steps=self._step,
            total_reward=self._total_reward,
            final_margin=final_conf_diff,
            final_unc=self._unc_curve[-1] if self._unc_curve else 0.0,
            initial_unc=self._initial_unc,
            correct=correct,
            objects_correct=objects_correct,
            objects_found=len(self._objects_found_ids),
            objects_in_scene=self._objects_in_scene,
            target_correct=(success and objects_correct > 0),
            initial_confidence=self._initial_conf,
            final_confidence=final_conf,
            initial_confidence_diff=self._initial_conf_diff,
            first_step_confidence_diff=self._conf_diff_curve[0] if self._conf_diff_curve else 0.0,
            final_confidence_diff=final_conf_diff,
            best_confidence_diff=best_conf_diff,
            steps_to_best_confidence_diff=steps_to_best,
            wang_macro_accuracy=wang["accuracy"],
            wang_macro_precision=wang["precision"],
            wang_macro_recall=wang["recall"],
            wang_macro_f1=wang["f1"],
            steps_to_all_objects=self._step_all_found,
            steps_to_all_correct=self._step if success else self.max_steps + 1,
            collision_steps=self._collision_steps,
            out_of_reach_steps=self._oob_steps,
            p_hidden_initial=self._p_hidden_initial,
            p_hidden_final=self._p_hidden_final,
            uncertainty_curve=list(self._unc_curve),
            confidence_diff_curve=list(self._conf_diff_curve),
            reward_curve=list(self._reward_curve),
        )
