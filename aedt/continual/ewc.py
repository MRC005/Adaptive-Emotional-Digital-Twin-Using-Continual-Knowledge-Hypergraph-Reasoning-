"""LAYER 1 / MODULE 6 -- Elastic Weight Consolidation.

Purpose  Let the emotional-pattern model keep learning from new episodes
         without discarding what earlier episodes taught it.
Input    A torch model, and a stream of tasks in chronological order.
Output   Fisher information, anchored parameters, and the EWC penalty.
Status   RESEARCH COMPONENT, trained offline. Not in the browser.

THE METHOD (Kirkpatrick et al., 2017)

    L_total = L_new  +  (lambda/2) * sum_i F_i (theta_i - theta*_i)^2

``theta*`` are the parameters after the previous task; ``F`` is the diagonal
of the Fisher information, estimated as the mean squared gradient of the
log-likelihood of the model's OWN predictions on that task's data. Parameters
the old task depended on get a stiff spring; the rest stay free.

WHY THE FISHER IS COMPUTED ON SAMPLED LABELS, NOT TRUE LABELS

The Fisher is an expectation under the MODEL's predictive distribution, so the
labels are sampled from ``softmax(logits)`` rather than taken from the dataset.
Using true labels computes the empirical Fisher, which is a different quantity
and is a common way to get EWC subtly wrong. ``empirical=True`` is offered for
comparison and is off by default.

THE DISTINCTION THIS MODULE EXISTS TO ENFORCE

  A. DATA MEMORY UPDATE  -- the twin stores another event. No parameters move.
                            That is ``aedt/knowledge/store.py``, and it is NOT
                            continual learning.
  B. MODEL CONTINUAL LEARNING -- parameters are updated on new data while a
                            penalty protects the old. That is this file.

Claiming (A) as (B) is the specific dishonesty this module is here to prevent.
The experiment in ``scripts/run_ewc_experiment.py`` measures forgetting on
sequential tasks and reports it whether or not EWC helps.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

__all__ = ["EWC", "forgetting_metrics"]


class EWC:
    """Fisher estimation and the quadratic penalty, over one or more tasks."""

    def __init__(self, model, lam: float = 1000.0):
        import torch
        self.model = model
        self.lam = float(lam)
        # task_id -> {param name: tensor}
        self.fisher: dict[str, dict] = {}
        self.anchor: dict[str, dict] = {}

    # ----------------------------------------------------------- estimation
    def consolidate(self, task_id: str, batches, *, n_samples: int | None = None,
                    empirical: bool = False) -> None:
        """Estimate the diagonal Fisher on ``batches`` and anchor the weights.

        ``batches`` yields ``(inputs_tuple, targets)`` or
        ``(inputs_tuple, targets, index)``. The optional ``index`` selects the
        rows the loss is taken over, which matters for a transductive model
        whose forward pass returns every node: without it the Fisher would be
        estimated over the held-out rows too, and would no longer describe what
        the task actually trained on.

        Called AFTER a task is learned, so the anchor is that task's solution.
        """
        import torch
        import torch.nn.functional as F

        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()
                  if p.requires_grad}
        seen = 0
        for batch in batches:
            inputs, targets = batch[0], batch[1]
            index = batch[2] if len(batch) > 2 else None
            self.model.zero_grad(set_to_none=True)
            logits = self.model(*inputs)
            if index is not None:
                logits = logits[index]
                targets = targets[index]
            if empirical:
                labels = targets
            else:
                # sample from the model's own predictive distribution
                with torch.no_grad():
                    probs = F.softmax(logits, dim=-1)
                labels = torch.multinomial(probs, 1).squeeze(-1)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
            seen += 1
            if n_samples and seen >= n_samples:
                break

        denom = max(seen, 1)
        self.fisher[task_id] = {n: (f / denom) for n, f in fisher.items()}
        self.anchor[task_id] = {n: p.detach().clone()
                                for n, p in self.model.named_parameters()
                                if p.requires_grad}
        self.model.zero_grad(set_to_none=True)
        log.info("EWC consolidated task %s over %d batches", task_id, denom)

    # -------------------------------------------------------------- penalty
    def penalty(self):
        """(lambda/2) * sum over consolidated tasks of F (theta - theta*)^2."""
        import torch
        if not self.fisher:
            return torch.zeros((), requires_grad=True)
        total = None
        params = dict(self.model.named_parameters())
        for task_id, fisher in self.fisher.items():
            anchor = self.anchor[task_id]
            for n, f in fisher.items():
                p = params.get(n)
                if p is None or not p.requires_grad:
                    continue
                term = (f * (p - anchor[n]) ** 2).sum()
                total = term if total is None else total + term
        if total is None:
            return torch.zeros((), requires_grad=True)
        return 0.5 * self.lam * total

    @property
    def n_tasks(self) -> int:
        return len(self.fisher)

    def fisher_summary(self) -> dict:
        """Sanity numbers: a Fisher of all zeros means EWC is doing nothing."""
        out = {}
        for task_id, f in self.fisher.items():
            vals = [float(t.abs().mean()) for t in f.values()]
            out[task_id] = {"n_params": len(vals),
                            "mean_abs_fisher": float(sum(vals) / max(len(vals), 1)),
                            "all_zero": all(v == 0.0 for v in vals)}
        return out


def forgetting_metrics(acc_matrix: list[list[float]]) -> dict:
    """Standard continual-learning summary from an accuracy matrix.

    ``acc_matrix[i][j]`` is accuracy on task j after training through task i.

      average accuracy  mean over j of acc[last][j]
      forgetting        mean over earlier j of (best earlier acc on j) - acc[last][j]
      backward transfer acc[last][j] - acc[j][j], averaged; negative = forgetting
    """
    if not acc_matrix:
        return {}
    T = len(acc_matrix)
    last = acc_matrix[-1]
    avg = sum(last) / len(last)
    forg, bwt = [], []
    for j in range(T - 1):
        best_earlier = max(acc_matrix[i][j] for i in range(j, T - 1))
        forg.append(best_earlier - last[j])
        bwt.append(last[j] - acc_matrix[j][j])
    return {
        "average_accuracy": float(avg),
        "forgetting": float(sum(forg) / len(forg)) if forg else 0.0,
        "backward_transfer": float(sum(bwt) / len(bwt)) if bwt else 0.0,
        "final_per_task": [float(v) for v in last],
    }
