"""
Self-contained `torchcrf` compatibility module.

Implements a batch_first-aware CRF that exactly matches the API of
kmkurn/pytorch-crf (the library that was used during training), so
models serialized with that library can be loaded and run without any
changes to model weights or calling code.

State-dict keys are identical: start_transitions, end_transitions, transitions.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class CRF(nn.Module):
    """Conditional Random Field with optional batch_first support.

    Parameters
    ----------
    num_tags : int
        Number of tags.
    batch_first : bool
        If True, the input emissions/tags tensors have shape
        ``(batch, seq_len, num_tags)`` / ``(batch, seq_len)``.
        If False (default), they are ``(seq_len, batch, num_tags)`` / ``(seq_len, batch)``.
    """

    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError(f"num_tags must be positive, got {num_tags}")
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _to_seq_first(
        self, emissions: torch.Tensor, tags_or_mask: Optional[torch.Tensor]
    ):
        """Transpose from batch-first to seq-first when batch_first=True."""
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            if tags_or_mask is not None:
                tags_or_mask = tags_or_mask.transpose(0, 1)
        return emissions, tags_or_mask

    def _validate(
        self,
        emissions: torch.Tensor,
        tags: Optional[torch.LongTensor] = None,
        mask: Optional[torch.BoolTensor] = None,
    ) -> None:
        if emissions.dim() != 3:
            raise ValueError(
                f"emissions must be 3-D, got {emissions.dim()}-D"
            )
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f"last dim of emissions ({emissions.size(2)}) != num_tags ({self.num_tags})"
            )
        if tags is not None:
            if tags.dim() != 2:
                raise ValueError(f"tags must be 2-D, got {tags.dim()}-D")
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    f"emissions[:2] {tuple(emissions.shape[:2])} != tags {tuple(tags.shape)}"
                )
        if mask is not None:
            if mask.dim() != 2:
                raise ValueError(f"mask must be 2-D, got {mask.dim()}-D")
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    f"emissions[:2] {tuple(emissions.shape[:2])} != mask {tuple(mask.shape)}"
                )
            no_empty_seq = mask[0].all()
            if not no_empty_seq:
                raise ValueError("mask of the first step must all be True")

    # ------------------------------------------------------------------
    # forward (log-likelihood)
    # ------------------------------------------------------------------

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.LongTensor,
        mask: Optional[torch.BoolTensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute the conditional log-likelihood of a sequence of tags.

        Parameters
        ----------
        emissions : Tensor
            ``(batch, seq_len, num_tags)`` if batch_first else ``(seq_len, batch, num_tags)``
        tags : LongTensor
            ``(batch, seq_len)`` if batch_first else ``(seq_len, batch)``
        mask : BoolTensor, optional
            Same shape as tags; 1 for valid positions.
        reduction : str
            One of ``'none'``, ``'sum'``, ``'mean'``, ``'token_mean'``.

        Returns
        -------
        Tensor
            Per-sample log-likelihood if ``reduction='none'``; scalar otherwise.
        """
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)

        self._validate(emissions, tags, mask)

        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.bool)

        numerator = self._compute_score(emissions, tags, mask)    # (batch,)
        denominator = self._compute_normalizer(emissions, mask)   # (batch,)
        llh = numerator - denominator                             # (batch,)

        if reduction == "none":
            return llh
        elif reduction == "sum":
            return llh.sum()
        elif reduction == "mean":
            return llh.mean()
        elif reduction == "token_mean":
            return llh.sum() / mask.float().sum()
        else:
            raise ValueError(f"Unknown reduction: {reduction!r}")

    def _compute_score(
        self,
        emissions: torch.Tensor,
        tags: torch.LongTensor,
        mask: torch.BoolTensor,
    ) -> torch.Tensor:
        # emissions: (seq_len, batch, num_tags)
        # tags:      (seq_len, batch)
        # mask:      (seq_len, batch)
        seq_len, batch_size = tags.shape

        score = self.start_transitions[tags[0]]                    # (batch,)
        score += emissions[0, torch.arange(batch_size), tags[0]]  # (batch,)

        for i in range(1, seq_len):
            score += (
                self.transitions[tags[i - 1], tags[i]] + emissions[i, torch.arange(batch_size), tags[i]]
            ) * mask[i].float()

        # end transition: pick the score at the actual last valid tag
        seq_ends = mask.long().sum(dim=0) - 1               # (batch,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        score += self.end_transitions[last_tags]
        return score

    def _compute_normalizer(
        self,
        emissions: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> torch.Tensor:
        # emissions: (seq_len, batch, num_tags)
        seq_len = emissions.size(0)

        # (batch, num_tags)
        log_prob = self.start_transitions + emissions[0]

        for i in range(1, seq_len):
            # (batch, num_tags, 1)
            broadcast_log_prob = log_prob.unsqueeze(2)
            # (1, num_tags, num_tags)
            broadcast_trans = self.transitions.unsqueeze(0)
            # (batch, 1, num_tags)
            broadcast_emit = emissions[i].unsqueeze(1)

            next_score = torch.logsumexp(
                broadcast_log_prob + broadcast_trans + broadcast_emit, dim=1
            )  # (batch, num_tags)
            log_prob = torch.where(mask[i].unsqueeze(1), next_score, log_prob)

        return torch.logsumexp(log_prob + self.end_transitions, dim=1)  # (batch,)

    # ------------------------------------------------------------------
    # decode (Viterbi)
    # ------------------------------------------------------------------

    def decode(
        self,
        emissions: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> List[List[int]]:
        """Viterbi decoding.

        Parameters
        ----------
        emissions : Tensor
            ``(batch, seq_len, num_tags)`` if batch_first else ``(seq_len, batch, num_tags)``
        mask : BoolTensor, optional

        Returns
        -------
        list of list of int
            Best tag sequence for each element in the batch.
        """
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)

        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.bool)

        return self._viterbi_decode(emissions, mask)

    def _viterbi_decode(
        self,
        emissions: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> List[List[int]]:
        # emissions: (seq_len, batch, num_tags)
        # mask:      (seq_len, batch)
        seq_len, batch_size, num_tags = emissions.shape

        # (batch, num_tags)
        viterbi_score = self.start_transitions + emissions[0]
        # list of (batch, num_tags) backpointer tensors
        history: list[torch.Tensor] = []

        for i in range(1, seq_len):
            # (batch, num_tags, 1) + (1, num_tags, num_tags) + (batch, 1, num_tags)
            broadcast_score = viterbi_score.unsqueeze(2)
            broadcast_emit = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emit
            best_score, best_tags = next_score.max(dim=1)  # (batch, num_tags)

            viterbi_score = torch.where(
                mask[i].unsqueeze(1), best_score, viterbi_score
            )
            history.append(best_tags)  # (batch, num_tags)

        # end transition
        viterbi_score += self.end_transitions

        # best last tag per batch element
        seq_ends = mask.long().sum(dim=0) - 1      # (batch,)
        best_last_tag = viterbi_score.argmax(dim=1)  # (batch,)

        best_path = [best_last_tag]
        for bp in reversed(history):
            # bp: (batch, num_tags)
            best_last_tag = bp[torch.arange(batch_size), best_last_tag]
            best_path.append(best_last_tag)
        best_path.reverse()

        # trim each path to its actual sequence length
        result: List[List[int]] = []
        for b in range(batch_size):
            length = int(seq_ends[b].item()) + 1
            result.append([best_path[t][b].item() for t in range(length)])
        return result

    def __repr__(self) -> str:
        return f"CRF(num_tags={self.num_tags}, batch_first={self.batch_first})"
