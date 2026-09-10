"""Conservative confirmation of a streaming CTC decoder's stable prefix."""

from __future__ import annotations


class StreamingCTCPrefix:
    """Commit only a confirmed, fully observed prefix for one utterance."""

    _LOOKAHEAD = 8

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Forget all evidence and irreversible commits for the utterance."""
        self._committed: list[str] = []
        self._committed_positions: list[int] = []
        self._previous: tuple[list[str], list[int], int] | None = None
        self._highest_evidence_end = 0

    @staticmethod
    def _validate(hypothesis: list[str], positions: list[int], evidence_end: int) -> None:
        if not isinstance(hypothesis, list) or not isinstance(positions, list):
            raise ValueError("hypothesis and positions must be lists")
        if not isinstance(evidence_end, int) or evidence_end <= 0:
            raise ValueError("evidence_end must be positive")
        if len(hypothesis) != len(positions):
            raise ValueError("hypothesis and positions must have the same length")
        previous = -1
        for label, position in zip(hypothesis, positions):
            if not isinstance(label, str) or not label:
                raise ValueError("hypothesis labels must be nonempty strings")
            if not isinstance(position, int) or position < 0 or position >= evidence_end:
                raise ValueError("positions must be nonnegative and before evidence_end")
            if position <= previous:
                raise ValueError("positions must be strictly increasing")
            previous = position

    def _result(self, *, conflict: bool = False, new_commits: list[str] | None = None) -> dict[str, object]:
        if conflict or self._previous is None:
            provisional: list[str] = []
        else:
            provisional = self._previous[0][len(self._committed):]
        return {
            "committed": list(self._committed),
            "provisional": provisional,
            "conflict": conflict,
            "new_commits": list(new_commits or []),
        }

    def update(
        self, hypothesis: list[str], positions: list[int], evidence_end: int
    ) -> dict[str, object]:
        """Return the agreement-confirmed prefix; agreement does not imply correctness."""
        self._validate(hypothesis, positions, evidence_end)
        labels, token_positions = list(hypothesis), list(positions)
        if evidence_end <= self._highest_evidence_end:
            return self._result()
        self._highest_evidence_end = evidence_end

        committed_length = len(self._committed)
        if (
            len(labels) < committed_length
            or labels[:committed_length] != self._committed
            or token_positions[:committed_length] != self._committed_positions
        ):
            return self._result(conflict=True)

        if self._previous is None:
            self._previous = (labels, token_positions, evidence_end)
            return self._result()

        previous_labels, previous_positions, _ = self._previous
        common_length = 0
        for old_label, old_position, label, position in zip(
            previous_labels, previous_positions, labels, token_positions
        ):
            if old_label != label or old_position != position:
                break
            common_length += 1
        safe_length = committed_length
        while (
            safe_length < common_length
            and token_positions[safe_length] < evidence_end - self._LOOKAHEAD
        ):
            safe_length += 1
        new_commits = labels[committed_length:safe_length]
        if new_commits:
            self._committed.extend(new_commits)
            self._committed_positions.extend(token_positions[committed_length:safe_length])
        self._previous = (labels, token_positions, evidence_end)
        return self._result(new_commits=new_commits)
