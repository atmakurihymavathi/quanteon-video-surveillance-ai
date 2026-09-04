"""Tests for MOT evaluation helpers."""

import os
import sys

import pytest

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from src.evaluation import (
    GTBox,
    PredBox,
    evaluate,
    load_mot_ground_truth,
)


class TestLoadMotGroundTruth:
    def test_loads_valid_mot_rows(self, tmp_path):
        path = tmp_path / "gt.txt"

        path.write_text(
            "1,7,10,20,30,40,1,1,1\n"
            "2,7,12,20,30,40,1,1,1\n",
            encoding="utf-8",
        )

        boxes = load_mot_ground_truth(str(path))

        assert len(boxes) == 2
        assert boxes[0].frame == 1
        assert boxes[0].track_id == 7
        assert boxes[0].bbox == (10.0, 20.0, 40.0, 60.0)

    def test_skips_malformed_rows(self, tmp_path):
        path = tmp_path / "gt.txt"

        path.write_text(
            "bad,row\n"
            "1,5,10,20,30,40,1,1,1\n",
            encoding="utf-8",
        )

        boxes = load_mot_ground_truth(str(path))

        assert len(boxes) == 1

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_mot_ground_truth(
                str(tmp_path / "missing.txt")
            )


class TestEvaluate:
    def test_perfect_match(self):
        gt = [
            GTBox(
                frame=1,
                track_id=1,
                bbox=(0, 0, 100, 100),
            )
        ]

        predictions = [
            PredBox(
                frame=1,
                track_id=10,
                bbox=(0, 0, 100, 100),
                confidence=0.9,
            )
        ]

        result = evaluate(gt, predictions)

        assert result.true_positives == 1
        assert result.false_positives == 0
        assert result.false_negatives == 0
        assert result.precision == pytest.approx(1.0)
        assert result.recall == pytest.approx(1.0)
        assert result.f1 == pytest.approx(1.0)
        assert result.id_switches == 0

    def test_false_positive_and_false_negative(self):
        gt = [
            GTBox(
                frame=1,
                track_id=1,
                bbox=(0, 0, 50, 50),
            )
        ]

        predictions = [
            PredBox(
                frame=1,
                track_id=10,
                bbox=(100, 100, 150, 150),
            )
        ]

        result = evaluate(gt, predictions)

        assert result.true_positives == 0
        assert result.false_positives == 1
        assert result.false_negatives == 1

    def test_id_switch_is_counted(self):
        gt = [
            GTBox(1, 1, (0, 0, 100, 100)),
            GTBox(2, 1, (0, 0, 100, 100)),
        ]

        predictions = [
            PredBox(1, 10, (0, 0, 100, 100)),
            PredBox(2, 20, (0, 0, 100, 100)),
        ]

        result = evaluate(gt, predictions)

        assert result.true_positives == 2
        assert result.id_switches == 1

    def test_empty_predictions(self):
        gt = [
            GTBox(1, 1, (0, 0, 100, 100))
        ]

        result = evaluate(gt, [])

        assert result.true_positives == 0
        assert result.false_positives == 0
        assert result.false_negatives == 1
        assert result.recall == 0.0
