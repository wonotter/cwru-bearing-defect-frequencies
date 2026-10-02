"""주파수 선택/진폭 정규화/원시 점수 ROC를 검증하는 독립적인 예제."""

import unittest

import numpy as np

from rule_classifier import (
    CLASS_NAMES, _dataset_keys, classify_segment, evaluate_predictions,
    predict_fault_or_normal, resample_to_12k, score_spectrum,
)


class FrequencyRuleTests(unittest.TestCase):
    def test_known_outer_race_harmonics(self):
        freqs = np.arange(0.0, 6001.0)
        amplitudes = np.ones_like(freqs)
        # 1800 RPM: BPFO = 107.544 Hz. 세 고조파만 강하게 만든다.
        amplitudes[[108, 215, 323]] = 30.0
        scores, peaks, background = score_spectrum(freqs, amplitudes, rpm=1800)
        self.assertEqual(CLASS_NAMES[np.argmax(scores)], "Outer Race")
        self.assertAlmostEqual(scores[0], 30.0)
        self.assertAlmostEqual(background, 1.0)
        self.assertEqual([p["peak_hz"] for p in peaks[0]], [108, 215, 323])

    def test_score_is_invariant_to_signal_amplitude(self):
        freqs = np.arange(0.0, 6001.0)
        amplitudes = np.ones_like(freqs)
        amplitudes[[162, 325, 487]] = [5, 10, 15]
        first, _, _ = score_spectrum(freqs, amplitudes, rpm=1800)
        scaled, _, _ = score_spectrum(freqs, 17 * amplitudes, rpm=1800)
        np.testing.assert_allclose(first, scaled)
        self.assertEqual(CLASS_NAMES[np.argmax(first)], "Inner Race")

    def test_no_signal_has_no_frequency_evidence(self):
        scores, _, _ = score_spectrum(np.arange(6001), np.zeros(6001), rpm=1800)
        np.testing.assert_array_equal(scores, [0, 0])
        result = classify_segment(np.zeros(12000), fs=12000, rpm=1800)
        self.assertEqual(result["predicted_label"], "Normal")
        np.testing.assert_array_equal(result["scores"], [0, 0, 0])

    def test_end_to_end_am_carrier_without_ground_truth(self):
        # 공진 주파수 3500 Hz에 내륜 주파수 162.456 Hz의 진폭 변조를 합성.
        fs = 12000
        t = np.arange(fs) / fs
        modulation = 1 + 0.6 * np.cos(2 * np.pi * 162.456 * t)
        signal = modulation * np.cos(2 * np.pi * 3500 * t)
        result = classify_segment(signal, fs=fs, rpm=1800)
        self.assertEqual(result["predicted_label"], "Inner Race")

    def test_invalid_spectrum_is_rejected(self):
        with self.assertRaises(ValueError):
            score_spectrum([0, 1, 0], [1, 1, 1], rpm=1800)
        with self.assertRaises(ValueError):
            score_spectrum(np.arange(6001), np.ones(6001), rpm=0)


class MetricTests(unittest.TestCase):
    def test_perfect_classification_with_nonprobability_scores(self):
        # 행의 합은 1이 아니다. 이진 OvR은 이 연속 점수를 직접 사용한다.
        scores = np.array([[40, 1, -40], [3, 50, -50], [2, 3, -3]])
        report, _ = evaluate_predictions(CLASS_NAMES, CLASS_NAMES, scores)
        self.assertEqual(report["confusion_matrix"], np.eye(3, dtype=int).tolist())
        self.assertEqual(report["macro_f1"], 1.0)
        self.assertEqual(report["macro_auc_ovr"], 1.0)

    def test_tied_scores_have_auc_one_half(self):
        truth = list(CLASS_NAMES) * 2
        predictions = ["Outer Race"] * 6
        report, curves = evaluate_predictions(truth, predictions, np.ones((6, 3)))
        self.assertEqual(report["confusion_matrix"], [[2, 0, 0], [2, 0, 0], [2, 0, 0]])
        self.assertAlmostEqual(report["per_class"]["Outer Race"]["f1"], 0.5)
        self.assertAlmostEqual(report["macro_f1"], 1 / 6)
        self.assertEqual(report["macro_auc_ovr"], 0.5)
        np.testing.assert_array_equal(curves["Normal"]["fpr"], [0, 1])


class NormalRuleTests(unittest.TestCase):
    def test_threshold_and_fault_selection(self):
        self.assertEqual(predict_fault_or_normal([3, 4], 10), "Normal")
        self.assertEqual(predict_fault_or_normal([10, 4], 10), "Outer Race")
        self.assertEqual(predict_fault_or_normal([4, 11], 10), "Inner Race")
        with self.assertRaises(ValueError):
            predict_fault_or_normal([3, 4], float("nan"))

    def test_ball_record_is_excluded_and_matching_normal_is_included(self):
        keys = _dataset_keys()
        self.assertIn("Normal_0", keys)
        self.assertIn("IR007_0", keys)
        self.assertIn("OR007@6_0", keys)
        self.assertNotIn("B007_0", keys)
        self.assertNotIn("Normal_1", keys)

    def test_downsampling_retains_in_band_signal_and_blocks_alias(self):
        # 3 kHz를 유지하고, 8 kHz가 12 kHz 변환 후 4 kHz로 접히지 않게 한다.
        t = np.arange(48000) / 48000
        signal = np.cos(2 * np.pi * 3000 * t) + np.cos(2 * np.pi * 8000 * t)
        downsampled = resample_to_12k(signal, 48000)
        self.assertEqual(len(downsampled), 12000)
        frequencies = np.fft.rfftfreq(len(downsampled), 1 / 12000)
        amplitude = 2 * np.abs(np.fft.rfft(downsampled)) / len(downsampled)
        self.assertGreater(amplitude[frequencies == 3000][0], 0.95)
        self.assertLess(amplitude[frequencies == 4000][0], 0.01)


if __name__ == "__main__":
    unittest.main()
