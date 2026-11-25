import numpy as np


def test_process_arrays_calls_assemble_in_frozen_order(monkeypatch):
    """Regression test for the former five-argument/reversed assemble call."""
    from app.a5_operator import A5Operator

    op = object.__new__(A5Operator)
    op.schema = {"channels": ["obs", "norm", "mask", "depth", "horizontal", "vertical", "compact", "preserve", "remove", "ambiguous", "edge"], "patch": 33}
    op.policy = {"annotation_threshold": .995, "print_threshold": .98, "print_margin": .96, "minimum_print_run": 2}
    op.device = "cpu"

    class RT:
        @staticmethod
        def make_padded(arrays, schema): return arrays
        @staticmethod
        def score_coordinates(model, calibrator, padded, schema, y, x, device, batch=512): return np.ones(len(y))
        @staticmethod
        def decisions(prob, policy, shape, y, x):
            final = np.array(["PRINT"] * len(y), dtype=object)
            return final.copy(), final
        @staticmethod
        def assemble(observed, a1, authoritative_mask, y, x, final):
            out = a1.copy(); out[y, x] = observed[y, x]
            pm = np.zeros(a1.shape, bool); pm[y, x] = True
            return out, pm

    op.rt = RT(); op.model = object(); op.calibrator = object()
    shape = (4, 4)
    arrays = {name: np.zeros(shape, np.uint8) for name in op.schema["channels"]}
    arrays["obs"][:] = 10
    arrays["mask"][:] = 1
    a1 = np.full(shape, 255, np.uint8)
    result = op.process_arrays(arrays, a1, authoritative_mask=arrays["mask"])
    assert np.array_equal(result["corrected"], arrays["obs"])
    assert result["print_mask"].all()
