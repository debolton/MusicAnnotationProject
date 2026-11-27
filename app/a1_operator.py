from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any
import hashlib
import sys
import importlib.util

import numpy as np


# ---------------------------------------------------------------------------
# Frozen paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

V89_DIR = ROOT / "frozen" / "v8_9"
V89_OPERATOR_PATH = V89_DIR / "v8_9_operator.py"

P_PRINT_BUNDLE = (
    ROOT
    / "frozen"
    / "p_print_provider_v8_2f"
)

REMOVE_BUNDLE = (
    ROOT
    / "frozen"
    / "notebook52f_remove_bundle"
)


# ---------------------------------------------------------------------------
# Public result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class A1PageResult:
    """
    Exact Notebook 87 A1 result and supporting audit arrays.

    A1 construction:

        1. Run the frozen V8.9 operator.
        2. Start from V89PageResult.corrected.
        3. Restore every frozen V7.8 AMBIGUOUS coordinate
           to the exact observed byte.
    """

    page: str | None

    observed: np.ndarray
    reviewed_seed: np.ndarray
    authoritative_mask: np.ndarray

    telea: np.ndarray
    paper: np.ndarray
    normalised: np.ndarray

    horizontal: np.ndarray
    vertical: np.ndarray
    compact: np.ndarray

    preserve: np.ndarray
    remove: np.ndarray
    ambiguous: np.ndarray

    baseline_v78: np.ndarray
    v8_9_corrected: np.ndarray
    a1: np.ndarray

    ring2: np.ndarray | None

    v89_operator_sha256: str
    p_print_bundle_root: str
    remove_bundle_root: str

    @property
    def corrected(self) -> np.ndarray:
        """
        Compatibility alias for callers expecting a corrected page.
        """

        return self.a1

    @property
    def mask(self) -> np.ndarray:
        """
        Compatibility alias for the prepared authoritative mask.
        """

        return self.authoritative_mask

    @property
    def geometry_t80(self) -> np.ndarray:
        """
        Compatibility alias for the frozen V7.8 baseline.
        """

        return self.baseline_v78

    @property
    def routes(self) -> dict[str, np.ndarray]:
        return {
            "preserve": self.preserve,
            "remove": self.remove,
            "ambiguous": self.ambiguous,
            "horizontal": self.horizontal,
            "vertical": self.vertical,
            "compact": self.compact,
            "paper": self.paper,
            "normalised": self.normalised,
        }

    @property
    def a1_delta_from_v89(self) -> np.ndarray:
        return (
            self.a1
            != self.v8_9_corrected
        )

    @property
    def a1_delta_from_v78(self) -> np.ndarray:
        return (
            self.a1
            != self.baseline_v78
        )

    def as_dict(self) -> dict[str, Any]:
        """
        Return a dictionary compatible with existing HF processing.
        """

        return {
            "page": self.page,

            "observed": self.observed,
            "obs": self.observed,

            "reviewed_seed": self.reviewed_seed,
            "seed": self.reviewed_seed,

            "authoritative_mask": self.authoritative_mask,
            "mask": self.authoritative_mask,

            "telea": self.telea,
            "paper": self.paper,
            "normalised": self.normalised,
            "norm": self.normalised,

            "horizontal": self.horizontal,
            "vertical": self.vertical,
            "compact": self.compact,

            "preserve": self.preserve,
            "remove": self.remove,
            "ambiguous": self.ambiguous,

            "routes": self.routes,

            "baseline_v78": self.baseline_v78,
            "geometry_t80": self.baseline_v78,
            "v7_8": self.baseline_v78,

            "v8_9": self.v8_9_corrected,
            "v8_9_corrected": self.v8_9_corrected,

            "a1": self.a1,
            "a1_notebook87": self.a1,
            "corrected": self.a1,

            "ring2": self.ring2,

            "a1_delta_from_v89": self.a1_delta_from_v89,
            "a1_delta_from_v7_8": self.a1_delta_from_v78,

            "a1_policy_id": (
                "NOTEBOOK87_A1_FROM_FROZEN_V8_9"
            ),

            "v89_operator_sha256": self.v89_operator_sha256,
            "p_print_bundle_root": self.p_print_bundle_root,
            "remove_bundle_root": self.remove_bundle_root,

            "a1_assembly": (
                "FROZEN_V8_9_CORRECTED_THEN_ALL_FROZEN_"
                "V7_8_AMBIGUOUS_COORDINATES_RESTORED_"
                "TO_EXACT_OBSERVED"
            ),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _load_module(
    path: Path,
) -> ModuleType:
    if not path.is_file():
        raise RuntimeError(
            "Missing frozen V8.9 operator source: "
            f"{path}"
        )

    module_name = (
        "hf_frozen_v8_9_operator"
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            "Unable to create a module specification for "
            f"{path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    # Register the dynamically loaded module before executing it.
    #
    # This is required by Python features such as @dataclass that
    # inspect sys.modules[cls.__module__] while the module body is
    # executing.
    previous_module = sys.modules.get(
        module_name
    )

    sys.modules[
        module_name
    ] = module

    try:
        spec.loader.exec_module(
            module
        )

    except Exception:
        # Do not leave a partially initialised module registered.
        if previous_module is None:
            sys.modules.pop(
                module_name,
                None,
            )

        else:
            sys.modules[
                module_name
            ] = previous_module

        raise

    return module


def _require_callable(
    module: ModuleType,
    name: str,
):
    value = getattr(
        module,
        name,
        None,
    )

    if not callable(value):
        raise RuntimeError(
            "Frozen V8.9 operator does not expose the "
            f"required callable {name!r}"
        )

    return value


def _result_value(
    result: Any,
    *names: str,
    required: bool = True,
):
    """
    Read a value from either a dataclass/object result or a dictionary.
    """

    for name in names:
        if isinstance(result, dict):
            if name in result:
                return result[name]
        elif hasattr(result, name):
            return getattr(
                result,
                name,
            )

    if required:
        raise RuntimeError(
            "Frozen V8.9 result omitted required field; "
            f"accepted names={names}"
        )

    return None


def _require_uint8_image(
    value,
    name: str,
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    array = np.asarray(value)

    if array.ndim != 2:
        raise RuntimeError(
            f"{name} must be two-dimensional; "
            f"found shape {array.shape}"
        )

    if (
        expected_shape is not None
        and array.shape != expected_shape
    ):
        raise RuntimeError(
            f"{name} shape mismatch: expected "
            f"{expected_shape}, found {array.shape}"
        )

    if not np.issubdtype(
        array.dtype,
        np.number,
    ):
        raise RuntimeError(
            f"{name} must contain numeric values"
        )

    if not np.isfinite(array).all():
        raise RuntimeError(
            f"{name} contains non-finite values"
        )

    if array.size and (
        array.min() < 0
        or array.max() > 255
    ):
        raise RuntimeError(
            f"{name} contains values outside 0..255"
        )

    return np.ascontiguousarray(
        array.astype(np.uint8)
    )


def _require_bool_mask(
    value,
    name: str,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    array = np.asarray(value)

    if array.ndim != 2:
        raise RuntimeError(
            f"{name} must be two-dimensional; "
            f"found shape {array.shape}"
        )

    if array.shape != expected_shape:
        raise RuntimeError(
            f"{name} shape mismatch: expected "
            f"{expected_shape}, found {array.shape}"
        )

    if (
        np.issubdtype(array.dtype, np.number)
        and not np.isfinite(array).all()
    ):
        raise RuntimeError(
            f"{name} contains non-finite values"
        )

    return np.ascontiguousarray(
        array.astype(bool)
    )


def _normalise_binary_u8(
    value,
    name: str,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    array = _require_uint8_image(
        value,
        name,
        expected_shape,
    )

    return np.ascontiguousarray(
        np.where(
            array > 0,
            255,
            0,
        ).astype(np.uint8)
    )


def _normalise_geometry_result(
    value,
) -> tuple[np.ndarray, dict[str, Any]]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
    ):
        raise RuntimeError(
            "Frozen V8.9 geometry_t80() must return "
            "(baseline, routes)"
        )

    baseline, routes = value

    if not isinstance(routes, dict):
        raise RuntimeError(
            "Frozen V8.9 geometry_t80() routes must be "
            "a dictionary"
        )

    return baseline, routes


# ---------------------------------------------------------------------------
# A1 operator
# ---------------------------------------------------------------------------

class A1Operator:
    """
    Exact Notebook 87 A1 generator.

    Authoritative lineage:

        observed + reviewed seed
            -> frozen V8.9 operator
            -> frozen V8.9 corrected page
            -> restore every frozen V7.8 AMBIGUOUS coordinate
               to the exact observed byte
            -> Notebook 87 A1

    The frozen V8.9 source and both frozen bundles are loaded
    without modifying their contents.
    """

    a1_policy_id = (
        "NOTEBOOK87_A1_FROM_FROZEN_V8_9"
    )

    def __init__(
        self,
        device: str | None = None,
        v89_operator_path: Path | str = V89_OPERATOR_PATH,
        p_print_bundle_root: Path | str = P_PRINT_BUNDLE,
        remove_bundle_root: Path | str = REMOVE_BUNDLE,
    ):
        self.v89_operator_path = Path(
            v89_operator_path
        )

        self.p_print_bundle_root = Path(
            p_print_bundle_root
        )

        self.remove_bundle_root = Path(
            remove_bundle_root
        )

        if not self.p_print_bundle_root.is_dir():
            raise RuntimeError(
                "Missing frozen V8.9 p_print bundle: "
                f"{self.p_print_bundle_root}"
            )

        if not self.remove_bundle_root.is_dir():
            raise RuntimeError(
                "Missing frozen V8.9 REMOVE bundle: "
                f"{self.remove_bundle_root}"
            )

        self.v89_operator_sha256 = _sha256_file(
            self.v89_operator_path
        )

        self.module = _load_module(
            self.v89_operator_path
        )

        self._prepare_mask = _require_callable(
               self.module,
            "prepare_mask",
        )

        self._telea_reconstruct = _require_callable(
            self.module,
            "telea_reconstruct",
        )

        self._geometry_t80 = _require_callable(
            self.module,
            "geometry_t80",
        )

        load_v89_operator = _require_callable(
            self.module,
            "load_operator",
        )

        try:
            self.v89 = load_v89_operator(
                p_print_bundle_root=(
                    self.p_print_bundle_root
                ),
                remove_bundle_root=(
                    self.remove_bundle_root
                ),
                device=device,
            )
        except TypeError:
            # Compatibility with a positional loader while retaining
            # the same frozen bundle order used by Notebook 87.
            self.v89 = load_v89_operator(
                self.p_print_bundle_root,
                self.remove_bundle_root,
                device,
            )

        if not callable(
            getattr(
                self.v89,
                "run_page",
                None,
            )
        ):
            raise RuntimeError(
                "Frozen V8.9 operator does not provide run_page()"
            )

    def run(
        self,
        observed: np.ndarray,
        reviewed_seed: np.ndarray,
        page: str | None = None,
    ) -> dict[str, Any]:
        """
        Run the exact Notebook 87 A1 construction and return a
        dictionary suitable for the HF application.
        """

        return self.run_page(
            observed=observed,
            reviewed_seed=reviewed_seed,
            page=page,
        ).as_dict()

    def run_page(
        self,
        observed: np.ndarray,
        reviewed_seed: np.ndarray,
        page: str | None = None,
    ) -> A1PageResult:
        observed_u8 = _require_uint8_image(
            observed,
            "observed page",
        )

        shape = observed_u8.shape

        seed_u8 = _normalise_binary_u8(
            reviewed_seed,
            "reviewed authoritative seed",
            shape,
        )

        page_name = (
            str(page)
            if page is not None
            else "page"
        )

        # -------------------------------------------------------
        # Run the exact hash-locked V8.9 implementation used by
        # Notebook 87.
        # -------------------------------------------------------

        try:
            v89_result = self.v89.run_page(
                observed_u8,
                seed_u8,
                page=page_name,
            )
        except TypeError:
            v89_result = self.v89.run_page(
                observed_u8,
                seed_u8,
            )

        v89_corrected = _require_uint8_image(
            _result_value(
                v89_result,
                "corrected",
            ),
            "frozen V8.9 corrected page",
            shape,
        )

        v89_baseline = _require_uint8_image(
            _result_value(
                v89_result,
                "baseline_v78",
                "geometry_t80",
                "baseline",
            ),
            "frozen V8.9 V7.8 baseline",
            shape,
        )

        v89_mask = _require_bool_mask(
            _result_value(
                v89_result,
                "authoritative_mask",
                "mask",
            ),
            "frozen V8.9 authoritative mask",
            shape,
        )

        ring2_value = _result_value(
            v89_result,
            "ring2",
            "ring_2",
            "ring2_mask",
            required=False,
        )

        ring2 = None

        if ring2_value is not None:
            ring2 = _require_bool_mask(
                ring2_value,
                "frozen V8.9 Ring-2 mask",
                shape,
            )

        # -------------------------------------------------------
        # Rebuild the exact V7.8 arrays through the functions
        # embedded in the same frozen V8.9 source.
        #
        # This avoids mixing the historical Notebook 87 lineage
        # with a later independent canonical_v78 implementation.
        # -------------------------------------------------------

        prepared_u8 = _normalise_binary_u8(
            self._prepare_mask(
                seed_u8
            ),
            "frozen V8.9 prepared mask",
            shape,
        )

        prepared = (
            prepared_u8
            > 0
        )

        if not np.array_equal(
            prepared,
            v89_mask,
        ):
            missing = int(
                np.sum(
                    v89_mask
                    & ~prepared
                )
            )

            unexpected = int(
                np.sum(
                    prepared
                    & ~v89_mask
                )
            )

            raise RuntimeError(
                "A1 wrapper prepared mask differs from the "
                "mask used by frozen V8.9: "
                f"missing={missing}, unexpected={unexpected}"
            )

        telea = _require_uint8_image(
            self._telea_reconstruct(
                observed_u8,
                prepared_u8,
            ),
            "frozen V8.9 Telea page",
            shape,
        )

        rebuilt_baseline_value, routes = (
            _normalise_geometry_result(
                self._geometry_t80(
                    observed_u8,
                    prepared_u8,
                    telea,
                )
            )
        )

        rebuilt_baseline = _require_uint8_image(
            rebuilt_baseline_value,
            "rebuilt frozen V8.9 V7.8 baseline",
            shape,
        )

        required_routes = {
            "preserve",
            "remove",
            "ambiguous",
            "horizontal",
            "vertical",
            "compact",
            "paper",
            "normalised",
        }

        missing_routes = sorted(
            required_routes - set(routes)
        )

        if missing_routes:
            raise RuntimeError(
                "Frozen V8.9 geometry_t80() omitted routes: "
                f"{missing_routes}"
            )

        preserve = _require_bool_mask(
            routes["preserve"],
            "V7.8 PRESERVE route",
            shape,
        )

        remove = _require_bool_mask(
            routes["remove"],
            "V7.8 REMOVE route",
            shape,
        )

        ambiguous = _require_bool_mask(
            routes["ambiguous"],
            "V7.8 AMBIGUOUS route",
            shape,
        )

        horizontal = _require_bool_mask(
            routes["horizontal"],
            "V7.8 horizontal support",
            shape,
        )

        vertical = _require_bool_mask(
            routes["vertical"],
            "V7.8 vertical support",
            shape,
        )

        compact = _require_bool_mask(
            routes["compact"],
            "V7.8 compact support",
            shape,
        )

        paper = _require_uint8_image(
            routes["paper"],
            "V7.8 paper estimate",
            shape,
        )

        normalised = _require_uint8_image(
            routes["normalised"],
            "V7.8 normalised page",
            shape,
        )

        # -------------------------------------------------------
        # V7.8 identity checks
        # -------------------------------------------------------

        if not np.array_equal(
            rebuilt_baseline,
            v89_baseline,
        ):
            different = int(
                np.sum(
                    rebuilt_baseline
                    != v89_baseline
                )
            )

            maximum_difference = int(
                np.max(
                    np.abs(
                        rebuilt_baseline.astype(
                            np.int16
                        )
                        - v89_baseline.astype(
                            np.int16
                        )
                    )
                )
            )

            raise RuntimeError(
                "A1 wrapper did not reproduce the exact V7.8 "
                "baseline used by frozen V8.9: "
                f"different_pixels={different}, "
                f"max_abs_difference={maximum_difference}"
            )

        if np.any(
            preserve & remove
        ):
            raise RuntimeError(
                "V7.8 PRESERVE and REMOVE routes overlap"
            )

        if np.any(
            preserve & ambiguous
        ):
            raise RuntimeError(
                "V7.8 PRESERVE and AMBIGUOUS routes overlap"
            )

        if np.any(
            remove & ambiguous
        ):
            raise RuntimeError(
                "V7.8 REMOVE and AMBIGUOUS routes overlap"
            )

        routed = (
            preserve
            | remove
            | ambiguous
        )

        if not np.array_equal(
            routed,
            prepared,
        ):
            missing = int(
                np.sum(
                    prepared
                    & ~routed
                )
            )

            escaped = int(
                np.sum(
                    routed
                    & ~prepared
                )
            )

            raise RuntimeError(
                "Frozen V7.8 routes do not exactly partition "
                "the authoritative mask: "
                f"missing={missing}, outside={escaped}"
            )

        if np.any(
            rebuilt_baseline[~prepared]
            != observed_u8[~prepared]
        ):
            raise RuntimeError(
                "Frozen V7.8 baseline has outside-mask drift"
            )

        if np.any(
            telea[~prepared]
            != observed_u8[~prepared]
        ):
            raise RuntimeError(
                "Frozen V8.9 Telea page has outside-mask drift"
            )

        # -------------------------------------------------------
        # Exact Notebook 87 A1 assembly
        #
        # Notebook 87:
        #
        #     a1 = np.asarray(
        #         result.corrected,
        #         np.uint8,
        #     ).copy()
        #
        #     a1[ambiguous] = observed[ambiguous]
        # -------------------------------------------------------

        a1 = v89_corrected.copy()

        a1[ambiguous] = (
            observed_u8[ambiguous]
        )

        # -------------------------------------------------------
        # Exact Notebook 87 A1 integrity checks
        # -------------------------------------------------------

        if np.any(
            a1[ambiguous]
            != observed_u8[ambiguous]
        ):
            raise RuntimeError(
                "Notebook 87 A1 AMBIGUOUS bytes do not equal "
                "the exact observed bytes"
            )

        if np.any(
            a1[~ambiguous]
            != v89_corrected[~ambiguous]
        ):
            raise RuntimeError(
                "Notebook 87 A1 differs from frozen V8.9 "
                "outside the V7.8 AMBIGUOUS route"
            )

        if np.any(
            a1[~prepared]
            != observed_u8[~prepared]
        ):
            raise RuntimeError(
                "Notebook 87 A1 changed pixels outside the "
                "authoritative mask"
            )

        actual_delta = (
            a1
            != v89_corrected
        )

        expected_delta = (
            ambiguous
            & (
                observed_u8
                != v89_corrected
            )
        )

        if not np.array_equal(
            actual_delta,
            expected_delta,
        ):
            unexpected = int(
                np.sum(
                    actual_delta
                    & ~expected_delta
                )
            )

            missing = int(
                np.sum(
                    expected_delta
                    & ~actual_delta
                )
            )

            raise RuntimeError(
                "Notebook 87 A1 delta does not equal the "
                "effective AMBIGUOUS-to-observed restoration: "
                f"unexpected={unexpected}, missing={missing}"
            )

        # Every byte that A1 changes relative to V8.9 must be an
        # exact observed byte.
        if np.any(
            actual_delta
            & (
                a1
                != observed_u8
            )
        ):
            raise RuntimeError(
                "Notebook 87 A1 changed a byte to a value "
                "other than the exact observed byte"
            )

        # The A1 wrapper must not manufacture any change outside
        # the union of:
        #
        #   - changes already made by frozen V8.9; and
        #   - AMBIGUOUS-to-observed restoration.
        v89_delta_from_v78 = (
            v89_corrected
            != rebuilt_baseline
        )

        a1_delta_from_v78 = (
            a1
            != rebuilt_baseline
        )

        permitted_a1_delta = (
            v89_delta_from_v78
            | expected_delta
        )

        if np.any(
            a1_delta_from_v78
            & ~permitted_a1_delta
        ):
            unexpected = int(
                np.sum(
                    a1_delta_from_v78
                    & ~permitted_a1_delta
                )
            )

            raise RuntimeError(
                "Notebook 87 A1 contains unexplained changes "
                "relative to frozen V7.8: "
                f"unexpected={unexpected}"
            )

        return A1PageResult(
            page=page,

            observed=observed_u8,
            reviewed_seed=seed_u8,
            authoritative_mask=prepared,

            telea=telea,
            paper=paper,
            normalised=normalised,

            horizontal=horizontal,
            vertical=vertical,
            compact=compact,

            preserve=preserve,
            remove=remove,
            ambiguous=ambiguous,

            baseline_v78=rebuilt_baseline,
            v8_9_corrected=v89_corrected,
            a1=a1,

            ring2=ring2,

            v89_operator_sha256=(
                self.v89_operator_sha256
            ),
            p_print_bundle_root=str(
                self.p_print_bundle_root
            ),
            remove_bundle_root=str(
                self.remove_bundle_root
            ),
        )