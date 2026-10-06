"""Compatibility imports for baseline scripts; implementation lives in frontend/."""
from .frontend.legacy import (
    Match, MotionModelDecision, registration_view, sift_inliers, register,
    register_translation_fallback, register_enhanced_adjacent,
    register_weak_signal_adjacent, select_supported_motion,
    _matrix_from_params, _params_from_matrix, _transform, _huber_residual,
    _solve, _overlap, _warped_correlation,
)
