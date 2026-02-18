"""Explainability modules: MC Dropout, Grad-CAM, LIME/SHAP."""

from explainability.mc_dropout import mc_dropout_predict, uncertainty_analysis, compute_calibration
from explainability.gradcam import grad_cam_1d, gradient_saliency, explain_model
from explainability.perturbation import lime_explain, shap_explain, batch_lime_explain, batch_shap_explain

__all__ = [
    'mc_dropout_predict', 'uncertainty_analysis', 'compute_calibration',
    'grad_cam_1d', 'gradient_saliency', 'explain_model',
    'lime_explain', 'shap_explain', 'batch_lime_explain', 'batch_shap_explain',
]
