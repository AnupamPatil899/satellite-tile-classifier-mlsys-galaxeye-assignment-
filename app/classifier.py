import os
import json
import time
import math
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from typing import Dict, Any, List, Tuple
from app.config import settings
from train import SatelliteCNN

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

class OfflineClassifier:
    """
    Production-grade offline CPU inference engine for satellite tiles.
    - Dual runtime: ONNX Runtime CPU (default high-perf) with PyTorch CPU fallback.
    - SIMD vectorized micro-batching for burst satellite downlinks.
    - Multi-dimensional uncertainty quantification: Softmax Confidence, Prediction Margin, and Shannon Entropy.
    """
    def __init__(self):
        self.classes = settings.CLASSES
        self.model_version = settings.VERSION
        self.device = torch.device("cpu")
        self.pytorch_model = None
        self.onnx_session = None
        self.runtime_engine = "PyTorch-CPU"
        
        self.transform = transforms.Compose([
            transforms.Resize(settings.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self._load_metadata()
        self._init_runtime()

    def _load_metadata(self):
        if os.path.exists(settings.METADATA_PATH):
            try:
                with open(settings.METADATA_PATH, "r") as f:
                    meta = json.load(f)
                    self.classes = meta.get("classes", self.classes)
                    self.model_version = meta.get("model_version", self.model_version)
            except Exception as e:
                print(f"Warning: Failed to load metadata: {e}")

    def _init_runtime(self):
        # 1. Attempt to load ONNX Runtime session
        if HAS_ONNX and os.path.exists(settings.ONNX_MODEL_PATH):
            try:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.onnx_session = ort.InferenceSession(
                    settings.ONNX_MODEL_PATH,
                    sess_options=opts,
                    providers=["CPUExecutionProvider"]
                )
                self.runtime_engine = "ONNXRuntime-CPU"
                print(f"Initialized inference engine with {self.runtime_engine}")
                return
            except Exception as e:
                print(f"Notice: Failed to initialize ONNX Runtime ({e}), falling back to PyTorch.")

        # 2. PyTorch Fallback
        self.pytorch_model = SatelliteCNN(num_classes=len(self.classes))
        if os.path.exists(settings.MODEL_PATH):
            state = torch.load(settings.MODEL_PATH, map_location=self.device)
            self.pytorch_model.load_state_dict(state)
            print(f"Initialized inference engine with PyTorch-CPU ({settings.MODEL_PATH})")
        else:
            print("Warning: Model weights not found. Running with initialized weights.")
            
        self.pytorch_model.to(self.device)
        self.pytorch_model.eval()

    def is_ready(self) -> bool:
        return (self.onnx_session is not None) or (self.pytorch_model is not None)

    def _compute_uncertainty_metrics(self, probs: np.ndarray) -> Tuple[str, float, float, float, str]:
        """
        Computes multi-dimensional uncertainty:
        - top-1 confidence
        - prediction margin (p1 - p2)
        - normalized Shannon entropy H in [0, 1]
        - status: 'CONFIDENT' or 'UNCERTAIN'
        """
        sorted_indices = np.argsort(probs)[::-1]
        top1_idx = sorted_indices[0]
        top2_idx = sorted_indices[1] if len(sorted_indices) > 1 else top1_idx
        
        confidence = float(probs[top1_idx])
        margin = float(probs[top1_idx] - probs[top2_idx])
        
        # Shannon Entropy: - sum(p * ln(p)) / ln(K)
        k = len(probs)
        log_k = math.log(k) if k > 1 else 1.0
        entropy_raw = -sum(p * math.log(p + 1e-12) for p in probs if p > 0)
        normalized_entropy = float(entropy_raw / log_k)
        
        predicted_class = self.classes[top1_idx]
        
        # Multi-factor decision boundary:
        # A prediction is only CONFIDENT if confidence >= threshold AND margin >= threshold AND entropy <= threshold
        is_confident = (
            confidence >= settings.CONFIDENCE_THRESHOLD and
            margin >= settings.MARGIN_THRESHOLD and
            normalized_entropy <= settings.MAX_ENTROPY_THRESHOLD
        )
        status = "CONFIDENT" if is_confident else "UNCERTAIN"
        
        return predicted_class, round(confidence, 4), round(margin, 4), round(normalized_entropy, 4), status

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Predict for a single PIL image.
        """
        batch_results = self.predict_batch([image])
        return batch_results[0]

    def predict_batch(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """
        Vectorized batch inference for satellite bursts.
        """
        if not images:
            return []
            
        start_time = time.perf_counter()
        
        tensors = []
        for img in images:
            if img.mode != "RGB":
                img = img.convert("RGB")
            tensors.append(self.transform(img))
            
        stacked = torch.stack(tensors)  # (N, 3, 64, 64)
        
        if self.onnx_session is not None:
            # ONNX Runtime inference
            ort_inputs = {self.onnx_session.get_inputs()[0].name: stacked.numpy()}
            ort_outs = self.onnx_session.run(None, ort_inputs)
            logits = ort_outs[0]
            # Softmax
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs_matrix = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        else:
            # PyTorch inference
            with torch.no_grad():
                logits = self.pytorch_model(stacked)
                probs = torch.softmax(logits, dim=1)
                probs_matrix = probs.numpy()

        total_time_ms = (time.perf_counter() - start_time) * 1000.0
        per_item_latency = round(total_time_ms / len(images), 2)
        
        results = []
        for i in range(len(images)):
            probs = probs_matrix[i]
            pred_class, conf, margin, entropy, status = self._compute_uncertainty_metrics(probs)
            
            prob_dict = {
                self.classes[c]: round(float(probs[c]), 4)
                for c in range(len(self.classes))
            }
            
            results.append({
                "predicted_class": pred_class,
                "confidence": conf,
                "margin": margin,
                "entropy": entropy,
                "class_probabilities": prob_dict,
                "status": status,
                "model_version": self.model_version,
                "inference_latency_ms": per_item_latency,
                "runtime_engine": self.runtime_engine
            })
            
        return results

# Global singleton
classifier = OfflineClassifier()
