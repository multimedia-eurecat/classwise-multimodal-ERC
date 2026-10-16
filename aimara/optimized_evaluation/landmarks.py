"""Sparse 68-point 3DDFA inference from explicit ONNX/BFM assets.

Uses the MIT-licensed geometry helpers in third_party/tddfa. No external source
checkout, dense BFM decoder, topology, native extension or working-directory
change is required for the sparse landmarks used by the ERC checkpoint.
"""
import pickle
import cv2
import numpy as np

from .third_party.tddfa.geometry import (
    crop_img, parse_roi_box_from_bbox, _parse_param, similar_transform,
)


class SparseLandmarkModel:
    def __init__(self, onnx_path, bfm_path, normalization_path):
        import onnxruntime as ort
        with open(bfm_path, "rb") as handle:
            bfm = pickle.load(handle)
        indices = bfm["keypoints"].astype(np.int64)
        self.u = bfm["u"].astype(np.float32)[indices].reshape(-1, 1)
        self.shape = bfm["w_shp"].astype(np.float32)[indices, :40]
        self.expression = bfm["w_exp"].astype(np.float32)[indices, :10]
        if self.u.shape != (204, 1) or self.shape.shape != (204, 40) or self.expression.shape != (204, 10):
            raise ValueError("Expected a 68-point BFM with 40 shape and 10 expression coefficients")
        with open(normalization_path, "rb") as handle:
            normalization = pickle.load(handle)
        self.mean = np.asarray(normalization["mean"], dtype=np.float32).reshape(-1)
        self.std = np.asarray(normalization["std"], dtype=np.float32).reshape(-1)
        if self.mean.shape != (62,) or self.std.shape != (62,):
            raise ValueError("Expected 62-parameter landmark normalization")
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def extract(self, crop_rgb):
        height, width = crop_rgb.shape[:2]
        if min(height, width) < 16:
            return None
        roi = parse_roi_box_from_bbox([0., 0., float(width - 1), float(height - 1), 1.])
        image = cv2.resize(crop_img(crop_rgb, roi), (120, 120), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32).transpose(2, 0, 1)[None, ...]
        image = (image - 127.5) / 128.
        parameters = self.session.run(None, {self.input_name:image})[0].flatten().astype(np.float32)
        if parameters.shape != (62,):
            raise ValueError("Landmark network must output 62 parameters")
        parameters = parameters * self.std + self.mean
        rotation, offset, shape, expression = _parse_param(parameters)
        points = rotation @ (self.u + self.shape @ shape + self.expression @ expression).reshape(3, -1, order="F") + offset
        points = similar_transform(points, roi, 120)
        landmarks = points.T
        # Same normalization as unimodal.visual.models, without importing the
        # face-detector/Transformer runtime for this sparse NumPy operation.
        scale = float(np.linalg.norm(landmarks[36] - landmarks[45])) + 1e-6
        return ((landmarks - landmarks[30]) / scale).astype(np.float32)
