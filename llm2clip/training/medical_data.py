"""
Medical dataset classes for chest X-ray datasets (MIMIC-CXR, IU X-Ray).
Supports both image loading and pre-extracted feature loading with ANAO metadata.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)


class MedicalJsonDataset(Dataset):
    """
    Dataset for medical image-text pairs with ANAO metadata support.

    Expects a JSON file with entries like:
    {
        "image": "path/to/chest_xray.jpg",
        "report": "Full radiology report text...",
        "findings": "Findings section...",
        "impression": "Impression section..."
    }

    Or can load pre-extracted features and metadata:
    - img_feature_path: path to .pt file with pre-extracted image features
    - text_feature_path: path to .pt file with pre-extracted text features
    - metadata_path: path to ANAO metadata JSON (from extract_medical_metadata.py)
    """

    def __init__(
        self,
        input_filename: str,
        transforms=None,
        img_root: Optional[str] = None,
        img_feature_path: Optional[str] = None,
        text_feature_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        tokenizer=None,
        report_key: str = "report",
    ):
        logger.debug(f"Loading medical data from {input_filename}")

        self.meta = json.load(open(input_filename, "r"))
        self.img_features = None
        self.text_features = None
        self.anao_metadata = None
        self.report_key = report_key

        # Load pre-extracted image features
        if img_feature_path:
            if Path(img_feature_path).suffix == ".npy":
                self.img_features = np.memmap(
                    img_feature_path, dtype="float32", mode="r",
                    shape=(len(self.meta), 1024)
                )
            elif Path(img_feature_path).suffix == ".pt":
                self.img_features = torch.load(img_feature_path, mmap=True)
            else:
                self.img_features = torch.load(img_feature_path, mmap=True)

        # Load pre-extracted text features
        if text_feature_path:
            text_features_list = []
            if isinstance(text_feature_path, list):
                self.random_text = True
                self.text_features = [
                    torch.load(path, mmap=True) for path in text_feature_path
                ]
            else:
                self.random_text = False
                self.text_features = torch.load(text_feature_path, mmap=True)

        # Load ANAO metadata
        if metadata_path:
            self._load_anao_metadata(metadata_path)

        self.img_root = img_root
        self.transforms = transforms
        self.tokenize = tokenizer

        logger.debug(f"Done loading medical data: {len(self.meta)} samples")

    def _load_anao_metadata(self, metadata_path: str):
        """Load pre-extracted anatomy/negation labels."""
        with open(metadata_path, "r") as f:
            data = json.load(f)
        # Build lookup by report_id
        self.anao_metadata = {}
        for item in data.get("metadata", data.get("data", [])):
            rid = item.get("report_id", item.get("id", ""))
            if rid:
                self.anao_metadata[rid] = item

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        images, texts = None, None
        anat_label = 5  # default: unspecified
        neg_label = [0] * 15  # default: not mentioned (treated as 0)

        # Load image
        if self.img_features is not None:
            images = self.img_features[str(idx)]
        else:
            image_path = os.path.join(self.img_root, self.meta[idx]["image"])
            images = self.transforms(Image.open(image_path))

        # Load text
        if self.text_features is not None:
            if self.random_text:
                texts = random.choice([fs[str(idx)] for fs in self.text_features])
            else:
                texts = self.text_features[str(idx)]
        else:
            # Find report text
            report_text = None
            for key in [self.report_key, "report", "findings", "caption", "text"]:
                if key in self.meta[idx]:
                    report_text = self.meta[idx][key]
                    break
            if report_text is None:
                report_text = ""
            if self.tokenize:
                texts = self.tokenize([report_text])[0]

        # Load ANAO metadata
        if self.anao_metadata is not None:
            report_id = self.meta[idx].get("id", str(idx))
            if report_id in self.anao_metadata:
                item = self.anao_metadata[report_id]
                anat_label = item.get("anat_label", 5)
                neg_label = item.get("neg_label", [0] * 15)

        return images, texts, anat_label, neg_label


class MIMICCXRDataConfig:
    """Standard MIMIC-CXR data configuration."""
    DEFAULT_IMG_ROOT = "data/mimic_cxr/images"
    DEFAULT_REPORT_FILE = "data/mimic_cxr/reports.json"
    DEFAULT_REPORT_KEY = "findings"


class IUXrayDataConfig:
    """Standard IU X-Ray data configuration."""
    DEFAULT_IMG_ROOT = "data/iu_xray/images"
    DEFAULT_REPORT_FILE = "data/iu_xray/reports.json"
    DEFAULT_REPORT_KEY = "findings"


def create_medical_dataset(
    dataset_name: str,
    input_filename: str,
    transforms=None,
    img_root: Optional[str] = None,
    img_feature_path: Optional[str] = None,
    text_feature_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
    tokenizer=None,
) -> MedicalJsonDataset:
    """
    Factory function for creating medical datasets.

    Args:
        dataset_name: "mimic_cxr" or "iu_xray"
        input_filename: path to JSON annotation file
        img_root: root directory for images
        img_feature_path: path to pre-extracted image features
        text_feature_path: path to pre-extracted text features
        metadata_path: path to ANAO metadata JSON
        tokenizer: text tokenizer function
    """
    if dataset_name == "mimic_cxr":
        report_key = MIMICCXRDataConfig.DEFAULT_REPORT_KEY
    elif dataset_name == "iu_xray":
        report_key = IUXrayDataConfig.DEFAULT_REPORT_KEY
    else:
        report_key = "findings"

    return MedicalJsonDataset(
        input_filename=input_filename,
        transforms=transforms,
        img_root=img_root,
        img_feature_path=img_feature_path,
        text_feature_path=text_feature_path,
        metadata_path=metadata_path,
        tokenizer=tokenizer,
        report_key=report_key,
    )
