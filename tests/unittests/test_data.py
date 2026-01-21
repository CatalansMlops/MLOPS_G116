import os

import pytest
from PIL import Image

_TEST_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")

RAW_DIR = os.path.join(_PATH_DATA, "raw", "brain_dataset")
DATA_AVAILABLE = os.path.isdir(RAW_DIR)

EXPECTED_SPLITS = ["Training", "Testing"]
EXPECTED_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


@pytest.mark.skipif(not DATA_AVAILABLE, reason="Data files not found")
def test_splits_exist() -> None:
    """Validate that required train/test splits exist."""
    splits = [f for f in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, f))]
    expected_splits = set(EXPECTED_SPLITS)
    missing_splits = expected_splits - set(splits)
    assert not missing_splits, f"❌ Splits missing: {missing_splits}"
    print(f"✅ Training and Testing splits found: {splits}")


@pytest.mark.skipif(not DATA_AVAILABLE, reason="Data files not found")
def test_classes_exist() -> None:
    """Validate that required class folders exist for each split."""
    for split in EXPECTED_SPLITS:
        split_path = os.path.join(RAW_DIR, split)
        classes = [f for f in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, f))]
        expected_classes = set(EXPECTED_CLASSES)
        missing_classes = expected_classes - set(classes)
        assert not missing_classes, f"❌ Missing classes in {split}: {missing_classes}"
        print(f"✅ All classes exist in {split}: {expected_classes}")


@pytest.mark.skipif(not DATA_AVAILABLE, reason="Data files not found")
def test_images_exist() -> None:
    """Validate that each class folder contains at least one image."""
    for split in EXPECTED_SPLITS:
        for cls in EXPECTED_CLASSES:
            folder_path = os.path.join(RAW_DIR, split, cls)
            files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            assert len(files) > 0, f"❌ No images found in {split}/{cls}"
            print(f"✅ {len(files)} images found in {split}/{cls}")


@pytest.mark.skipif(not DATA_AVAILABLE, reason="Data files not found")
def test_images_readable() -> None:
    """Validate that each image can be opened by PIL without errors."""
    files_count = 0
    for split in EXPECTED_SPLITS:
        for cls in EXPECTED_CLASSES:
            folder_path = os.path.join(RAW_DIR, split, cls)
            files_count += len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
            for f in os.listdir(folder_path):
                path = os.path.join(folder_path, f)
                try:
                    Image.open(path).verify()
                except Exception:
                    raise AssertionError(f"❌ Corrupted image: {path}")
    print(f"✅ All {files_count} images are readable")
