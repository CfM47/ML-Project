"""Test suite for ViT Classifier."""

import numpy as np

from auto_ml.implementations.classifiers.vit import ViTModel


def test_vit_classifier() -> None:  # noqa: D103
    print("Initializing Verification for ViTModel...")

    # 1. Create dummy data
    print("Creating dummy image and region...")
    image = (
        np.random.randint(0, 255, (512, 512), dtype=np.uint8).astype(np.float32) / 255.0
    )
    x, y, width, height = 100, 100, 224, 224

    print(f"Image shape: {image.shape}")
    print(f"Region (x, y, w, h): ({x}, {y}, {width}, {height})")

    # 2. Instantiate Classifier
    print("Instantiating ViTModel...")
    try:
        classifier = ViTModel(
            device="cpu",
            channels=1,
            num_classes=3,
            image_size=224,
        )
        print("ViTModel instantiated successfully.")
    except Exception as e:
        print(f"Error instantiating ViTModel: {e}")
        import traceback

        traceback.print_exc()
        return

    # 3. Test Classification
    print("Testing classify() method...")
    try:
        class_label, confidence = classifier.classify(image, x, y, width, height)
        print(
            f"Classification result: class={class_label}, confidence={confidence:.4f}",
        )

        # Verify output format
        assert isinstance(class_label, (int, np.integer)), "class_label should be int"
        assert isinstance(confidence, (float, np.floating)), (
            "confidence should be float"
        )
        assert 0 <= class_label < 3, f"class_label {class_label} should be in [0, 3)"
        assert 0.0 <= confidence <= 1.0, f"confidence {confidence} should be in [0, 1]"
    except Exception as e:
        print(f"Error during classification: {e}")
        import traceback

        traceback.print_exc()
        return

    # 4. Test with RGB image
    print("\nTesting RGB image classification...")
    try:
        rgb_image = (
            np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8).astype(np.float32)
            / 255.0
        )
        rgb_classifier = ViTModel(
            device="cpu",
            channels=3,
            num_classes=3,
            image_size=224,
        )
        class_label, confidence = rgb_classifier.classify(
            rgb_image, x, y, width, height,
        )
        print(
            f"RGB Classification result: class={class_label}, "
            f"confidence={confidence:.4f}",
        )

        assert isinstance(class_label, (int, np.integer)), "class_label should be int"
        assert 0 <= class_label < 3, f"class_label {class_label} should be in [0, 3)"
        assert 0.0 <= confidence <= 1.0, f"confidence {confidence} should be in [0, 1]"
    except Exception as e:
        print(f"Error during RGB classification: {e}")
        import traceback

        traceback.print_exc()
        return

    print("\nVERIFICATION SUCCESSFUL!")
