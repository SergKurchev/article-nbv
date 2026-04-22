"""Test and visualize PointNet predictions.

Usage:
    uv run python scripts/test_pointnet.py
    uv run python scripts/test_pointnet.py --visualize  # Show 3D point clouds
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
import config
from src.vision.models import PointNet
import argparse


def load_model(weights_path):
    """Load trained PointNet model."""
    model = PointNet(
        num_classes=config.NUM_CLASSES,
        vector_dim=15,
        num_points=config.POINTNET_NUM_POINTS,
        use_vector=True
    )

    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()

    return model


def test_sample(model, sample_dir):
    """Test model on a single sample."""
    # Load point clouds
    point_clouds = np.load(sample_dir / "point_clouds.npy")
    vectors = np.load(sample_dir / "vectors.npy")

    # Load metadata
    with open(sample_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)

    true_class = metadata['class_id']

    # Test on all views
    predictions = []
    confidences = []

    for i in range(len(point_clouds)):
        points = torch.from_numpy(point_clouds[i].T).unsqueeze(0).float()
        vector = torch.from_numpy(vectors[i]).unsqueeze(0).float()

        with torch.no_grad():
            logits = model(points, vector)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()

        predictions.append(pred_class)
        confidences.append(confidence)

    # Majority vote
    pred_class = max(set(predictions), key=predictions.count)
    avg_confidence = np.mean(confidences)

    return {
        'true_class': true_class,
        'pred_class': pred_class,
        'predictions': predictions,
        'confidences': confidences,
        'avg_confidence': avg_confidence,
        'correct': pred_class == true_class
    }


def visualize_point_cloud(points, title="Point Cloud"):
    """Visualize 3D point cloud using matplotlib.

    Args:
        points: [N, 3] numpy array
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("Error: matplotlib not installed. Install with: pip install matplotlib")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=points[:, 2], cmap='viridis', s=1, alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Equal aspect ratio
    max_range = np.array([points[:, 0].max()-points[:, 0].min(),
                          points[:, 1].max()-points[:, 1].min(),
                          points[:, 2].max()-points[:, 2].min()]).max() / 2.0

    mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Test PointNet model")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights")
    parser.add_argument("--visualize", action="store_true", help="Visualize point clouds")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to test")

    args = parser.parse_args()

    print("="*60)
    print("PointNet Testing")
    print("="*60)

    # Load model
    if args.weights is None:
        weights_path = config.CNN_WEIGHTS_DIR / "pointnet_best.pt"
    else:
        weights_path = Path(args.weights)

    if not weights_path.exists():
        print(f"Error: Model weights not found: {weights_path}")
        print("Please train the model first: uv run python src/vision/train_pointnet.py")
        return

    print(f"\nLoading model from: {weights_path}")
    model = load_model(weights_path)
    print("Model loaded successfully")

    # Get dataset
    dataset_dir = config.BASE_DIR / "dataset_pointnet" / config.OBJECT_MODE
    if not dataset_dir.exists():
        print(f"Error: Dataset not found: {dataset_dir}")
        return

    samples = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("sample_")])

    if len(samples) == 0:
        print(f"Error: No samples found in {dataset_dir}")
        return

    print(f"Found {len(samples)} samples")

    # Test on random samples
    test_samples = np.random.choice(samples, min(args.num_samples, len(samples)), replace=False)

    results = []
    for sample_dir in test_samples:
        result = test_sample(model, sample_dir)
        results.append(result)

        status = "✓" if result['correct'] else "✗"
        print(f"\n{status} Sample: {sample_dir.name}")
        print(f"  True class: {result['true_class']}")
        print(f"  Pred class: {result['pred_class']}")
        print(f"  Avg confidence: {result['avg_confidence']:.3f}")

        # Visualize if requested
        if args.visualize:
            point_clouds = np.load(sample_dir / "point_clouds.npy")
            # Show first view
            visualize_point_cloud(
                point_clouds[0],
                title=f"Sample {sample_dir.name} - Class {result['true_class']}"
            )

    # Summary
    accuracy = sum(r['correct'] for r in results) / len(results) * 100
    avg_confidence = np.mean([r['avg_confidence'] for r in results])

    print(f"\n{'='*60}")
    print(f"Test Results:")
    print(f"  Samples tested: {len(results)}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Avg confidence: {avg_confidence:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
