"""Verification script to test NBV project setup.

Checks:
1. Texture files exist
2. All imports work
3. Environment can be created
4. Asset loader can load objects with textures
"""

import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def check_textures():
    """Verify texture files exist."""
    print("Checking textures...")
    import config

    texture_dir = config.DATA_DIR / "objects" / "textures"
    required_textures = ["red.png", "mixed.png", "green.png"]

    for texture in required_textures:
        texture_path = texture_dir / texture
        if not texture_path.exists():
            print(f"  ❌ Missing: {texture}")
            return False
        else:
            print(f"  ✓ Found: {texture}")

    print("  ✓ All textures present\n")
    return True


def check_imports():
    """Verify all modules can be imported."""
    print("Checking imports...")

    modules = [
        ("config", "config"),
        ("texture_generator", "src.vision.texture_generator"),
        ("asset_loader", "src.simulation.asset_loader"),
        ("environment", "src.simulation.environment"),
        ("dataset", "src.vision.dataset"),
        ("models", "src.vision.models"),
    ]

    for name, module_path in modules:
        try:
            __import__(module_path)
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            return False

    print("  ✓ All imports successful\n")
    return True


def check_environment():
    """Test environment creation."""
    print("Checking environment...")

    try:
        from src.simulation.environment import NBVEnv

        # Create environment in headless mode
        env = NBVEnv(headless=True)
        print("  ✓ Environment created")

        # Test reset
        obs, info = env.reset()
        print(f"  ✓ Environment reset successful")
        print(f"    - Image shape: {obs['image'].shape}")
        print(f"    - Vector shape: {obs['vector'].shape}")

        env.close()
        print("  ✓ Environment closed\n")
        return True

    except Exception as e:
        print(f"  ❌ Environment test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def check_texture_loading():
    """Test texture loading in asset loader."""
    print("Checking texture loading...")

    try:
        import pybullet as p
        from src.simulation.asset_loader import AssetLoader

        # Connect to PyBullet
        client_id = p.connect(p.DIRECT)

        loader = AssetLoader(client_id)

        # Test loading objects with different textures
        texture_types = ["red", "mixed", "green"]

        for i, texture_type in enumerate(texture_types):
            obj_id = loader.load_target_object(class_id=i, texture_type=texture_type)
            print(f"  ✓ Loaded Object_{i+1:02d} with {texture_type} texture")
            p.removeBody(obj_id, physicsClientId=client_id)

        p.disconnect(client_id)
        print("  ✓ Texture loading successful\n")
        return True

    except Exception as e:
        print(f"  ❌ Texture loading failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("NBV Project Setup Verification")
    print("=" * 60)
    print()

    checks = [
        ("Textures", check_textures),
        ("Imports", check_imports),
        ("Environment", check_environment),
        ("Texture Loading", check_texture_loading),
    ]

    results = []

    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} check crashed: {e}\n")
            results.append((name, False))

    print("=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
        if not result:
            all_passed = False

    print()

    if all_passed:
        print("✓ All checks passed! Project is ready.")
        print("\nNext steps:")
        print("  1. Generate textures: uv run python src/vision/texture_generator.py")
        print("  2. Test textures: uv run python scripts/test_textures.py")
        print("  3. Generate dataset: uv run python src/vision/dataset.py")
        print("  4. Train CNN: uv run python src/vision/train_cnn.py")
        print("  5. Train RL agent: uv run python train.py")
        return 0
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
