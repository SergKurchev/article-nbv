"""Take screenshots of PyBullet window for analysis."""
import time
import pyautogui
from pathlib import Path

# Wait for window to open
print("Waiting 5 seconds for PyBullet window...")
time.sleep(5)

# Take screenshots from different angles
screenshots = []
for i in range(3):
    screenshot_path = Path(f"texture_view_{i+1}.png")
    pyautogui.screenshot(screenshot_path)
    screenshots.append(screenshot_path)
    print(f"Screenshot {i+1} saved: {screenshot_path}")

    if i < 2:
        # Rotate camera (simulate arrow key presses)
        pyautogui.press('right', presses=10, interval=0.1)
        time.sleep(1)

print("\nScreenshots saved:")
for path in screenshots:
    print(f"  - {path}")
