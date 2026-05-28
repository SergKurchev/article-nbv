"""
Patch requirements.txt before pip install:
  - Remove packages installed separately (detectron2, pytorch3d, waspinator, ai2thor, prior)
  - Upgrade outdated pinned versions (pyyaml 5.3.1 -> >=5.4.1, Pillow 9.1.0 -> >=10.2.0)
"""

skip = ['waspinator', 'detectron2', 'pytorch3d', 'ai2thor', 'prior']

lines = open('requirements.txt').readlines()
with open('requirements.txt', 'w') as f:
    for line in lines:
        lc = line.strip().lower()
        if any(x in lc for x in skip):
            print(f'  Skipping: {line.strip()}')
            continue
        if 'pyyaml==5.3.1' in lc:
            f.write('pyyaml>=5.4.1\n')
        elif 'pillow==9.1.0' in lc:
            f.write('Pillow>=10.2.0\n')
        else:
            f.write(line)

print('requirements.txt patched successfully')
