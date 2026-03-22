import os

def fix_imports():
    for root, _, files in os.walk('src'):
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Replace bad imports
                content = content.replace('from simulation.', 'from src.simulation.')
                content = content.replace('from utils.', 'from src.utils.')
                content = content.replace('from rl.', 'from src.rl.')
                content = content.replace('from vision.', 'from src.vision.')
                
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)

if __name__ == '__main__':
    fix_imports()
