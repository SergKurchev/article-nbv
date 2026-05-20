import json

notebook_path = 'notebooks/kaggle_odin_rl_train.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

modified = False
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        source_str = ''.join(source)
        if 'FREEZE_BACKBONE  = True' in source_str:
            new_source = []
            for line in source:
                if 'FREEZE_BACKBONE  = True' in line:
                    new_source.append(line)
                    new_source.append('TRAIN_LAST_TRANSFORMER_BLOCK = True  # True: train last block of transformer in backbone\n')
                elif 'if FREEZE_BACKBONE:' in line:
                    new_source.append(line)
                elif 'train_cmd.append("--freeze_backbone")' in line:
                    new_source.append(line)
                    new_source.append('if TRAIN_LAST_TRANSFORMER_BLOCK:\n')
                    new_source.append('    train_cmd.append("--train_last_transformer_block")\n')
                elif 'print(f"  Freeze:       {FREEZE_BACKBONE}")' in line:
                    new_source.append(line)
                    new_source.append('print(f"  Train Last Transformer Block: {TRAIN_LAST_TRANSFORMER_BLOCK}")\n')
                else:
                    new_source.append(line)
            cell['source'] = new_source
            modified = True
            print('Successfully modified the training cell in the notebook!')
            break

if modified:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
else:
    print('Failed to locate target cell in the notebook.')
