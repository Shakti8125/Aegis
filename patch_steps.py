import json
import glob

files = glob.glob('notebooks/*.ipynb')

for fpath in files:
    with open(fpath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    modified = False
    for cell in data.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            new_source = []
            
            for line in source:
                if "\"--total-env-steps\", \"50000\"," in line:
                    new_source.append(line.replace("50000", "1000000"))
                    modified = True
                else:
                    new_source.append(line)
            
            cell['source'] = new_source
            
    if modified:
        print(f"Modified {fpath}")
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
