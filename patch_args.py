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
            
            i = 0
            while i < len(source):
                line = source[i]
                if "sys.executable, \"-m\", \"marl.train\"," in line:
                    new_source.append(line)
                    # We expect the next lines to be the bad arguments
                    # Let's peek ahead and replace them
                elif "\"--total-steps\", \"50000\"," in line:
                    new_source.append(line.replace("--total-steps", "--total-env-steps"))
                    modified = True
                elif "\"--n-envs\", \"4\"," in line:
                    new_source.append(line.replace("--n-envs", "--envs"))
                    modified = True
                elif "\"--save-dir\", f\"marl/checkpoints/{RUN_ID}\"" in line:
                    new_source.append("    \"--checkpoint-dir\", \"marl/checkpoints\",\n")
                    new_source.append("    \"--run-id\", RUN_ID\n")
                    modified = True
                else:
                    new_source.append(line)
                i += 1
            
            cell['source'] = new_source
            
    if modified:
        print(f"Modified {fpath}")
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
