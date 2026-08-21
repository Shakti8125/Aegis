import json
import glob
import os

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
                
                # Replace git clone condition
                if "# 2. Clone the Aegis repo" in line:
                    new_source.append("# 2. Clone the Aegis repo unconditionally (force fresh clone)\n")
                    new_source.append("REPO_URL = \"https://github.com/Shakti8125/Aegis.git\"\n")
                    new_source.append("import shutil\n")
                    new_source.append("if os.path.exists(\"/content/Aegis\"):\n")
                    new_source.append("    shutil.rmtree(\"/content/Aegis\")\n")
                    new_source.append("elif os.path.exists(\"Aegis\"):\n")
                    new_source.append("    shutil.rmtree(\"Aegis\")\n")
                    new_source.append("subprocess.check_call([\"git\", \"clone\", REPO_URL, \"/content/Aegis\"])\n")
                    
                    # skip the old lines
                    i += 1
                    while i < len(source) and ("REPO_URL =" in source[i] or "if not os.path.exists" in source[i] or "subprocess.check_call([\"git\", \"clone\"" in source[i]):
                        i += 1
                    modified = True
                    continue
                
                # Remove normalization_state_dict
                if "normalization_state_dict" in line:
                    modified = True
                    i += 1
                    continue
                
                if "\"state_dict\": encoder.state_dict()," in line:
                    # check if the next line is the normalization one
                    if i + 1 < len(source) and "normalization_state_dict" in source[i+1]:
                        new_source.append("    \"state_dict\": encoder.state_dict()\n")
                    else:
                        new_source.append(line)
                    modified = True
                    i += 1
                    continue
                
                new_source.append(line)
                i += 1
            
            cell['source'] = new_source
            
    if modified:
        print(f"Modified {fpath}")
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
