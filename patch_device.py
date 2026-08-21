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
                original_line = line
                if "rtg = torch.tensor(traj[\"returns_to_go\"][:seq_len]" in line and ".to(device)" not in line:
                    line = line.replace(".unsqueeze(-1)\\n", ".unsqueeze(-1).to(device)\\n").replace(".unsqueeze(-1)", ".unsqueeze(-1).to(device)")
                if "st = torch.tensor(traj[\"states\"][:seq_len]" in line and ".to(device)" not in line:
                    line = line.replace(".unsqueeze(0)\\n", ".unsqueeze(0).to(device)\\n").replace(".unsqueeze(0)", ".unsqueeze(0).to(device)")
                if "ts = torch.tensor(traj[\"timesteps\"][:seq_len]" in line and ".to(device)" not in line:
                    line = line.replace(".unsqueeze(0)\\n", ".unsqueeze(0).to(device)\\n").replace(".unsqueeze(0)", ".unsqueeze(0).to(device)")
                if "act = torch.tensor(traj[\"actions\"][:seq_len, 0]" in line and ".to(device)" not in line:
                    line = line.replace(".unsqueeze(0)\\n", ".unsqueeze(0).to(device)\\n").replace(".unsqueeze(0)", ".unsqueeze(0).to(device)")
                
                if line != original_line:
                    modified = True
                new_source.append(line)
            
            cell['source'] = new_source
            
    if modified:
        print(f"Modified {fpath}")
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
