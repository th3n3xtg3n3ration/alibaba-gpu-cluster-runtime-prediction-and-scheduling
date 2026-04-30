import json
import os

def update_notebook(filepath):
    print(f"Updating {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
            
        source = cell['source']
        
        # 1. Update imports
        if any('from src.simulation.scheduler_simulator import FIFOScheduler' in line for line in source):
            for i, line in enumerate(source):
                if 'from src.simulation.scheduler_simulator import' in line and 'SRFScheduler' not in line:
                    source[i] = line.replace('SJFPredScheduler', 'SJFPredScheduler, SRFScheduler')
                    print("Updated imports.")
                    
        # 2. Update run_policy
        if any('def run_policy' in line for line in source):
            for i, line in enumerate(source):
                if 'elif policy_name == "SJF-Oracle":' in line:
                    # Check if SRF already added to avoid duplication
                    if i+2 < len(source) and 'SRF (Heuristic)' in source[i+2]:
                        break
                    source.insert(i+2, '    elif policy_name == "SRF (Heuristic)":\n')
                    source.insert(i+3, '        scheduler = SRFScheduler()\n')
                    print("Updated run_policy.")
                    break

        # 3. Update POLICIES
        if any('POLICIES = [' in line for line in source):
            for i, line in enumerate(source):
                if '"FIFO",' in line:
                    if i+1 < len(source) and '"SRF (Heuristic)",' in source[i+1]:
                        break
                    source.insert(i+1, '    "SRF (Heuristic)",\n')
                    print("Updated POLICIES list.")
                    break
                    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Done {filepath}.")

if __name__ == '__main__':
    base = "/Users/hasanugurcelebi/Thesis/alibaba-gpu-runtime-prediction-and-scheduling"
    update_notebook(f"{base}/notebooks/tr/05_gorev_zamanlayici_degerlendirme.ipynb")
    update_notebook(f"{base}/notebooks/en/05_scheduler_evaluation.ipynb")
