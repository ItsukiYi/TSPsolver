"""Test import for improvement #2."""
import sys, os
_p = r'D:\cs240project'
# Clear any existing problematic paths
sys.path = [p for p in sys.path if 'src' not in p and 'DualOpt' not in p and 'difusco' not in p.lower()]
sys.path.insert(0, os.path.join(_p, 'DualOpt-improved'))
sys.path.append(os.path.join(_p, 'src'))
print('path[0]:', sys.path[0])
from utils.difusco_pipeline import run_difusco_dualopt_pipeline
print('Import OK!')
