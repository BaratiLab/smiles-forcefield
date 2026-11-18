"""Guard an absent optional completion file in the conda-forge GROMACS package.

Noninteractive Micromamba activation uses `set -e`; sourcing an unmatched
completion glob otherwise prevents Docker build and container startup.
This changes shell completion only, not simulation code or parameters.
"""
from pathlib import Path

for path in Path('/opt/conda').glob('bin*/GMXRC.bash'):
    original=path.read_text()
    fixed=original.replace('source $cfile', 'if [ -f "$cfile" ]; then source "$cfile"; fi')
    if fixed!=original:
        path.write_text(fixed)
        print('Guarded optional GROMACS completion:',path)
