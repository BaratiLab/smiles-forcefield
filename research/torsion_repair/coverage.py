"""Outcome-blind Sage parameter coverage audit of candidate structures."""
import argparse, hashlib, json
from pathlib import Path
from openff.toolkit import ForceField, Molecule


def normalized(path):
    return min(tuple(path), tuple(path[::-1]))


def run(candidates, out):
    source = Path(candidates)
    rows = json.loads(source.read_text())
    ff = ForceField("openff_unconstrained-2.2.1.offxml")
    result = []
    eligible = []
    for r in rows:
        try:
            m = Molecule.from_smiles(r["smiles"], allow_undefined_stereo=False)
            labels = ff.label_molecules(m.to_topology())[0]
            expected = {
                "vdW": {(i,) for i in range(m.n_atoms)},
                "Bonds": {
                    normalized(
                        (term.atom1.molecule_atom_index, term.atom2.molecule_atom_index)
                    )
                    for term in m.bonds
                },
                "Angles": {
                    normalized(tuple(a.molecule_atom_index for a in term))
                    for term in m.angles
                },
                "ProperTorsions": {
                    normalized(tuple(a.molecule_atom_index for a in term))
                    for term in m.propers
                },
            }
            missing = {
                name: sorted(paths - {normalized(k) for k in labels[name]})
                for name, paths in expected.items()
            }
            if any(missing.values()):
                raise ValueError("Missing parameter assignments: " + str(missing))
            eligible.append(r)
            result.append(
                {
                    "entry": r["entry"],
                    "coverage_passed": True,
                    "atoms": m.n_atoms,
                    "proper_torsion_instances": len(expected["ProperTorsions"]),
                }
            )
        except Exception as error:
            result.append(
                {
                    "entry": r["entry"],
                    "coverage_passed": False,
                    "error_type": type(error).__name__,
                    "error": str(error)[:1000],
                }
            )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "status": "structure and topology coverage only; no charge assignment or QM outcome retrieval",
        "forcefield": "openff_unconstrained-2.2.1.offxml",
        "forcefield_sha256": hashlib.sha256(ff.to_string().encode()).hexdigest(),
        "candidate_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "input_parents": len(rows),
        "coverage_passed_parents": len(eligible),
        "coverage_failed_parents": len(rows) - len(eligible),
        "limitations": [
            "Improper assignment multiplicity is not audited by this simple topological count",
            "Successful labels do not guarantee AM1-BCC convergence",
            "QM scan completeness and chemical-state consistency still need fixed-rule checks after model/protocol freeze",
        ],
        "results": result,
    }
    (out / "coverage.json").write_text(json.dumps(record, indent=2))
    (out / "coverage-eligible-structures.json").write_text(
        json.dumps(eligible, indent=2)
    )
    print("Coverage", len(eligible), "/", len(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    run(a.candidates, a.out)
