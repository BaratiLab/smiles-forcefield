"""Structure-only candidate cohort screening. Never downloads QM outcomes."""
import argparse, hashlib, io, json, sys, tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def identity(m):
    m = Chem.RemoveHs(Chem.Mol(m))
    Chem.RemoveStereochemistry(m)
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmiles(m, canonical=True)


def scaffold(m):
    m = Chem.RemoveHs(Chem.Mol(m))
    Chem.RemoveStereochemistry(m)
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    s = MurckoScaffold.GetScaffoldForMol(m)
    return (
        ("murcko:" + Chem.MolToSmiles(s))
        if s.GetNumAtoms()
        else ("acyclic:" + Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(m)))
    )


def run(entries, seen_dirs, archive, out, pilot=None, status=None):
    entries = Path(entries)
    seen_parents = set()
    seen_scaffolds = set()
    files = []
    for folder in seen_dirs:
        for p in Path(folder).glob("*.json"):
            d = json.loads(p.read_text())
            if "molecule" not in d:
                continue
            m = Chem.MolFromSmiles(d["molecule"]["smiles"])
            seen_parents.add(identity(m))
            seen_scaffolds.add(scaffold(m))
            files.append(p)
    if pilot:
        for r in json.loads(Path(pilot).read_text()):
            m = Chem.MolFromSmiles(r["smiles"])
            seen_parents.add(identity(m))
            seen_scaffolds.add(scaffold(m))
    statuses = (
        {r[0]: r[2] for r in json.loads(Path(status).read_text())} if status else {}
    )
    fitted = set()
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf:
            if member.isfile() and member.name.endswith("/input.sdf"):
                m = next(
                    iter(
                        Chem.ForwardSDMolSupplier(
                            io.BytesIO(tf.extractfile(member).read()), removeHs=False
                        )
                    ),
                    None,
                )
                if m is not None:
                    fitted.add(identity(m))
    rows = []
    eligible = []
    parents = set()
    for key, r in sorted(json.loads(entries.read_text()).items()):
        smiles = r["attributes"]["canonical_isomeric_smiles"]
        m = Chem.MolFromSmiles(smiles)
        parent = identity(m)
        sc = scaffold(m)
        reasons = []
        if status and statuses.get(key) != "complete":
            reasons.append("not_complete_at_inventory_freeze")
        if parent in seen_parents:
            reasons.append("previously_inspected_parent")
        if sc in seen_scaffolds:
            reasons.append("previously_inspected_scaffold")
        if parent in fitted:
            reasons.append("Sage_2_2_1_archive_fitting_parent")
        if parent in parents:
            reasons.append("duplicate_parent_in_candidate_source")
        # Neutral, connected, conventional-element subset matches intended first-stage scope.
        if (
            len(Chem.GetMolFrags(m)) != 1
            or Chem.GetFormalCharge(m) != 0
            or any(
                a.GetAtomicNum() not in (1, 6, 7, 8, 9, 15, 16, 17, 35, 53)
                for a in m.GetAtoms()
            )
        ):
            reasons.append("outside_initial_scope")
        row = {
            "entry": key,
            "parent": parent,
            "scaffold": sc,
            "smiles": smiles,
            "exclusions": reasons,
        }
        rows.append(row)
        if not reasons:
            eligible.append(row)
            parents.add(parent)
    # Seal membership by structure hash; it is not yet a registered confirmatory cohort.
    eligible.sort(
        key=lambda r: hashlib.sha256(
            ("repair-new-source-v1:" + r["parent"]).encode()
        ).hexdigest()
    )
    result = {
        "status": "structure-only inventory, outcomes untouched; confirmation protocol not yet frozen",
        "source": "QCArchive dataset472 TorsionNet500 Re-optimization TorsionDrives v4.0",
        "source_entries": len(rows),
        "eligible_unique_parents": len(eligible),
        "eligible_scaffold_groups": len({r["scaffold"] for r in eligible}),
        "previously_inspected_parents": len(seen_parents),
        "previously_inspected_scaffolds": len(seen_scaffolds),
        "archive_fitting_parents": len(fitted),
        "exclusion_counts": {
            reason: sum(reason in r["exclusions"] for r in rows)
            for reason in sorted({x for r in rows for x in r["exclusions"]})
        },
        "candidate_membership_sha256": hashlib.sha256(
            json.dumps(eligible, sort_keys=True).encode()
        ).hexdigest(),
        "structure_entries_sha256": hashlib.sha256(entries.read_bytes()).hexdigest(),
        "fitting_archive_sha256": hashlib.sha256(
            Path(archive).read_bytes()
        ).hexdigest(),
        "qm_outcome_records_retrieved": False,
        "limitations": [
            "Not an audit of every charge/nonbonded fitting source",
            "Pretrained LLM exposure to original TorsionNet is unknown",
            "Entry count is not number of complete converged 24-frame scans",
            "Acyclic scaffold grouping follows the prior development convention",
            "No claim that this reaches the proposed 300-parent/150-scaffold target",
        ],
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "inventory.json").write_text(json.dumps(result, indent=2))
    (out / "screened-entries.json").write_text(json.dumps(rows, indent=2))
    (out / "candidate-structures.json").write_text(json.dumps(eligible, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--entries", required=True)
    p.add_argument("--seen", nargs="+", required=True)
    p.add_argument("--archive", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pilot")
    p.add_argument("--status")
    a = p.parse_args()
    run(a.entries, a.seen, a.archive, a.out, a.pilot, a.status)
