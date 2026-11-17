from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors

ELEMENTS={1,6,7,8,9,15,16,17,35,53}

def inspect(smiles):
    if not isinstance(smiles,str) or not 1<=len(smiles.strip())<=1000:
        raise ValueError('Provide a SMILES string of 1–1000 characters.')
    # Disable SMILES name parsing and extended CXSMILES properties.
    params=Chem.SmilesParserParams(); params.parseName=False; params.allowCXSMILES=False
    mol=Chem.MolFromSmiles(smiles.strip(),params)
    if mol is None:raise ValueError('Invalid SMILES.')
    if len(Chem.GetMolFrags(mol))!=1:raise ValueError('Provide one connected molecule, without salts or mixtures.')
    if any(a.GetAtomicNum() not in ELEMENTS for a in mol.GetAtoms()):raise ValueError('Supported elements: H, C, N, O, F, P, S, Cl, Br, I.')
    if not any(a.GetAtomicNum()==6 for a in mol.GetAtoms()):raise ValueError('This release supports organic molecules containing carbon.')
    if Chem.GetFormalCharge(mol)!=0:raise ValueError('This first release supports neutral molecules only. Specify the intended neutral state.')
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):raise ValueError('Radicals are not supported.')
    if mol.GetNumHeavyAtoms()>40 or Chem.AddHs(mol).GetNumAtoms()>100:raise ValueError('Limit: 40 heavy atoms and 100 total atoms.')
    for s in Chem.FindPotentialStereo(mol):
        if str(s.specified)=='Unspecified':raise ValueError('Specify all stereochemistry explicitly before submitting.')
    for a in mol.GetAtoms():a.SetAtomMapNum(0)
    canonical=Chem.MolToSmiles(mol,isomericSmiles=True)
    return {'canonical_smiles':canonical,'formula':rdMolDescriptors.CalcMolFormula(mol),
            'heavy_atoms':mol.GetNumHeavyAtoms(),'atoms':Chem.AddHs(mol).GetNumAtoms(),'formal_charge':0}

def svg(smiles):
    canonical=inspect(smiles)['canonical_smiles']
    drawer=Draw.rdMolDraw2D.MolDraw2DSVG(520,300)
    drawer.drawOptions().clearBackground=False
    Draw.rdMolDraw2D.PrepareAndDrawMolecule(drawer,Chem.MolFromSmiles(canonical))
    drawer.FinishDrawing()
    return drawer.GetDrawingText()
