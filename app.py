from collections import Counter
from io import BytesIO

import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Draw, rdMolDescriptors
import py3Dmol
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Molecule Atlas | Interactive Explorer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Physical values are rounded educational reference values. Conditions are stated
# where important; use the linked PubChem record for source details and updates.
# "Cyanide" is represented as hydrogen cyanide (HCN), a discrete molecule. The
# cyanide ion (CN-) is a different chemical species.
MOLECULES = {
    "Acetic acid": {
        "smiles": "CC(=O)O",
        "iupac": "Ethanoic acid",
        "class": "Organic compound · carboxylic acid",
        "cas": "64-19-7",
        "phase": "Liquid (pure acid; may solidify in a cool room)",
        "transition_label": "Melting point",
        "transition_value": "16.6 °C (pure acetic acid, at about 1 atm)",
        "boiling": "117.9 °C (at 1 atm)",
        "density": "About 1.049 g/cm³ (at 20 °C)",
        "solubility": "Miscible with water; also mixes with ethanol",
        "description": "A two-carbon carboxylic acid responsible for the characteristic acidity of vinegar. Concentrated, glacial acetic acid has a much higher hazard than diluted household vinegar.",
        "uses": "Food acidity regulation in suitable dilute formulations, laboratory reagent, and manufacture of acetate salts, polymers and other chemicals.",
        "safety": "Concentrated acetic acid is corrosive and can cause serious skin and eye burns. Handle laboratory-grade material with appropriate protective equipment.",
        "pubchem_cid": "176",
    },
    "Methyl alcohol": {
        "smiles": "CO",
        "iupac": "Methanol",
        "class": "Organic compound · alcohol",
        "cas": "67-56-1",
        "phase": "Liquid",
        "transition_label": "Melting point",
        "transition_value": "−97.6 °C (approx.)",
        "boiling": "64.7 °C (at 1 atm)",
        "density": "About 0.792 g/cm³ (at 20 °C)",
        "solubility": "Miscible with water",
        "description": "The simplest alcohol, containing one carbon atom and one hydroxyl group. It is also called wood alcohol.",
        "uses": "Industrial solvent, chemical feedstock, and component in some fuels and antifreeze formulations.",
        "safety": "Highly flammable and toxic. Swallowing, inhaling high concentrations, or significant skin exposure can cause severe poisoning, blindness or death. Never drink it.",
        "pubchem_cid": "887",
    },
    "Nitrobenzene": {
        "smiles": "O=[N+]([O-])c1ccccc1",
        "iupac": "Nitrobenzene",
        "class": "Organic compound · nitroaromatic",
        "cas": "98-95-3",
        "phase": "Liquid (may solidify below about 5.7 °C)",
        "transition_label": "Melting point",
        "transition_value": "About 5.7 °C",
        "boiling": "About 210.8 °C (at 1 atm)",
        "density": "About 1.204 g/cm³ (at 20 °C)",
        "solubility": "Very slightly soluble in water; more soluble in many organic solvents",
        "description": "An aromatic benzene ring bearing a nitro group (−NO₂). The nitro group is represented with formal charges in the molecular structure.",
        "uses": "Primarily an industrial intermediate for producing aniline; used in controlled manufacture of dyes, rubber chemicals and other products.",
        "safety": "Toxic by inhalation and skin absorption and can affect the blood's ability to carry oxygen. Avoid all direct contact and do not handle outside an appropriately equipped laboratory.",
        "pubchem_cid": "7416",
    },
    "Camphor": {
        "smiles": "CC1(C2CCC1(C(=O)C2)C)C",
        "iupac": "1,7,7-Trimethylbicyclo[2.2.1]heptan-2-one",
        "class": "Organic compound · bicyclic ketone / monoterpenoid",
        "cas": "76-22-2",
        "phase": "Solid (crystalline; can slowly sublime)",
        "transition_label": "Melting point",
        "transition_value": "About 174–180 °C",
        "boiling": "About 204 °C; can sublime near its boiling point",
        "density": "About 0.99 g/cm³ (solid, near room temperature)",
        "solubility": "Slightly soluble in water (around 1.2–1.6 g/L at 25 °C); soluble in ethanol",
        "description": "A bicyclic molecule containing a ketone group. Camphor is a naturally occurring monoterpenoid found in some plants and has a strong characteristic aroma.",
        "uses": "Used in some topical products, fragrances, and moth-repellent formulations; actual use depends on local product regulations.",
        "safety": "Harmful if swallowed in sufficient quantities and can cause serious poisoning, especially in children. Keep away from children and follow product labels.",
        "pubchem_cid": "2537",
    },
    "Propene": {
        "smiles": "C=CC",
        "iupac": "Prop-1-ene (propene)",
        "class": "Organic compound · alkene",
        "cas": "115-07-1",
        "phase": "Gas (at 25 °C and 1 atm)",
        "transition_label": "Melting point",
        "transition_value": "−185.3 °C (approx.)",
        "boiling": "−47.7 °C (at 1 atm)",
        "density": "About 1.7 kg/m³ as a gas (at 25 °C and 1 atm)",
        "solubility": "Poorly soluble in water; more soluble in some organic liquids",
        "description": "A three-carbon alkene containing one carbon–carbon double bond. It is also known as propylene.",
        "uses": "A major industrial feedstock for making polypropylene plastics and other chemicals.",
        "safety": "Extremely flammable gas stored under pressure. Leaks can displace oxygen and create fire or explosion hazards; liquefied propene can cause cold burns.",
        "pubchem_cid": "8252",
    },
    "Hydrogen cyanide (cyanide)": {
        "smiles": "C#N",
        "iupac": "Hydrogen cyanide",
        "class": "Small molecule · nitrile / hydrogen cyanide",
        "cas": "74-90-8",
        "formula_override": "HCN",
        "phase": "Highly volatile liquid near 25 °C at 1 atm (boiling point is close to room temperature)",
        "transition_label": "Melting point",
        "transition_value": "About −13.3 °C",
        "boiling": "About 25.6 °C (at 1 atm); very volatile",
        "density": "About 0.69 g/cm³ as a liquid near 20 °C",
        "solubility": "Miscible with water",
        "description": "A very small, linear molecule with one hydrogen, one carbon and one nitrogen atom. Here, ‘cyanide’ is represented as hydrogen cyanide (HCN); the cyanide ion (CN⁻) is a different species.",
        "uses": "Used only in strictly controlled industrial chemical synthesis and other specialized processes. This listing is for chemistry education, not handling guidance.",
        "safety": "Extremely toxic and potentially rapidly fatal by inhalation, ingestion or skin exposure. Never attempt to make, obtain, smell, or handle it; any suspected exposure is an emergency.",
        "pubchem_cid": "768",
    },
}

# ----- Styling -----
st.markdown(
    """
    <style>
      .block-container {max-width: 1450px; padding-top: 1.5rem; padding-bottom: 3rem;}
      [data-testid="stSidebar"] {border-right: 1px solid rgba(130,150,180,.22);}
      .atlas-eyebrow {font-size: .74rem; letter-spacing: .16em; text-transform: uppercase; font-weight: 750; color: #27a99a; margin-bottom: .35rem;}
      .atlas-title {font-size: clamp(2rem, 4vw, 3.3rem); font-weight: 800; letter-spacing: -.045em; line-height: 1.02; margin: 0;}
      .atlas-subtitle {color: var(--text-color); opacity: .73; font-size: 1rem; margin-top: .8rem; max-width: 720px;}
      .atlas-hero {border: 1px solid rgba(120,150,180,.27); background: linear-gradient(135deg, rgba(42,166,153,.10), rgba(88,112,209,.08)); border-radius: 22px; padding: 1.65rem 1.8rem; margin-bottom: 1rem;}
      .atlas-chip {display:inline-block; border:1px solid rgba(100,135,160,.34); background:rgba(130,160,180,.09); border-radius:999px; padding:.27rem .65rem; font-size:.78rem; margin: .15rem .25rem .15rem 0;}
      .atlas-note {border-left: 3px solid #27a99a; background: rgba(39,169,154,.08); padding: .8rem 1rem; border-radius: 0 10px 10px 0; font-size: .9rem;}
      .atlas-small-label {font-size:.76rem; color:var(--text-color); opacity:.68; text-transform:uppercase; letter-spacing:.08em; font-weight:700;}
      .atlas-value {font-size:1.1rem; font-weight:750; margin-top:.15rem; overflow-wrap:anywhere;}
      .atlas-section {font-size:1.18rem; font-weight:750; letter-spacing:-.02em; margin: .4rem 0 .55rem 0;}
      .atlas-muted {color:var(--text-color); opacity:.7; font-size:.9rem;}
      a {text-decoration:none;}
      div[data-testid="stMetric"] {border:1px solid rgba(130,150,180,.23); border-radius:14px; padding: .75rem .85rem; background: rgba(140,160,180,.045);}
    </style>
    """,
    unsafe_allow_html=True,
)


def get_molecule(name: str):
    """Return a sanitized RDKit molecule for the selected compound."""
    mol = Chem.MolFromSmiles(MOLECULES[name]["smiles"])
    if mol is None:
        raise ValueError(f"Could not parse SMILES for {name}")
    return mol


def make_3d_molblock(name: str) -> str:
    """Generate a 3D conformer for teaching/visualization, not experimental geometry."""
    mol = Chem.AddHs(get_molecule(name))
    result = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=False)
    if result != 0:
        result = AllChem.EmbedMolecule(mol, randomSeed=7, useRandomCoords=True)
    if result != 0:
        raise RuntimeError("A 3D conformer could not be generated for this molecule.")
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=250)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=250)
    except Exception:
        # The embedded conformer is still viewable if optimization is unavailable.
        pass
    return Chem.MolToMolBlock(mol)


def property_rows(name: str) -> dict:
    mol = get_molecule(name)
    expanded = Chem.AddHs(mol)
    data = MOLECULES[name]
    hbd = data.get("hbd_override", rdMolDescriptors.CalcNumHBD(mol))
    hba = data.get("hba_override", rdMolDescriptors.CalcNumHBA(mol))
    element_count = Counter(atom.GetSymbol() for atom in expanded.GetAtoms())
    return {
        "Molecule": name,
        "IUPAC name": data["iupac"],
        "Molecular formula": data.get("formula_override", rdMolDescriptors.CalcMolFormula(mol)), 
        "Molar mass (g/mol)": round(Descriptors.MolWt(mol), 3),
        "Monoisotopic exact mass (Da)": round(Descriptors.ExactMolWt(mol), 4),
        "RDKit XLogP (estimate)": round(Crippen.MolLogP(mol), 3),
        "Topological polar surface area (Å²)": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "Hydrogen-bond donors": int(hbd),
        "Hydrogen-bond acceptors": int(hba),
        "Rotatable bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "Heavy atoms": int(rdMolDescriptors.CalcNumHeavyAtoms(mol)),
        "Aromatic rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "Total atoms (incl. H)": int(expanded.GetNumAtoms()),
        "Total bonds (incl. H)": int(expanded.GetNumBonds()),
        "Formal charge": int(Chem.GetFormalCharge(mol)),
        "Element composition": ", ".join(f"{element}: {count}" for element, count in sorted(element_count.items())),
        "SMILES": data["smiles"],
        "Canonical SMILES": Chem.MolToSmiles(mol, canonical=True),
        "CAS Registry Number": data["cas"],
        "Physical state (approx. 25 °C)": data["phase"],
        data["transition_label"]: data["transition_value"],
        "Boiling / thermal behaviour": data["boiling"],
        "Density": data["density"],
        "Water solubility": data["solubility"],
    }


def render_atom_legend():
    st.markdown(
        '<span class="atlas-chip">C · carbon</span>'
        '<span class="atlas-chip">H · hydrogen</span>'
        '<span class="atlas-chip">O · oxygen</span>'
        '<span class="atlas-chip">N · nitrogen</span>',
        unsafe_allow_html=True,
    )

# ----- Sidebar -----
st.sidebar.markdown("## 🧪 Molecule Atlas")
st.sidebar.caption("A compact molecular reference lab")
st.sidebar.divider()
search_term = st.sidebar.text_input("Find a molecule", placeholder="e.g. camphor")
all_names = list(MOLECULES.keys())
filtered_names = [n for n in all_names if search_term.strip().lower() in n.lower()] if search_term.strip() else all_names
if not filtered_names:
    st.sidebar.warning("No matching molecule. Showing the full library.")
    filtered_names = all_names
selected_name = st.sidebar.selectbox("Molecule library", filtered_names, index=0)
selected = MOLECULES[selected_name]

st.sidebar.divider()
st.sidebar.markdown("**Included in this app**")
st.sidebar.markdown("2D structure · interactive 3D model · physical properties · calculated descriptors")
st.sidebar.caption("Educational reference only. Not a substitute for laboratory or clinical data.")

# ----- Header -----
st.markdown(
    f"""
    <div class="atlas-hero">
      <div class="atlas-eyebrow">INTERACTIVE CHEMISTRY REFERENCE</div>
      <div class="atlas-title">Molecule Atlas</div>
      <div class="atlas-subtitle">Explore the 2D structure, interactive 3D geometry and key chemical and physical properties of six selected molecules.</div>
      <div style="margin-top:1rem">
        <span class="atlas-chip">6 molecules</span>
        <span class="atlas-chip">2D structures</span>
        <span class="atlas-chip">Rotatable 3D models</span>
        <span class="atlas-chip">Property explorer</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

mol = get_molecule(selected_name)
props = property_rows(selected_name)
formula = props["Molecular formula"]

head_col1, head_col2 = st.columns([3, 1])
with head_col1:
    st.markdown(f"<div class='atlas-eyebrow'>SELECTED MOLECULE</div><h2 style='margin:.1rem 0 .2rem 0'>{selected_name}</h2>", unsafe_allow_html=True)
    st.markdown(f"<div class='atlas-muted'>{selected['iupac']} · {selected['class']}</div>", unsafe_allow_html=True)
with head_col2:
    st.markdown(f"<div class='atlas-small-label'>Molecular formula</div><div class='atlas-value'>{formula}</div><div class='atlas-muted' style='margin-top:.4rem'>CAS {selected['cas']}</div>", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Molar mass", f"{props['Molar mass (g/mol)']:.3f} g/mol")
m2.metric("Atoms incl. H", props["Total atoms (incl. H)"])
m3.metric("Polar surface area", f"{props['Topological polar surface area (Å²)']:.2f} Å²")
m4.metric("Physical state", selected["phase"])

st.divider()
overview_tab, structure_tab, model_tab, properties_tab = st.tabs(["Overview", "2D structure", "3D model", "All properties"])

with overview_tab:
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown("<div class='atlas-section'>2D molecular structure</div>", unsafe_allow_html=True)
        draw_mol = Chem.Mol(mol)
        AllChem.Compute2DCoords(draw_mol)
        image = Draw.MolToImage(draw_mol, size=(760, 500), kekulize=True, wedgeBonds=True)
        st.image(image, use_container_width=True)
        render_atom_legend()
        st.caption("2D diagram generated from the molecule's SMILES representation using RDKit.")
    with right:
        st.markdown("<div class='atlas-section'>Molecule profile</div>", unsafe_allow_html=True)
        st.write(selected["description"])
        st.markdown("**Common applications**")
        st.write(selected["uses"])
        st.markdown("**At a glance**")
        st.markdown(f"<span class='atlas-chip'>{selected['class']}</span><span class='atlas-chip'>{selected['phase']}</span>", unsafe_allow_html=True)
        quick = pd.DataFrame(
            [("Melting / transition", selected["transition_value"]), ("Boiling / thermal behaviour", selected["boiling"]), ("Density", selected["density"]), ("Water solubility", selected["solubility"])],
            columns=["Property", "Value"],
        )
        st.dataframe(quick, hide_index=True, use_container_width=True)
        st.markdown("**SMILES notation**")
        st.code(props["SMILES"], language="text")
        st.caption("Canonical form: " + props["Canonical SMILES"])
        st.link_button("Open PubChem compound record ↗", f"https://pubchem.ncbi.nlm.nih.gov/compound/{selected['pubchem_cid']}", use_container_width=True)

with structure_tab:
    st.markdown("<div class='atlas-section'>2D structure and formula</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.25, 1], gap="large")
    with c1:
        st.image(image, use_container_width=True)
        render_atom_legend()
    with c2:
        st.markdown(f"<div class='atlas-small-label'>Molecular formula</div><div style='font-size:2rem;font-weight:800;margin:.2rem 0 1rem 0'>{formula}</div>", unsafe_allow_html=True)
        st.write("The 2D diagram shows the atom connectivity and bond pattern. It does not represent the full three-dimensional shape.")
        st.markdown("**SMILES notation**")
        st.code(props["SMILES"], language="text")
        st.markdown("**Canonical SMILES**")
        st.code(props["Canonical SMILES"], language="text")
        st.markdown("**Element composition**")
        st.write(props["Element composition"])
        st.markdown("**Structure file**")
        two_d = Chem.Mol(mol)
        AllChem.Compute2DCoords(two_d)
        st.download_button(
            "Download 2D MOL file",
            data=Chem.MolToMolBlock(two_d),
            file_name=f"{selected_name.lower().replace(' ', '_')}_2d.mol",
            mime="chemical/x-mdl-molfile",
            use_container_width=True,
        )

with model_tab:
    st.markdown("<div class='atlas-section'>Interactive 3D molecular model</div>", unsafe_allow_html=True)
    st.write("Drag to rotate, scroll or pinch to zoom, and use the viewer controls to explore the generated geometry.")
    try:
        molblock = make_3d_molblock(selected_name)
        view = py3Dmol.view(width=1000, height=520)
        view.addModel(molblock, "mol")
        view.setStyle({"stick": {"radius": 0.16}, "sphere": {"scale": 0.23}})
        view.setBackgroundColor("0x101827")
        view.zoomTo()
        components.html(view._make_html(), height=530, scrolling=False)
        render_atom_legend()
        st.caption("The 3D geometry is generated computationally for visualization. It is not an experimentally determined crystal structure.")
        st.download_button(
            "Download generated 3D MOL file",
            data=molblock,
            file_name=f"{selected_name.lower().replace(' ', '_')}_3d.mol",
            mime="chemical/x-mdl-molfile",
        )
    except Exception as exc:
        st.error("The 3D viewer could not create a model in this environment.")
        st.caption(f"Technical detail: {exc}")
        st.info("Check that the packages listed in requirements.txt are installed, then restart the app.")

with properties_tab:
    st.markdown("<div class='atlas-section'>Chemical and physical properties</div>", unsafe_allow_html=True)
    st.caption("Calculated descriptors are produced from the molecular graph by RDKit. Physical values are rounded reference values and can vary with temperature, pressure, purity and source.")
    rows = pd.DataFrame([{"Property": key, "Value": value} for key, value in props.items() if key != "Molecule"])
    st.dataframe(rows, hide_index=True, use_container_width=True, height=570)
    csv = pd.DataFrame([property_rows(name) for name in all_names]).to_csv(index=False).encode("utf-8")
    st.download_button("Download all six molecules as CSV", data=csv, file_name="molecule_atlas_properties.csv", mime="text/csv")

st.divider()
with st.expander("Data sources and interpretation"):
    st.write("Compound identity, synonyms and general reference information are linked to PubChem records. This educational application combines selected rounded reference values with molecular descriptors calculated locally by RDKit.")
    st.markdown(f"- **{selected_name}:** [PubChem record](https://pubchem.ncbi.nlm.nih.gov/compound/{selected['pubchem_cid']})")
    st.markdown("- **RDKit:** [Documentation](https://www.rdkit.org/docs/)")
    st.markdown("- **3Dmol.js / py3Dmol:** [3Dmol.js documentation](https://3dmol.org/) · [py3Dmol project](https://github.com/3dmol/3Dmol.js/tree/master/py3Dmol)")
    st.markdown("- LogP, polar surface area and hydrogen-bond counts are computed descriptors, not direct measurements. Different software and counting conventions may produce different values.")

st.markdown("<div class='atlas-muted' style='text-align:center;padding-top:.4rem'>Molecule Atlas · educational chemistry project · reference data linked to PubChem</div>", unsafe_allow_html=True)
