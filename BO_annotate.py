"""
EDRIXS-BO symmetry label annotator  for initial Hamiltonian eigenvalues
-------------------------------------
Usage:
    python eval_single_params.py --config-py NiCl2_config.py --params params_test.json

This loads your material-specific config, uses its RIXS_runner instance (rixs_funV)
and experimental reference map (rixs_ref), and evaluates the positive L1 distance
between L1-normalized simulation and reference at a single parameter set.
"""

import argparse
import importlib.util
import json
import numpy as np
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Dict, Any,  Optional

from annotation import constructMockH, getWeight, getWeightL
from edrixs_python import get_fock_bin_by_N

def load_config_module(path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location("cfg_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load config from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_params_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Params JSON not found: {p}")
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {p}: {e}") from e
    if not isinstance(data, dict):
        raise TypeError(f"Top-level JSON must be an object/dict, got: {type(data).__name__}")
    # If user wrapped under {"params": {...}}, allow it.
    if "params" in data and isinstance(data["params"], dict):
        data = data["params"]
    # Ensure values are JSON scalars; numpy will handle numeric casting later.
    for k, v in list(data.items()):
        if isinstance(v, (list, dict)):
            raise TypeError(f"Parameter '{k}' must be a scalar (number/string), got {type(v).__name__}")
    return data

import re
from datetime import datetime

def _fmt_terms_t2g(raw: str) -> str:
    """
    Convert strings like '0.34*3T2(t2)^3+0.12*1A1(t2)^2' into LaTeX math:
    $0.34\,{}^{3}T_{2}(t_{2})^{3} + 0.12\,{}^{1}A_{1}(t_{2})^{2}$
    """
    if not raw:
        return r"\textit{(none above cutoff)}"
    parts = []
    for chunk in raw.split('+'):
        m = re.fullmatch(r'(?P<w>\d+(?:\.\d+)?)\*(?P<spin>\d+)(?P<sym>A1|A2|E|T1|T2)\(t2\)\^(?P<n>\d+)', chunk)
        if not m:
            # Fallback: escape safely
            parts.append(re.sub(r'([_&#%$])', r'\\\1', chunk))
            continue
        w = m['w']
        spin = m['spin']
        sym = m['sym']
        n = m['n']
        # add subscript to trailing digit in the irrep (A1->A_{1}, T2->T_{2})
        if sym.endswith(('1','2')):
            sym_tex = f"{sym[:-1]}_{{{sym[-1]}}}"
        else:
            sym_tex = sym
        parts.append(fr"{w}\,{{}}^{{{spin}}}{sym_tex}(t_{{2}})^{{{n}}}")
    return r"$" + r" + ".join(parts) + r"$"

def _fmt_terms_L(raw: str) -> str:
    """
    Convert strings like '0.41*3D+0.21*1S' into:
    $0.41\,{}^{3}\mathrm{D} + 0.21\,{}^{1}\mathrm{S}$
    """
    if not raw:
        return r"\textit{(none above cutoff)}"
    parts = []
    for chunk in raw.split('+'):
        m = re.fullmatch(r'(?P<w>\d+(?:\.\d+)?)\*(?P<spin>\d+)(?P<L>[SPDFG])', chunk)
        if not m:
            parts.append(re.sub(r'([_&#%$])', r'\\\1', chunk))
            continue
        w = m['w']
        spin = m['spin']
        L = m['L']
        parts.append(fr"{w}\,{{}}^{{{spin}}}\mathrm{{{L}}}")
    return r"$" + r" + ".join(parts) + r"$"

from string import Template
from datetime import datetime
import numpy as np
from pathlib import Path

def print_latex(
    energies_rel,
    t2g_terms,
    L_terms,
    *,
    title=r"Initial-state symmetry analysis (single-ion model)",
    material_name=None,
    decimals_energy=6,
    write_to=None,  # Optional[str]
):
    if len(energies_rel) != len(t2g_terms) or len(energies_rel) != len(L_terms):
        raise ValueError("Lengths of energies_rel, t2g_terms, and L_terms must match.")

    # sort by energy (ascending)
    order = np.argsort(np.asarray(energies_rel))
    E = [float(np.round(energies_rel[i], decimals_energy)) for i in order]
    T = [t2g_terms[i] for i in order]
    L = [L_terms[i] for i in order]

    # rows
    rows_t = "\n".join(
        f"{i} & {E[i]:.{decimals_energy}f} & {_fmt_terms_t2g(T[i])} \\\\"
        for i in range(len(E))
    )
    rows_L = "\n".join(
        f"{i} & {E[i]:.{decimals_energy}f} & {_fmt_terms_L(L[i])} \\\\"
        for i in range(len(E))
    )

    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    colspec = f"+1.{decimals_energy}"

    from string import Template

    from string import Template

    tpl = Template(r"""
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{longtable}
\usepackage{amsmath}

\title{$title}
\author{Generated on $today}
\date{$material}

\begin{document}
\maketitle

\section*{Cubic-basis decomposition ($$t_2$$/$$e_g$$)}
Energies are shown relative to the ground state.

\begin{longtable}{|r|r|p{0.70\textwidth}|}
\hline
\# & Energy (eV) & Symmetry content \\
\hline
$rows_t
\end{longtable}

\section*{L-term decomposition}
Same eigenstates, shown with spectroscopic L terms.

\begin{longtable}{|r|r|p{0.70\textwidth}|}
\hline
\# & Energy (eV) & Symmetry content \\
\hline
$rows_L
\end{longtable}

\end{document}
""")






    doc = tpl.substitute(
        title=title,
        today=today,
        material=(material_name or ""),
        dec=decimals_energy,
        colspec=colspec,
        rows_t=rows_t,
        rows_L=rows_L,
    )

    if write_to:
        Path(write_to).write_text(doc, encoding="utf-8")
        print(f"LaTeX written to: {write_to}")
        return None
    return doc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-py", required=True, help="Path to material config (e.g., NiCl2_config.py)")
    ap.add_argument("--params", required=True, help="Path to JSON file containing the parameter set")
    args = ap.parse_args()

    cfg = load_config_module(args.config_py)

    # --- Required objects from config ---
    # rixs_funV: an instance of RIXS_runner configured for this material
    # rixs_ref: experimental reference map (2D numpy array)
    try:
        greedy_bounds = cfg.greedy_bounds
        true_values = cfg.true_values              # dict: {param: value}
        rixs_funV = cfg.rixs_funV

    except AttributeError as e:
        raise RuntimeError(
            "Config must define 'greedy_bounds' (dict), 'true_values' (dict), and 'funGreedy' (callable)."
        ) from e
    if not isinstance(greedy_bounds, dict) or not isinstance(true_values, dict):
        raise TypeError("'greedy_bounds' and 'true_values' must both be dicts.")

    # --- Load parameter set from JSON ---
    params = load_params_json(args.params)

    # --- Key validation against greedy_bounds ---
    ordered_keys = list(greedy_bounds.keys())  # preserves intended argument order
    bounds_set = set(ordered_keys)
    json_set = set(params.keys())

    # (c) Any mismatching (extra) keys -> error
    extra = sorted(json_set - bounds_set)
    if extra:
        raise KeyError(
            "JSON contains keys not present in greedy_bounds: "
            + ", ".join(extra)
        )

    # (b) Missing keys -> warn and fill from true_values
    missing = sorted(bounds_set - json_set)
    if missing:
        # ensure true_values supplies them all
        missing_missing = [k for k in missing if k not in true_values]
        if missing_missing:
            raise KeyError(
                "Missing keys not found in true_values for backfill: "
                + ", ".join(missing_missing)
            )
        warnings.warn(
            "Missing keys in JSON were filled from true_values: "
            + ", ".join(missing),
            RuntimeWarning,
        )
        for k in missing:
            params[k] = true_values[k]

    # (a) Keys now match; build ordered positional args for funGreedy
    ordered_args = [params[k] for k in ordered_keys]

    # --- Call funGreedy ---
    #result = funGreedy(*ordered_args)

    # Print results
    for k in ordered_keys:
        print(f"  {k:>8s} = {params[k]}")
    print(f"\nAnnotation:\n")

    annstr=['A1','A2','E','T1','T2']
    Lstr=['S','P','D','F','G']

    v_norb=10
    c_norb=6
    basis_i = get_fock_bin_by_N(v_norb, cfg.n_occu, c_norb, c_norb)
    basis_n = get_fock_bin_by_N(v_norb, cfg.n_occu+1, c_norb, c_norb - 1)

    anstringtab_t=[]
    anstringtab_L=[]

    out = rixs_funV({**params,**(cfg.fixed_params_greedy)},ed_only=True)
    eval_i, eval_n, trans_op, evec_i, evec_n = out

    evs_i,Ops_i, annots_i,sumIrreps0=constructMockH(1.0,0.1*np.sqrt(2),0.01*np.sqrt(3),'d','p',cfg.n_occu,0,t_or_L='t');
    weight_i=np.array([[[getWeight(annots_i,evs_i,evec_i,S2Q=i,AnnotQ=k,t2gQ=j+1)for k in range(5)] for j in range(1,6)] for i in [0,2,4]])
    weight_i=weight_i.reshape([75,-1])
    wranks=np.flip(np.argsort(weight_i.T,axis=1),axis=1)
    wvals=np.flip(np.sort(weight_i.T,axis=1),axis=1)

    for j in range(len(basis_i)):
        annot_string=''
        for i in range(len(wvals[0])):
            weight=np.round(wvals[j][i],decimals=2)
            if weight >= 0.03:
                if (i > 0):
                    annot_string = annot_string+'+'
                annot_string=annot_string+str(weight)+'*'+str(1+2*(wranks[j][i]//25))+annstr[wranks[j][i]%5]+'(t2)^'+str(2+(wranks[j][i]%25)//5)
        anstringtab_t.append(annot_string)

    #print([[np.round(eval_i[j]-eval_i[0],decimals=6),anstringtab[j]] for j in range(len(basis_i))])

    evs_i,Ops_i, annots_i,sumIrreps0=constructMockH(1.0,0.1*np.sqrt(2),0.01*np.sqrt(3),'d','p',cfg.n_occu,0,t_or_L='L');

    weight_i=np.array([[getWeightL(annots_i,evs_i,evec_i,S2Q=i,L2Q=j) for j in range(0,5)] for i in [0,2,4]])
    weight_i=weight_i.reshape([15,-1])
    wranks=np.flip(np.argsort(weight_i.T,axis=1),axis=1)
    wvals=np.flip(np.sort(weight_i.T,axis=1),axis=1)


    for j in range(len(basis_i)):
        annot_string=''
        for i in range(len(wvals[0])):
            weight=np.round(wvals[j][i],decimals=2)
            if weight >= 0.03:
                if (i > 0):
                    annot_string = annot_string+'+'
                annot_string=annot_string+str(weight)+'*'+str(1+2*(wranks[j][i]//5))+Lstr[(wranks[j][i]%5)]
        anstringtab_L.append(annot_string)

    #print([[np.round(eval_i[j]-eval_i[0],decimals=6),anstringtab[j]] for j in range(len(basis_i))])
    energies_rel = [float(np.round(eval_i[j]-eval_i[0],decimals=6)) for j in range(len(basis_i))]
    print_latex(
        energies_rel,
        anstringtab_t,
        anstringtab_L,
        title=r"Initial-state symmetry analysis (single-ion, octahedral)",
        material_name=getattr(cfg, "material_name", None) if hasattr(cfg, "material_name") else None,
        decimals_energy=6,
        write_to="annotation_tables.tex",  # set to None to get the LaTeX as a return string instead
    )

if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    sys.exit(main())