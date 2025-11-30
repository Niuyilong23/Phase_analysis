# @Time    : 2025/8/11 15:03
# @Author  : JunFei Cai
# @File    : ce_mc.py.py

#!/usr/bin/env python3
"""icet + Ewald (pymatgen) example: fit a cluster expansion to electrostatic (Ewald) energies
and run canonical Monte Carlo to probe finite-temperature ordering on the Mn sublattice.

Assumptions (from the user):
 - Primitive structure is in POSCAR_522 (VASP POSCAR) in the same folder as this script.
 - The primitive cell contains three kinds of sites:
     * Mn-sites: these should be allowed to be occupied by Li(+1), Ni(+2), Mn(+4)
     * Li-sites: fixed Li(+1)
     * O-sites: fixed O(-2)
 - We will use the Ewald electrostatic energy (via pymatgen) as the "reference" energy
   to fit the cluster expansion. This is an approximation but lets us build a CE and
   run MC to explore finite-temperature ordering driven by long-range electrostatics.
 - This is a worked example (toy model). For production-quality CE you would replace
   the Ewald energies by DFT energies, and carefully validate the CE (CV, hold-out set, etc.).

Requirements (install with pip if needed):
    pip install icet ase pymatgen numpy scipy scikit-learn matplotlib

Run:
    python icet_ewald_ce_mc_example.py

Output:
 - Prints training / optimization summary to stdout.
 - Saves two figures: 'sro_vs_temperature.png' and 'energy_vs_temperature.png'.

Notes and caveats:
 - The script is intended as a clear, commented example. You should adapt cutoffs, the
   number of training structures, CE complexity, MC cell size and MC length to your system.
 - Units: pymatgen.EwaldSummation returns energy in eV (if species oxidation states are set to integer charges).
 - The CE parameters and MC are purely driven by the electrostatic model here.
"""

import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt

# ASE for reading POSCAR and handling Atoms objects
from ase.io import read
from ase.build import make_supercell

# pymatgen for Ewald sums (needs oxidation states on sites)
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.core import Species

# icet: cluster expansion toolbox
from icet import ClusterSpace, StructureContainer, ClusterExpansion, Optimizer
# MCHAMMER ( MC module of icet)
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import CanonicalEnsemble
from mchammer.observers import ClusterCountObserver, ClusterExpansionObserver

# ---- USER PARAMETERS (edit if you want different behavior) ----
POSCAR_PATH = 'POSCAR_522'              # path to the user's uploaded POSCAR (primitive cell)
CUT_OFFS = [5.0]                        # cluster-space cutoffs (Å) -- tune for your system
N_TRAIN = 120                           # number of random structures used for fitting (toy example)
TRAIN_SUPERCELL_REPEAT = (2, 2, 2)      # repeat of primitive cell used to generate training structures
MC_TARGET_MN_SITES = 128                # aim for at least this many Mn-sub-lattice sites in MC supercell
MC_SUPERCELL_REPEAT_MIN = 2             # minimum repeat for MC supercell (will be auto-adapted)
MC_SWEEPS = 150                         # Monte Carlo sweeps (1 sweep = Ntrial = number of sites)
TEMPERATURES = [50, 100, 200, 300, 500, 800]  # temperatures (K) to probe in MC

# Oxidation states (used by EwaldSummation)
OXIDATION_STATES = {'Li': +1, 'Ni': +2, 'Mn': +4, 'O': -2}

# Mn-sub-lattice target fractions (the user requested): Li:0.2, Ni:0.2, Mn:0.6 on Mn sites
MN_SUBL_FRACS = {'Li': 0.2, 'Ni': 0.2, 'Mn': 0.6}

# Random seed for reproducibility
SEED = 12345
np.random.seed(SEED)
random.seed(SEED)

# ------------------------------------------------------------------
# Helper functions
def read_primitive(path):
    """Read the primitive cell POSCAR using ASE."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"POSCAR not found: {path}")
    prim = read(path, format='vasp')
    prim.set_pbc((True, True, True))
    return prim

def build_site_allowed_species_list(atoms):
    """Create per-site allowed species lists for ClusterSpace:
       - Mn sites: ['Li', 'Ni', 'Mn']
       - Li sites: ['Li']
       - O sites: ['O']
       - Other (unexpected): keep the native symbol as the only allowed species
    Returns a list of lists (length == number of sites in primitive cell).
    """
    allowed = []
    for a in atoms:
        sym = a.symbol
        if sym == 'Mn':
            allowed.append(['Li', 'Ni', 'Mn'])
        elif sym == 'Li':
            allowed.append(['Li'])
        elif sym == 'O':
            allowed.append(['O'])
        else:
            # fallback: single species
            allowed.append([sym])
    return allowed

def random_decorate_mn_subs(supercell, mn_indices, fracs, rng=None):
    """Decorate the given ASE supercell in-place by randomizing occupations on the
    mn_indices according to fractions in fracs (dictionary with keys 'Li','Ni','Mn').
    The function guarantees integer counts by rounding; small deviations may occur due to rounding.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(mn_indices)
    n_Li = int(round(fracs['Li'] * n))
    n_Ni = int(round(fracs['Ni'] * n))
    n_Mn = n - n_Li - n_Ni
    if n_Mn < 0:
        # adjust in pathological rounding cases
        n_Mn = 0
        if n_Ni > n:
            n_Ni = max(0, n - n_Li)
    occ = ['Li'] * n_Li + ['Ni'] * n_Ni + ['Mn'] * n_Mn
    if len(occ) != n:
        # fix by trimming/appending Mn (should be rare)
        occ = (occ + ['Mn'] * n)[:n]
    rng.shuffle(occ)
    # assign into ASE Atoms symbols (mutates supercell in place)
    symbols = list(supercell.get_chemical_symbols())
    for idx, val in zip(mn_indices, occ):
        symbols[idx] = val
    supercell.set_chemical_symbols(symbols)

def ewald_energy_from_ase(atoms, ox_states_map):
    """Compute the Ewald (electrostatic) energy for a structure (ASE Atoms) by:
       - converting to pymatgen Structure,
       - assigning oxidation states (= ionic charges) via pymatgen Species,
       - running EwaldSummation and returning total energy (in eV).
       Note: EwaldSummation assumes integer charges and proper periodic cell.
    """
    s = AseAtomsAdaptor.get_structure(atoms, primitive=False)
    # assign oxidation states (site-by-site)
    new_sites = []
    for site in s:
        sym = site.specie.symbol if hasattr(site.specie, 'symbol') else str(site.specie)
        if sym not in ox_states_map:
            raise ValueError(f"Missing oxidation state for element: {sym}")
        sp = Species(sym, ox_states_map[sym])
        new_sites.append(sp)
    # replace species on structure (do a shallow copy to avoid mutating original)
    s2 = s.copy()
    for i, sp in enumerate(new_sites):
        s2[i] = sp
    ewald = EwaldSummation(s2)
    return ewald.total_energy

def find_mn_indices(atoms):
    """Return indices of sites whose element symbol is 'Mn' in an ASE Atoms object."""
    return [i for i, a in enumerate(atoms) if a.symbol == 'Mn']

def pick_supercell_repeat_for_mc(prim_atoms, min_mn_sites):
    """Choose cubic repeat (n,n,n) for MC supercell so that the total number of Mn sites
       >= min_mn_sites. Returns tuple (n,n,n)."""
    n_prim_atoms = len(prim_atoms)
    mn_in_prim = sum(1 for a in prim_atoms if a.symbol == 'Mn')
    if mn_in_prim == 0:
        raise ValueError('No Mn sites found in primitive cell.')
    # minimal cubic repeat estimate
    n = math.ceil((min_mn_sites / mn_in_prim) ** (1/3))
    n = max(n, MC_SUPERCELL_REPEAT_MIN)
    return (n, n, n)

# ------------------------------------------------------------------
def main():
    print('Reading primitive cell...')
    prim = read_primitive(POSCAR_PATH)
    print(f'Primitive cell: {len(prim)} atoms')

    # build cluster space (allowed species per site)
    print('Building ClusterSpace (defining which species can occupy each site)...')
    allowed_symbols = build_site_allowed_species_list(prim)
    print('Allowed species per site (primitive cell):', allowed_symbols)
    cs = ClusterSpace(structure=prim, cutoffs=CUT_OFFS, chemical_symbols=allowed_symbols)
    print('ClusterSpace created. Number of orbits (clusters):', len(cs.orbit_list))

    # Prepare structure container to hold training structures + energies
    sc = StructureContainer(cluster_space=cs)

    # Create a training supercell template and find Mn sublattice indices within it
    train_sc = prim.repeat(TRAIN_SUPERCELL_REPEAT)
    mn_indices_train = find_mn_indices(train_sc)
    print(f'Using training supercell repeat {TRAIN_SUPERCELL_REPEAT} -> {len(train_sc)} atoms, Mn sites: {len(mn_indices_train)}')

    # Generate N_TRAIN randomized structures at the target Mn-sub-lattice composition and compute Ewald energies
    print(f'Generating {N_TRAIN} randomized training structures (Ewald energies) ...')
    for i in range(N_TRAIN):
        s = train_sc.copy()
        # randomize occupations on Mn-sublattice according to requested fractions
        random_decorate_mn_subs(s, mn_indices_train, MN_SUBL_FRACS, rng=np.random.default_rng())
        # compute Ewald energy for this decorated structure
        E = ewald_energy_from_ase(s, OXIDATION_STATES)
        # Add structure & energy to the StructureContainer
        sc.add_structure(s, properties={'energy': float(E)})
        if (i + 1) % 20 == 0:
            print(f'  - generated {i+1}/{N_TRAIN} structures')

    print('Training set complete. Number of structures in StructureContainer:', len(sc.structures))

    # Fit CE using icet Optimizer (default optimizer + settings). This will print progress.
    print('Constructing fit data and running Optimizer to obtain ECIs...')
    fit_data = sc.get_fit_data()   # returns list-of-properties for optimizer
    opt = Optimizer(fit_data)
    opt.train()
    # Build ClusterExpansion from the optimized parameters
    ce = ClusterExpansion(cluster_space=cs, parameters=opt.parameters)
    print('Fitted cluster expansion:')
    print(ce)
    # NOTE: at this point you may want to inspect cross-validation, parameter sparsity, etc.

    # ------------------------------------------------------------------
    # Monte Carlo sampling in canonical ensemble at different temperatures
    # ------------------------------------------------------------------
    print('Preparing Monte Carlo supercell...')
    mc_repeat = pick_supercell_repeat_for_mc(prim, MC_TARGET_MN_SITES)
    mc_cell = prim.repeat(mc_repeat)
    print(f'MC supercell repeat {mc_repeat} -> {len(mc_cell)} atoms')

    # Identify Mn indices in MC cell and decorate initial configuration (random)
    mn_indices_mc = find_mn_indices(mc_cell)
    random_decorate_mn_subs(mc_cell, mn_indices_mc, MN_SUBL_FRACS, rng=np.random.default_rng(SEED+1))

    # Build a cluster expansion calculator (fast local energy changes)
    calc = ClusterExpansionCalculator(mc_cell, ce)

    # We'll collect simple observables vs temperature
    temps = TEMPERATURES
    energy_vs_T = []
    sro_Li_Mn_vs_T = []     # SRO Li - Mn on Mn-sub-lattice (first pair shell)
    sro_Ni_Mn_vs_T = []      # SRO Ni - Mn
    sro_Li_Ni_vs_T = []      # SRO Li - Ni

    # Prepare a helper cluster-count observer (we use it to compute SRO from snapshots after MC)
    cc_observer = ClusterCountObserver(cluster_space=cs, structure=mc_cell, interval=len(mc_cell))

    for T in temps:
        print(f'Running canonical Monte Carlo at T = {T} K ... (this may take a while)')
        mc = CanonicalEnsemble(structure=mc_cell.copy(), calculator=calc, temperature=float(T),
                               dc_filename=None)  # no disk data container, in-memory only
        # Attach observers to record energy + cluster counts periodically
        ce_obs = ClusterExpansionObserver(ce, interval=len(mc_cell))   # energy-like quantity
        mc.attach_observer(ce_obs)
        mc.attach_observer(cc_observer)

        n_steps = int(len(mc_cell) * MC_SWEEPS)  # number of trial swaps
        mc.run(n_steps)

        # after the run: take the current structure (mc.structure) and evaluate
        final_structure = mc.structure  # ASE Atoms object
        # energy from CE observer (may be per-site or total depending on CE scaling) -- we record the value
        E_pred = ce_obs.get_observable(final_structure)
        # If get_observable returns a dict (sometimes), try to convert to float sensibly:
        if isinstance(E_pred, dict):
            # pick first numeric value
            E_val = list(E_pred.values())[0]
        else:
            E_val = float(E_pred)

        energy_vs_T.append(E_val / len(final_structure))  # energy per atom (rough normalization)

        # compute cluster counts (pairs) for the final structure and derive SRO for first pair orbit
        df_counts = cc_observer.get_cluster_counts(final_structure)  # pandas DataFrame
        # restrict to pair (order == 2) orbits and pick the first orbit index (nearest neighbor shell)
        pair_orbits = sorted(df_counts.loc[df_counts['order'] == 2]['orbit_index'].unique().tolist())
        if len(pair_orbits) == 0:
            print('  Warning: no pair orbits found in cluster counts. Skipping SRO calculation.')
            sro_Li_Mn_vs_T.append(np.nan)
            sro_Ni_Mn_vs_T.append(np.nan)
            sro_Li_Ni_vs_T.append(np.nan)
            continue

        first_orbit = pair_orbits[0]
        orbit_df = df_counts.loc[df_counts['orbit_index'] == first_orbit]

        # get symbol counts on the active sublattice (the one that has Li,Ni,Mn)
        # find which sublattice contains Mn by looking into cluster_space sublattices
        sublattices = cs.get_sublattices(mc_cell)
        active_subl = None
        for sub in sublattices:
            allowed = sub.chemical_symbols
            if 'Mn' in allowed and len(set(allowed).intersection({'Li', 'Ni', 'Mn'})) > 0:
                active_subl = sub
                break
        if active_subl is None:
            print('  Warning: could not locate active Mn sublattice in ClusterSpace; SRO skipped.')
            sro_Li_Mn_vs_T.append(np.nan)
            sro_Ni_Mn_vs_T.append(np.nan)
            sro_Li_Ni_vs_T.append(np.nan)
            continue

        # count atoms of each symbol on that sublattice
        occupation = np.array(final_structure.get_chemical_symbols())
        indices = active_subl.indices
        symbol_counts = {}
        for sym in active_subl.chemical_symbols:
            symbol_counts[sym] = occupation[indices].tolist().count(sym)
        N_total = sum(symbol_counts.values())
        total_count = orbit_df['cluster_count'].sum()  # number of clusters for that orbit (in supercell)

        def compute_pair_count(a, b):
            """Return number of a-b pairs in this orbit as described by orbit_df."""
            A_B = 0
            for i, row in orbit_df.iterrows():
                occu = row['occupation']  # list-like, e.g. ['Li', 'Mn'] depending on cluster
                # cluster_count is how many such clusters appear in the supercell
                if a in occu and b in occu:
                    A_B += row['cluster_count']
            return A_B

        # compute SRO for each pair using Warren-Cowley formula:
        # alpha_{A-B} = 1 - P(B|A)/c_B
        # where P(B|A) = (n_{A-B} / N_A) / Z and Z = 2*total_count / N_total
        def compute_alpha(a, b):
            if a not in symbol_counts or b not in symbol_counts:
                return np.nan
            N_A = symbol_counts[a]
            N_B = symbol_counts[b]
            if N_A == 0 or N_B == 0 or total_count == 0:
                return np.nan
            n_AB = compute_pair_count(a, b)
            Z = 2.0 * total_count / float(N_total)   # average neighbors per site for this shell
            c_B = float(N_B) / float(N_total)
            P_B_given_A = (n_AB / float(N_A)) / Z
            alpha = 1.0 - (P_B_given_A / c_B) if c_B > 0 else np.nan
            return alpha

        sro_Li_Mn_vs_T.append(compute_alpha('Li', 'Mn'))
        sro_Ni_Mn_vs_T.append(compute_alpha('Ni', 'Mn'))
        sro_Li_Ni_vs_T.append(compute_alpha('Li', 'Ni'))

        print(f'  T={T}K: energy/atom (CE) = {E_val/len(final_structure):.5f} eV, SRO(Li-Mn)={sro_Li_Mn_vs_T[-1]:.4f}')

    # ------------------------------------------------------------------
    # Plot results
    # ------------------------------------------------------------------
    plt.figure(figsize=(6,4))
    plt.plot(temps, energy_vs_T, marker='o')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Energy per atom (CE prediction, eV)')
    plt.title('Energy vs Temperature (CE-driven by Ewald energies)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('energy_vs_temperature.png', dpi=300)
    print('Saved: energy_vs_temperature.png')

    plt.figure(figsize=(6,4))
    plt.plot(temps, sro_Li_Mn_vs_T, marker='o', label='Li-Mn')
    plt.plot(temps, sro_Ni_Mn_vs_T, marker='s', label='Ni-Mn')
    plt.plot(temps, sro_Li_Ni_vs_T, marker='^', label='Li-Ni')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Warren-Cowley SRO (first shell)')
    plt.title('Short-range order vs Temperature (first neighbor shell)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('sro_vs_temperature.png', dpi=300)
    print('Saved: sro_vs_temperature.png')

    print('Done. Note: this is an illustrative workflow. For production-quality results, '
          'replace Ewald energies by DFT references and validate CE thoroughly.')

if __name__ == '__main__':
    main()

