"""
qmri_utils.py — shared utilities.
Functions here are the source of truth.
Functions of Quantum functional similarity: taken from the Hoa's verify_rewrite_pairs.ipynb


Used by: 
qmri_compute.ipynb, for qMRI computation 
rewrite_demo.ipynb, for Rewrite Equivalence

"""

from __future__ import annotations

import ast
import contextlib
import io
import math
import re
from collections import Counter
from typing import Optional

import numpy as np
try:
    import zss as _zss_mod
except ImportError:
    _zss_mod = None

from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Statevector, random_unitary, state_fidelity, Operator, process_fidelity as _process_fidelity
from qiskit_aer import AerSimulator

# Single simulator instance reused across all TVD calls
_aer_sim = AerSimulator()


# ── Argument resolution ────────────────────────────────────────────────────────
#
# Two-layer system for supplying call arguments to entry_point functions:
#
# Layer 1 — TASK_EXTRA_ARGS  (what args each task needs, declared statically)
#   Maps task_id → {kwarg_name: value_or_token}.
#   Values that are JSON-safe primitives (int, float, list[float]) are stored
#   directly — resolve_arg returns them unchanged.
#   Values that are complex Python objects (Operator, QuantumCircuit, nx.Graph)
#   are stored as double-underscore token strings, e.g. '__random_unitary_4__'.
#   Reason: TASK_EXTRA_ARGS is JSON-serialised inside exc_arr records
#   (field: kwargs_tokens), so live objects cannot be stored here.
#
# Layer 2 — resolve_arg  (how each token becomes a live Python object, at runtime)
#   Called by build_exc_record:  kwargs = {k: resolve_arg(v) for k, v in kwargs_tokens}
#   Primitives pass through as-is; token strings are reconstructed into objects.
#   Add a new branch here whenever a new token is introduced in TASK_EXTRA_ARGS.
#
# Usage order:
#   TASK_EXTRA_ARGS  →  build_exc_record(kwargs_tokens=...)
#                    →  resolve_arg(token)  →  actual Python object
#                    →  exec_code(entry_point, kwargs, code)

TASK_EXTRA_ARGS: dict = {
    # ── QHE tasks that need explicit call arguments ──────────────────────────
    # Primitive value — resolve_arg returns it unchanged.
    'qiskitHumanEval/65':  {'n': 3},
    # Token — resolve_arg builds a 4-qubit random Operator at runtime.
    # MUST resolve once and reuse; random_unitary differs on every call.
    'qiskitHumanEval/117': {'unitary': '__random_unitary_4__'},

    # ── QB tasks with Qiskit Parameter objects ───────────────────────────────
    # Token — resolve_arg builds ParameterVector at runtime.
    '39': {'parameters': '__parametervector_2__'},
    '40': {'parameters': '__parametervector_8__'},

    # ── QB tasks patched for Qiskit 1.x API compatibility (§13-C) ───────────
    # Graph token — resolve_arg builds the 6-edge bipartite nx.Graph.
    '04': {'G': '__nx_graph_qaoa__', 'beta': [0.5]*5, 'gamma': [0.5]*5},
    # QuantumCircuit token — resolve_arg builds a 1-qubit H circuit.
    '06': {'unknown_state': '__qc_1q_h__'},
    # Primitive values — resolve_arg returns them unchanged.
    '29': {'alice': 1, 'bob': 1},
    '41': {'param': [0.5]*6},
    '42': {'theta': 0.5, 'phi': 0.5, 'lam': 0.5},
}


def resolve_arg(token):
    """
    Convert a TASK_EXTRA_ARGS value into a live Python object.

    Primitive values (int, float, list, bool, None) pass through unchanged.
    Token strings (__name__) are reconstructed into objects that cannot be
    stored in JSON — called once per task inside build_exc_record so the
    same objects are reused across all 50 param samples for that task.
    """
    if token == '__random_unitary_4__':
        return random_unitary(4)
    if token == '__parametervector_2__':
        from qiskit.circuit import ParameterVector
        return ParameterVector('p', 2)
    if token == '__parametervector_8__':
        from qiskit.circuit import ParameterVector
        return ParameterVector('p', 8)
    if token == '__random_beta_5__':
        import numpy as np
        return list(np.random.uniform(0, 3.14159, 5))
    if token == '__random_gamma_5__':
        import numpy as np
        return list(np.random.uniform(0, 3.14159, 5))
    if token == '__random_6_floats__':
        import numpy as np
        return list(np.random.uniform(0, 2 * 3.14159, 6))
    if token == '__random_8_floats__':
        import numpy as np
        return list(np.random.uniform(0, 2 * 3.14159, 8))
    if token == '__nx_graph_qaoa__':
        import networkx as nx
        G = nx.Graph()
        G.add_edges_from([[0, 3], [0, 4], [1, 3], [1, 4], [2, 3], [2, 4]])
        return G
    if token == '__qc_1q_h__':
        qc = QuantumCircuit(1)
        qc.h(0)
        return qc
    return token


# ── Code reconstruction ────────────────────────────────────────────────────────

def reconstruct_full_code(prompt: str, raw_body: str) -> str:
    """
    Combine prompt (imports + function signature + docstring) with a StarCoder
    function body completion.

    StarCoder returns only the body; the prompt ends with the closing docstring
    quote. We prepend four spaces so the first body line is correctly indented
    inside the function.

    Example:
        prompt   = "def f(n):\\n    \\"\\"\\"docstring.\\"\\"\\"\\""
        raw_body = "qc = QuantumCircuit(n)\\n    return qc"
        result   = "def f(n):\\n    \\"\\"\\"docstring.\\"\\"\\"\\n    qc = QuantumCircuit(n)\\n    return qc"
    """
    return prompt + '\n    ' + raw_body


def extract_body(code: str) -> tuple:
    """
    Split a complete Python function into (prefix, body).

    prefix = import lines + def line  (no docstring)
    body   = function body after the docstring (4-space indented, stripped)

    When no 'def' line is found (e.g. QHE canonical_solution which is already
    body-only) returns ('', code.strip()).

    QB canonical_solution includes the full function definition; QHE
    canonical_solution is body-only. Both return body-only from [1].
    """
    lines = code.splitlines()
    def_idx = next(
        (i for i, ln in enumerate(lines)
         if ln.lstrip().startswith('def ') and not ln[:1].isspace()),
        None,
    )
    if def_idx is None:
        return '', code.strip()
    prefix = '\n'.join(lines[:def_idx + 1])
    rest   = '\n'.join(lines[def_idx + 1:])
    try:
        tree = ast.parse('def _f():\n' + rest)
        node = tree.body[0]
        if (node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            rest = '\n'.join(rest.splitlines()[node.body[0].end_lineno - 1:])
    except SyntaxError:
        pass
    return prefix, rest.strip()


# ── Circuit execution ──────────────────────────────────────────────────────────

def exec_code(
    entry_point: str,
    kwargs: dict,
    full_code: str,
) -> tuple[Optional[QuantumCircuit], Optional[str]]:
    """
    Execute full_code in an isolated namespace, call entry_point(**kwargs).

    Returns
    -------
    (QuantumCircuit, None)  on success
    (None, error_msg)       on any failure
    """
    ns: dict = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            exec(full_code, ns, ns)  # noqa: S102
        fn = ns.get(entry_point)
        if fn is None:
            return None, f"entry_point '{entry_point}' not found"
        result = fn(**kwargs)
        qc = result[0] if isinstance(result, tuple) else result
        if not isinstance(qc, QuantumCircuit):
            return None, f"returned {type(qc).__name__}, expected QuantumCircuit"
        return qc, None
    except BaseException as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


# ── AST similarity ─────────────────────────────────────────────────────────────

def _ast_to_zss(node: ast.AST):
    """Recursively convert a Python AST node to a zss.Node tree (label = node class name)."""
    z = _zss_mod.Node(type(node).__name__)
    for child in ast.iter_child_nodes(node):
        z.addkid(_ast_to_zss(child))
    return z


def compute_ast_sim(code1: str, code2: str) -> Optional[float]:
    """
    Normalised tree edit distance (Zhang-Shasha) on Python ASTs.

    sim = 1 - TED(T1, T2) / (|T1| + |T2|)

    Normalisation by |T1|+|T2| guarantees sim ∈ [0, 1] for unit edit costs
    (insert=delete=relabel=1), since the maximum TED equals |T1|+|T2|
    (delete all of T1, insert all of T2).  Symmetric.

    Clone type interpretation (Roy et al. 2009):
      - Type-2 (renamed identifiers only)  → sim ≈ 1.0
      - Type-3 (added/removed statements)  → sim ≈ 0.5–0.9
      - Unrelated code                      → sim < 0.5

    Returns None if either code string has a SyntaxError.
    """
    def _parse(code: str):
        try:
            return ast.parse(code)
        except SyntaxError:
            try:
                # Body format: first line 0-indent, rest 4-space (StarCoder / _extract_body).
                # Prepend 'def _f():\n    ' to make the whole block valid Python.
                return ast.parse('def _f():\n    ' + code)
            except SyntaxError:
                return None

    t1_raw = _parse(code1)
    t2_raw = _parse(code2)
    if t1_raw is None or t2_raw is None:
        return None

    n1 = sum(1 for _ in ast.walk(t1_raw))
    n2 = sum(1 for _ in ast.walk(t2_raw))
    denom = n1 + n2
    if denom == 0:
        return 1.0

    if _zss_mod is None:
        return None
    ted = _zss_mod.simple_distance(_ast_to_zss(t1_raw), _ast_to_zss(t2_raw))
    return 1.0 - ted / denom


# ── Quantum functional similarity ──────────────────────────────────────────────

# ── Rewrite equivalence verification ─────────────────────────────────────────
# Following Hoa (verify_rewrite_pairs_8-3Hoa.ipynb):
#   no-measurement  → process_fidelity(Operator(qc1), Operator(qc2))
#                     verifies full unitary (all input states), not just |0⟩
#   with-measurement → separate seeded AerSimulator per circuit, raw counts,
#                      no transpile, no key normalisation
#                      (wrong classical register structure → TVD=1.0, correctly fails)

SHOTS           = 32_768
N_PARAM_SAMPLES = 50
SEED_BASE       = 42


def _tvd_threshold(n_qubits: int, shots: int) -> float:
    return min(0.02, 3 * math.sqrt(2 ** n_qubits / shots))


def circuit_fidelity(qc1: QuantumCircuit, qc2: QuantumCircuit) -> float:
    """Process fidelity via full Operator — verifies all input states."""
    try:
        return float(_process_fidelity(Operator(qc1), Operator(qc2)))
    except Exception:
        raise
    except BaseException as e:
        # Qiskit Rust backend (PyO3) can raise PanicException (BaseException, not
        # Exception) when EquivalenceLibrary hits unreachable code during unitary
        # synthesis. Re-wrap so all callers' `except Exception` blocks catch it.
        raise RuntimeError(f"circuit_fidelity Rust panic: {e}") from e


def tvd(qc1: QuantumCircuit, qc2: QuantumCircuit,
        shots: int = SHOTS) -> float:
    """Sampled TVD with separate seeded simulators; raw counts, no normalisation.
    Transpiles gates only (preserves classical register structure so wrong-register
    rewrites still produce TVD=1.0 as intended)."""
    sim1 = AerSimulator(seed_simulator=SEED_BASE)
    sim2 = AerSimulator(seed_simulator=SEED_BASE + 1)
    try:
        tc1 = transpile(qc1, sim1)
        tc2 = transpile(qc2, sim2)
    except Exception:
        raise
    except BaseException as e:
        raise RuntimeError(f"transpile Rust panic: {e}") from e
    r1 = sim1.run(tc1, shots=shots).result().get_counts()
    r2 = sim2.run(tc2, shots=shots).result().get_counts()
    keys = set(r1) | set(r2)
    return 0.5 * sum(abs(r1.get(k, 0) - r2.get(k, 0)) / shots for k in keys)


def verify_equivalence(orig: QuantumCircuit, rew: QuantumCircuit, task: dict) -> dict:
    """
    Route equivalence check by circuit type (measurement / parameterized / plain).

    Returns a dict with at minimum {'status': 'pass'|'fail'|'error', 'method': ...}.
    """
    n_q      = orig.num_qubits
    is_meas  = task.get('is_measurement', False)
    is_param = task.get('is_parameterized', False) or len(orig.parameters) > 0
    thresh   = _tvd_threshold(n_q, SHOTS)
    try:
        if is_param:
            np.random.seed(SEED_BASE)
            n_fails, worst = 0, None
            metric = 'max_tvd' if is_meas else 'min_fidelity'
            for _ in range(N_PARAM_SAMPLES):
                vals   = np.random.uniform(0, 2 * np.pi, len(orig.parameters))
                bind   = dict(zip(orig.parameters, vals))
                o_bnd  = orig.assign_parameters(bind)
                r_bind = {p: bind[p] for p in rew.parameters if p in bind}
                if not r_bind and len(rew.parameters) == len(orig.parameters):
                    r_bind = dict(zip(sorted(rew.parameters, key=lambda p: p.name), vals))
                r_bnd  = rew.assign_parameters(r_bind)
                if is_meas:
                    d = tvd(o_bnd, r_bnd)
                    n_fails += d >= thresh
                    worst   = d if worst is None else max(worst, d)
                else:
                    fid = circuit_fidelity(o_bnd, r_bnd)
                    n_fails += fid < 1 - 1e-6
                    worst   = fid if worst is None else min(worst, fid)
            return {'status': 'pass' if n_fails == 0 else 'fail',
                    'method': 'parameterized_sampling', 'n_samples': N_PARAM_SAMPLES,
                    'n_fails': n_fails, metric: worst,
                    'threshold': thresh if is_meas else 1 - 1e-6}
        elif is_meas:
            d = tvd(orig, rew)
            return {'status': 'pass' if d < thresh else 'fail', 'method': 'tvd_sampled',
                    'value': d, 'threshold': thresh, 'shots': SHOTS}
        else:
            fid = circuit_fidelity(orig, rew)
            return {'status': 'pass' if fid >= 1 - 1e-6 else 'fail', 'method': 'process_fidelity',
                    'value': fid, 'threshold': 1 - 1e-6}
    except Exception as e:
        return {'status': 'error', 'error_msg': f'{type(e).__name__}: {e}'}
# ── Parameterized circuit functional score ────────────────────────────────────

def compute_f_parameterized(
    qc: QuantumCircuit,
    canonical_qc: QuantumCircuit,
    is_measurement: bool,
    n_samples: int = 50,
    shots: int = 8_192,
    seed: int = 42,
) -> Optional[float]:
    """
    Functional correctness score for a parameterized circuit vs canonical.

    Samples n_samples random θ ∈ [0, 2π]^n, binds both circuits, computes
    per-θ score (1 − TVD for measurement; circuit_fidelity for no-measurement),
    and returns the mean over successful samples.

    Parameter alignment: name-match first; falls back to positional alignment
    by sorted name if no names overlap.

    Returns float ∈ [0, 1], or None when alignment / exec fails for all samples.
    """
    if qc is None or canonical_qc is None:
        return None

    canon_params = sorted(canonical_qc.parameters, key=lambda p: p.name)
    n_params = len(canon_params)
    if n_params == 0:
        try:
            return 1.0 - tvd(qc, canonical_qc, shots=shots) if is_measurement \
                else circuit_fidelity(qc, canonical_qc)
        except Exception:  # noqa: BLE001
            return None

    qc_params = sorted(qc.parameters, key=lambda p: p.name)
    if len(qc_params) != n_params:
        return None

    rng = np.random.default_rng(seed)
    scores: list[float] = []
    for _ in range(n_samples):
        vals = rng.uniform(0, 2 * np.pi, n_params)
        canon_bind = dict(zip(canon_params, vals))
        qc_bind = {p: canon_bind[p] for p in qc_params if p in canon_bind}
        if not qc_bind:
            qc_bind = dict(zip(qc_params, vals))
        try:
            canon_bound = canonical_qc.assign_parameters(canon_bind)
            qc_bound    = qc.assign_parameters(qc_bind)
            score = 1.0 - tvd(qc_bound, canon_bound, shots=shots) if is_measurement \
                else circuit_fidelity(qc_bound, canon_bound)
            scores.append(score)
        except Exception:  # noqa: BLE001
            pass

    return float(np.mean(scores)) if scores else None



# ══════════════════════════════════════════════════════════════════════════════
#  Execution Argument Store (exc_arr)
# ══════════════════════════════════════════════════════════════════════════════
#
# Pre-computes and serialises the kwargs needed to call entry_point(**kwargs)
# for every benchmark task.  Run once in qmri_strategy.ipynb; subsequent
# runs load the saved file instead of re-deriving arguments.
#
# Why it exists
# ─────────────
# Some tasks require non-trivial inputs (random unitary matrices, ParameterVectors,
# graphs).  Without a shared store, each consumer re-generates these randomly,
# producing different circuits on each run and making results non-reproducible.
#
# Consumers (all read exp1_results/exc_arr.json via load_exc_arr + deserialize_kwargs)
# ──────────────────────────────────────────────────────────────────────────────
#   s_func             exec canonical → QC → AerSimulator probability distribution
#   s_struct           exec canonical → QC → gate-dependency DAG
#   rewrite-equivalence exec canonical + rewrite → compare QCs;
#                       param_samples provides shared θ vectors for parameterised tasks
#   qMRI f_ori/f_rew   exec generated code → QC; param_samples replaces the
#                       internal seed=42 sampling in compute_f_parameterized
#
# Error classes (error_class field in each record)
# ─────────────────────────────────────────────────
#   missing_args      TypeError: missing required positional argument
#   no_call_captured  recorder ran; test never called candidate with args
#   wrong_return_type function ran but returned a non-QuantumCircuit object
#   timeout           recorder thread timed out
#   account_required  AccountNotFoundError (needs IBMQ credentials)
#   missing_library   MissingOptionalLibraryError / ModuleNotFoundError
#   circuit_too_wide  CircuitTooWideForTarget
#   exec_error        all other execution failures
# ══════════════════════════════════════════════════════════════════════════════

import inspect
import json
import threading
from pathlib import Path


# ── Serialisation layer ───────────────────────────────────────────────────────

def serialize_kwarg(val) -> dict:
    """Convert one resolved kwarg value to a JSON-safe dict (type-tagged)."""
    if isinstance(val, (int, float, str, bool)) or val is None:
        return {'type': 'primitive', 'data': val}
    if isinstance(val, list):
        return {'type': 'list', 'data': [serialize_kwarg(item) for item in val]}
    if isinstance(val, np.ndarray):
        return {'type': 'ndarray', 'data': val.tolist()}
    try:
        from qiskit.quantum_info import Operator
        if isinstance(val, Operator):
            m = val.data
            return {
                'type': 'unitary',
                'dim': m.shape[0],
                'data': [[float(m[i, j].real), float(m[i, j].imag)]
                         for i in range(m.shape[0]) for j in range(m.shape[1])],
            }
    except Exception:  # noqa: BLE001
        pass
    try:
        from qiskit.circuit import ParameterVector
        if isinstance(val, ParameterVector):
            return {'type': 'parameter_vector', 'name': val.name, 'length': len(val)}
    except Exception:  # noqa: BLE001
        pass
    try:
        import networkx as nx
        if isinstance(val, nx.Graph):
            return {'type': 'nx_graph', 'edges': [list(e) for e in val.edges()]}
    except Exception:  # noqa: BLE001
        pass
    try:
        from qiskit import QuantumCircuit as _QC
        if isinstance(val, _QC):
            from qiskit.qasm2 import dumps
            return {'type': 'qasm2', 'data': dumps(val)}
    except Exception:  # noqa: BLE001
        pass
    try:
        from qiskit.circuit import Gate
        if isinstance(val, Gate):
            # Wrap in a minimal circuit so we can round-trip via QASM
            from qiskit import QuantumCircuit as _QC2
            from qiskit.qasm2 import dumps as _dumps
            _qc = _QC2(val.num_qubits)
            _qc.append(val, range(val.num_qubits))
            return {'type': 'gate_circuit', 'data': _dumps(_qc)}
    except Exception:  # noqa: BLE001
        pass
    return {'type': 'repr', 'data': repr(val)}


def deserialize_kwarg(serial):
    """Restore one serialised kwarg to a real Python object."""
    # Legacy format: inner items of list-of-list were stored as raw lists/primitives
    # without type-tag wrapping (exc_arr built before serialize_kwarg was recursive).
    if isinstance(serial, list):
        return [deserialize_kwarg(item) for item in serial]
    if not isinstance(serial, dict):
        return serial
    t = serial.get('type')
    if t == 'primitive':
        return serial['data']
    if t == 'list':
        return [deserialize_kwarg(item) for item in serial['data']]
    if t == 'ndarray':
        return np.array(serial['data'])
    if t == 'unitary':
        from qiskit.quantum_info import Operator
        dim  = serial['dim']
        flat = serial['data']
        m    = np.array([complex(r, i) for r, i in flat]).reshape(dim, dim)
        return Operator(m)
    if t == 'parameter_vector':
        from qiskit.circuit import ParameterVector
        return ParameterVector(serial['name'], serial['length'])
    if t == 'nx_graph':
        import networkx as nx
        G = nx.Graph()
        G.add_edges_from(serial['edges'])
        return G
    if t == 'qasm2':
        from qiskit.qasm2 import loads, LEGACY_CUSTOM_INSTRUCTIONS
        return loads(serial['data'], custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS)
    if t == 'gate_circuit':
        from qiskit.qasm2 import loads, LEGACY_CUSTOM_INSTRUCTIONS
        return loads(serial['data'], custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS).data[0].operation
    if t == 'repr':
        return serial['data']  # returns the repr string; object cannot be reconstructed
    raise ValueError(f'Cannot deserialise exc_arr type: {t!r}')


def deserialize_kwargs(kwargs_serial: dict) -> dict:
    """Restore all serialised kwargs to real Python objects.
    Primary entry point for all exc_arr consumers."""
    return {k: deserialize_kwarg(v) for k, v in (kwargs_serial or {}).items()}


def serialize_kwargs(kwargs: dict) -> dict:
    """Serialise all resolved kwargs to JSON-safe form. Inverse of deserialize_kwargs."""
    return {k: serialize_kwarg(v) for k, v in (kwargs or {}).items()}


# ── Param samples ─────────────────────────────────────────────────────────────

def generate_param_samples(
    n_params: int,
    n_samples: int = 50,
    seed: int = 42,
) -> list:
    """Pre-generate fixed θ samples for parameterised circuits.
    Returns a list of n_samples float lists, each of length n_params.
    Shared by rewrite-equivalence (multi-θ check) and qMRI (f_ori/f_rew scoring)
    so both steps exercise circuits under identical parameter distributions."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 2 * np.pi, (n_samples, n_params)).tolist()


# ── Error classification ──────────────────────────────────────────────────────

def classify_exc_error(exc_type: str, msg: str) -> str:
    """Map a raw exception type + message to a canonical error_class string."""
    m = (msg or '').lower()
    if exc_type == 'TypeError' and 'missing' in m and 'positional argument' in m:
        return 'missing_args'
    if exc_type == 'TypeError':
        return 'wrong_return_type'
    if exc_type in ('TimeoutExpired', 'TimeoutException'):
        return 'timeout'
    if exc_type == 'AccountNotFoundError':
        return 'account_required'
    if exc_type in ('MissingOptionalLibraryError', 'ModuleNotFoundError'):
        return 'missing_library'
    if exc_type == 'CircuitTooWideForTarget':
        return 'circuit_too_wide'
    return 'exec_error'


# ── Recorder: auto-extract args from QHE test field ──────────────────────────

class _ArgsCaptured(Exception):
    """Raised by recorder to abort the test immediately after capturing args.
    Unwinds the stack cleanly — no zombie threads left behind."""
    def __init__(self, args, kwargs):
        self.captured_args   = args
        self.captured_kwargs = kwargs


def extract_args_recorder(
    code: str,
    test_src: str,
    entry_point: str,
    timeout: float = 8.0,
) -> tuple:
    """
    Auto-extract call args from a QHE task's test field (QHE version of GLOBAL_INPUTS).

    Replaces entry_point with a recorder in the test namespace.  When the test
    calls check(recorder), the recorder intercepts the first invocation, raises
    _ArgsCaptured to abort immediately (no lingering computation), and maps
    positional args to named kwargs via inspect.signature.

    Runs in a daemon thread with a hard timeout so a slow test setup cannot
    block the batch loop indefinitely.

    Returns: (named_kwargs: dict | None, status: str, error_msg: str | None)
      status: 'ok' | 'no_call_captured' | 'entry_point_not_found' | 'timeout' | 'error'
    """
    result: dict = {}

    def _run() -> None:
        ns: dict = {}
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                exec(code, ns, ns)  # noqa: S102

            real_fn = ns.get(entry_point)
            if not callable(real_fn):
                result['status'] = 'entry_point_not_found'
                return

            def recorder(*args, **kwargs):
                raise _ArgsCaptured(args, kwargs)

            test_ns = dict(ns)
            test_ns[entry_point] = recorder

            try:
                with contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(io.StringIO()):
                    exec(test_src, test_ns)  # noqa: S102
                    check_fn = test_ns.get('check')
                    if callable(check_fn):
                        check_fn(recorder)
                result['status'] = 'no_call_captured'
            except _ArgsCaptured as e:
                # Map positional args to named kwargs via the real function's signature
                try:
                    sig        = inspect.signature(real_fn)
                    param_names = list(sig.parameters.keys())
                    named = {param_names[i]: v
                             for i, v in enumerate(e.captured_args)
                             if i < len(param_names)}
                    named.update(e.captured_kwargs)
                except Exception:  # noqa: BLE001
                    named = {f'arg_{i}': v for i, v in enumerate(e.captured_args)}
                result['status'] = 'ok'
                result['kwargs'] = named
        except Exception as exc:  # noqa: BLE001
            result['status']    = 'error'
            result['exc_type']  = type(exc).__name__
            result['error_msg'] = str(exc)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        return None, 'timeout', f'recorder timed out after {timeout}s'

    status = result.get('status', 'error')
    if status == 'ok':
        return result['kwargs'], 'ok', None
    return None, status, result.get('error_msg')


# ── Build one exc_arr record ──────────────────────────────────────────────────

_MEASURE_KW = ('.measure(', '.measure_all(', 'ClassicalRegister(', 'measure_all(')

def _static_is_measurement(code: str) -> bool:
    """Detect measurement from source text without execution."""
    return any(kw in code for kw in _MEASURE_KW)


def build_exc_record(
    task_id: str,
    source: str,
    entry_point: str,
    code: str,
    kwargs_tokens: dict,
    test_src: Optional[str] = None,
    n_param_samples: int = 50,
    param_seed: int = 42,
    framework: str = 'qiskit',
    original_prompt_text: str = '',
    canonical_solution_text: str = '',
) -> dict:
    """
    Build one exc_arr record for a task.

    Flow
    ────
    1. Resolve kwargs_tokens → real objects via resolve_arg.
    2. exec_code(entry_point, kwargs, code) → QuantumCircuit.
    3. If step 2 fails with missing_args AND test_src is provided:
       call extract_args_recorder to capture named kwargs from the test field,
       then retry exec_code with the captured kwargs.
    4. On success: serialise kwargs, extract circuit metadata.
    5. On any failure: record error_class + error_msg, leave exc_status='failed'.

    Redundant benchmark fields (framework, original_prompt_text,
    canonical_solution_text, test) are stored verbatim so downstream consumers
    (rewrite_demo.ipynb) can read from exc_arr without touching the raw datasets.
    """
    record: dict = {
        'task_id':                  task_id,
        'source':                   source,
        'entry_point':              entry_point,
        'framework':                framework,
        'original_prompt_text':     original_prompt_text,
        'canonical_solution_text':  canonical_solution_text,
        'test':                     test_src,
        'exc_status':               'failed',
        'error_class':              None,
        'error_msg':                None,
        'kwargs_tokens':            kwargs_tokens or {},
        'kwargs_serial':            {},
        'n_params':                 None,
        'n_qubits':                 None,
        'is_measurement':           _static_is_measurement(code),
        'is_parameterized':         None,
    }

    # Step 1 – resolve tokens
    try:
        kwargs = {k: resolve_arg(v) for k, v in (kwargs_tokens or {}).items()}
    except Exception as e:  # noqa: BLE001
        record['error_class'] = 'exec_error'
        record['error_msg']   = f'resolve_arg: {e}'
        return record

    # Step 2 – initial execution attempt
    qc, err = exec_code(entry_point, kwargs, code)

    # Step 3 – recorder fallback for QHE missing-args failures
    if qc is None and err and 'missing' in err.lower() and 'positional argument' in err.lower():
        if test_src:
            rec_kwargs, rec_status, rec_err = extract_args_recorder(
                code, test_src, entry_point
            )
            if rec_status == 'ok' and rec_kwargs is not None:
                kwargs = rec_kwargs
                record['kwargs_tokens'] = {k: f'__recorder_{k}__' for k in kwargs}
                qc, err = exec_code(entry_point, kwargs, code)
            else:
                record['error_class'] = rec_status if rec_status != 'error' \
                    else classify_exc_error('', rec_err or '')
                record['error_msg'] = rec_err
                return record
        else:
            record['error_class'] = 'missing_args'
            record['error_msg']   = err
            return record

    # Step 4a – handle remaining failures
    if qc is None:
        exc_type = (err or '').split(':')[0].strip() if err and ':' in err else 'exec_error'
        record['error_class'] = classify_exc_error(exc_type, err or '')
        record['error_msg']   = err
        return record

    # Step 4b – success: serialise and populate metadata
    record['exc_status']       = 'ok'
    record['kwargs_serial']    = serialize_kwargs(kwargs)
    record['n_qubits']         = qc.num_qubits
    record['is_measurement']   = any(d.operation.name == 'measure' for d in qc.data)
    record['is_parameterized'] = len(qc.parameters) > 0
    record['n_params']         = len(qc.parameters) if record['is_parameterized'] else 0
    return record


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_exc_arr(path) -> tuple:
    """Load exc_arr.json; return (records: list, lookup: dict[task_id → record]).
    Returns ([], {}) if the file does not exist or is corrupted (backs up bad file)."""
    p = Path(path)
    if not p.exists():
        return [], {}
    try:
        with open(p, encoding='utf-8') as f:
            records = json.load(f)
        return records, {r['task_id']: r for r in records}
    except json.JSONDecodeError as e:
        # File was truncated mid-write; try to recover the last complete record
        import shutil
        bak = p.with_suffix('.json.bak')
        shutil.copy(p, bak)
        print(f'[load_exc_arr] WARNING: JSON parse error ({e}). Attempting recovery...')
        with open(p, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        for end in range(len(text), 0, -1):
            if text[end - 1] == '}':
                try:
                    records = json.loads(text[:end] + '\n]')
                    print(f'[load_exc_arr] Recovered {len(records)} records. Backup: {bak}')
                    save_exc_arr(p, records)
                    return records, {r['task_id']: r for r in records}
                except json.JSONDecodeError:
                    continue
        print(f'[load_exc_arr] Could not recover. Starting fresh. Backup: {bak}')
        return [], {}


def save_exc_arr(path, records: list) -> None:
    """Persist exc_arr records to JSON atomically (write to tmp then rename)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.json.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    tmp.replace(p)  # atomic on POSIX; avoids partial-write corruption


# ── Pass@k ─────────────────────────────────────────────────────────────────────

def pass_at_k(n: int, c: int, k: int = 1) -> float:
    """Unbiased pass@k estimator for a single task (HumanEval formula).

    Parameters
    ----------
    n : total number of samples generated for the task
    c : number of samples that passed (exec_ok)
    k : k value (default 1)

    Returns
    -------
    float in [0, 1]
    """
    if n == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def execute_code(code: str, test_code: str, entry_point: str,
                  canonical_solution: str = '',
                  timeout: float = 10.0) -> dict:
    """Run test_code against the implementation in code.

    Handles two test formats automatically:
    - ``def check(candidate):`` style (QHE) — calls check(entry_point_fn)
    - ``unittest.TestCase`` style (QB)   — builds check program per Quanbench
      logic: canonical → cir_solution, generated → cir_generated, then runs
      the unittest suite; all assertions in the test are honoured as-is.

    Parameters
    ----------
    code               : full solution code (prompt + generated body combined)
    test_code          : test string from exc_arr['test']
    entry_point        : function name defined by code
    canonical_solution : full canonical solution code (required for unittest
                         style so cir_solution can be defined)
    timeout            : seconds before daemon thread is abandoned
    """
    if not test_code:
        return {'success': False, 'error': 'no test_code', 'output': ''}

    result = [{'success': False, 'error': '', 'output': ''}]
    is_unittest = 'unittest' in test_code and 'def check(' not in test_code

    def _run():
        import matplotlib
        matplotlib.use('Agg')

        try:
            if not is_unittest:
                # ── QHE style: def check(candidate) ──────────────────────
                import builtins as _builtins
                _BLOCKED = {'exec', 'eval', 'compile', 'breakpoint', 'input'}
                restricted = {k: v for k, v in vars(_builtins).items()
                              if k not in _BLOCKED}
                full = f"{code}\n\n{test_code}"
                ns: dict = {'__builtins__': restricted}
                with contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(io.StringIO()):
                    exec(full, ns)  # noqa: S102
                ns['check'](ns[entry_point])

            else:
                # ── QB style: unittest.TestCase ───────────────────────────
                # Mirrors Quanbench's build_check_program + unsafe_execute.
                # canonical → cir_solution; generated (code) → cir_generated.
                import types, unittest as _ut, cmath, numpy as _np
                import sys as _sys

                # pre-seed QB utility functions if Quanbench is importable
                _utils: dict = {'np': _np, 'cmath': cmath}
                try:
                    _repo = str(Path(__file__).parent)
                    if _repo not in _sys.path:
                        _sys.path.insert(0, _repo)
                    from Quanbench.Quanbench_eval.test_matrix import (  # noqa
                        compute_KL, run_circuit, compare_depth_gatecount,
                        unitary_equivalent, compute_matrix_similarity,
                        compute_KL_noexecute, check_phase, is_gate_count_subset,
                    )
                    _utils.update({
                        'compute_KL': compute_KL,
                        'run_circuit': run_circuit,
                        'compare_depth_gatecount': compare_depth_gatecount,
                        'unitary_equivalent': unitary_equivalent,
                        'compute_matrix_similarity': compute_matrix_similarity,
                        'compute_KL_noexecute': compute_KL_noexecute,
                        'check_phase': check_phase,
                        'is_gate_count_subset': is_gate_count_subset,
                    })
                except ImportError:
                    pass

                # build check program: canonical → cir_solution,
                #                      generated  → cir_generated
                parts = []
                if canonical_solution:
                    parts.append(canonical_solution)
                    parts.append(f"cir_solution = {entry_point}")
                parts.append(code)
                parts.append(f"cir_generated = {entry_point}")
                parts.append(test_code)
                check_program = '\n'.join(parts)

                mod = types.ModuleType('__qmri_test__')
                mod.__dict__.update(_utils)
                with contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(io.StringIO()):
                    exec(check_program, mod.__dict__)  # noqa: S102
                    suite = _ut.defaultTestLoader.loadTestsFromModule(mod)
                    test_result = _ut.TestResult()
                    suite.run(test_result)

                issues = test_result.failures + test_result.errors
                if issues:
                    raise AssertionError(issues[0][1][:300])

            result[0] = {'success': True, 'error': '', 'output': ''}

        except TimeoutError as e:
            result[0] = {'success': False, 'error': str(e), 'output': 'TimeoutError'}
        except AssertionError as e:
            result[0] = {'success': False, 'error': str(e), 'output': 'AssertionError'}
        except Exception as e:  # noqa: BLE001
            result[0] = {'success': False, 'error': str(e), 'output': 'Exception'}

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout)
    return result[0]


# ══════════════════════════════════════════════════════════════════════════════
#  Gate Similarity (S_gate)
#  Source: R2_quantify.ipynb §18A–18B, adapted for generated-vs-canonical pairs.
#  Original computed corpus vs canonical; here both sides are known code strings.
# ══════════════════════════════════════════════════════════════════════════════

GATE_VOCAB: list[str] = [
    # single-qubit
    'h', 'x', 'y', 'z', 's', 'sdg', 't', 'tdg', 'sx', 'sxdg', 'id',
    'rx', 'ry', 'rz', 'p', 'u', 'u1', 'u2', 'u3',
    # two-qubit
    'cx', 'cnot', 'cy', 'cz', 'ch', 'crx', 'cry', 'crz', 'cp', 'cu', 'cu1', 'cu3',
    'swap', 'rxx', 'ryy', 'rzz', 'rzx', 'ecr',
    # three-qubit / multi-controlled
    'ccx', 'toffoli', 'cswap', 'mcx', 'mct', 'mcp', 'mcrx', 'mcry', 'mcrz',
    # non-unitary but common circuit ops
    'measure', 'barrier', 'reset', 'initialize',
]
GATE_VOCAB_SET: set[str] = set(GATE_VOCAB)

# Map Qiskit Gate class names → unified vocabulary entry (for qc.append(HGate(), ...) style)
GATE_CLASS_TO_NAME: dict[str, str] = {
    'HGate': 'h', 'XGate': 'x', 'YGate': 'y', 'ZGate': 'z',
    'SGate': 's', 'SdgGate': 'sdg', 'TGate': 't', 'TdgGate': 'tdg',
    'SXGate': 'sx', 'SXdgGate': 'sxdg', 'IGate': 'id',
    'RXGate': 'rx', 'RYGate': 'ry', 'RZGate': 'rz', 'PhaseGate': 'p',
    'UGate': 'u', 'U1Gate': 'u1', 'U2Gate': 'u2', 'U3Gate': 'u3',
    'CXGate': 'cx', 'CYGate': 'cy', 'CZGate': 'cz', 'CHGate': 'ch',
    'CRXGate': 'crx', 'CRYGate': 'cry', 'CRZGate': 'crz', 'CPhaseGate': 'cp',
    'CUGate': 'cu', 'CU1Gate': 'cu1', 'CU3Gate': 'cu3',
    'SwapGate': 'swap', 'RXXGate': 'rxx', 'RYYGate': 'ryy', 'RZZGate': 'rzz',
    'RZXGate': 'rzx', 'ECRGate': 'ecr',
    'CCXGate': 'ccx', 'CSwapGate': 'cswap',
    'MCXGate': 'mcx', 'MCXGrayCode': 'mcx', 'MCXRecursive': 'mcx', 'MCXVChain': 'mcx',
    'MCPhaseGate': 'mcp',
    'Measure': 'measure', 'Barrier': 'barrier', 'Reset': 'reset',
}


def extract_gate_counts(code: str) -> tuple[Counter, bool]:
    """AST-scan a code string for gate calls (both qc.h(0) and HGate() forms).

    Returns (Counter of gate_name → count, parse_ok).
    parse_ok=False when the code has a SyntaxError (truncated / invalid snippet).
    Note: high-level library functions (QFT(), EfficientSU2(), …) are not decomposed
    by static scanning — such tasks will produce an empty counter and S_gate=NaN.
    """
    counts: Counter = Counter()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        try:
            tree = ast.parse('def _f():\n    ' + code)
        except SyntaxError:
            return counts, False

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr in GATE_VOCAB_SET:
            counts[node.func.attr] += 1
        elif isinstance(node.func, ast.Name) and node.func.id in GATE_CLASS_TO_NAME:
            counts[GATE_CLASS_TO_NAME[node.func.id]] += 1
    return counts, True


def _cosine(v1: list[int], v2: list[int]) -> float:
    """Cosine similarity for two integer vectors (no sklearn dependency)."""
    n1 = sum(x * x for x in v1) ** 0.5
    n2 = sum(x * x for x in v2) ** 0.5
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(v1, v2)) / (n1 * n2)


def compute_s_gate_pair(code_gen: str, code_canon: str) -> Optional[float]:
    """S_gate between a generated code snippet and the canonical solution.

    Both sides are scanned via AST; cosine similarity of their GATE_VOCAB histograms
    is returned. Returns None when either side has zero recognised gate calls
    (high-level library circuit, empty body, or SyntaxError).
    """
    counts_g, ok_g = extract_gate_counts(code_gen)
    counts_c, ok_c = extract_gate_counts(code_canon)
    vec_g = [counts_g.get(g, 0) for g in GATE_VOCAB]
    vec_c = [counts_c.get(g, 0) for g in GATE_VOCAB]
    if sum(vec_g) == 0 or sum(vec_c) == 0:
        return None
    return _cosine(vec_g, vec_c)


# ══════════════════════════════════════════════════════════════════════════════
#  Structure Similarity (S_struct)
#  Source: R2_quantify.ipynb §18I–18J, simplified for generated-vs-canonical.
#  Both sides are real QuantumCircuit objects (already executed), so we skip the
#  corpus snippet-extraction step and call dag_wire_structure directly on both.
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_gate_name(name: str) -> str:
    """Strip Qiskit's auto-increment suffix from composite gate names (e.g. 'circuit-53' → 'circuit')."""
    return re.sub(r'-\d+$', '', name)


def dag_wire_structure(circuit: QuantumCircuit) -> dict[int, list[str]]:
    """Return per-wire gate-name sequences from the circuit's DAG.

    Returns {qubit_index: [gate_name, …]} for every qubit that has at least one operation.
    Gate names are normalised to strip Qiskit's auto-increment instance suffixes.
    """
    dag = circuit_to_dag(circuit)
    wire_sequences: dict[int, list[str]] = {}
    for qubit in circuit.qubits:
        idx = circuit.find_bit(qubit).index
        wire_sequences[idx] = [
            _normalize_gate_name(node.op.name)
            for node in dag.nodes_on_wire(qubit, only_ops=True)
        ]
    return wire_sequences


def _kgram_coverage(seq_c: list[str], seq_p: list[str], k: int = 2) -> tuple[float, dict]:
    """Canonical-anchored k-gram coverage: fraction of canonical's k-grams present in generated.

    Falls back to k=1 (plain gate multiset) when the wire is shorter than k.
    Returns (coverage ∈ [0,1], missing_kgrams dict).
    """
    if len(seq_c) < k:
        cnt_c: Counter = Counter(seq_c)
        cnt_p: Counter = Counter(seq_p)
    else:
        cnt_c = Counter(tuple(seq_c[i:i + k]) for i in range(len(seq_c) - k + 1))
        cnt_p = Counter(tuple(seq_p[i:i + k]) for i in range(len(seq_p) - k + 1))
    total = sum(cnt_c.values())
    if total == 0:
        return 1.0, {}
    covered = sum(min(cnt_c[g], cnt_p.get(g, 0)) for g in cnt_c)
    missing = {str(k): v for k, v in (cnt_c - cnt_p).items()}
    return covered / total, missing


def _lcs_length(seq_c: list[str], seq_p: list[str]) -> int:
    """Standard O(n·m) longest-common-subsequence length (order-preserving, gaps allowed)."""
    n, m = len(seq_c), len(seq_p)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq_c[i - 1] == seq_p[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def _lcs_coverage(seq_c: list[str], seq_p: list[str]) -> float:
    """Canonical-anchored LCS coverage: LCS length / len(canonical wire). Order-preserving."""
    if not seq_c:
        return 1.0
    return _lcs_length(seq_c, seq_p) / len(seq_c)


def compute_s_struct_pair(
    qc_gen: Optional[QuantumCircuit],
    qc_canon: Optional[QuantumCircuit],
) -> Optional[float]:
    """S_struct between a generated circuit and the canonical circuit.

    Uses the real DAG (circuit_to_dag) on both sides. Canonical is the reference:
    S_struct = max(mean_kgram_coverage, mean_lcs_coverage) over canonical's wires.
    Extra gates in generated beyond canonical are NOT penalised.

    Returns None if either circuit is None or has no gate operations.
    """
    if qc_gen is None or qc_canon is None:
        return None
    try:
        seq_c = dag_wire_structure(qc_canon)
        seq_g = dag_wire_structure(qc_gen)
        if not seq_c:
            return None
        kgram_scores: list[float] = []
        lcs_scores:   list[float] = []
        for idx, gates_c in seq_c.items():
            gates_g = seq_g.get(idx, [])
            kg, _   = _kgram_coverage(gates_c, gates_g)
            kgram_scores.append(kg)
            lcs_scores.append(_lcs_coverage(gates_c, gates_g))
        s_kgram = sum(kgram_scores) / len(kgram_scores)
        s_lcs   = sum(lcs_scores)   / len(lcs_scores)
        return max(s_kgram, s_lcs)
    except Exception:
        return None
