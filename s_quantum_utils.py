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

# Equivalence thresholds used by verify_equivalence() and functional_similarity()
_FIDELITY_EPSILON   = 1e-6   # process_fidelity: pass when score >= 1 - epsilon
_TVD_EXACT_EPSILON  = 1e-9   # exact statevector TVD: pass when score >= 1 - epsilon
_TAU_MID_DEFAULT    = 0.05   # shot-based fallback; MUST be replaced with calibrated value


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


def verify_equivalence(
    orig: QuantumCircuit,
    rew: QuantumCircuit,
    task: dict,
    *,
    tau_mid: float = _TAU_MID_DEFAULT,
) -> dict:
    """Equivalence check via the unified functional_similarity() entry point.

    Routing is determined by circuit structure, not by task flags.
    prepare_functional_circuit() is applied inside functional_similarity().

    Thresholds:
      process_fidelity                  : score >= 1 - _FIDELITY_EPSILON  (1e-6)
      exact_terminal_output_probability : score >= 1 - _TVD_EXACT_EPSILON (1e-9)
      sampled_*_tvd                     : score >= 1 - tau_mid
                                          (tau_mid default 0.05 is conservative;
                                          replace with calibrated value before use)

    For parameterized circuits: uses min_score so ALL parameter samples must pass.

    Returns a dict with {'status': 'pass'|'fail'|'error', 'method': ...} plus details.
    """
    is_param = task.get('is_parameterized', False) or len(orig.parameters) > 0
    n_param_samples = N_PARAM_SAMPLES if is_param else 1

    try:
        result = functional_similarity(
            reference_qc=orig,
            candidate_qc=rew,
            n_parameter_samples=n_param_samples,
            parameter_seed=SEED_BASE,
        )

        if result["status"] in ("missing_circuit", "parameter_mismatch"):
            return {"status": "error", "error_msg": result["status"],
                    "functional_result": result}

        method      = result.get("method")
        score       = result.get("score")
        # For parameterized circuits, every sample must pass → use worst-case min_score
        check_score = result.get("min_score") if is_param else score

        if check_score is None:
            return {"status": "error",
                    "error_msg": result.get("status", "no_score"),
                    "functional_result": result}

        if method == "process_fidelity":
            threshold = 1.0 - _FIDELITY_EPSILON
        elif method == "exact_terminal_output_probability":
            threshold = 1.0 - _TVD_EXACT_EPSILON
        elif method in ("sampled_mid_circuit_output_tvd", "sampled_reset_output_tvd"):
            threshold = 1.0 - tau_mid
        else:
            return {"status": "error",
                    "error_msg": f"unhandled_method:{method}",
                    "functional_result": result}

        return {
            "status":             "pass" if check_score >= threshold else "fail",
            "method":             method,
            "score":              score,
            "check_score":        check_score,
            "threshold":          threshold,
            "n_parameter_samples": result.get("n_parameter_samples", 0),
            "functional_result":  result,
        }
    except Exception as exc:
        return {"status": "error", "error_msg": f"{type(exc).__name__}: {exc}"}
# ── Parameterized circuit functional score ────────────────────────────────────

def compute_f_parameterized(
    qc: QuantumCircuit,
    canonical_qc: QuantumCircuit,
    is_measurement: bool,
    n_samples: int = 50,
    shots: int = SHOTS,
    seed: int = 42,
) -> Optional[float]:
    """Thin wrapper around functional_similarity() for backward compatibility.

    is_measurement is ignored: routing (process fidelity / exact TVD / shot TVD)
    is determined automatically by circuit structure inside functional_similarity().

    shots default updated from 8_192 to SHOTS (32_768) for consistency.
    Parameter alignment and barrier/delay removal are handled internally.

    Returns mean score ∈ [0, 1], or None when all samples fail.
    """
    if qc is None or canonical_qc is None:
        return None
    result = functional_similarity(
        reference_qc=canonical_qc,
        candidate_qc=qc,
        n_parameter_samples=n_samples,
        parameter_seed=seed,
        shots=shots,
    )
    return result.get("score")



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
    include_qc: bool = False,
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
    if include_qc:
        record['_qc'] = qc
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

# Non-gate operations excluded from S_gate across all computation paths
# (AST, raw circuit, transpiled circuit).  Measurements and barriers carry
# no gate-type information and would bias cosine similarity toward zero.
SG_EXCLUDED_OPS: frozenset[str] = frozenset({
    "measure",
    "reset",
    "barrier",
    "delay",
})

# Operations removed from circuits before S_struct DAG comparison.
# All non-unitary and timing/annotation ops are stripped; only quantum gates remain.
STRUCTURAL_EXCLUDED_OPS: frozenset[str] = frozenset({
    "measure",
    "reset",
    "barrier",
    "delay",
})

# Operations removed before functional comparison.
# Measurements and reset define observable semantics and are kept;
# only timing/ordering annotations are dropped.
FUNCTIONAL_EXCLUDED_OPS: frozenset[str] = frozenset({
    "barrier",
    "delay",
})


def _has_reset(qc: QuantumCircuit) -> bool:
    """Return True if qc contains any reset instruction."""
    return any(_instruction_parts(item)[0].name == "reset" for item in qc.data)


def prepare_functional_circuit(qc: QuantumCircuit) -> QuantumCircuit:
    """Remove FUNCTIONAL_EXCLUDED_OPS (barrier, delay) from circuit.

    Measurements and reset are preserved — they define the observable semantics
    of the circuit.
    """
    return _remove_excluded_ops(qc, FUNCTIONAL_EXCLUDED_OPS)


def extract_gate_counts(code: str) -> tuple[Counter, bool]:
    """AST-scan a code string for gate calls (both qc.h(0) and HGate() forms).

    Returns (Counter of gate_name → count, parse_ok).
    parse_ok=False when the code has a SyntaxError (truncated / invalid snippet).

    Note: AST result is diagnostic only.
    Loops, append, compose, and library circuits may be under-counted.
    High-level library functions (QFT(), EfficientSU2(), …) are not decomposed
    by static scanning — such tasks will produce an empty counter and S_gate=NaN.
    Non-gate ops (measure, reset, barrier, delay) are excluded via SG_EXCLUDED_OPS.
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

    for op_name in SG_EXCLUDED_OPS:
        counts.pop(op_name, None)

    return counts, True


def _cosine(v1: list[int], v2: list[int]) -> float:
    """Cosine similarity for two integer vectors (no sklearn dependency)."""
    n1 = sum(x * x for x in v1) ** 0.5
    n2 = sum(x * x for x in v2) ** 0.5
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(v1, v2)) / (n1 * n2)


def _s_gate_from_counts(
    reference_counts: Counter,
    candidate_counts: Counter,
) -> dict:
    """Compute S_gate metrics from two gate-count Counters.

    Returns a dict with three keys:
      s_gate        – primary value; None when a scale anomaly is detected.
      s_gate_cosine – cosine similarity (always populated when both sides non-empty).
      s_gate_ratio  – symmetric scale factor max(r, 1/r) where r=cand_total/ref_total,
                      populated only when histograms are strictly proportional;
                      None when the shapes differ.

    Symmetric means: both "candidate is 2× reference" and "candidate is 0.5× reference"
    give s_gate_ratio = 2.0.  Only ratio == 1.0 means no scale anomaly.

    Strict proportionality is checked via integer cross-multiply (exact, no float):
      cand_vec[i] * ref_total == ref_vec[i] * cand_total  for all i.

    Scale-anomaly rule:
      ratio is not None and ratio != 1  →  s_gate = None
                                            (cosine stays; ratio records the scale)
    """
    vocabulary = sorted(set(reference_counts) | set(candidate_counts))
    if not vocabulary:
        return {"s_gate": None, "s_gate_cosine": None, "s_gate_ratio": None}

    ref_vec  = [reference_counts.get(g, 0) for g in vocabulary]
    cand_vec = [candidate_counts.get(g, 0)  for g in vocabulary]

    ref_total  = sum(ref_vec)
    cand_total = sum(cand_vec)
    if ref_total == 0 or cand_total == 0:
        return {"s_gate": None, "s_gate_cosine": None, "s_gate_ratio": None}

    cosine = float(_cosine(ref_vec, cand_vec))

    # Strict proportionality via integer cross-multiply (avoids floating-point drift)
    is_proportional = all(
        cand_vec[i] * ref_total == ref_vec[i] * cand_total
        for i in range(len(vocabulary))
    )

    if is_proportional:
        directional  = cand_total / ref_total
        s_gate_ratio = max(directional, 1.0 / directional)  # symmetric
        s_gate = cosine if s_gate_ratio == 1.0 else None
    else:
        s_gate_ratio = None
        s_gate = cosine

    return {
        "s_gate":        s_gate,
        "s_gate_cosine": cosine,
        "s_gate_ratio":  s_gate_ratio,
    }


def compute_s_gate_ast_pair(code_gen: str, code_canon: str) -> dict:
    """S_gate between two code snippets using AST-scanned gate histograms.

    Returns the full _s_gate_from_counts dict (s_gate, s_gate_cosine, s_gate_ratio).
    All keys are None when either side has zero recognised gate calls.
    Note: AST result is diagnostic only — see extract_gate_counts docstring.
    """
    counts_g, _ok_g = extract_gate_counts(code_gen)
    counts_c, _ok_c = extract_gate_counts(code_canon)
    return _s_gate_from_counts(counts_g, counts_c)


def compute_s_gate_pair(code_gen: str, code_canon: str) -> Optional[float]:
    """Backward-compatible wrapper — returns only s_gate (cosine or None).

    Prefer compute_s_gate_ast_pair() for new code that needs all three fields.
    """
    return compute_s_gate_ast_pair(code_gen, code_canon)["s_gate"]


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


# ── Weisfeiler-Lehman qubit-interaction kernel ────────────────────────────────

def qubit_interaction_graph(qc: QuantumCircuit) -> dict[int, set[int]]:
    """Build an undirected simple qubit-interaction graph.

    Nodes are qubit indices; an edge connects two qubits that share a
    multi-qubit gate.  For gates on k>2 qubits, all k*(k-1)/2 pairs are
    connected (clique).  Single-qubit gates and STRUCTURAL_EXCLUDED_OPS
    are skipped even if the circuit has not been pre-processed.
    """
    adjacency: dict[int, set[int]] = {i: set() for i in range(qc.num_qubits)}
    for item in qc.data:
        op, qubits, _ = _instruction_parts(item)
        if op.name in STRUCTURAL_EXCLUDED_OPS:
            continue
        qubit_indices = sorted({qc.find_bit(q).index for q in qubits})
        if len(qubit_indices) < 2:
            continue
        for a_idx, a in enumerate(qubit_indices):
            for b in qubit_indices[a_idx + 1:]:
                adjacency[a].add(b)
                adjacency[b].add(a)
    return adjacency


def _wl_feature_counts(adjacency: dict[int, set[int]], *, h: int = 2) -> Counter:
    """WL subtree features up to height h on a qubit-interaction graph.

    Initial labels are node degrees (standard unlabeled-graph initialisation
    per Shervashidze et al. 2011, footnote 2).  Labels are tuples — no lossy
    hash compression — so identical WL signatures across different circuits
    produce identical feature keys.
    """
    labels: dict[int, Any] = {
        node: f"degree:{len(nbrs)}" for node, nbrs in adjacency.items()
    }
    features: Counter = Counter()
    for iteration in range(h + 1):
        for label in labels.values():
            features[(iteration, repr(label))] += 1
        if iteration == h:
            break
        new_labels: dict[int, Any] = {}
        for node, nbrs in adjacency.items():
            neighbour_labels = tuple(sorted((repr(labels[n]) for n in nbrs)))
            new_labels[node] = (labels[node], neighbour_labels)
        labels = new_labels
    return features


def _counter_dot(a: Counter, b: Counter) -> float:
    """Inner product of two Counters (shared keys only)."""
    return float(sum(a[k] * b[k] for k in set(a) & set(b)))


def wl_interaction_similarity(
    reference_qc: Optional[QuantumCircuit],
    candidate_qc: Optional[QuantumCircuit],
    *,
    h: int = 2,
) -> dict:
    """Normalised WL subtree kernel on qubit-interaction graphs.

    Returns both the raw (unnormalised) kernel K_WL and the cosine-normalised
    similarity score in [0,1].  The normalisation

        S_WL = K_WL(G,G') / sqrt(K_WL(G,G) * K_WL(G',G'))

    is *not* from Shervashidze et al. (2011) Eq.(2); it is an adaptation to
    obtain a [0,1] similarity.  Both values are stored separately.
    """
    _empty = {
        "score": None, "kernel": None,
        "reference_self_kernel": None, "candidate_self_kernel": None,
        "h": h,
    }
    if reference_qc is None or candidate_qc is None:
        return {**_empty, "status": "missing_circuit"}

    ref_graph = qubit_interaction_graph(reference_qc)
    cand_graph = qubit_interaction_graph(candidate_qc)

    ref_feat  = _wl_feature_counts(ref_graph,  h=h)
    cand_feat = _wl_feature_counts(cand_graph, h=h)

    kernel      = _counter_dot(ref_feat, cand_feat)
    ref_self    = _counter_dot(ref_feat, ref_feat)
    cand_self   = _counter_dot(cand_feat, cand_feat)
    denominator = math.sqrt(ref_self * cand_self)

    if denominator == 0:
        return {**_empty,
                "kernel": kernel,
                "reference_self_kernel": ref_self,
                "candidate_self_kernel": cand_self,
                "status": "empty_graph_features"}

    score = kernel / denominator
    ref_edges  = sum(len(v) for v in ref_graph.values()) // 2
    cand_edges = sum(len(v) for v in cand_graph.values()) // 2
    return {
        "score":                    float(max(0.0, min(1.0, score))),
        "kernel":                   kernel,
        "reference_self_kernel":    ref_self,
        "candidate_self_kernel":    cand_self,
        "status":                   "ok",
        "h":                        h,
        "reference_n_qubits":       reference_qc.num_qubits,
        "candidate_n_qubits":       candidate_qc.num_qubits,
        "reference_n_edges":        ref_edges,
        "candidate_n_edges":        cand_edges,
    }


def edge_jaccard_similarity(
    reference_qc: Optional[QuantumCircuit],
    candidate_qc: Optional[QuantumCircuit],
) -> Optional[float]:
    """Jaccard similarity of undirected qubit-interaction edge sets.

    Returns None when either circuit is missing or when both circuits have no
    multi-qubit gates (union of edge sets is empty — Jaccard is undefined, not 1).
    """
    if reference_qc is None or candidate_qc is None:
        return None

    def _edges(qc: QuantumCircuit) -> set:
        graph = qubit_interaction_graph(qc)
        return {
            (min(a, b), max(a, b))
            for a, neighbours in graph.items()
            for b in neighbours
            if a != b
        }

    ref_edges  = _edges(reference_qc)
    cand_edges = _edges(candidate_qc)
    union = ref_edges | cand_edges
    if not union:
        return None  # both circuits have no multi-qubit gates; Jaccard undefined
    return len(ref_edges & cand_edges) / len(union)


def summarize_wl_saturation(
    pair_records: list[dict],
    *,
    field: str = "s_struct_wl_transpiled",
) -> dict:
    """Diagnostic: proportion of WL similarity values equal to 1.

    Use this on the 64-pair validation set before deciding whether to include
    S_WL in the composite S_struct score.
    """
    values = [
        float(r[field])
        for r in pair_records
        if r.get(field) is not None and math.isfinite(float(r[field]))
    ]
    if not values:
        return {"n_valid": 0, "n_exactly_one": 0, "proportion_exactly_one": None}
    exactly_one = [v for v in values if np.isclose(v, 1.0, rtol=0.0, atol=1e-12)]
    return {
        "n_valid":                  len(values),
        "n_exactly_one":            len(exactly_one),
        "proportion_exactly_one":   len(exactly_one) / len(values),
        "n_unique_rounded_6dp":     len({round(v, 6) for v in values}),
        "minimum":                  min(values),
        "median":                   float(np.median(values)),
        "maximum":                  max(values),
    }


def compute_s_struct_pair(
    qc_gen: Optional[QuantumCircuit],
    qc_canon: Optional[QuantumCircuit],
) -> Optional[dict]:
    """S_struct between a generated circuit and the canonical circuit.

    Uses the real DAG (circuit_to_dag) on both sides. Canonical is the reference:
    S_struct = mean(mean_kgram_coverage, mean_lcs_coverage) over canonical's wires.
    Extra gates in generated beyond canonical are NOT penalised.

    Returns dict with s_struct/s_struct_lcs/s_struct_kgram, or None if either
    circuit is None or has no gate operations.
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
        return {
            "s_struct":        (s_kgram + s_lcs) / 2.0,
            "s_struct_kgram":  s_kgram,
            "s_struct_lcs":    s_lcs,
        }
    except Exception:
        return None

"""task->gate after Transpile"""
ANALYSIS_BASIS = {
    "rz",
    "sx",
    "x",
    "cx",
}

DECOMPOSE_TO_ANALYSIS_BASIS = {
    # Single-qubit unitary gates
    "h", "p", "r", "rx", "ry",
    "s", "sdg", "sxdg",
    "t", "tdg",
    "u", "u1", "u2", "u3",
    "y", "z",

    # Two-qubit unitary gates
    "cp", "crx", "cry", "crz",
    "csx",
    "cu", "cu1", "cu2", "cu3",
    "cy", "cz", "ecr",
    "rxx", "ryy", "rzx", "rzz",
    "swap",

    # Three- and multi-qubit gates
    "ccx", "ccz", "cswap",
    "mcp", "mcphase", "mcr",
    "mcrx", "mcry", "mcrz",
    "mcswap", "mcsx",
    "mcu", "mcu1", "mcu2", "mcu3",
    "mcx", "mcx_gray",
    "mcy", "mcz",

    # Composite unitary operations
    "diagonal",
    "multiplexer",
    "pauli",
    "unitary",
}

PRESERVED_SEMANTIC_OPS = {
    "measure",
    "reset",
    "delay",
    "if_else",
    "for_loop",
    "while_loop",
    "switch_case",
    "break_loop",
    "continue_loop",
    "store",
    "barrier",
}

NONUNITARY_OR_CHANNEL_OPS = {
    "initialize",
    "kraus",
    "quantum_channel",
    "superop",
    "qerror_loc",
    "roerror",
}

IGNORED_OPS = {
    "id",
}

AER_META_OPS = {
    "save_amplitudes",
    "save_amplitudes_sq",
    "save_clifford",
    "save_density_matrix",
    "save_expval",
    "save_expval_var",
    "save_matrix_product_state",
    "save_probabilities",
    "save_probabilities_dict",
    "save_stabilizer",
    "save_state",
    "save_statevector",
    "save_statevector_dict",
    "save_superop",
    "save_unitary",
    "set_density_matrix",
    "set_matrix_product_state",
    "set_stabilizer",
    "set_statevector",
    "set_superop",
    "set_unitary",
}


# ══════════════════════════════════════════════════════════════════════════════
# Unified quantum-metric pipeline (append-only revision)
# ══════════════════════════════════════════════════════════════════════════════

from typing import Any, Iterable, Mapping, Sequence


COMMON_BASIS: tuple[str, ...] = ("rz", "sx", "x", "cx")
COMMON_OPTIMIZATION_LEVEL = 0
COMMON_TRANSPILE_SEED = 42
DEFAULT_PARAMETER_SAMPLES = 50
DEFAULT_PARAMETER_SEED = 42


def _instruction_parts(item):
    """Return (operation, qubits, clbits) across Qiskit instruction formats."""
    if hasattr(item, "operation"):
        return item.operation, tuple(item.qubits), tuple(item.clbits)
    operation, qubits, clbits = item
    return operation, tuple(qubits), tuple(clbits)


def _terminal_measurement_indices(qc: QuantumCircuit) -> list[int]:
    """Indices of measurements in the final measurement suffix.

    Final barriers may occur inside that suffix but are retained.  A measurement
    before any later quantum/classical operation is treated as mid-circuit.
    """
    indices: list[int] = []
    suffix_is_terminal = True
    for index in range(len(qc.data) - 1, -1, -1):
        operation, _, _ = _instruction_parts(qc.data[index])
        if operation.name == "measure" and suffix_is_terminal:
            indices.append(index)
        elif operation.name == "barrier" and suffix_is_terminal:
            continue
        else:
            suffix_is_terminal = False
    return sorted(indices)


def classify_measurements(qc: QuantumCircuit) -> str:
    """Classify a circuit as 'none', 'terminal_only', or 'mid_circuit'."""
    n_measurements = sum(
        1 for item in qc.data if _instruction_parts(item)[0].name == "measure"
    )
    if n_measurements == 0:
        return "none"
    if n_measurements == len(_terminal_measurement_indices(qc)):
        return "terminal_only"
    return "mid_circuit"


def strip_terminal_measurements(qc: QuantumCircuit) -> QuantumCircuit:
    """Copy a circuit and remove only terminal measurements.

    Mid-circuit measurements, barriers, resets, delays, and all other operations
    are retained.
    """
    remove = set(_terminal_measurement_indices(qc))
    if not remove:
        return qc.copy()

    stripped = qc.copy_empty_like()
    for index, item in enumerate(qc.data):
        if index in remove:
            continue
        operation, qubits, clbits = _instruction_parts(item)
        mapped_qubits = [stripped.qubits[qc.find_bit(bit).index] for bit in qubits]
        mapped_clbits = [stripped.clbits[qc.find_bit(bit).index] for bit in clbits]
        stripped.append(operation, mapped_qubits, mapped_clbits)
    return stripped


def _remove_excluded_ops(qc: QuantumCircuit, excluded: frozenset) -> QuantumCircuit:
    """Return a copy of qc with every instruction whose name is in excluded removed."""
    out = qc.copy_empty_like()
    for item in qc.data:
        op, qubits, clbits = _instruction_parts(item)
        if op.name not in excluded:
            mapped_qubits = [out.qubits[qc.find_bit(bit).index] for bit in qubits]
            mapped_clbits = [out.clbits[qc.find_bit(bit).index] for bit in clbits]
            out.append(op, mapped_qubits, mapped_clbits)
    return out


def prepare_structural_circuit(
    qc: QuantumCircuit,
    *,
    transpile_to_common_basis: bool,
    basis_gates: Sequence[str] = COMMON_BASIS,
    optimization_level: int = COMMON_OPTIMIZATION_LEVEL,
    seed_transpiler: int = COMMON_TRANSPILE_SEED,
) -> QuantumCircuit:
    """Prepare the circuit used by s_gate and s_struct.

    Removes all STRUCTURAL_EXCLUDED_OPS (measure/reset/barrier/delay) so that only
    gate operations remain.  Common-basis transpilation is explicit so raw and
    transpiled results remain separately auditable.
    """
    prepared = _remove_excluded_ops(qc, STRUCTURAL_EXCLUDED_OPS)
    if not transpile_to_common_basis:
        return prepared
    return transpile(
        prepared,
        basis_gates=list(basis_gates),
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )


def _normalised_operation_counts(qc: QuantumCircuit) -> Counter:
    """Count top-level circuit operations, excluding non-gate ops in SG_EXCLUDED_OPS.

    Excludes measure/reset/barrier/delay so that structural preparation need not
    be perfect; also guards against callers that pass an unstripped circuit.
    """
    counts: Counter = Counter()
    for name, count in qc.count_ops().items():
        normalized_name = _normalize_gate_name(name)
        if normalized_name in SG_EXCLUDED_OPS:
            continue
        counts[normalized_name] += int(count)
    return counts


def s_gate_metrics_from_circuits(
    *,
    reference_qc: Optional[QuantumCircuit],
    candidate_qc: Optional[QuantumCircuit],
) -> dict:
    """Gate-count similarity metrics (s_gate, s_gate_cosine, s_gate_ratio) for two circuits.

    Returns all-None dict when either circuit is None or an exception occurs.
    Terminal measurements should be removed by prepare_structural_circuit() upstream;
    _normalised_operation_counts() also filters SG_EXCLUDED_OPS as a second guard.
    """
    _empty = {"s_gate": None, "s_gate_cosine": None, "s_gate_ratio": None}
    if reference_qc is None or candidate_qc is None:
        return _empty
    try:
        reference_counts = _normalised_operation_counts(reference_qc)
        candidate_counts = _normalised_operation_counts(candidate_qc)
        return _s_gate_from_counts(reference_counts, candidate_counts)
    except Exception:
        return _empty


def s_gate_from_circuits(
    *,
    reference_qc: Optional[QuantumCircuit],
    candidate_qc: Optional[QuantumCircuit],
) -> Optional[float]:
    """Backward-compatible wrapper — returns only s_gate (cosine or None).

    Prefer s_gate_metrics_from_circuits() for new code that needs all three fields.
    """
    return s_gate_metrics_from_circuits(
        reference_qc=reference_qc,
        candidate_qc=candidate_qc,
    )["s_gate"]


def s_struct_metrics_from_circuits(
    *,
    reference_qc: Optional[QuantumCircuit],
    candidate_qc: Optional[QuantumCircuit],
) -> dict:
    """Canonical-anchored k-gram/LCS structure similarity — full metric dict.

    Returns {s_struct, s_struct_lcs, s_struct_kgram} (all None on failure).
    The keyword-only interface removes the positional ambiguity in the earlier
    compute_s_struct_pair(candidate, canonical) function.
    """
    _empty = {"s_struct": None, "s_struct_lcs": None, "s_struct_kgram": None}
    if reference_qc is None or candidate_qc is None:
        return _empty
    if len(reference_qc.data) == 0 or len(candidate_qc.data) == 0:
        return _empty
    result = compute_s_struct_pair(candidate_qc, reference_qc)
    return result if result is not None else _empty


def s_struct_from_circuits(
    *,
    reference_qc: Optional[QuantumCircuit],
    candidate_qc: Optional[QuantumCircuit],
) -> Optional[float]:
    """Backward-compatible wrapper — returns only s_struct (mean of lcs/kgram).

    Prefer s_struct_metrics_from_circuits() for new code that needs all three fields.
    """
    return s_struct_metrics_from_circuits(
        reference_qc=reference_qc,
        candidate_qc=candidate_qc,
    )["s_struct"]


def _terminal_measurement_map(qc: QuantumCircuit) -> list[tuple[int, int]]:
    """Return terminal (qubit_index, clbit_index) measurement pairs."""
    pairs: list[tuple[int, int]] = []
    for index in _terminal_measurement_indices(qc):
        _, qubits, clbits = _instruction_parts(qc.data[index])
        if len(qubits) != 1 or len(clbits) != 1:
            raise ValueError("Each measurement must map one qubit to one classical bit")
        pairs.append((qc.find_bit(qubits[0]).index, qc.find_bit(clbits[0]).index))
    return pairs


def exact_terminal_output_probabilities(
    qc: QuantumCircuit,
    *,
    measurement_map: Optional[Sequence[tuple[int, int]]] = None,
) -> dict[str, float]:
    """Exact probabilities over measured outputs, without shots.

    Only measured qubits are retained, and their output order follows the
    qubit-to-classical-bit mapping.  Unused classical registers are ignored.
    """
    measurement_type = classify_measurements(qc)
    if measurement_type == "mid_circuit":
        raise ValueError("Exact Statevector probabilities do not support mid-circuit measurement")

    pairs = list(measurement_map) if measurement_map is not None else _terminal_measurement_map(qc)
    if not pairs:
        raise ValueError("No terminal measurement map is available")

    # If a classical bit is written more than once, its last terminal write wins.
    qubit_for_clbit: dict[int, int] = {}
    for qubit_index, clbit_index in pairs:
        qubit_for_clbit[int(clbit_index)] = int(qubit_index)
    if max(qubit_for_clbit.values()) >= qc.num_qubits:
        raise ValueError("Measurement map refers to a missing candidate qubit")

    stripped = strip_terminal_measurements(qc)
    if any(_instruction_parts(item)[0].name == "measure" for item in stripped.data):
        raise ValueError("A non-terminal measurement remains after stripping")
    if len(stripped.parameters) > 0:
        raise ValueError("Bind circuit parameters before computing Statevector probabilities")

    # Statevector needs a quantum-only instruction stream.  Classical registers
    # are metadata after terminal measurements have been removed.
    state_circuit = QuantumCircuit(stripped.num_qubits)
    state_circuit.global_phase = stripped.global_phase
    for item in stripped.data:
        operation, qubits, clbits = _instruction_parts(item)
        if clbits or getattr(operation, "condition", None) is not None:
            raise ValueError("A classical dependency remains in the Statevector circuit")
        mapped_qubits = [state_circuit.qubits[stripped.find_bit(bit).index] for bit in qubits]
        state_circuit.append(operation, mapped_qubits, [])

    probabilities = np.asarray(Statevector(state_circuit).probabilities(), dtype=float)
    ordered_clbits = sorted(qubit_for_clbit, reverse=True)
    output: dict[str, float] = {}
    for basis_index, probability in enumerate(probabilities):
        if probability <= 0.0:
            continue
        key = "".join(
            str((basis_index >> qubit_for_clbit[clbit_index]) & 1)
            for clbit_index in ordered_clbits
        )
        output[key] = output.get(key, 0.0) + float(probability)
    return output


def probability_tvd(
    first: Mapping[str, float],
    second: Mapping[str, float],
) -> float:
    """Total variation distance between two exact probability mappings."""
    keys = set(first) | set(second)
    return 0.5 * sum(abs(float(first.get(k, 0.0)) - float(second.get(k, 0.0))) for k in keys)


def _bind_parameter_pair(
    reference_qc: QuantumCircuit,
    candidate_qc: QuantumCircuit,
    *,
    n_samples: int,
    seed: int,
) -> list[tuple[QuantumCircuit, QuantumCircuit]]:
    """Create reproducible, aligned parameter bindings for a circuit pair."""
    reference_parameters = sorted(reference_qc.parameters, key=lambda p: p.name)
    candidate_parameters = sorted(candidate_qc.parameters, key=lambda p: p.name)
    if not reference_parameters and not candidate_parameters:
        return [(reference_qc, candidate_qc)]
    if len(reference_parameters) != len(candidate_parameters):
        return []

    rng = np.random.default_rng(seed)
    pairs: list[tuple[QuantumCircuit, QuantumCircuit]] = []
    candidate_by_name = {p.name: p for p in candidate_parameters}
    same_names = {p.name for p in reference_parameters} == set(candidate_by_name)
    for _ in range(n_samples):
        values = rng.uniform(0.0, 2.0 * np.pi, len(reference_parameters))
        reference_bind = dict(zip(reference_parameters, values))
        if same_names:
            candidate_bind = {
                candidate_by_name[parameter.name]: value
                for parameter, value in zip(reference_parameters, values)
            }
        else:
            candidate_bind = dict(zip(candidate_parameters, values))
        pairs.append((
            reference_qc.assign_parameters(reference_bind),
            candidate_qc.assign_parameters(candidate_bind),
        ))
    return pairs


def _functional_similarity_bound(
    reference_qc: QuantumCircuit,
    candidate_qc: QuantumCircuit,
    shots: int = SHOTS,
) -> dict[str, Any]:
    """Functional similarity for a parameter-free circuit pair.

    Three routing paths:
      Path 3 — mid-circuit measurement OR reset present → shot-based TVD (shots param)
      Path 1 — no measurements, no reset             → process fidelity (exact)
      Path 2 — terminal-only measurements, no reset  → exact statevector TVD
    """
    reference_type = classify_measurements(reference_qc)
    candidate_type = classify_measurements(candidate_qc)
    ref_has_reset  = _has_reset(reference_qc)
    cand_has_reset = _has_reset(candidate_qc)

    # Path 3: mid-circuit measurement or reset — fall back to shot-based TVD
    if (reference_type == "mid_circuit" or candidate_type == "mid_circuit"
            or ref_has_reset or cand_has_reset):
        if reference_type == "none" and candidate_type == "none":
            # reset-only circuits have no observable output
            return {
                "score": None,
                "status": "reset_no_measurement",
                "method": "sampled_output_tvd",
                "tvd": None,
                "process_fidelity": None,
            }
        method = (
            "sampled_reset_output_tvd"
            if (ref_has_reset or cand_has_reset)
            else "sampled_mid_circuit_output_tvd"
        )
        try:
            tvd_value = tvd(reference_qc, candidate_qc, shots=shots)
            return {
                "score": float(max(0.0, min(1.0, 1.0 - tvd_value))),
                "tvd": tvd_value,
                "status": "ok",
                "method": method,
                "process_fidelity": None,
            }
        except Exception as exc:
            return {
                "score": None,
                "status": "functional_error",
                "method": method,
                "tvd": None,
                "process_fidelity": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    # Path 1: no measurements, no reset → process fidelity
    if reference_type == "none":
        if candidate_type != "none":
            return {
                "score": None,
                "status": "measurement_type_mismatch",
                "method": None,
                "tvd": None,
                "process_fidelity": None,
            }
        try:
            fid = circuit_fidelity(reference_qc, candidate_qc)
            return {
                "score": float(fid),
                "process_fidelity": float(fid),
                "tvd": None,
                "status": "ok",
                "method": "process_fidelity",
            }
        except Exception as exc:
            return {
                "score": None,
                "status": "functional_error",
                "method": "process_fidelity",
                "process_fidelity": None,
                "tvd": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    # Path 2: terminal-only measurements, no reset → exact statevector TVD
    try:
        reference_map = _terminal_measurement_map(reference_qc)
        reference_probabilities = exact_terminal_output_probabilities(reference_qc)
        candidate_probabilities = exact_terminal_output_probabilities(
            candidate_qc,
            measurement_map=reference_map if candidate_type == "none" else None,
        )
        tvd_value = probability_tvd(reference_probabilities, candidate_probabilities)
        return {
            "score": float(max(0.0, min(1.0, 1.0 - tvd_value))),
            "tvd": tvd_value,
            "status": "ok",
            "method": "exact_terminal_output_probability",
            "process_fidelity": None,
        }
    except Exception as exc:
        return {
            "score": None,
            "status": "functional_error",
            "method": "exact_terminal_output_probability",
            "tvd": None,
            "process_fidelity": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def functional_similarity(
    *,
    reference_qc: Optional[QuantumCircuit],
    candidate_qc: Optional[QuantumCircuit],
    n_parameter_samples: int = DEFAULT_PARAMETER_SAMPLES,
    parameter_seed: int = DEFAULT_PARAMETER_SEED,
    shots: int = SHOTS,
) -> dict[str, Any]:
    """Compute f using process fidelity, exact TVD, or shot-based TVD.

    Applies prepare_functional_circuit() internally (removes barrier/delay) so
    all callers — compute_pair_metrics, verify_equivalence, compute_f_parameterized
    — get consistent preprocessing regardless of call path.

    Parameterized circuits are sampled n_parameter_samples times.
    shots is forwarded to _functional_similarity_bound for mid/reset paths.
    """
    if reference_qc is None or candidate_qc is None:
        return {"score": None, "status": "missing_circuit", "method": None,
                "min_score": None, "tvd": None, "process_fidelity": None}
    # Uniform preprocessing: remove barrier/delay on every call path
    reference_qc = prepare_functional_circuit(reference_qc)
    candidate_qc = prepare_functional_circuit(candidate_qc)

    bound_pairs = _bind_parameter_pair(
        reference_qc,
        candidate_qc,
        n_samples=n_parameter_samples,
        seed=parameter_seed,
    )
    if not bound_pairs:
        return {"score": None, "status": "parameter_mismatch", "method": None,
                "min_score": None, "tvd": None, "process_fidelity": None}

    results = [_functional_similarity_bound(ref, cand, shots=shots) for ref, cand in bound_pairs]
    scores = [result["score"] for result in results if result.get("score") is not None]
    if not scores:
        first = results[0]
        return {
            **first,
            "n_parameter_samples": len(bound_pairs) if len(bound_pairs) > 1 else 0,
            "n_valid_samples": 0,
            "min_score": None,
            "sample_scores": [],
        }

    valid_results = [r for r in results if r.get("score") is not None]
    methods = {r.get("method") for r in valid_results}
    tvd_vals = [r["tvd"] for r in valid_results if r.get("tvd") is not None]
    pf_vals  = [r["process_fidelity"] for r in valid_results if r.get("process_fidelity") is not None]
    return {
        "score":              float(np.mean(scores)),
        "min_score":          float(min(scores)),
        "sample_scores":      scores,
        "status":             "ok" if len(scores) == len(bound_pairs) else "partial",
        "method":             methods.pop() if len(methods) == 1 else "mixed",
        "tvd":                float(np.mean(tvd_vals)) if tvd_vals else None,
        "process_fidelity":   float(np.mean(pf_vals)) if pf_vals else None,
        "n_parameter_samples": len(bound_pairs) if len(bound_pairs) > 1 else 0,
        "n_valid_samples":    len(scores),
    }


def compute_pair_metrics(
    *,
    reference_qc: Optional[QuantumCircuit],
    candidate_qc: Optional[QuantumCircuit],
    reference_code: Optional[str] = None,
    candidate_code: Optional[str] = None,
    basis_gates: Sequence[str] = COMMON_BASIS,
    optimization_level: int = COMMON_OPTIMIZATION_LEVEL,
    seed_transpiler: int = COMMON_TRANSPILE_SEED,
    n_parameter_samples: int = DEFAULT_PARAMETER_SAMPLES,
    parameter_seed: int = DEFAULT_PARAMETER_SEED,
) -> dict[str, Any]:
    """Compute raw, and transpiled s_gate/s_struct, and f.
    reference_code / candidate_code: optional source strings for AST s_gate diagnostic; computed regardless of whether circuits are available.
    """
    result: dict[str, Any] = {
        "candidate_exec_ok": reference_qc is not None and candidate_qc is not None,
        # AST diagnostic (computed from source code , without execution)
        "s_gate_ast":        None,
        "s_gate_ast_cosine": None,
        "s_gate_ast_ratio":  None,
        # Executed circuit, before transpilation
        "s_gate_raw":        None,
        "s_gate_raw_cosine": None,
        "s_gate_raw_ratio":  None,
        "s_struct_raw":      None,
        "s_struct_raw_lcs":  None,
        "s_struct_raw_kgram": None,
        # Executed circuit, after common-basis transpilation (primary for paper)
        "s_gate_transpiled":        None,
        "s_gate_transpiled_cosine": None,
        "s_gate_transpiled_ratio":  None,
        "s_struct_transpiled":      None,
        "s_struct_transpiled_lcs":  None,
        "s_struct_transpiled_kgram": None,
        # WL qubit-interaction unlabelled (diagnostic; not yet merged into s_struct)
        "s_struct_wl_raw":              None,
        "s_struct_wl_kernel_raw":       None,
        "s_struct_wl_status_raw":       "not_run",
        "s_struct_wl_transpiled":       None,
        "s_struct_wl_kernel_transpiled": None,
        "s_struct_wl_status_transpiled": "not_run",
        "wl_iterations":                2,
        # Edge Jaccard qubit-interaction (diagnostic; not yet merged into s_struct)
        "s_struct_edge_jaccard_raw":        None,
        "s_struct_edge_jaccard_transpiled": None,
        "f":              None,
        "f_tvd":          None,
        "f_min_score":    None,
        "reference_measurement_type": None,
        "candidate_measurement_type": None,
        "raw_status":        "not_run",
        "transpiled_status": "not_run",
        "functional_status": "not_run",
        "functional_method": None,
        "basis_gates":        list(basis_gates),
        "optimization_level": optimization_level,
        "seed_transpiler":    seed_transpiler,
        "strip_policy":       "structural_excluded_ops",
    }

    # ── AST diagnostic (runs even when circuits are missing) ─────────────────
    if reference_code is not None and candidate_code is not None:
        ast_metrics = compute_s_gate_ast_pair(candidate_code, reference_code)
        result["s_gate_ast"]        = ast_metrics["s_gate"]
        result["s_gate_ast_cosine"] = ast_metrics["s_gate_cosine"]
        result["s_gate_ast_ratio"]  = ast_metrics["s_gate_ratio"]

    if reference_qc is None or candidate_qc is None:
        result.update({
            "raw_status":        "missing_circuit",
            "transpiled_status": "missing_circuit",
            "functional_status": "missing_circuit",
        })
        return result

    result["reference_measurement_type"] = classify_measurements(reference_qc)
    result["candidate_measurement_type"] = classify_measurements(candidate_qc)

    try:
        reference_raw = prepare_structural_circuit(
            reference_qc, transpile_to_common_basis=False,
        )
        candidate_raw = prepare_structural_circuit(
            candidate_qc, transpile_to_common_basis=False,
        )
        gate_raw = s_gate_metrics_from_circuits(
            reference_qc=reference_raw, candidate_qc=candidate_raw,
        )
        result["s_gate_raw"]        = gate_raw["s_gate"]
        result["s_gate_raw_cosine"] = gate_raw["s_gate_cosine"]
        result["s_gate_raw_ratio"]  = gate_raw["s_gate_ratio"]
        struct_raw = s_struct_metrics_from_circuits(
            reference_qc=reference_raw, candidate_qc=candidate_raw,
        )
        result["s_struct_raw"]       = struct_raw["s_struct"]
        result["s_struct_raw_lcs"]   = struct_raw["s_struct_lcs"]
        result["s_struct_raw_kgram"] = struct_raw["s_struct_kgram"]
        wl_raw = wl_interaction_similarity(reference_raw, candidate_raw, h=2)
        result["s_struct_wl_raw"]          = wl_raw["score"]
        result["s_struct_wl_kernel_raw"]   = wl_raw["kernel"]
        result["s_struct_wl_status_raw"]        = wl_raw["status"]
        result["s_struct_wl_details_raw"]       = wl_raw
        result["s_struct_edge_jaccard_raw"]     = edge_jaccard_similarity(reference_raw, candidate_raw)
        result["raw_status"] = "ok"
    except Exception as exc:
        result["raw_status"] = f"error: {type(exc).__name__}: {exc}"

    try:
        reference_transpiled = prepare_structural_circuit(
            reference_qc,
            transpile_to_common_basis=True,
            basis_gates=basis_gates,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
        )
        candidate_transpiled = prepare_structural_circuit(
            candidate_qc,
            transpile_to_common_basis=True,
            basis_gates=basis_gates,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
        )
        gate_transpiled = s_gate_metrics_from_circuits(
            reference_qc=reference_transpiled,
            candidate_qc=candidate_transpiled,
        )
        result["s_gate_transpiled"]        = gate_transpiled["s_gate"]
        result["s_gate_transpiled_cosine"] = gate_transpiled["s_gate_cosine"]
        result["s_gate_transpiled_ratio"]  = gate_transpiled["s_gate_ratio"]
        struct_transpiled = s_struct_metrics_from_circuits(
            reference_qc=reference_transpiled,
            candidate_qc=candidate_transpiled,
        )
        result["s_struct_transpiled"]       = struct_transpiled["s_struct"]
        result["s_struct_transpiled_lcs"]   = struct_transpiled["s_struct_lcs"]
        result["s_struct_transpiled_kgram"] = struct_transpiled["s_struct_kgram"]
        wl_tr = wl_interaction_similarity(reference_transpiled, candidate_transpiled, h=2)
        result["s_struct_wl_transpiled"]          = wl_tr["score"]
        result["s_struct_wl_kernel_transpiled"]   = wl_tr["kernel"]
        result["s_struct_wl_status_transpiled"]        = wl_tr["status"]
        result["s_struct_wl_details_transpiled"]       = wl_tr
        result["s_struct_edge_jaccard_transpiled"]     = edge_jaccard_similarity(reference_transpiled, candidate_transpiled)
        result["transpiled_status"] = "ok"
    except Exception as exc:
        result["transpiled_status"] = f"error: {type(exc).__name__}: {exc}"

    functional = functional_similarity(
        reference_qc=prepare_functional_circuit(reference_qc),
        candidate_qc=prepare_functional_circuit(candidate_qc),
        n_parameter_samples=n_parameter_samples,
        parameter_seed=parameter_seed,
    )
    result["f"]              = functional.get("score")
    result["f_tvd"]          = functional.get("tvd")
    result["f_min_score"]    = functional.get("min_score")
    result["functional_status"] = functional.get("status")
    result["functional_method"] = functional.get("method")
    result["functional_details"] = functional
    return result


def make_failed_pair_metrics(error: Optional[str] = None) -> dict[str, Any]:
    """Create a NaN/None metric record for an execution failure."""
    result = compute_pair_metrics(reference_qc=None, candidate_qc=None)
    result["candidate_exec_ok"] = False
    result["execution_error"] = error
    return result


def s_quantum_axes(
    pair_metrics: Mapping[str, Any],
    *,
    representation: str = "transpiled",
) -> dict[str, Optional[float]]:
    """Return S_quantum as three separate axes, never as a scalar maximum."""
    suffix = "transpiled" if representation == "transpiled" else "raw"
    return {
        "s_gate": pair_metrics.get(f"s_gate_{suffix}"),
        "s_struct": pair_metrics.get(f"s_struct_{suffix}"),
        "f": pair_metrics.get("f"),
    }


def aggregate_valid_generations(
    samples: Iterable[Mapping[str, Any]],
    *,
    metric_names: Sequence[str] = (
        "s_gate_raw",
        "s_struct_raw",
        "s_struct_raw_lcs",
        "s_struct_raw_kgram",
        "s_struct_wl_raw",
        "s_gate_transpiled",
        "s_struct_transpiled",
        "s_struct_transpiled_lcs",
        "s_struct_transpiled_kgram",
        "s_struct_wl_transpiled",
        "f",
        "f_min_score",
    ),
) -> dict[str, Any]:
    """Aggregate generation metrics over finite values and retain valid indices.

    In addition to per-metric mean/n_valid/valid_indices, emits scale-anomaly
    diagnostics for s_gate_raw and s_gate_transpiled:
      s_gate_{suffix}_ratio_n        – number of samples with a non-None ratio
      s_gate_{suffix}_ratio_indices  – their indices
      s_gate_{suffix}_scale_anomaly_n – subset where ratio != 1 (scale mismatch)
    """
    rows = list(samples)
    output: dict[str, Any] = {
        "n_generations": len(rows),
        "n_executable": sum(bool(row.get("candidate_exec_ok", False)) for row in rows),
    }
    for metric in metric_names:
        valid = [
            (index, float(row[metric]))
            for index, row in enumerate(rows)
            if row.get(metric) is not None
            and isinstance(row.get(metric), (int, float, np.number))
            and math.isfinite(float(row[metric]))
        ]
        output[f"{metric}_mean"] = (
            float(np.mean([value for _, value in valid])) if valid else None
        )
        output[f"{metric}_n_valid"] = len(valid)
        output[f"{metric}_valid_indices"] = [index for index, _ in valid]

    # ── S_gate ratio / scale-anomaly diagnostics ──────────────────────────────
    for suffix in ("raw", "transpiled"):
        ratio_key = f"s_gate_{suffix}_ratio"
        ratio_entries = [
            (i, float(row[ratio_key]))
            for i, row in enumerate(rows)
            if row.get(ratio_key) is not None
        ]
        anomaly_indices = [i for i, r in ratio_entries if abs(r - 1.0) > 1e-9]
        output[f"s_gate_{suffix}_ratio_n"]         = len(ratio_entries)
        output[f"s_gate_{suffix}_ratio_indices"]   = [i for i, _ in ratio_entries]
        output[f"s_gate_{suffix}_scale_anomaly_n"] = len(anomaly_indices)

    return output
