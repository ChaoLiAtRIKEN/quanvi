from __future__ import annotations
import string
import inspect
from functools import wraps
from typing import Any, Optional
import numpy as np
import opt_einsum as oe
import torch
from tensor_factory import conj_transpose, LocalMBuilder

def local_operator_support(D: int, local_idx: int) -> tuple[set[int], list[int]]:
    """Physical sites covered by the local term and op_sites order for string building."""
    if local_idx == 0:
        return {0, 1}, [0, 1]
    if local_idx == D - 1:
        return {D - 2, D - 1}, [D - 2, D - 1]
    sites = [local_idx - 1, local_idx, local_idx + 1]
    return set(sites), sites


def first_core_index_overlapping_local(Nk: int, Num_gates: int, local_range: set[int]) -> Optional[int]:
    """Smallest idx whose core window intersects local_range; None if no gate overlaps."""
    for idx in range(Num_gates):
        core_sites = range(idx, idx + Nk)
        if local_range.intersection(core_sites):
            return idx
    return None


def num_state_replicas(idx: int, Num_gates: int, Nk: int) -> int:
    """How many state / conj(state) tensors are contracted at this core step (matches inference)."""
    return Nk if idx == Num_gates - 1 else 1


def local_gate_attachment(
    D: int, Nk: int, Num_gates: int, local_idx: int, idx: int
) -> tuple[Optional[int], bool]:
    """
    idx_first: smallest core index overlapping this local term.
    attach_m: whether this (idx, local_idx) step should include the local operator tensor.
    """
    local_range, _ = local_operator_support(D, local_idx)
    idx_first = first_core_index_overlapping_local(Nk, Num_gates, local_range)
    attach = idx_first is not None and idx == idx_first
    return idx_first, attach


def default_local_m_operators_torch(
    D: int, K: int, dtype: torch.dtype, device: torch.device
) -> list[torch.Tensor]:
    """
    Placeholder local blocks (identity on the appropriate product space).
    Shapes match tensor_factory.build_M_loc_3local_new documentation: (K^2,K^2) at
    boundaries and (K^3,K^3) in the bulk, reshaped as (K,)*4 / (K,)*6.
    """
    out: list[torch.Tensor] = []
    eye2 = torch.eye(K * K, dtype=dtype, device=device).reshape(K, K, K, K)
    eye3 = torch.eye(K * K * K, dtype=dtype, device=device).reshape((K,) * 6)
    out.append(eye2)
    for _ in range(1, D - 1):
        out.append(eye3)
    out.append(eye2)
    return out


def element_string_builder(D: int, Nk: int, idx: int, Num_gates: int, local_idx: int) -> dict[str, Any]:
    pool = list(string.ascii_lowercase + string.ascii_uppercase)

    def new_index() -> str:
        return pool.pop(0)
    # build core string
    core_left = [new_index() for _ in range(Nk * 2)]
    core_right = [new_index() for _ in range(Nk * 2)]

    # build state string
    count = num_state_replicas(idx, Num_gates, Nk)
    state_left = ",".join(core_left[i] for i in range(count))
    state_right = ",".join(core_right[Nk + i] for i in range(count))

    # build local operator string 
    core_sites = list(range(idx, idx + Nk))
    core_range = set(core_sites)
    core_site_to_offset = {q: q - idx for q in core_sites}

    local_range, op_sites = local_operator_support(D, local_idx)
    has_local_overlap = bool(core_range & local_range)

    idx_first, attach_m = local_gate_attachment(D, Nk, Num_gates, local_idx, idx)
    op_str: Optional[str] = None
    touched_offsets: set[int] = set()
    if attach_m:
        op_left: list[str] = []
        op_right: list[str] = []
        for q in op_sites:
            if q in core_range:
                off = core_site_to_offset[q]
                op_left.append(core_left[Nk + off])
                op_right.append(core_right[off])
                touched_offsets.add(off)
            else:
                op_left.append(new_index())
                op_right.append(new_index())
        op_str = "".join(op_left + op_right)

    if idx == 0:
        candidate_offsets = range(Nk)
    else:
        candidate_offsets = [Nk - 1]
    for off in candidate_offsets:
        if off not in touched_offsets and (idx + off) not in local_range:
            core_right[off] = core_left[Nk + off]

    return {
        "core_left": core_left,
        "core_right": core_right,
        "state_left": state_left,
        "state_right": state_right,
        "core_sites": core_sites,
        "core_range": core_range,
        "local_range": local_range,
        "op_sites": op_sites,
        "has_local_overlap": has_local_overlap,
        "idx_first_overlap": idx_first,
        "build_local_here": attach_m,
        "candidate_offsets": list(candidate_offsets),
        "touched_offsets": touched_offsets,
        "op_str": op_str,
        "num_state_replicas": count,
    }


def free_indices_after_contraction(parts: list[str]) -> str:
    """
    Subscript for einsum output: labels that appear exactly once across all inputs
    (standard numpy/opt_einsum rule — those axes are not summed and remain open).
    Order follows first occurrence when scanning operands left-to-right.
    """
    counts: dict[str, int] = {}
    for p in parts:
        for ch in p:
            if ch.isalpha():
                counts[ch] = counts.get(ch, 0) + 1
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        for ch in p:
            if ch.isalpha() and counts.get(ch, 0) == 1 and ch not in seen:
                seen.add(ch)
                out.append(ch)
    return "".join(out)

def canonical_free_order(Nk: int, op: dict, free: str) -> list[str]:
    """Order free indices deterministically for any Nk >= 1 and state count >= 0:
1. Uncontracted upper bonds, core_left[count:Nk].
2. Uncontracted lower bonds, core_right[Nk+count:2*Nk].
3. Remaining free indices from core_left, in their original order.
4. Remaining free indices from core_right, in their original order.
5. Remaining free indices from the local operator, in its original order."""
    core_left = op["core_left"]
    core_right = op["core_right"]
    op_str = op["op_str"]
    count = op["num_state_replicas"]
    free_set = set(free)

    order: list[str] = []

    upper_bonds = core_left[count : Nk]
    lower_bonds = core_right[Nk + count : 2 * Nk]
    for ch in list(upper_bonds) + list(lower_bonds):
        if ch in free_set and ch not in order:
            order.append(ch)

    for ch in core_left:
        if ch in free_set and ch not in order:
            order.append(ch)
    for ch in core_right:
        if ch in free_set and ch not in order:
            order.append(ch)
    if op_str:
        for ch in op_str:
            if ch in free_set and ch not in order:
                order.append(ch)

    return order


def build_axis_schema(Nk: int, op: dict, canonical: list[str], idx: int) -> list[tuple]:
    """Label each axis of F, whose axes follow canonical_free_order, by its source:
- ("ket_ax", k): axis k from core_left.
- ("bra_ax", k): an axis from core_right mapped back to core axis k.
- ("op_bra", q) or ("op_ket", q): an operator axis at site q.
After conj_transpose with permutation [Nk..2Nk-1] + [0..Nk-1],
core_right[pos] maps to original core axis (pos + Nk) mod (2*Nk)."""
    core_left = op["core_left"]
    core_right = op["core_right"]
    op_str = op["op_str"]
    op_sites = op["op_sites"]

    schema: list[tuple] = []
    for ch in canonical:
        role: Optional[tuple] = None
        for pos, c in enumerate(core_left):
            if c == ch:
                role = ("ket_ax", pos)
                break
        if role is None:
            for pos, c in enumerate(core_right):
                if c == ch:
                    core_axis = (pos + Nk) % (2 * Nk)
                    role = ("bra_ax", core_axis)
                    break
        if role is None and op_str:
            n_sites = len(op_sites)
            for k, c in enumerate(op_str):
                if c == ch:
                    if k < n_sites:
                        role = ("op_bra", op_sites[k])
                    else:
                        role = ("op_ket", op_sites[k - n_sites])
                    break
        if role is None:
            role = ("unknown", -1)
        schema.append(role)
    return schema


def match_schemas(
    prev_schema: list[tuple],
    cur_schema: list[tuple],
    count_prev: int,
    Nk: int,
    idx_cur: int,
) -> list[tuple[int, int]]:
    """Match axes shared by adjacent gates and therefore contracted by einsum.
For a sliding window:
1. Pair ket and bra sliding bonds between the previous gate's right outputs
   and the current gate's left inputs.
2. Pair operator legs extending beyond one gate with the corresponding core
   axes in the next gate.
3. Carry unmatched operator legs forward with the same site and side labels."""
    pairs: list[tuple[int, int]] = []
    used_prev: set[int] = set()
    used_cur: set[int] = set()

    for side in ("ket_ax", "bra_ax"):
        for k_slide in range(Nk - count_prev):
            prev_axis = count_prev + k_slide
            cur_axis = Nk + k_slide
            pi = ci = None
            for i, sf in enumerate(prev_schema):
                if i in used_prev:
                    continue
                if sf == (side, prev_axis):
                    pi = i
                    break
            for j, sc in enumerate(cur_schema):
                if j in used_cur:
                    continue
                if sc == (side, cur_axis):
                    ci = j
                    break
            if pi is not None and ci is not None:
                pairs.append((pi, ci))
                used_prev.add(pi)
                used_cur.add(ci)

    op_to_ax = {"op_bra": "ket_ax", "op_ket": "bra_ax"}
    for i, sf in enumerate(prev_schema):
        if i in used_prev:
            continue
        kind, val = sf
        if kind not in op_to_ax:
            continue
        q = val
        target_axis = Nk + (q - idx_cur)
        if not (0 <= target_axis < 2 * Nk):
            continue
        target = (op_to_ax[kind], target_axis)
        for j, sc in enumerate(cur_schema):
            if j in used_cur:
                continue
            if sc == target:
                pairs.append((i, j))
                used_prev.add(i)
                used_cur.add(j)
                break

    for i, sf in enumerate(prev_schema):
        if i in used_prev:
            continue
        if sf[0] not in ("op_bra", "op_ket"):
            continue
        for j, sc in enumerate(cur_schema):
            if j in used_cur:
                continue
            if sc == sf:
                pairs.append((i, j))
                used_prev.add(i)
                used_cur.add(j)
                break

    return pairs


def build_chain_einsum(
    prev_schema: list[tuple],
    cur_schema: list[tuple],
    count_prev: int,
    Nk: int,
    idx_cur: int,
) -> tuple[str, list[tuple]]:
    """Build the chain einsum string from matched axes and return its output schema.
Unmatched axes remain ordered with previous-gate axes before current-gate axes."""
    pool = list(string.ascii_lowercase + string.ascii_uppercase)
    pairs = match_schemas(prev_schema, cur_schema, count_prev, Nk, idx_cur)
    matched_prev = {i for i, _ in pairs}
    matched_cur = {j for _, j in pairs}

    prev_letters = [""] * len(prev_schema)
    cur_letters = [""] * len(cur_schema)
    for i, j in pairs:
        ch = pool.pop(0)
        prev_letters[i] = ch
        cur_letters[j] = ch
    for i in range(len(prev_schema)):
        if prev_letters[i] == "":
            prev_letters[i] = pool.pop(0)
    for j in range(len(cur_schema)):
        if cur_letters[j] == "":
            cur_letters[j] = pool.pop(0)

    out_letters: list[str] = []
    new_schema: list[tuple] = []
    for i, sf in enumerate(prev_schema):
        if i not in matched_prev:
            out_letters.append(prev_letters[i])
            new_schema.append(sf)
    for j, sc in enumerate(cur_schema):
        if j not in matched_cur:
            out_letters.append(cur_letters[j])
            new_schema.append(sc)

    einsum_str = (
        "".join(prev_letters) + "," + "".join(cur_letters) + "->" + "".join(out_letters)
    )
    return einsum_str, new_schema

def build_einsum_string_for_core_layer(Nk: int, op: dict[str, Any]) -> str:
    """
    Single einsum path consistent with prepare_einsum_operation tensor order:
    core, conj_core, state (count copies), conj(state) (count copies), optional local m.

    Output subscript after ``->`` lists every index that appears exactly once on the
    LHS (uncontracted / carried intermediate), not an implicit full trace to scalar.
    """
    core_left: list[str] = op["core_left"]
    core_right: list[str] = op["core_right"]
    state_left: str = op["state_left"]
    state_right: str = op["state_right"]
    op_str: Optional[str] = op["op_str"]
    count: int = op["num_state_replicas"]

    parts: list[str] = [
        "".join(core_left),
        "".join(core_right),
        *state_left.split(","),
        *state_right.split(","),
    ]
    if op_str is not None:
        parts.append(op_str)
    free = free_indices_after_contraction(parts)
    return ",".join(parts) + "->" + free


def build_close_einsum(schema: list[tuple]) -> tuple[Optional[str], list[tuple]]:
    """Build the final trace einsum from the remaining axis schema.

Pair matching ket and bra core axes, matching operator bra and ket legs at
the same site, and compatible operator/core legs. The plan depends only on
the schema, so it can be cached with the einsum string.

Returns the einsum string (or None) and the schema left after tracing."""
    pairs: list[tuple[int, int]] = []
    used: set[int] = set()

    def _try_pair(pred):
        for i in range(len(schema)):
            if i in used:
                continue
            for j in range(len(schema)):
                if j in used or j == i:
                    continue
                if pred(schema[i], schema[j]):
                    pairs.append((i, j))
                    used.add(i)
                    used.add(j)
                    break

    _try_pair(lambda a, b: a[0] == "ket_ax" and b[0] == "bra_ax" and a[1] == b[1])
    _try_pair(lambda a, b: a[0] == "op_bra" and b[0] == "op_ket" and a[1] == b[1])
    _try_pair(lambda a, b: a[0] == "op_bra" and b[0] == "ket_ax")
    _try_pair(lambda a, b: a[0] == "op_ket" and b[0] == "bra_ax")

    if not pairs:
        return None, schema

    letters = list(string.ascii_lowercase + string.ascii_uppercase)
    axis_letter = [""] * len(schema)
    for i, j in pairs:
        ch = letters.pop(0)
        axis_letter[i] = ch
        axis_letter[j] = ch
    for k in range(len(schema)):
        if axis_letter[k] == "":
            axis_letter[k] = letters.pop(0)
    remaining = [axis_letter[k] for k in range(len(schema)) if k not in used]
    einsum_str = "".join(axis_letter) + "->" + "".join(remaining)
    return einsum_str, [schema[k] for k in range(len(schema)) if k not in used]


def fuse_gate_with_chain(einsum_str, perm, chain_str):
    """Substitute the gate expression into the chain, preserving canonical axes.

    Contracting the ket and bra in isolation can form an outer product with
    K**(4*Nk-2) entries. Let the left environment consume their open legs first.
    Only dummy index names and contraction order change; no truncation occurs.
    """
    lhs, free = einsum_str.split("->")
    canonical = "".join(free[i] for i in perm) if perm is not None else free
    chain_lhs, output = chain_str.split("->")
    previous, current = chain_lhs.split(",")
    names = dict(zip(canonical, current))
    used = set(chain_lhs.replace(",", "") + output)
    next_symbol = 0
    for ch in lhs.replace(",", ""):
        if ch not in names:
            while oe.get_symbol(next_symbol) in used:
                next_symbol += 1
            names[ch] = oe.get_symbol(next_symbol)
            used.add(names[ch])
    operands = ["".join(names[ch] for ch in term) for term in lhs.split(",")]
    return previous + "," + ",".join(operands) + "->" + output


_PLAN_CACHE: dict[tuple, list[dict[str, Any]]] = {}


def build_loss_plan(D: int, E: int, Nk: int, Num_gates: int) -> list[dict[str, Any]]:
    """Precompute and cache the contraction plan for each local loss term.

The einsum strings, permutations, and schema matches depend only on
(D, E, Nk, Num_gates), not on tensor values. Build them once and reuse them
throughout optimization."""
    key = (D, E, Nk, Num_gates)
    if key in _PLAN_CACHE:
        return _PLAN_CACHE[key]

    # Gates without m depend only on the cores and state, so their results
    # can be shared across local terms when the einsum and permutation match.
    # Chain prefixes before m is attached are shareable for the same reason.
    # Assign IDs here so get_loss can memoize these intermediates per call.
    gate_ids: dict[tuple, int] = {}
    chain_ids: dict[tuple, int] = {}

    plans: list[dict[str, Any]] = []
    for local_idx in range(D):
        gates: list[dict[str, Any]] = []
        former_schema: list[tuple] = []
        former_count = 0
        running: Optional[tuple] = None  # Shared chain key, or None when not reusable.
        for idx in range(Num_gates):
            op = element_string_builder(D, Nk, idx, Num_gates, local_idx)
            einsum_str = build_einsum_string_for_core_layer(Nk, op)

            # Build the structural part of the former permute_F_to_canonical step.
            free = einsum_str.split("->")[1]
            canonical = canonical_free_order(Nk, op, free)
            perm = None if list(free) == canonical else [free.index(c) for c in canonical]

            cur_schema = build_axis_schema(Nk, op, canonical, idx)
            cur_count = op["num_state_replicas"]

            chain_str: Optional[str] = None
            if idx == 0:
                former_schema = cur_schema
            else:
                chain_str, former_schema = build_chain_einsum(
                    former_schema, cur_schema, former_count, Nk, idx
                )
            # The chain output's right side comes from this gate; the next slide uses cur_count.
            former_count = cur_count

            attach_m = op["op_str"] is not None
            if attach_m:
                gate_id = None
            else:
                gkey = (idx, einsum_str, tuple(perm or ()))
                gate_id = gate_ids.setdefault(gkey, len(gate_ids))

            if chain_str is None:
                # idx == 0: former_F is this gate's output.
                chain_id = None
                running = None if gate_id is None else ("g", gate_id)
            elif running is None or gate_id is None:
                # Once m is in the chain (or attached here), later results depend on local_idx.
                chain_id = None
                running = None
            else:
                running = ("c", running, gate_id, chain_str)
                chain_id = chain_ids.setdefault(running, len(chain_ids))

            gates.append(
                {
                    "einsum_str": einsum_str,
                    "perm": perm,
                    "chain_str": chain_str,
                    "fused_str": (fuse_gate_with_chain(einsum_str, perm, chain_str)
                                  if chain_str is not None else None),
                    "count": cur_count,
                    "attach_m": attach_m,
                    "gate_id": gate_id,
                    "chain_id": chain_id,
                }
            )

        close_str, remaining_schema = build_close_einsum(former_schema)
        plans.append(
            {"gates": gates, "close_str": close_str, "schema": remaining_schema}
        )

    _PLAN_CACHE[key] = plans
    return plans


_EXPR_CACHE: dict[tuple, Any] = {}


def cached_contract(einsum_str: str, *operands, optimize: str = "greedy"):
    """Contract operands using a cached opt_einsum path for each expression and shape tuple."""
    shapes = tuple(tuple(o.shape) for o in operands)
    key = (einsum_str, shapes, optimize)
    expr = _EXPR_CACHE.get(key)
    if expr is None:
        expr = oe.contract_expression(einsum_str, *shapes, optimize=optimize)
        _EXPR_CACHE[key] = expr
    return expr(*operands)


def _compatible_loss_init(init):
    """Accept both the current API and the pre-projection-removal API."""
    signature = inspect.signature(init)
    @wraps(init)
    def wrapped(self, *args, **kwargs):
        # Old positional calls contain R between B and dtype, and M after cores.
        if len(args) > 5 and not isinstance(args[5], torch.dtype):
            names = ("D", "E", "K", "Nk", "B", "R", "dtype", "state",
                     "einsum_builder", "core_tensor", "M", "optimizer", "init_m",
                     "local_m_operators")
            if len(args) > len(names):
                raise TypeError("Too many positional arguments for LossCalculator")
            for name, value in zip(names, args):
                if name in kwargs:
                    raise TypeError(f"LossCalculator got multiple values for argument '{name}'")
                kwargs[name] = value
            args = ()
        rank = kwargs.pop("R", None)
        matrix = kwargs.pop("M", None)
        bound = signature.bind(self, *args, **kwargs)
        init(*bound.args, **bound.kwargs)
        self.R = self.K if rank is None else rank
        self.M = matrix
    return wrapped


class LossCalculator:
    """Core-local loss contraction driven by operation_builder (indices) + prepare_einsum_operation (tensors)."""

    @_compatible_loss_init
    def __init__(
        self,
        D: int,
        E: int,
        K: int,
        Nk: int,
        B: int,
        dtype: torch.dtype,
        state: torch.Tensor,
        einsum_builder: Any,
        core_tensor: torch.nn.ParameterList,
        optimizer: str = "greedy",
        init_m: Any | None = None,
        local_m_operators: list[torch.Tensor] | None = None,
    ):
        self.D = D
        self.E = E
        self.K = K
        self.Nk = Nk
        self.B = B
        self.dtype = dtype
        self.state = state
        self.einsum_builder = einsum_builder
        self.core_tensor = core_tensor
        self.optimizer = optimizer
        self.init_m = init_m
        self.local_m_operators = local_m_operators
        self.Num_gates = D + E - Nk + 1

    def _build_local_m(self, device) -> list[torch.Tensor]:
        """Resample from the proposal and estimate the D local operators M^(d)."""
        from torch.distributions import Independent, Uniform
        from utils.basis_function import HermitePolynomials

        lo, hi = float(self.init_m.x_range[0]), float(self.init_m.x_range[1])
        low = torch.full((self.D,), lo, dtype=torch.float32, device=device)
        high = torch.full((self.D,), hi, dtype=torch.float32, device=device)
        proposal_dist = Independent(Uniform(low, high), 1)
        basis_function = HermitePolynomials(dim=self.D, num_basis_functions=self.K)
        builder_kwargs = dict(
            D=self.D, K=self.K, proposal_dist=proposal_dist,
            basis_function=basis_function,
            transform=getattr(self.init_m, "transform", None),
            x_range=self.init_m.x_range,
        )
        # Servers may still have the original tensor_factory.py. Preserve its
        # projection and sampling arguments instead of shifting positional args.
        if "M" in inspect.signature(LocalMBuilder).parameters:
            builder_kwargs.update(M=self.M, B=self.B, R=self.R,
                                  proj_matrix=getattr(self.init_m, "proj_matrix", None))
        elif self.R != self.K or (
            getattr(self.init_m, "proj_matrix", None) is not None
            and not np.array_equal(self.init_m.proj_matrix, np.eye(self.K))
        ):
            raise ValueError("Projected models require the original tensor_factory.py")
        lmb = LocalMBuilder(**builder_kwargs)
        num_samples = max(self.B, self.D * 100)
        m_list_np = lmb.build_M_loc_3local_new(
            self.init_m.target_score,
            num_samples=num_samples,
            symmetrize=True,
            importance_weight=True,
        )
        return [
            torch.as_tensor(np.asarray(a), dtype=self.dtype, device=device)
            for a in m_list_np
        ]

    def local_m_operator_list(self) -> list[torch.Tensor]:
        device = self.state.device
        if self.local_m_operators is not None:
            if len(self.local_m_operators) != self.D:
                raise ValueError(
                    "local_m_operators must contain one tensor per physical "
                    f"dimension; expected {self.D}, got "
                    f"{len(self.local_m_operators)}."
                )
            return [
                torch.as_tensor(m, dtype=self.dtype, device=device)
                for m in self.local_m_operators
            ]

        if self.init_m is None:
            bond_dim = int(self.core_tensor[0].shape[0])
            return default_local_m_operators_torch(
                self.D, bond_dim, self.dtype, device
            )

        # Re-estimate each local operator at every call. The loss is linear in M, so
        # fresh unbiased Monte Carlo estimates yield an unbiased stochastic gradient.
        # Freezing one estimate would instead optimize a fixed noisy objective.
        return self._build_local_m(device)

    def get_loss(self) -> torch.Tensor:
        device = self.state.device
        m_list = self.local_m_operator_list()
        plans = build_loss_plan(self.D, self.E, self.Nk, self.Num_gates)

        # These tensors do not depend on local_idx, so compute them once per step.
        conj_cores = [conj_transpose(c, self.Nk) for c in self.core_tensor]
        conj_state = torch.conj(self.state)

        # Core tensors stay fixed within get_loss, so m-independent intermediates can be reused.
        gate_cache: dict[int, torch.Tensor] = {}
        chain_cache: dict[int, torch.Tensor] = {}

        total: torch.Tensor = torch.tensor(0.0, device=device, dtype=self.dtype)
        for local_idx in range(self.D):
            m = m_list[local_idx]
            plan = plans[local_idx]
            former_F: Optional[torch.Tensor] = None

            for idx, gate in enumerate(plan["gates"]):
                # Once a left environment exists, contract the next gate into
                # it directly. This avoids materializing the ket/bra gate
                # outer product for every Nk; it changes only contraction order.
                if former_F is not None:
                    chain_id = gate["chain_id"]
                    if chain_id is not None and chain_id in chain_cache:
                        former_F = chain_cache[chain_id]
                        continue
                    operation = [former_F, self.core_tensor[idx], conj_cores[idx]]
                    operation.extend([self.state] * gate["count"])
                    operation.extend([conj_state] * gate["count"])
                    if gate["attach_m"]:
                        operation.append(m)
                    former_F = cached_contract(gate["fused_str"], *operation, optimize=self.optimizer)
                    if chain_id is not None:
                        chain_cache[chain_id] = former_F
                    continue
                gate_id = gate["gate_id"]
                F = gate_cache.get(gate_id) if gate_id is not None else None
                if F is None:
                    # core, conj(core), state * count, conj(state) * count, [local m]
                    operation = [self.core_tensor[idx], conj_cores[idx]]
                    operation.extend([self.state] * gate["count"])
                    operation.extend([conj_state] * gate["count"])
                    if gate["attach_m"]:
                        operation.append(m)

                    F = cached_contract(
                        gate["einsum_str"], *operation, optimize=self.optimizer
                    )
                    if gate["perm"] is not None:
                        F = F.permute(*gate["perm"]).contiguous()
                    if gate_id is not None:
                        gate_cache[gate_id] = F

                if former_F is None:
                    former_F = F
                else:
                    chain_id = gate["chain_id"]
                    cached = chain_cache.get(chain_id) if chain_id is not None else None
                    if cached is not None:
                        former_F = cached
                    else:
                        former_F = cached_contract(
                            gate["chain_str"], former_F, F, optimize=self.optimizer
                        )
                        if chain_id is not None:
                            chain_cache[chain_id] = former_F

            if plan["close_str"] is not None:
                former_F = cached_contract(
                    plan["close_str"], former_F, optimize=self.optimizer
                )
            if former_F.ndim != 0:
                raise AssertionError(
                    f"[local_idx={local_idx}] {former_F.ndim} axes remain open after chain contraction; "
                    f"remaining schema = {plan['schema']}. Check the index alias rules or operator support."
                )
            total = total + former_F
        return total.real
