"""Safe, deterministic interpreter for DSL programs (INTERFACES.md §1, ``interpreter.py``).

``execute(prog, grid)`` type-checks the AST, evaluates it under step / cell / wall-clock limits and returns a
valid ``Grid`` or raises :class:`ExecError`; it never returns ``None`` and never hangs.  ``typecheck(prog)``
returns the program's output type or raises :class:`DSLTypeError` (a ``TypeError`` subclass).
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional, Tuple

from arcjepa.core.types import Grid, MAX_SIDE, validate_grid
from arcjepa.dsl.ast import Node
from arcjepa.dsl.primitives import MAX_OBJECTS, REGISTRY, ExecContext, ExecError, Primitive
from arcjepa.dsl.types import LEAF_INPUT, LEAF_OBJ, LITERAL_TYPES, Object, T, is_literal_of

__all__ = ["ExecError", "DSLTypeError", "typecheck", "infer_types", "execute", "evaluate", "value_type_ok"]


class DSLTypeError(TypeError):
    """Raised by :func:`typecheck` when an AST violates the typed signatures."""


# ======================================================================================= type checking

def _lookup(node: Node) -> Primitive:
    prim = REGISTRY.get(node.op)
    if prim is None:
        raise DSLTypeError(f"unknown primitive {node.op!r}")
    if len(node.args) != prim.arity:
        raise DSLTypeError(f"{node.op} expects {prim.arity} args, got {len(node.args)}")
    return prim


def _typecheck(node: Node, in_lambda: bool, table: Optional[Dict[Tuple[int, ...], Tuple[T, bool]]],
               path: Tuple[int, ...]) -> T:
    if node.op == LEAF_INPUT:
        if node.args:
            raise DSLTypeError("INPUT takes no arguments")
        out = T.GRID
    elif node.op == LEAF_OBJ:
        if node.args:
            raise DSLTypeError("OBJ takes no arguments")
        if not in_lambda:
            raise DSLTypeError("OBJ used outside a PROGRAM body")
        out = T.OBJECT
    else:
        prim = _lookup(node)
        for i, (arg, t) in enumerate(zip(node.args, prim.arg_types)):
            if t is T.PROGRAM:
                if not isinstance(arg, Node):
                    raise DSLTypeError(f"{node.op} arg {i}: PROGRAM body must be a node")
                bt = _typecheck(arg, True, table, path + (i,))
                if bt not in (T.OBJECT, T.PROGRAM):
                    raise DSLTypeError(f"{node.op} arg {i}: PROGRAM body must produce OBJECT, got {bt.value}")
            elif isinstance(arg, Node):
                at = _typecheck(arg, in_lambda, table, path + (i,))
                if at is not t:
                    raise DSLTypeError(f"{node.op} arg {i}: expected {t.value}, got {at.value}")
            else:
                if t not in LITERAL_TYPES:
                    raise DSLTypeError(f"{node.op} arg {i}: type {t.value} has no literals (got {arg!r})")
                if not is_literal_of(arg, t):
                    raise DSLTypeError(f"{node.op} arg {i}: {arg!r} is not a {t.value} literal")
                allowed = prim.literal_args.get(i)
                if allowed is not None and not _in_domain(arg, allowed):
                    raise DSLTypeError(f"{node.op} arg {i}: literal {arg!r} not in the allowed domain")
        out = prim.out_type
    if table is not None:
        table[path] = (out, in_lambda)
    return out


def _in_domain(value: Any, allowed: Any) -> bool:
    for v in allowed:
        if type(v) is type(value) and v == value:
            return True
    return False


def typecheck(prog: Node, *, in_lambda: bool = False) -> T:
    """Return the output type of ``prog`` or raise :class:`DSLTypeError` (a ``TypeError``)."""
    if not isinstance(prog, Node):
        raise DSLTypeError(f"program must be a Node, got {type(prog).__name__}")
    return _typecheck(prog, in_lambda, None, ())


def infer_types(prog: Node, *, in_lambda: bool = False) -> Dict[Tuple[int, ...], Tuple[T, bool]]:
    """Map every sub-node path to ``(type, inside_lambda)``; raises :class:`DSLTypeError` on mismatch."""
    table: Dict[Tuple[int, ...], Tuple[T, bool]] = {}
    _typecheck(prog, in_lambda, table, ())
    return table


# ======================================================================================= value checks

def value_type_ok(value: Any, t: T, max_cells: int = MAX_SIDE * MAX_SIDE) -> bool:
    """Cheap runtime check that ``value`` is a well-formed value of type ``t``."""
    if t is T.GRID:
        if not isinstance(value, list) or not value or not isinstance(value[0], list) or not value[0]:
            return False
        h, w = len(value), len(value[0])
        if h > MAX_SIDE or w > MAX_SIDE or h * w > max_cells:
            return False
        return validate_grid(value)
    if t is T.OBJECT_SET:
        return isinstance(value, list) and len(value) <= MAX_OBJECTS and all(
            isinstance(o, Object) and len(o.cells) > 0 for o in value)
    if t is T.OBJECT:
        return isinstance(value, Object) and len(value.cells) > 0
    if t is T.MASK:
        return isinstance(value, list) and bool(value) and isinstance(value[0], list)
    if t is T.COLOR:
        return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 9
    if t is T.INTEGER:
        return isinstance(value, int) and not isinstance(value, bool)
    if t is T.POSITION:
        return isinstance(value, str) or (isinstance(value, tuple) and len(value) == 2)
    if t is T.BOOLEAN:
        return isinstance(value, bool)
    if t is T.RELATION:
        return isinstance(value, str)
    if t is T.PROGRAM:
        return isinstance(value, Node)
    return False  # pragma: no cover


# ======================================================================================= evaluation

class _Evaluator:
    """Recursive evaluator with step / cell / time budgets."""

    def __init__(self, ctx: ExecContext, max_steps: int, max_cells: int, deadline: float) -> None:
        self.ctx = ctx
        self.max_steps = max_steps
        self.max_cells = max_cells
        self.deadline = deadline
        self.steps = 0

    def eval(self, node: Node, env: Optional[Object]) -> Any:
        self.steps += 1
        if self.steps > self.max_steps:
            raise ExecError(f"step budget {self.max_steps} exceeded")
        if time.perf_counter() > self.deadline:
            raise ExecError("time budget exceeded")
        op = node.op
        if op == LEAF_INPUT:
            return self.ctx.input
        if op == LEAF_OBJ:
            if env is None:
                raise ExecError("OBJ evaluated outside a PROGRAM body")
            return env
        prim = REGISTRY.get(op)
        if prim is None or len(node.args) != prim.arity:
            raise ExecError(f"bad node {op}")
        if prim.lazy:  # IF: evaluate the condition, then only the taken branch
            cond = self._arg(node.args[0], prim.arg_types[0], env)
            branch = node.args[1] if cond else node.args[2]
            if not isinstance(branch, Node):
                raise ExecError("IF branches must be nodes")
            return self.eval(branch, env)
        args = [self._arg(a, t, env) for a, t in zip(node.args, prim.arg_types)]
        if prim.needs_ctx:
            args.insert(0, self.ctx)
        if prim.higher_order:
            args.append(self._lambda_evaluator())
        try:
            out = prim.fn(*args)
        except ExecError:
            raise
        except (RecursionError, MemoryError) as e:  # pragma: no cover - defensive
            raise ExecError(f"{op}: {type(e).__name__}") from e
        except Exception as e:
            raise ExecError(f"{op}: {type(e).__name__}: {e}") from e
        if not value_type_ok(out, prim.out_type, self.max_cells):
            raise ExecError(f"{op} produced an invalid {prim.out_type.value}")
        return out

    def _arg(self, arg: Any, t: T, env: Optional[Object]) -> Any:
        if t is T.PROGRAM:
            if not isinstance(arg, Node):
                raise ExecError("PROGRAM argument must be a node")
            prim = REGISTRY.get(arg.op)
            if prim is not None and prim.out_type is T.PROGRAM:
                return self.eval(arg, env)  # e.g. COMPOSE -> lambda body
            return arg
        if isinstance(arg, Node):
            return self.eval(arg, env)
        return arg

    def _lambda_evaluator(self) -> Callable[[Node, Object], Any]:
        def ev(body: Node, obj: Object) -> Any:
            return self.eval(body, obj)
        return ev


def evaluate(prog: Node, grid: Grid, *, max_steps: int = 10_000, max_cells: int = 900,
             timeout_s: float = 0.5) -> Any:
    """Evaluate any well-typed expression on ``grid`` and return its (typed) value; raises ExecError."""
    if not validate_grid(grid):
        raise ExecError("input is not a valid grid")
    try:
        typecheck(prog)
    except TypeError as e:
        raise ExecError(f"type error: {e}") from e
    ctx = ExecContext.of(grid)
    ev = _Evaluator(ctx, max_steps, max_cells, time.perf_counter() + timeout_s)
    return ev.eval(prog, None)


def execute(prog: Node, grid: Grid, *, max_steps: int = 10_000, max_cells: int = 900,
            timeout_s: float = 0.5) -> Grid:
    """Run a GRID-typed program on ``grid``; returns a valid Grid or raises :class:`ExecError`."""
    if not validate_grid(grid):
        raise ExecError("input is not a valid grid")
    try:
        t = typecheck(prog)
    except TypeError as e:
        raise ExecError(f"type error: {e}") from e
    if t is not T.GRID:
        raise ExecError(f"program produces {t.value}, not GRID")
    ctx = ExecContext.of(grid)
    ev = _Evaluator(ctx, max_steps, max_cells, time.perf_counter() + timeout_s)
    out = ev.eval(prog, None)
    if not value_type_ok(out, T.GRID, max_cells):
        raise ExecError("program produced an invalid grid")
    return out
