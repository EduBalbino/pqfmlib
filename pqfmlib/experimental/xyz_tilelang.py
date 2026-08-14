"""Fused XYZ expectation evaluation on CUDA with TileLang."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import tilelang
import tilelang.language as T
import torch
from numpy.typing import ArrayLike, NDArray

RX, RY, RXX, RYY, PHASE_RUN = range(5)
AXIS = {"x": 0, "y": 1, "z": 2}
BATCH_SIZE = 128
FWHT_TILE_BITS = 12
FWHT_TILE_SIZE = 1 << FWHT_TILE_BITS
WALSH_BITS = FWHT_TILE_BITS // 2
WALSH_SIZE = 1 << WALSH_BITS
CUT_QUBITS = 18
CUT_HALF_BITS = 9
CUT_HALF_WIDTH = 1 << CUT_HALF_BITS
CUT_MAX_BRANCHES = 64
SCHMIDT_RANK = 8
FUSED_GEMM_TILE = WALSH_SIZE
CUT_TILE_COUNT = (1 << CUT_QUBITS) // FWHT_TILE_SIZE
CUT_MAX_MASKS = CUT_QUBITS * (CUT_QUBITS + 1) // 2
WALSH_64 = np.asarray(
    [
        [
            1 - 2 * ((row & column).bit_count() & 1)
            for column in range(WALSH_SIZE)
        ]
        for row in range(WALSH_SIZE)
    ],
    np.float16,
)


def _normalize_edges(edges, n_qubits):
    normalized = []
    for edge in edges:
        try:
            first, second = sorted(map(int, edge))
        except (TypeError, ValueError):
            raise ValueError("Each edge must contain two qubit indices") from None
        if first < 0 or first == second or second >= n_qubits:
            raise ValueError(f"Invalid edge {edge!r} for {n_qubits} qubits")
        normalized.append((first, second))
    return tuple(sorted(normalized))


def _q18_cut_layout(parameter_count, edges):
    """Build the one 9+9 layout implemented by the q18 kernel."""
    parameters_per_layer = 3 * (CUT_QUBITS + len(edges))
    if parameter_count % parameters_per_layer:
        raise ValueError("Theta columns do not fit diagonal XYZ layers")
    layers = parameter_count // parameters_per_layer
    locations = np.asarray(
        [divmod(qubit, CUT_HALF_BITS) for qubit in range(CUT_QUBITS)],
        np.int32,
    )
    cut_edges = []
    crossing_edges = 0
    for first, second in edges:
        first_side, first_position = locations[first]
        second_side, second_position = locations[second]
        if first_side == second_side:
            cut_edges.append((first_side, first_position, second_position))
        else:
            crossing_edges += 1
            cut_edges.append((-1, first_position, second_position))
    if 3 * layers * crossing_edges > CUT_MAX_BRANCHES.bit_length() - 1:
        raise ValueError("q18 requires the fixed 9+9 cut with at most 64 branches")
    return layers, locations, np.asarray(cut_edges, np.int32)


def _operation_plan(n_qubits, parameter_count, axes, edges):
    """Recover PQFMLib XYZ's logical circuit from its native theta layout."""
    axis_count = len(axes)
    edge_count = len(edges)
    parameters_per_layer = axis_count * (n_qubits + edge_count)
    if parameter_count % parameters_per_layer:
        raise ValueError(
            "Theta columns do not fit PQFMLib's diagonal XYZ parameter layout"
        )
    layers = parameter_count // parameters_per_layer
    local_size = layers * n_qubits
    pair_offset = axis_count * local_size
    pair_size = layers * edge_count
    local_kinds = {"x": RX, "y": RY}
    pair_kinds = {"x": RXX, "y": RYY}
    plan, phase_masks, phase_parameters = [], [], []

    def append_phase_run(entries):
        start = len(phase_masks)
        for mask, parameter in entries:
            phase_masks.append(mask)
            phase_parameters.append(parameter)
        count = len(phase_masks) - start
        phase_masks.extend([0] * (n_qubits - count))
        phase_parameters.extend([0] * (n_qubits - count))
        plan.append((PHASE_RUN, start, count))

    for layer in range(layers):
        for axis_position, axis in enumerate(axes):
            offset = axis_position * local_size + layer * n_qubits
            entries = [
                (1 << qubit, offset + qubit) for qubit in range(n_qubits)
            ]
            if axis == "z":
                append_phase_run(entries)
            else:
                plan.extend(
                    (local_kinds[axis], mask, parameter)
                    for mask, parameter in entries
                )
        for axis_position, axis in enumerate(axes):
            offset = pair_offset + axis_position * pair_size + layer * edge_count
            entries = [
                ((1 << left) | (1 << right), offset + position)
                for position, (left, right) in enumerate(edges)
            ]
            if axis == "z":
                append_phase_run(entries)
            else:
                plan.extend(
                    (pair_kinds[axis], mask, parameter)
                    for mask, parameter in entries
                )
    return (
        np.asarray(plan, np.int32),
        np.asarray(phase_masks, np.int32),
        np.asarray(phase_parameters, np.int32),
    )


def _inputs(parameter_values, n_qubits, axes, observable_edges):
    values = np.asarray(parameter_values, np.float32)
    if values.ndim == 1:
        values = values[None]
    if values.ndim != 2:
        raise ValueError("Expected a two-dimensional PQFMLib theta matrix")
    axes = tuple(str(axis).lower() for axis in axes)
    if invalid := sorted(set(axes) - AXIS.keys()):
        raise ValueError(f"Unsupported observable axes: {invalid}")
    pairs = (
        observable_edges
        if observable_edges is not None
        else (
            (i, j)
            for i in range(n_qubits)
            for j in range(i + 1, n_qubits)
        )
    )
    pairs = tuple(dict.fromkeys(_normalize_edges(pairs, n_qubits)))
    # Qiskit labels Pauli masks in the opposite order from statevector bits.
    masks = [1 << (n_qubits - 1 - qubit) for qubit in range(n_qubits)]
    masks += [
        (1 << (n_qubits - 1 - i)) | (1 << (n_qubits - 1 - j))
        for i, j in pairs
    ]
    return (
        values,
        np.asarray([AXIS[axis] for axis in axes], np.int32),
        np.asarray(masks, np.int32),
    )


def _pair(pair, bit):
    lower = T.bitwise_and(pair, bit - 1)
    zero = lower + T.shift_left(T.bitwise_and(pair, T.bitwise_not(bit - 1)), 1)
    return zero, zero + bit


@T.macro
def _cut_single(states, theta, row, branches, axis, bit, capacity):
    """Apply one local XYZ rotation to every active cut branch."""
    cosine, sine = T.__cos(theta), T.__sin(theta)
    for work in T.Parallel(capacity * CUT_HALF_WIDTH // 2):
        branch = work // (CUT_HALF_WIDTH // 2)
        pair = work % (CUT_HALF_WIDTH // 2)
        if branch < branches:
            zero, one = _pair(pair, bit)
            zr, zi = states[row, branch, 0, zero], states[row, branch, 1, zero]
            or_, oi = states[row, branch, 0, one], states[row, branch, 1, one]
            if axis == 0:
                states[row, branch, 0, zero] = cosine * zr + sine * oi
                states[row, branch, 1, zero] = cosine * zi - sine * or_
                states[row, branch, 0, one] = cosine * or_ + sine * zi
                states[row, branch, 1, one] = cosine * oi - sine * zr
            elif axis == 1:
                states[row, branch, 0, zero] = cosine * zr - sine * or_
                states[row, branch, 1, zero] = cosine * zi - sine * oi
                states[row, branch, 0, one] = sine * zr + cosine * or_
                states[row, branch, 1, one] = sine * zi + cosine * oi
            else:
                states[row, branch, 0, zero] = cosine * zr + sine * zi
                states[row, branch, 1, zero] = cosine * zi - sine * zr
                states[row, branch, 0, one] = cosine * or_ - sine * oi
                states[row, branch, 1, one] = cosine * oi + sine * or_
    T.sync_threads()


@T.macro
def _cut_pair(
    states, theta, row, branches, axis, first_bit, second_bit, capacity
):
    """Apply an intra-partition same-axis Pauli-pair rotation."""
    mask = first_bit | second_bit
    pivot = T.min(first_bit, second_bit)
    cosine, sine = T.__cos(theta), T.__sin(theta)
    for work in T.Parallel(capacity * CUT_HALF_WIDTH // 2):
        branch = work // (CUT_HALF_WIDTH // 2)
        pair = work % (CUT_HALF_WIDTH // 2)
        if branch < branches:
            zero, _ = _pair(pair, pivot)
            one = T.bitwise_xor(zero, mask)
            zr, zi = states[row, branch, 0, zero], states[row, branch, 1, zero]
            or_, oi = states[row, branch, 0, one], states[row, branch, 1, one]
            if axis == 2:
                selected = T.bitwise_and(zero, mask)
                odd = T.bitwise_xor(selected != 0, selected == mask)
                signed_z = T.if_then_else(odd, sine, -sine)
                states[row, branch, 0, zero] = cosine * zr - signed_z * zi
                states[row, branch, 1, zero] = signed_z * zr + cosine * zi
                states[row, branch, 0, one] = cosine * or_ - signed_z * oi
                states[row, branch, 1, one] = signed_z * or_ + cosine * oi
            else:
                signed_xy = T.if_then_else(
                    axis == 1,
                    T.if_then_else(
                        T.bitwise_and(zero, T.bitwise_xor(mask, pivot)) == 0,
                        -sine,
                        sine,
                    ),
                    sine,
                )
                states[row, branch, 0, zero] = cosine * zr + signed_xy * oi
                states[row, branch, 1, zero] = cosine * zi - signed_xy * or_
                states[row, branch, 0, one] = signed_xy * zi + cosine * or_
                states[row, branch, 1, one] = -signed_xy * zr + cosine * oi
    T.sync_threads()


@T.macro
def _cut_expand(
    left, right, theta, row, branches, axis, left_bit, right_bit, capacity
):
    """Expand exp(-i theta P_left P_right) into two product branches."""
    sine = T.__sin(theta)
    for work in T.Parallel(capacity * CUT_HALF_WIDTH):
        branch = work // CUT_HALF_WIDTH
        index = work % CUT_HALF_WIDTH
        if branch < branches:
            left_partner = T.if_then_else(axis == 2, index, index ^ left_bit)
            right_partner = T.if_then_else(axis == 2, index, index ^ right_bit)
            lar = T.alloc_var(
                T.float32, left[row, branch, 0, left_partner]
            )
            lai = T.alloc_var(
                T.float32, left[row, branch, 1, left_partner]
            )
            rbr = T.alloc_var(
                T.float32, right[row, branch, 0, right_partner]
            )
            rbi = T.alloc_var(
                T.float32, right[row, branch, 1, right_partner]
            )
            if axis == 2:
                left_sign = T.if_then_else(
                    T.bitwise_and(index, left_bit) != 0, -1.0, 1.0
                )
                right_sign = T.if_then_else(
                    T.bitwise_and(index, right_bit) != 0, -1.0, 1.0
                )
                lar, lai = left_sign * lar, left_sign * lai
                rbr, rbi = right_sign * rbr, right_sign * rbi
            elif axis == 1:
                left_positive = T.bitwise_and(index, left_bit) != 0
                right_positive = T.bitwise_and(index, right_bit) != 0
                lar, lai = (
                    T.if_then_else(left_positive, -lai, lai),
                    T.if_then_else(left_positive, lar, -lar),
                )
                rbr, rbi = (
                    T.if_then_else(right_positive, -rbi, rbi),
                    T.if_then_else(right_positive, rbr, -rbr),
                )
            left[row, branch + branches, 0, index] = sine * lai
            left[row, branch + branches, 1, index] = -sine * lar
            right[row, branch + branches, 0, index] = rbr
            right[row, branch + branches, 1, index] = rbi
    T.sync_threads()
    cosine = T.__cos(theta)
    for work in T.Parallel(capacity * CUT_HALF_WIDTH):
        branch = work // CUT_HALF_WIDTH
        index = work % CUT_HALF_WIDTH
        if branch < branches:
            left[row, branch, 0, index] *= cosine
            left[row, branch, 1, index] *= cosine
    T.sync_threads()


@T.macro
def _rotate_cut_basis(states, row, axis, branches):
    """Rotate each nine-qubit branch into an X or Y measurement basis."""
    scale = 0.7071067811865476
    for qubit in T.serial(CUT_HALF_BITS):
        bit = 1 << qubit
        for work in T.Parallel(
            branches * CUT_HALF_WIDTH // 2
        ):
            branch = work // (CUT_HALF_WIDTH // 2)
            pair = work % (CUT_HALF_WIDTH // 2)
            zero, one = _pair(pair, bit)
            zr = states[row, branch, 0, zero]
            zi = states[row, branch, 1, zero]
            or_ = states[row, branch, 0, one]
            oi = states[row, branch, 1, one]
            states[row, branch, 0, zero] = scale * T.if_then_else(
                axis == 0, zr + or_, zr + oi
            )
            states[row, branch, 1, zero] = scale * T.if_then_else(
                axis == 0, zi + oi, zi - or_
            )
            states[row, branch, 0, one] = scale * T.if_then_else(
                axis == 0, zr - or_, zr - oi
            )
            states[row, branch, 1, one] = scale * T.if_then_else(
                axis == 0, zi - oi, zi + or_
            )
        T.sync_threads()


@T.macro
def _reconstruct_cut_tiles(
    left,
    right,
    state,
    row,
    ar,
    ai,
    br,
    bi,
    real,
    imag,
    temporary,
    branches,
):
    """Reconstruct one selected product basis with complex Tensor Core GEMM."""
    reconstruction_tiles = CUT_HALF_WIDTH // FUSED_GEMM_TILE
    for right_tile in T.serial(reconstruction_tiles):
        for left_tile in T.serial(reconstruction_tiles):
            for branch, position in T.Parallel(
                branches, FUSED_GEMM_TILE
            ):
                left_index = left_tile * FUSED_GEMM_TILE + position
                right_index = right_tile * FUSED_GEMM_TILE + position
                ar[branch, position] = left[row, branch, 0, left_index]
                ai[branch, position] = left[row, branch, 1, left_index]
                br[position, branch] = right[
                    row, branch, 0, right_index
                ]
                bi[position, branch] = right[
                    row, branch, 1, right_index
                ]
            T.sync_threads()

            T.gemm(br, ar, real, clear_accum=True)
            T.gemm(bi, ai, temporary, clear_accum=True)
            for m, n in T.Parallel(FUSED_GEMM_TILE, FUSED_GEMM_TILE):
                real[m, n] -= temporary[m, n]
            T.gemm(br, ai, imag, clear_accum=True)
            T.gemm(bi, ar, temporary, clear_accum=True)
            for m, n in T.Parallel(FUSED_GEMM_TILE, FUSED_GEMM_TILE):
                right_index = right_tile * FUSED_GEMM_TILE + m
                left_index = left_tile * FUSED_GEMM_TILE + n
                index = (right_index << CUT_HALF_BITS) | left_index
                state[row, 0, index] = real[m, n]
                state[row, 1, index] = imag[m, n] + temporary[m, n]
            T.sync_threads()
@T.macro
def _rotate_basis(
    state: T.Buffer,
    width: T.int32,
    n_qubits: T.int32,
    axis: T.int32,
    inverse: bool,
):
    """Apply (or undo) the X/Y measurement basis rotation."""
    scale = 0.7071067811865476
    for qubit in T.serial(n_qubits):
        bit = 1 << qubit
        for pair in T.Parallel(width // 2):
            zero, one = _pair(pair, bit)
            zr, zi = state[0, zero], state[1, zero]
            or_, oi = state[0, one], state[1, one]
            state[0, zero] = scale * (
                zr + T.if_then_else(axis == 0, or_, T.if_then_else(inverse, or_, oi))
            )
            state[1, zero] = scale * (
                zi + T.if_then_else(axis == 0, oi, T.if_then_else(inverse, oi, -or_))
            )
            state[0, one] = scale * T.if_then_else(
                axis == 0, zr - or_, T.if_then_else(inverse, oi - zi, zr - oi)
            )
            state[1, one] = scale * T.if_then_else(
                axis == 0, zi - oi, T.if_then_else(inverse, zr - or_, zi + or_)
            )
        T.sync_threads()


@T.macro
def _measure_cut(
    state: T.Buffer,
    coefficients: T.Buffer,
    output: T.Buffer,
    scratch: T.Buffer,
    hadamard: T.Buffer,
    hadamard_shared: T.Buffer,
    probability_tile: T.Buffer,
    intermediate: T.Buffer,
    transformed: T.Buffer,
    row: T.int32,
    width: T.int32,
    axis_position: T.int32,
    masks: T.Buffer,
    mask_count: T.int32,
):
    """Measure selected q18 parities with two 64x64 Tensor Core transforms."""
    T.copy(hadamard, hadamard_shared)
    for tile in T.serial(CUT_TILE_COUNT):
        base = tile * FWHT_TILE_SIZE
        for matrix_row, matrix_column in T.Parallel(WALSH_SIZE, WALSH_SIZE):
            offset = matrix_row * WALSH_SIZE + matrix_column
            index = base + offset
            real, imag = state[row, 0, index], state[row, 1, index]
            probability_tile[matrix_row, matrix_column] = width * (
                real * real + imag * imag
            )
        T.sync_threads()
        T.gemm(hadamard_shared, probability_tile, transformed, clear_accum=True)
        T.copy(transformed, intermediate)
        T.gemm(intermediate, hadamard_shared, transformed, clear_accum=True)
        T.copy(transformed, scratch)
        T.sync_threads()
        for mask_position in T.Parallel(mask_count):
            low_mask = T.bitwise_and(masks[mask_position], FWHT_TILE_SIZE - 1)
            coefficients[row, tile, mask_position] = scratch[
                low_mask >> WALSH_BITS,
                T.bitwise_and(low_mask, WALSH_SIZE - 1),
            ] / width
        T.sync_threads()

    # Complete the six high-bit parity signs without materializing their FWHT.
    for mask_position in T.Parallel(mask_count):
        mask = masks[mask_position]
        high_mask = mask >> FWHT_TILE_BITS
        expectation = T.alloc_var(T.float32, 0.0)
        for tile in T.serial(CUT_TILE_COUNT):
            selected = T.bitwise_and(tile, high_mask)
            odd = T.if_then_else(
                T.bitwise_and(high_mask, high_mask - 1) == 0,
                selected != 0,
                T.bitwise_xor(selected != 0, selected == high_mask),
            )
            coefficient = coefficients[row, tile, mask_position]
            expectation += T.if_then_else(odd, -coefficient, coefficient)
        output[row, axis_position * mask_count + mask_position] = expectation
    T.sync_threads()


@T.macro
def _measure_cut_axes(
    left,
    right,
    state,
    coefficients,
    output,
    scratch,
    hadamard,
    ar,
    ai,
    br,
    bi,
    real,
    imag,
    temporary,
    measurement_hadamard,
    measurement_probability,
    measurement_intermediate,
    row,
    width,
    masks,
    mask_count,
    branches,
):
    """Reconstruct and measure Z, X, and Y from cut-state factors."""
    _reconstruct_cut_tiles(
        left,
        right,
        state,
        row,
        ar,
        ai,
        br,
        bi,
        real,
        imag,
        temporary,
        branches,
    )
    _measure_cut(
        state,
        coefficients,
        output,
        scratch,
        hadamard,
        measurement_hadamard,
        measurement_probability,
        measurement_intermediate,
        real,
        row,
        width,
        2,
        masks,
        mask_count,
    )

    _rotate_cut_basis(left, row, 0, branches)
    _rotate_cut_basis(right, row, 0, branches)
    _reconstruct_cut_tiles(
        left,
        right,
        state,
        row,
        ar,
        ai,
        br,
        bi,
        real,
        imag,
        temporary,
        branches,
    )
    _measure_cut(
        state,
        coefficients,
        output,
        scratch,
        hadamard,
        measurement_hadamard,
        measurement_probability,
        measurement_intermediate,
        real,
        row,
        width,
        0,
        masks,
        mask_count,
    )

    # Hadamard is self-inverse; return to Z before applying the Y basis.
    _rotate_cut_basis(left, row, 0, branches)
    _rotate_cut_basis(right, row, 0, branches)
    _rotate_cut_basis(left, row, 1, branches)
    _rotate_cut_basis(right, row, 1, branches)
    _reconstruct_cut_tiles(
        left,
        right,
        state,
        row,
        ar,
        ai,
        br,
        bi,
        real,
        imag,
        temporary,
        branches,
    )
    _measure_cut(
        state,
        coefficients,
        output,
        scratch,
        hadamard,
        measurement_hadamard,
        measurement_probability,
        measurement_intermediate,
        real,
        row,
        width,
        1,
        masks,
        mask_count,
    )


@tilelang.jit(
    target="cuda",
    execution_backend="cython",
    pass_configs={"tl.disable_data_race_check": True},
)
def _simulate(
    plan, phase_masks, phase_parameters, theta, axes, masks, n_qubits: int
):
    """Evolve and contract each row without materializing a statevector."""
    rows, gates, phase_count, parameters, axis_count, mask_count = T.const(
        "rows gates phase_count parameters axis_count mask_count"
    )
    width = 1 << n_qubits
    plan: T.Tensor((gates, 3), T.int32)
    phase_masks: T.Tensor((phase_count,), T.int32)
    phase_parameters: T.Tensor((phase_count,), T.int32)
    theta: T.Tensor((rows, parameters), T.float32)
    axes: T.Tensor((axis_count,), T.int32)
    masks: T.Tensor((mask_count,), T.int32)
    output = T.empty((rows, axis_count * mask_count), T.float32)

    with T.Kernel(rows, threads=256) as row:
        state = T.alloc_shared((2, width), T.float32)
        amplitude = 1.0 / T.sqrt(T.float32(width))
        for index in T.Parallel(width):
            state[0, index] = amplitude
            state[1, index] = 0.0
        T.sync_threads()

        for gate in T.serial(gates):
            kind, first, third = plan[gate, 0], plan[gate, 1], plan[gate, 2]
            if kind == PHASE_RUN:
                for index in T.Parallel(width):
                    phase = T.alloc_var(T.float32, 0.0)
                    for position in T.serial(n_qubits):
                        if position < third:
                            phase_position = first + position
                            mask = phase_masks[phase_position]
                            selected = T.bitwise_and(index, mask)
                            odd = T.if_then_else(
                                T.bitwise_and(mask, mask - 1) == 0,
                                selected != 0,
                                T.bitwise_xor(selected != 0, selected == mask),
                            )
                            angle = theta[row, phase_parameters[phase_position]]
                            phase += T.if_then_else(odd, angle, -angle)
                    cosine, sine = T.__cos(phase), T.__sin(phase)
                    real, imag = state[0, index], state[1, index]
                    state[0, index] = cosine * real - sine * imag
                    state[1, index] = sine * real + cosine * imag
            else:
                mask, parameter = first, third
                angle = theta[row, parameter]
                cosine, sine = T.__cos(angle), T.__sin(angle)
                if kind == RY:
                    for pair in T.Parallel(width // 2):
                        zero, one = _pair(pair, mask)
                        zr, zi = state[0, zero], state[1, zero]
                        or_, oi = state[0, one], state[1, one]
                        state[0, zero], state[1, zero] = (
                            cosine * zr - sine * or_,
                            cosine * zi - sine * oi,
                        )
                        state[0, one], state[1, one] = (
                            sine * zr + cosine * or_,
                            sine * zi + cosine * oi,
                        )
                else:
                    pivot = T.bitwise_and(mask, -mask)
                    for pair in T.Parallel(width // 2):
                        zero, _ = _pair(pair, pivot)
                        one = T.bitwise_xor(zero, mask)
                        zr, zi = state[0, zero], state[1, zero]
                        or_, oi = state[0, one], state[1, one]
                        signed_sine = T.if_then_else(
                            kind == RYY,
                            T.if_then_else(
                                T.bitwise_and(
                                    zero, T.bitwise_xor(mask, pivot)
                                )
                                == 0,
                                -sine,
                                sine,
                            ),
                            sine,
                        )
                        state[0, zero], state[1, zero] = (
                            cosine * zr + signed_sine * oi,
                            cosine * zi - signed_sine * or_,
                        )
                        state[0, one], state[1, one] = (
                            signed_sine * zi + cosine * or_,
                            -signed_sine * zr + cosine * oi,
                        )
            T.sync_threads()

        probabilities = T.alloc_shared((width,), T.float32)
        for axis_position in T.serial(axis_count):
            axis = axes[axis_position]
            if axis != 2:
                _rotate_basis(state, width, n_qubits, axis, False)

            for index in T.Parallel(width):
                probabilities[index] = (
                    state[0, index] * state[0, index]
                    + state[1, index] * state[1, index]
                )
            T.sync_threads()
            for qubit in T.serial(n_qubits):
                bit = 1 << qubit
                for pair in T.Parallel(width // 2):
                    left, right = _pair(pair, bit)
                    a, b = probabilities[left], probabilities[right]
                    probabilities[left], probabilities[right] = a + b, a - b
                T.sync_threads()
            for mask_position in T.Parallel(mask_count):
                output[row, axis_position * mask_count + mask_position] = probabilities[
                    masks[mask_position]
                ]

            if axis != 2:
                _rotate_basis(state, width, n_qubits, axis, True)
    return output


@tilelang.jit(
    target="cuda",
    execution_backend="cython",
    pass_configs={"tl.disable_data_race_check": True},
)
def _simulate_cut_global(
    theta,
    locations,
    cut_edges,
    axes,
    masks,
    hadamard,
    left,
    right,
    state,
    coefficients,
    layers: int,
    threads: int,
):
    """Exact two-way q18 XYZ evolution through product-state branching."""
    rows, parameters, edge_count, axis_count, mask_count = T.const(
        "rows parameters edge_count axis_count mask_count"
    )
    width = 1 << CUT_QUBITS
    theta: T.Tensor((rows, parameters), T.float32)
    locations: T.Tensor((CUT_QUBITS, 2), T.int32)
    cut_edges: T.Tensor((edge_count, 3), T.int32)
    axes: T.Tensor((axis_count,), T.int32)
    masks: T.Tensor((mask_count,), T.int32)
    hadamard: T.Tensor((WALSH_SIZE, WALSH_SIZE), T.float16)
    left: T.Tensor(
        (rows, CUT_MAX_BRANCHES, 2, CUT_HALF_WIDTH), T.float32
    )
    right: T.Tensor(
        (rows, CUT_MAX_BRANCHES, 2, CUT_HALF_WIDTH), T.float32
    )
    state: T.Tensor((rows, 2, width), T.float32)
    coefficients: T.Tensor(
        (rows, CUT_TILE_COUNT, CUT_MAX_MASKS), T.float32
    )
    output = T.empty((rows, axis_count * mask_count), T.float32)

    with T.Kernel(rows, threads=threads) as row:
        amplitude = 1.0 / T.sqrt(T.float32(CUT_HALF_WIDTH))
        for index in T.Parallel(CUT_HALF_WIDTH):
            left[row, 0, 0, index] = amplitude
            left[row, 0, 1, index] = 0.0
            right[row, 0, 0, index] = amplitude
            right[row, 0, 1, index] = 0.0
        T.sync_threads()

        branches = T.alloc_var(T.int32, 1)
        local_size = layers * CUT_QUBITS
        pair_size = layers * edge_count
        pair_offset = 3 * local_size
        for layer in T.serial(layers):
            for axis in T.serial(3):
                parameter_offset = axis * local_size + layer * CUT_QUBITS
                for qubit in T.serial(CUT_QUBITS):
                    side, position = locations[qubit, 0], locations[qubit, 1]
                    angle = theta[row, parameter_offset + qubit]
                    if side == 0:
                        _cut_single(
                            left,
                            angle,
                            row,
                            branches,
                            axis,
                            1 << position,
                            CUT_MAX_BRANCHES,
                        )
                    else:
                        _cut_single(
                            right,
                            angle,
                            row,
                            branches,
                            axis,
                            1 << position,
                            CUT_MAX_BRANCHES,
                        )

            for axis in T.serial(3):
                parameter_offset = pair_offset + axis * pair_size + layer * edge_count
                for edge in T.serial(edge_count):
                    side = cut_edges[edge, 0]
                    first_bit = 1 << cut_edges[edge, 1]
                    second_bit = 1 << cut_edges[edge, 2]
                    angle = theta[row, parameter_offset + edge]
                    if side == 0:
                        _cut_pair(
                            left,
                            angle,
                            row,
                            branches,
                            axis,
                            first_bit,
                            second_bit,
                            CUT_MAX_BRANCHES,
                        )
                    elif side == 1:
                        _cut_pair(
                            right,
                            angle,
                            row,
                            branches,
                            axis,
                            first_bit,
                            second_bit,
                            CUT_MAX_BRANCHES,
                        )
                    else:
                        _cut_expand(
                            left,
                            right,
                            angle,
                            row,
                            branches,
                            axis,
                            first_bit,
                            second_bit,
                            CUT_MAX_BRANCHES,
                        )
                        branches *= 2

        # Reconstruct each requested product basis from the much smaller
        # nine-qubit branches. This removes full q18 X/Y basis rotations.
        ar = T.alloc_shared(
            (CUT_MAX_BRANCHES, FUSED_GEMM_TILE), T.float16
        )
        ai = T.alloc_shared(
            (CUT_MAX_BRANCHES, FUSED_GEMM_TILE), T.float16
        )
        br = T.alloc_shared(
            (FUSED_GEMM_TILE, CUT_MAX_BRANCHES), T.float16
        )
        bi = T.alloc_shared(
            (FUSED_GEMM_TILE, CUT_MAX_BRANCHES), T.float16
        )
        real = T.alloc_fragment(
            (FUSED_GEMM_TILE, FUSED_GEMM_TILE), T.float32
        )
        imag = T.alloc_fragment(
            (FUSED_GEMM_TILE, FUSED_GEMM_TILE), T.float32
        )
        temporary = T.alloc_fragment(
            (FUSED_GEMM_TILE, FUSED_GEMM_TILE), T.float32
        )
        scratch = T.alloc_shared((WALSH_SIZE, WALSH_SIZE), T.float32)
        _measure_cut_axes(
            left,
            right,
            state,
            coefficients,
            output,
            scratch,
            hadamard,
            ar,
            ai,
            br,
            bi,
            real,
            imag,
            temporary,
            ar,
            ai,
            br,
            row,
            width,
            masks,
            mask_count,
            CUT_MAX_BRANCHES,
        )
    return output


@tilelang.jit(
    target="cuda",
    execution_backend="cython",
    pass_configs={"tl.disable_data_race_check": True},
)
def _evolve_schmidt_stage(
    theta, locations, cut_edges, stage_info, left, right
):
    """Apply one q18 local/intra-half stage, excluding its crossing gate."""
    rows, parameters, edge_count = T.const("rows parameters edge_count")
    theta: T.Tensor((rows, parameters), T.float32)
    locations: T.Tensor((CUT_QUBITS, 2), T.int32)
    cut_edges: T.Tensor((edge_count, 3), T.int32)
    stage_info: T.Tensor((2,), T.int32)
    left: T.Tensor(
        (rows, SCHMIDT_RANK, 2, CUT_HALF_WIDTH), T.float32
    )
    right: T.Tensor(
        (rows, SCHMIDT_RANK, 2, CUT_HALF_WIDTH), T.float32
    )
    output_left = T.empty(
        (rows, SCHMIDT_RANK, 2, CUT_HALF_WIDTH), T.float32
    )
    output_right = T.empty(
        (rows, SCHMIDT_RANK, 2, CUT_HALF_WIDTH), T.float32
    )

    with T.Kernel(rows, threads=256) as row:
        for work in T.Parallel(SCHMIDT_RANK * 2 * CUT_HALF_WIDTH):
            branch = work // (2 * CUT_HALF_WIDTH)
            component = (work // CUT_HALF_WIDTH) % 2
            index = work % CUT_HALF_WIDTH
            output_left[row, branch, component, index] = left[
                row, branch, component, index
            ]
            output_right[row, branch, component, index] = right[
                row, branch, component, index
            ]
        T.sync_threads()

        stage = stage_info[0]
        branches = stage_info[1]
        layer = stage // 3
        pair_axis = stage % 3
        layers = parameters // (3 * (CUT_QUBITS + edge_count))
        local_size = layers * CUT_QUBITS
        pair_size = layers * edge_count

        if pair_axis == 0:
            for axis in T.serial(3):
                parameter_offset = axis * local_size + layer * CUT_QUBITS
                for qubit in T.serial(CUT_QUBITS):
                    side, position = locations[qubit, 0], locations[qubit, 1]
                    angle = theta[row, parameter_offset + qubit]
                    if side == 0:
                        _cut_single(
                            output_left,
                            angle,
                            row,
                            branches,
                            axis,
                            1 << position,
                            SCHMIDT_RANK,
                        )
                    else:
                        _cut_single(
                            output_right,
                            angle,
                            row,
                            branches,
                            axis,
                            1 << position,
                            SCHMIDT_RANK,
                        )

        parameter_offset = (
            3 * local_size + pair_axis * pair_size + layer * edge_count
        )
        for edge in T.serial(edge_count):
            side = cut_edges[edge, 0]
            first_bit = 1 << cut_edges[edge, 1]
            second_bit = 1 << cut_edges[edge, 2]
            angle = theta[row, parameter_offset + edge]
            if side >= 0:
                if side == 0:
                    _cut_pair(
                        output_left,
                        angle,
                        row,
                        branches,
                        pair_axis,
                        first_bit,
                        second_bit,
                        SCHMIDT_RANK,
                    )
                else:
                    _cut_pair(
                        output_right,
                        angle,
                        row,
                        branches,
                        pair_axis,
                        first_bit,
                        second_bit,
                        SCHMIDT_RANK,
                    )
    return output_left, output_right


@tilelang.jit(
    target="cuda",
    execution_backend="cython",
    pass_configs={"tl.disable_data_race_check": True},
)
def _measure_schmidt(
    left,
    right,
    axes,
    masks,
    hadamard,
    state,
    coefficients,
    threads: int,
):
    """Reconstruct and measure a canonical rank-8 q18 state."""
    rows, axis_count, mask_count = T.const("rows axis_count mask_count")
    width = 1 << CUT_QUBITS
    left: T.Tensor(
        (rows, SCHMIDT_RANK, 2, CUT_HALF_WIDTH), T.float32
    )
    right: T.Tensor(
        (rows, SCHMIDT_RANK, 2, CUT_HALF_WIDTH), T.float32
    )
    axes: T.Tensor((axis_count,), T.int32)
    masks: T.Tensor((mask_count,), T.int32)
    hadamard: T.Tensor((WALSH_SIZE, WALSH_SIZE), T.float16)
    state: T.Tensor((rows, 2, width), T.float32)
    coefficients: T.Tensor(
        (rows, CUT_TILE_COUNT, CUT_MAX_MASKS), T.float32
    )
    output = T.empty((rows, axis_count * mask_count), T.float32)

    with T.Kernel(rows, threads=threads) as row:
        ar = T.alloc_shared((SCHMIDT_RANK, FUSED_GEMM_TILE), T.float16)
        ai = T.alloc_shared((SCHMIDT_RANK, FUSED_GEMM_TILE), T.float16)
        br = T.alloc_shared((FUSED_GEMM_TILE, SCHMIDT_RANK), T.float16)
        bi = T.alloc_shared((FUSED_GEMM_TILE, SCHMIDT_RANK), T.float16)
        real = T.alloc_fragment(
            (FUSED_GEMM_TILE, FUSED_GEMM_TILE), T.float32
        )
        imag = T.alloc_fragment(
            (FUSED_GEMM_TILE, FUSED_GEMM_TILE), T.float32
        )
        temporary = T.alloc_fragment(
            (FUSED_GEMM_TILE, FUSED_GEMM_TILE), T.float32
        )
        measurement_hadamard = T.alloc_shared(
            (WALSH_SIZE, WALSH_SIZE), T.float16
        )
        measurement_probability = T.alloc_shared(
            (WALSH_SIZE, WALSH_SIZE), T.float16
        )
        measurement_intermediate = T.alloc_shared(
            (WALSH_SIZE, WALSH_SIZE), T.float16
        )
        scratch = T.alloc_shared((WALSH_SIZE, WALSH_SIZE), T.float32)
        _measure_cut_axes(
            left,
            right,
            state,
            coefficients,
            output,
            scratch,
            hadamard,
            ar,
            ai,
            br,
            bi,
            real,
            imag,
            temporary,
            measurement_hadamard,
            measurement_probability,
            measurement_intermediate,
            row,
            width,
            masks,
            mask_count,
            SCHMIDT_RANK,
        )
    return output


def _complex_factors(storage, rank):
    return torch.complex(
        storage[:, :rank, 0], storage[:, :rank, 1]
    ).transpose(1, 2)


def _factor_storage(factors):
    rows, _, rank = factors.shape
    storage = torch.zeros(
        (rows, SCHMIDT_RANK, 2, CUT_HALF_WIDTH),
        dtype=torch.float32,
        device=factors.device,
    )
    storage[:, :rank] = torch.view_as_real(factors).permute(0, 2, 3, 1)
    return storage


@torch.no_grad()
def _gram_basis(states, transformed):
    overlap = states.mH @ transformed
    overlap = 0.5 * (overlap + overlap.mH)
    eigenvalues, eigenvectors = torch.linalg.eigh(overlap)
    eigenvalues.clamp_(-1.0 + 1e-6, 1.0 - 1e-6)
    states_eigen = states @ eigenvectors
    transformed_eigen = transformed @ eigenvectors
    plus = (states_eigen + transformed_eigen) / torch.sqrt(
        2.0 * (1.0 + eigenvalues)
    )[:, None]
    minus = (states_eigen - transformed_eigen) / torch.sqrt(
        2.0 * (1.0 - eigenvalues)
    )[:, None]
    basis = torch.cat((plus, minus), dim=2)

    eigenvectors_h = eigenvectors.mH / np.sqrt(2.0)
    plus_core = torch.cat((eigenvectors_h, eigenvectors_h), dim=2)
    minus_core = torch.cat((eigenvectors_h, -eigenvectors_h), dim=2)
    transform = torch.cat(
        (
            torch.sqrt(1.0 + eigenvalues)[:, :, None] * plus_core,
            torch.sqrt(1.0 - eigenvalues)[:, :, None] * minus_core,
        ),
        dim=1,
    )
    return basis, transform


@torch.no_grad()
def _schmidt_update(left, right, weights, left_pauli, right_pauli, angles):
    """Apply one crossing Pauli rotation and retain eight Schmidt terms."""
    left_basis, left_transform = _gram_basis(left, left_pauli)
    right_basis, right_transform = _gram_basis(right, right_pauli)
    coefficients = torch.cat(
        (
            torch.cos(angles)[:, None] * weights,
            -1j * torch.sin(angles)[:, None] * weights,
        ),
        dim=1,
    )
    core = (left_transform * coefficients[:, None]) @ right_transform.transpose(
        1, 2
    )
    core_left, singular_values, core_right_h = torch.linalg.svd(
        core, full_matrices=False
    )
    rank = min(SCHMIDT_RANK, singular_values.shape[1])
    singular_values = singular_values[:, :rank]
    singular_values /= torch.linalg.vector_norm(
        singular_values, dim=1, keepdim=True
    )
    return (
        left_basis @ core_left[:, :, :rank],
        right_basis @ core_right_h[:, :rank].transpose(1, 2),
        singular_values,
    )


@torch.no_grad()
def _canonicalize_branches(left, right):
    """Convert eight product branches to canonical Schmidt form."""

    def orthonormalize(states):
        gram = states.mH @ states
        gram = 0.5 * (gram + gram.mH)
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        eigenvalues.clamp_min_(1e-8)
        basis = (states @ eigenvectors) / torch.sqrt(eigenvalues)[:, None]
        transform = (
            torch.sqrt(eigenvalues)[:, :, None] * eigenvectors.mH
        )
        return basis, transform

    left_basis, left_transform = orthonormalize(left)
    right_basis, right_transform = orthonormalize(right)
    core_left, singular_values, core_right_h = torch.linalg.svd(
        left_transform @ right_transform.transpose(1, 2),
        full_matrices=False,
    )
    singular_values /= torch.linalg.vector_norm(
        singular_values, dim=1, keepdim=True
    )
    return (
        left_basis @ core_left,
        right_basis @ core_right_h.transpose(1, 2),
        singular_values,
    )


def _pauli_action(states, axis, bit):
    indices = torch.arange(CUT_HALF_WIDTH, device=states.device)
    if axis == 2:
        signs = torch.where(indices & bit == 0, 1.0, -1.0)
        return states * signs[None, :, None]
    transformed = states[:, indices ^ bit]
    if axis == 1:
        phases = torch.where(indices & bit == 0, -1j, 1j)
        transformed = transformed * phases[None, :, None]
    return transformed


class XYZTileLangBackend:
    """TileLang executor for q1–12 and the specialized q18 cut kernel."""

    name = "xyz-tilelang"

    def __init__(self, *, cut_threads: int = 512, schmidt_rank: int | None = 8):
        if cut_threads not in (256, 512):
            raise ValueError("cut_threads must be 256 or 512")
        if schmidt_rank not in (None, SCHMIDT_RANK):
            raise ValueError("schmidt_rank must be 8 or None")
        self.cut_threads = cut_threads
        self.schmidt_rank = schmidt_rank
        self._cut_workspace = None
        self._schmidt_workspace = None
        self._hadamard = None

    def _run_schmidt(self, theta, static, layers):
        locations, cut_edges, axes, masks, hadamard = static
        crossing = torch.nonzero(cut_edges[:, 0] < 0).flatten()
        if len(crossing) != 1:
            raise ValueError("The rank-8 q18 path requires one crossing edge")
        crossing_position = int(crossing[0])
        left_position = int(cut_edges[crossing_position, 1])
        right_position = int(cut_edges[crossing_position, 2])
        rows, edge_count = len(theta), len(cut_edges)

        left = torch.zeros(
            (rows, SCHMIDT_RANK, 2, CUT_HALF_WIDTH), device=theta.device
        )
        right = torch.zeros_like(left)
        left[:, 0, 0] = CUT_HALF_WIDTH**-0.5
        right[:, 0, 0] = CUT_HALF_WIDTH**-0.5
        active_rank = 1
        pair_size = layers * edge_count
        pair_offset = 3 * layers * CUT_QUBITS

        for stage in range(3 * layers):
            stage_info = torch.tensor(
                (stage, active_rank),
                dtype=torch.int32,
                device=theta.device,
            )
            left, right = _evolve_schmidt_stage(
                theta, locations, cut_edges, stage_info, left, right
            )
            axis = stage % 3
            layer = stage // 3
            parameter = (
                pair_offset
                + axis * pair_size
                + layer * edge_count
                + crossing_position
            )
            left_basis = _complex_factors(left, active_rank)
            right_basis = _complex_factors(right, active_rank)
            if layer == 0:
                angles = theta[:, parameter]
                left_basis = torch.cat(
                    (
                        torch.cos(angles)[:, None, None] * left_basis,
                        -1j
                        * torch.sin(angles)[:, None, None]
                        * _pauli_action(
                            left_basis, axis, 1 << left_position
                        ),
                    ),
                    dim=2,
                )
                right_basis = torch.cat(
                    (
                        right_basis,
                        _pauli_action(
                            right_basis, axis, 1 << right_position
                        ),
                    ),
                    dim=2,
                )
                active_rank *= 2
                if stage < 2:
                    left = _factor_storage(left_basis)
                    right = _factor_storage(right_basis)
                    continue
                left_basis, right_basis, weights = _canonicalize_branches(
                    left_basis, right_basis
                )
                left = _factor_storage(left_basis)
                right = _factor_storage(right_basis)
                continue
            left_basis, right_basis, weights = _schmidt_update(
                left_basis,
                right_basis,
                weights,
                _pauli_action(left_basis, axis, 1 << left_position),
                _pauli_action(right_basis, axis, 1 << right_position),
                theta[:, parameter],
            )
            active_rank = weights.shape[1]
            left = _factor_storage(left_basis)
            right = _factor_storage(right_basis)

        left = _factor_storage(left_basis * weights[:, None])
        right = _factor_storage(right_basis)
        if self._schmidt_workspace is None:
            self._schmidt_workspace = (
                torch.empty((rows, 2, 1 << CUT_QUBITS), device=theta.device),
                torch.empty(
                    (rows, CUT_TILE_COUNT, CUT_MAX_MASKS), device=theta.device
                ),
            )
        return _measure_schmidt(
            left,
            right,
            axes,
            masks,
            hadamard,
            *self._schmidt_workspace,
            self.cut_threads,
        )

    def run(
        self,
        parameter_values: ArrayLike,
        *,
        n_qubits: int,
        axes: Sequence[str] = ("x", "y", "z"),
        evolution_edges: Sequence[tuple[int, int]],
        observable_edges: Sequence[tuple[int, int]] | None = None,
    ) -> NDArray[np.float32]:
        if not (1 <= n_qubits <= 12 or n_qubits == CUT_QUBITS):
            raise ValueError("XYZTileLangBackend supports q1–12 and q18")
        axis_names = tuple(str(axis).lower() for axis in axes)
        values, axis_values, masks = _inputs(
            parameter_values, n_qubits, axis_names, observable_edges
        )
        if not len(axis_values):
            return np.empty((len(values), 0), np.float32)
        if not len(values):
            return np.empty((0, len(axis_values) * len(masks)), np.float32)
        device = torch.device("cuda")
        edges = _normalize_edges(evolution_edges, n_qubits)
        if n_qubits <= 12:
            plan, phase_masks, phase_parameters = _operation_plan(
                n_qubits, values.shape[1], axis_names, edges
            )
            static = tuple(
                torch.as_tensor(value, device=device)
                for value in (
                    plan,
                    phase_masks,
                    phase_parameters,
                    axis_values,
                    masks,
                )
            )
        else:
            if axis_names != ("x", "y", "z"):
                raise ValueError("The q18 cut kernel requires XYZ axes")
            layers, locations, cut_edges = _q18_cut_layout(
                values.shape[1], edges
            )
            if self._hadamard is None:
                self._hadamard = torch.as_tensor(WALSH_64, device=device)
            static = tuple(
                torch.as_tensor(value, device=device)
                for value in (locations, cut_edges, axis_values, masks)
            ) + (self._hadamard,)
        batch_size = (
            BATCH_SIZE if n_qubits == CUT_QUBITS else min(BATCH_SIZE, len(values))
        )
        result = np.empty(
            (len(values), len(axis_values) * len(masks)), dtype=np.float32
        )
        for start in range(0, len(values), batch_size):
            count = min(batch_size, len(values) - start)
            launch = np.empty((batch_size, values.shape[1]), np.float32)
            launch[:count] = values[start : start + count]
            launch[count:] = values[start]
            theta = torch.as_tensor(launch, device=device)
            if n_qubits <= 12:
                inputs = (*static[:3], theta, *static[3:])
                output = _simulate(*inputs, n_qubits)
            else:
                width = 1 << CUT_QUBITS
                if self.schmidt_rank is not None:
                    output = self._run_schmidt(theta, static, layers)
                    result[start : start + count] = output[:count].cpu().numpy()
                    continue
                if self._cut_workspace is None:
                    self._cut_workspace = (
                        torch.empty(
                            (
                                batch_size,
                                CUT_MAX_BRANCHES,
                                2,
                                CUT_HALF_WIDTH,
                            ),
                            device=device,
                        ),
                        torch.empty(
                            (
                                batch_size,
                                CUT_MAX_BRANCHES,
                                2,
                                CUT_HALF_WIDTH,
                            ),
                            device=device,
                        ),
                        torch.empty((batch_size, 2, width), device=device),
                        torch.empty(
                            (batch_size, CUT_TILE_COUNT, CUT_MAX_MASKS),
                            device=device,
                        ),
                    )
                output = _simulate_cut_global(
                    theta,
                    *static,
                    *self._cut_workspace,
                    layers,
                    self.cut_threads,
                )
            result[start : start + count] = output[:count].cpu().numpy()
        return result


__all__ = ["XYZTileLangBackend"]
