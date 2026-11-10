from std.math import exp, sqrt, tanh
from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from mojo_dllm.kernels.moe import (
    Routing,
    ExpertWeights,
    MoEWorkspace,
    route_tokens,
    grouped_experts,
    gelu_tanh,
)
from mojo_dllm.sys.mem import Floats
from mojo_dllm.sys.mmap import Mapping
from mojo_dllm.formats.gguf import ggml_row_bytes
from mojo_dllm.quant.kquants import dequant_row


def test_router_renormalizes_selected_probabilities() raises:
    var x = Floats(2 * 8)
    var scales = Floats(8)
    var matrix = Floats(4 * 8)
    for d in range(8):
        x[d] = Float32(d + 1) / 8
        x[8 + d] = -x[d]
        scales[d] = Float32(d + 2) / 9
        for e in range(4):
            matrix[e * 8 + d] = Float32((e * 7 + d * 3) % 13 - 6) / 10
    var routes = Routing(2, 4, 2)
    route_tokens(x.ptr(), 8, scales.ptr(), matrix.ptr(), 1e-6, routes)
    for t in range(2):
        var sumsq = Float64(0)
        for d in range(8):
            sumsq += Float64(x[t * 8 + d]) ** 2
        var logits = List[Float64]()
        for e in range(4):
            var score = Float64(0)
            for d in range(8):
                score += (
                    Float64(x[t * 8 + d])
                    / sqrt(sumsq / 8 + 1e-6)
                    / sqrt(8.0)
                    * Float64(scales[d])
                    * Float64(matrix[e * 8 + d])
                )
            logits.append(score)
        var first = 0
        for e in range(1, 4):
            if logits[e] > logits[first]:
                first = e
        var second = 0 if first != 0 else 1
        for e in range(4):
            if e != first and logits[e] > logits[second]:
                second = e
        var a = exp(logits[first])
        var b = exp(logits[second])
        assert_equal(Int(routes.ids.i32().unsafe_load(t * 2)), first)
        assert_equal(Int(routes.ids.i32().unsafe_load(t * 2 + 1)), second)
        assert_true(abs(Float64(routes.weights[t * 2]) - a / (a + b)) < 1e-6)
        assert_true(
            abs(Float64(routes.weights[t * 2 + 1]) - b / (a + b)) < 1e-6
        )
        assert_true(
            abs(routes.weights[t * 2] + routes.weights[t * 2 + 1] - 1) < 1e-6
        )


def test_router_ties_choose_lower_expert_ids() raises:
    var x = Floats(8)
    var scales = Floats(8)
    var matrix = Floats(4 * 8)
    var routes = Routing(1, 4, 2)
    route_tokens(x.ptr(), 8, scales.ptr(), matrix.ptr(), 1e-6, routes)
    assert_equal(routes.ids.i32().unsafe_load(0), 0)
    assert_equal(routes.ids.i32().unsafe_load(1), 1)
    assert_equal(routes.weights[0], 0.5)
    assert_equal(routes.weights[1], 0.5)


def test_gelu_tanh_matches_scalar_formula() raises:
    for v in [-4.0, -1.0, 0.0, 0.25, 1.5, 4.0]:
        var want = (
            0.5
            * v
            * (
                1
                + tanh(
                    sqrt(2.0 / 3.141592653589793) * (v + 0.044715 * v * v * v)
                )
            )
        )
        assert_true(abs(Float64(gelu_tanh(Float32(v))) - want) < 1e-6)


def _scalar_routes(
    x: Floats, scale: Floats, matrix: Floats, n: Int, hidden: Int
) raises -> Routing:
    var result = Routing(n, 4, 2)
    for t in range(n):
        var ss = Float64(0)
        for d in range(hidden):
            ss += Float64(x[t * hidden + d]) ** 2
        var score = List[Float64]()
        for e in range(4):
            var dot = Float64(0)
            for d in range(hidden):
                dot += (
                    Float64(x[t * hidden + d])
                    * Float64(scale[d])
                    * Float64(matrix[e * hidden + d])
                )
            score.append(
                dot / sqrt(ss / Float64(hidden) + 1e-6) / sqrt(Float64(hidden))
            )
        var best = 0
        for e in range(1, 4):
            if score[e] > score[best]:
                best = e
        var next = 1 if best == 0 else 0
        for e in range(4):
            if e != best and score[e] > score[next]:
                next = e
        var other = exp(score[next] - score[best])
        result.ids.i32().unsafe_store(t * 2, Int32(best))
        result.ids.i32().unsafe_store(t * 2 + 1, Int32(next))
        result.weights[t * 2] = Float32(1.0 / (1.0 + other))
        result.weights[t * 2 + 1] = Float32(other / (1.0 + other))
    return result^


def _scalar_experts(
    raw_addr: Int,
    gate_kind: Int,
    down_kind: Int,
    x: Floats,
    routes: Routing,
    expert_scales: Floats,
    hidden: Int,
    ffn: Int,
) raises -> Floats:
    var gate_row_bytes = ggml_row_bytes(gate_kind, hidden)
    var down_row_bytes = ggml_row_bytes(down_kind, ffn)
    var gate_stride = 2 * ffn * gate_row_bytes
    var down_addr = raw_addr + 4 * gate_stride
    var result = Floats(routes.n_tokens * hidden)
    var gate = Floats(hidden)
    var up = Floats(hidden)
    var down = Floats(ffn)
    var intermediate = List[Float64]()
    for _ in range(ffn):
        intermediate.append(0)
    for t in range(routes.n_tokens):
        for s in range(routes.top_k):
            var e = Int(routes.ids.i32().unsafe_load(t * routes.top_k + s))
            for f in range(ffn):
                dequant_row(
                    gate_kind,
                    raw_addr + e * gate_stride + f * gate_row_bytes,
                    gate.ptr(),
                    hidden,
                )
                dequant_row(
                    gate_kind,
                    raw_addr + e * gate_stride + (ffn + f) * gate_row_bytes,
                    up.ptr(),
                    hidden,
                )
                var g = Float64(0)
                var u = Float64(0)
                for d in range(hidden):
                    g += Float64(gate[d]) * Float64(x[t * hidden + d])
                    u += Float64(up[d]) * Float64(x[t * hidden + d])
                intermediate[f] = (
                    0.5
                    * g
                    * (
                        1.0
                        + tanh(
                            sqrt(2.0 / 3.141592653589793)
                            * (g + 0.044715 * g * g * g)
                        )
                    )
                    * u
                )
            for d in range(hidden):
                dequant_row(
                    down_kind,
                    down_addr + (e * hidden + d) * down_row_bytes,
                    down.ptr(),
                    ffn,
                )
                var dot = Float64(0)
                for f in range(ffn):
                    dot += Float64(down[f]) * intermediate[f]
                result[t * hidden + d] += Float32(
                    dot
                    * Float64(expert_scales[e])
                    * Float64(routes.weights[t * routes.top_k + s])
                )
    return result^


def _q8_experts(raw: Mapping, scales: Floats) raises -> ExpertWeights:
    """The 4 experts of `moe-q8.bin`: 64 gate/up rows of 256 and 256 down rows of 32.
    """
    var gate_bytes = 4 * 64 * ggml_row_bytes(8, 256)
    var down_bytes = 4 * 256 * ggml_row_bytes(8, 32)
    return ExpertWeights(
        raw.addr,
        gate_bytes,
        8,
        raw.addr + gate_bytes,
        down_bytes,
        8,
        256,
        32,
        4,
        scales.ptr(),
    )


def _check_moe_scalar(
    gate_kind: Int, down_kind: Int, suffix: String, mixed_routes: Bool = False
) raises:
    var hidden = 256
    var ffn = 32
    var n = 16
    var raw = Mapping("tests/fixtures/moe-" + suffix + ".bin")
    var scales = Floats(4)
    scales[0] = 0.5
    scales[1] = 1.5
    scales[2] = -0.25
    scales[3] = 2.0
    var gate_bytes = 4 * 2 * ffn * ggml_row_bytes(gate_kind, hidden)
    var down_bytes = 4 * hidden * ggml_row_bytes(down_kind, ffn)
    var weights = ExpertWeights(
        raw.addr,
        gate_bytes,
        gate_kind,
        raw.addr + gate_bytes,
        down_bytes,
        down_kind,
        hidden,
        ffn,
        4,
        scales.ptr(),
    )
    var ws = MoEWorkspace(n, weights)
    var x = Floats(n * hidden)
    var router_scale = Floats(hidden)
    var router_matrix = Floats(4 * hidden)
    for d in range(hidden):
        router_scale[d] = Float32(d % 5 + 1) / 5
        router_matrix[d] = 0.02
        router_matrix[hidden + d] = 0.01
        router_matrix[2 * hidden + d] = -0.01
        router_matrix[3 * hidden + d] = -0.02
    for t in range(n):
        for d in range(hidden):
            x[t * hidden + d] = Float32((t * 17 + d * 7) % 127 + 1) / 64
    var got = Floats(n * hidden)
    for tokens in [16, 3]:
        var routing = Routing(tokens, 4, 2)
        route_tokens(
            x.ptr(),
            hidden,
            router_scale.ptr(),
            router_matrix.ptr(),
            1e-6,
            routing,
            2,
        )
        var reference_routes = _scalar_routes(
            x, router_scale, router_matrix, tokens, hidden
        )
        for pair in range(tokens * 2):
            assert_equal(
                routing.ids.i32().unsafe_load(pair),
                reference_routes.ids.i32().unsafe_load(pair),
            )
            assert_true(
                abs(routing.weights[pair] - reference_routes.weights[pair])
                < 1e-6
            )
        # Every token selects experts 0 and 1; experts 2 and 3 receive no rows.
        for t in range(tokens):
            assert_equal(routing.ids.i32().unsafe_load(t * 2), 0)
            assert_equal(routing.ids.i32().unsafe_load(t * 2 + 1), 1)
        if mixed_routes:
            # Unequal groups preserve token order through gather/scatter.
            # Expert 0 sees every token; the other counts differ.
            for t in range(tokens):
                var e = 1 if t < 4 else (2 if t % 2 == 0 else 3)
                routing.ids.i32().unsafe_store(t * 2 + 1, Int32(e))
                reference_routes.ids.i32().unsafe_store(t * 2 + 1, Int32(e))
        for i in range(got.n):
            got[i] = 12345
        grouped_experts(weights, x.ptr(), routing, ws, got.ptr(), 2)
        var want = _scalar_experts(
            raw.addr,
            gate_kind,
            down_kind,
            x,
            reference_routes,
            scales,
            hidden,
            ffn,
        )
        var num = Float64(0)
        var den = Float64(0)
        for i in range(want.n):
            num += (Float64(got[i]) - Float64(want[i])) ** 2
            den += Float64(want[i]) ** 2
        var relative = sqrt(num / den)
        # Two separately quantized GEMMs with GELU between them: use twice
        # the single-GEMM relative bound, while routing is checked at 1e-6.
        assert_true(
            relative < 0.02,
            suffix + " MoE scalar relative error " + String(relative),
        )
        for i in range(want.n, got.n):
            assert_equal(got[i], 12345)


def test_moe_q8_scalar_reference_and_expert_scale() raises:
    _check_moe_scalar(8, 8, "q8")


def test_moe_q5_scalar_reference_and_expert_scale() raises:
    _check_moe_scalar(6, 6, "q5")


def test_moe_q4_gate_q5_down_matches_scalar() raises:
    _check_moe_scalar(12, 6, "q4-q5")


def test_moe_q4_gate_q8_down_matches_scalar() raises:
    _check_moe_scalar(12, 8, "q4-q8")


def test_moe_scatter_handles_unequal_expert_groups() raises:
    _check_moe_scalar(12, 8, "q4-q8", True)


def test_expert_weights_refuse_a_source_of_the_wrong_size() raises:
    var raw = Mapping("tests/fixtures/moe-q8.bin")
    var scales = Floats(4)
    var gate_bytes = 4 * 64 * ggml_row_bytes(8, 256)
    var down_bytes = 4 * 256 * ggml_row_bytes(8, 32)
    var weights = _q8_experts(raw, scales)
    assert_equal(weights.n_experts, 4)
    # The file holds both tensors, so a guard that is mutated away reads valid
    # memory and the test fails by assertion.
    with assert_raises(
        contains=(
            "the gate/up tensor holds "
            + String(gate_bytes - 1)
            + " bytes, but 4 experts of 64 rows of 256 Q8_0 values take "
            + String(gate_bytes)
        )
    ):
        _ = ExpertWeights(
            raw.addr,
            gate_bytes - 1,
            8,
            raw.addr + gate_bytes,
            down_bytes,
            8,
            256,
            32,
            4,
            scales.ptr(),
        )
    with assert_raises(
        contains=(
            "the down tensor holds "
            + String(down_bytes + 34)
            + " bytes, but 4 experts of 256 rows of 32 Q8_0 values take "
            + String(down_bytes)
        )
    ):
        _ = ExpertWeights(
            raw.addr,
            gate_bytes,
            8,
            raw.addr + gate_bytes,
            down_bytes + 34,
            8,
            256,
            32,
            4,
            scales.ptr(),
        )


def test_routing_set_len_is_checked_against_the_capacity() raises:
    var routing = Routing(4, 4, 2)
    assert_equal(routing.capacity, 4)
    assert_equal(routing.n_tokens, 4)
    routing.set_len(2)
    assert_equal(routing.n_tokens, 2)
    assert_equal(routing.capacity, 4)
    routing.set_len(4)
    with assert_raises(
        contains="MoE routing holds 4 tokens, cannot be set to 5"
    ):
        routing.set_len(5)
    with assert_raises(
        contains="MoE routing holds 4 tokens, cannot be set to -1"
    ):
        routing.set_len(-1)
    assert_equal(routing.n_tokens, 4)


def test_route_tokens_refuses_a_length_beyond_the_capacity() raises:
    # `n_tokens` is a public field: a caller that writes past `set_len` would
    # make `route_tokens` store ids past the buffers it owns.
    var routing = Routing(2, 4, 2)
    routing.n_tokens = 3
    var x = Floats(3 * 256)
    var router_matrix = Floats(4 * 256)
    var router_scale = Floats(256)
    with assert_raises(contains="MoE routing holds 2 tokens, cannot route 3"):
        route_tokens(
            x.ptr(), 256, router_scale.ptr(), router_matrix.ptr(), 1e-6, routing
        )


def test_route_tokens_names_the_token_with_non_finite_scores() raises:
    var hidden = 256
    var x = Floats(3 * hidden)
    var router_matrix = Floats(4 * hidden)
    var router_scale = Floats(hidden)
    for i in range(x.n):
        x[i] = Float32(i % 7) - 3.0
    for i in range(router_scale.n):
        router_scale[i] = 1.0
    for i in range(router_matrix.n):
        router_matrix[i] = Float32((i * 5) % 11) / 11.0
    var routing = Routing(3, 4, 2)
    route_tokens(
        x.ptr(), hidden, router_scale.ptr(), router_matrix.ptr(), 1e-6, routing
    )
    for i in range(6):
        assert_true(routing.weights[i] > 0.0 and routing.weights[i] < 1.0)
    x[hidden + 5] = Float32(Float64("nan"))
    with assert_raises(contains="non-finite router scores for token 1"):
        route_tokens(
            x.ptr(),
            hidden,
            router_scale.ptr(),
            router_matrix.ptr(),
            1e-6,
            routing,
        )
    x[hidden + 5] = 1.0
    router_matrix[3] = Float32(1e30 * 1e30)
    with assert_raises(contains="non-finite router scores for token"):
        route_tokens(
            x.ptr(),
            hidden,
            router_scale.ptr(),
            router_matrix.ptr(),
            1e-6,
            routing,
        )


def test_routing_rejects_invalid_top_k() raises:
    with assert_raises(
        contains="invalid MoE routing dimensions: tokens=16, experts=4, top_k=5"
    ):
        _ = Routing(16, 4, 5)


def test_moe_empty_canvas_and_invalid_route_guards() raises:
    var raw = Mapping("tests/fixtures/moe-q8.bin")
    var scales = Floats(4)
    for e in range(4):
        scales[e] = 1
    var weights = _q8_experts(raw, scales)
    var empty_ws = MoEWorkspace(0, weights)
    var ws = MoEWorkspace(1, weights)
    var x = Floats(256)
    var output = Floats(256)
    for i in range(256):
        output[i] = 12345
    var router_matrix = Floats(4 * 256)
    var router_scale = Floats(256)
    var empty = Routing(0, 4, 2)
    route_tokens(
        x.ptr(), 256, router_scale.ptr(), router_matrix.ptr(), 1e-6, empty
    )
    grouped_experts(weights, x.ptr(), empty, empty_ws, output.ptr(), 1)
    for i in range(256):
        assert_equal(output[i], 12345)
    var routing = Routing(1, 4, 2)
    # Duplicate ids would exceed the full-canvas per-expert scratch capacity.
    with assert_raises(contains="MoE routing repeats expert 0 for token 0"):
        grouped_experts(weights, x.ptr(), routing, ws, output.ptr(), 1)
    routing.ids.i32().unsafe_store(1, 4)
    with assert_raises(contains="MoE routing expert id out of range: 4"):
        grouped_experts(weights, x.ptr(), routing, ws, output.ptr(), 1)
    routing.ids.i32().unsafe_store(1, 1)
    with assert_raises(
        contains="MoE routing or workspace does not match expert dimensions"
    ):
        grouped_experts(weights, x.ptr(), routing, empty_ws, output.ptr(), 1)
    for i in range(256):
        assert_equal(output[i], 12345)


def _check_moe_dimension_guard(field: Int) raises:
    var raw = Mapping("tests/fixtures/moe-q8.bin")
    var scales = Floats(4)
    for e in range(4):
        scales[e] = 1
    var weights = _q8_experts(raw, scales)
    # Retain valid backing buffers and alter only one advertised dimension.
    # If its guard is removed, the kernel can run safely and assert_raises
    # fails instead of a buffer overrun disguising the missing validation.
    var ws = MoEWorkspace(2, weights)
    if field == 0:
        ws.hidden = 257
    elif field == 1:
        ws.ffn = 33
    elif field == 2:
        ws.n_experts = 5
    var routing = Routing(1, 5 if field == 3 else 4, 1)
    routing.ids.i32().unsafe_store(0, 0)
    routing.weights[0] = 1
    var x = Floats(256)
    var output = Floats(256)
    for i in range(256):
        output[i] = 12345
    with assert_raises(contains="workspace does not match expert dimensions"):
        grouped_experts(weights, x.ptr(), routing, ws, output.ptr(), 1)
    for i in range(256):
        assert_equal(output[i], 12345)


def test_moe_rejects_workspace_hidden_mismatch() raises:
    _check_moe_dimension_guard(0)


def test_moe_rejects_workspace_ffn_mismatch() raises:
    _check_moe_dimension_guard(1)


def test_moe_rejects_workspace_expert_count_mismatch() raises:
    _check_moe_dimension_guard(2)


def test_moe_rejects_routing_expert_count_mismatch() raises:
    _check_moe_dimension_guard(3)


def test_moe_rejects_negative_expert_id() raises:
    var raw = Mapping("tests/fixtures/moe-q8.bin")
    var scales = Floats(4)
    for e in range(4):
        scales[e] = 1
    var weights = _q8_experts(raw, scales)
    # The expected Error must be reported before this negative id reaches
    # the grouping Lists, whose bounds check otherwise traps.
    var ws = MoEWorkspace(2, weights)
    var routing = Routing(1, 4, 1)
    routing.ids.i32().unsafe_store(0, -1)
    routing.weights[0] = 1
    var x = Floats(256)
    var output = Floats(256)
    for i in range(256):
        output[i] = 12345
    with assert_raises(contains="expert id out of range: -1"):
        grouped_experts(weights, x.ptr(), routing, ws, output.ptr(), 1)
    for i in range(256):
        assert_equal(output[i], 12345)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
