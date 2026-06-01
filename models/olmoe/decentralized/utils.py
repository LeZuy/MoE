import torch
import torch.nn.functional as F

def sample_next_token(logits_last, temperature=0.8, top_p=0.9):
    """logits_last: [1, vocab_size]"""
    logits_last = logits_last / temperature

    # Top-p (nucleus) sampling
    sorted_logits, sorted_indices = torch.sort(logits_last, descending=True)
    cumprobs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    sorted_logits[cumprobs - F.softmax(sorted_logits, dim=-1) > top_p] = -float("inf")
    probs = F.softmax(sorted_logits, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1)
    return sorted_indices.gather(-1, sampled) 

### DEBUG CODE BY CLAUDE ###
def verify_attention_output(model, simulator, tokenizer, prompt, device, layer_idx=0):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    layer = model.model.layers[layer_idx]

    attn_output_ref = {}
    layer_input_ref = {}
    moe_input_ref   = {}

    def grab_first_arg(store):
        def hook(module, args, kwargs):
            if args:
                store["x"] = args[0].detach()
            elif "hidden_states" in kwargs:
                store["x"] = kwargs["hidden_states"].detach()
        return hook

    def grab_output(store):
        def hook(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            store["x"] = out.detach()
        return hook

    hooks = [
        layer.self_attn.register_forward_pre_hook(grab_first_arg(attn_output_ref), with_kwargs=True),
        layer.register_forward_pre_hook(grab_first_arg(layer_input_ref),           with_kwargs=True),
        layer.mlp.register_forward_pre_hook(grab_first_arg(moe_input_ref),         with_kwargs=True),
    ]

    attn_out_store = {}
    hooks.append(layer.self_attn.register_forward_hook(grab_output(attn_out_store)))

    with torch.no_grad():
        model(input_ids, attention_mask=attention_mask)

    for h in hooks:
        h.remove()

    # Chạy simulator
    hidden_states = layer_input_ref["x"]
    seq_len = hidden_states.shape[1]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    causal_mask = simulator._prepare_causal_mask(attention_mask, hidden_states)

    node = simulator.nodes[layer_idx]
    with torch.no_grad():
        residual, moe_input = node.run_attention(
            hidden_states, causal_mask, position_ids
        )

    # So sánh output attention (trước residual)
    attn_out_ref = attn_out_store["x"]
    # attn_out_ref là output self_attn, chưa cộng residual
    # residual sau attention = hidden_states + attn_out
    # => attn_out = residual - layer_input (vì residual = layer_input + attn_out)
    sim_attn_out = residual - hidden_states

    diff_attn = (attn_out_ref - sim_attn_out).abs()
    print(f"=== Attention output diff ===")
    print(f"max diff:  {diff_attn.max().item():.6f}")
    print(f"mean diff: {diff_attn.mean().item():.8f}")

    # So sánh moe_input (sau post_attn_layernorm)
    diff_moe_in = (moe_input_ref["x"] - moe_input).abs()
    print(f"\n=== MoE input diff ===")
    print(f"max diff:  {diff_moe_in.max().item():.6f}")
    print(f"mean diff: {diff_moe_in.mean().item():.8f}")
    print(f"ref sample: {moe_input_ref['x'][0, 0, :6]}")
    print(f"sim sample: {moe_input[0, 0, :6]}")

def verify_layer_by_layer(model, simulator, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Chạy model gốc, hook lấy output từng layer
    ref_outputs = {}
    hooks = []
    for i, layer in enumerate(model.model.layers):
        def make_hook(idx):
            def hook(module, input, output):
                ref_outputs[idx] = output[0].detach()
            return hook
        hooks.append(layer.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        model(input_ids, attention_mask=attention_mask)
    for h in hooks:
        h.remove()

    # Chạy simulator layer by layer
    hidden_states = model.model.embed_tokens(input_ids)
    seq_len = hidden_states.shape[1]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    causal_mask = simulator._prepare_causal_mask(attention_mask, hidden_states)

    for layer_idx in range(simulator.num_layers):
        hidden_states = simulator.forward_layer(
            layer_idx, hidden_states, causal_mask, position_ids
        )

        diff = (ref_outputs[layer_idx] - hidden_states).abs()
        print(f"Layer {layer_idx:2d} | max diff: {diff.max().item():.6f} | mean diff: {diff.mean().item():.8f}")

        # Dừng lại ở layer đầu tiên bị sai
        if diff.max().item() > 0.01:
            print(f"  ^^^ Layer {layer_idx} bắt đầu sai!")
            break

def verify_moe_output(model, simulator, tokenizer, prompt, device, layer_idx=0):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    layer = model.model.layers[layer_idx]
    moe_block = layer.mlp

    moe_input_ref = {}
    moe_output_ref = {}

    # ✅ pre_hook chỉ có 2 args: module, args
    def pre_hook(module, args):
        moe_input_ref["x"] = args[0].detach()

    def post_hook(module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        moe_output_ref["x"] = out.detach()

    h1 = moe_block.register_forward_pre_hook(pre_hook)
    h2 = moe_block.register_forward_hook(post_hook)

    with torch.no_grad():
        model(input_ids, attention_mask=attention_mask)

    h1.remove()
    h2.remove()

    moe_in      = moe_input_ref["x"]
    moe_out_ref = moe_output_ref["x"]

    node       = simulator.nodes[layer_idx]
    dispatcher = simulator.dispatchers[layer_idx]
    moe_out_sim = node.run_moe(moe_in, dispatcher, parallel=False)

    diff = (moe_out_ref - moe_out_sim).abs()
    print(f"MoE layer {layer_idx} | max diff: {diff.max().item():.6f}")
    print(f"ref sample: {moe_out_ref[0, 0, :8]}")
    print(f"sim sample: {moe_out_sim[0, 0, :8]}")

def verify_moe_internals(model, simulator, tokenizer, prompt, device, layer_idx=0):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    layer = model.model.layers[layer_idx]
    moe_block = layer.mlp

    moe_input_ref = {}
    def pre_hook(module, args):
        moe_input_ref["x"] = args[0].detach()
    h = moe_block.register_forward_pre_hook(pre_hook)
    with torch.no_grad():
        model(input_ids, attention_mask=attention_mask)
    h.remove()

    moe_in = moe_input_ref["x"]
    B, S, H = moe_in.shape
    flat = moe_in.view(-1, H)

    with torch.no_grad():
        router_out = moe_block.gate(flat)
        ref_logits = router_out[0] if isinstance(router_out, tuple) else router_out

    ref_weights = torch.softmax(ref_logits, dim=-1, dtype=torch.float32)
    ref_weights, ref_experts = torch.topk(ref_weights, simulator.num_experts_per_tok, dim=-1)
    ref_weights = ref_weights / ref_weights.sum(dim=-1, keepdim=True)

    print(f"flat shape:        {flat.shape}")
    print(f"ref_experts shape: {ref_experts.shape}")
    print(f"ref_weights shape: {ref_weights.shape}")

    # --- Gốc ---
    with torch.no_grad():
        ref_moe_out = moe_block.experts(flat, ref_experts, ref_weights)
    print(f"\nref out[0,:6]: {ref_moe_out[0, :6]}")

    # --- Từng client ---
    dispatcher = simulator.dispatchers[layer_idx]
    client_outputs = []

    for cid, client in enumerate(dispatcher.clients):
        try:
            with torch.no_grad():
                out = client.forward(flat, ref_experts, ref_weights)
            nonzero = (out.abs().sum(dim=-1) > 0).sum().item()
            print(f"Client {cid} | nonzero: {nonzero}/{flat.shape[0]} | sample: {out[0, :6]}")
            client_outputs.append(out)
        except Exception as e:
            print(f"Client {cid} ERROR: {e}")

    if client_outputs:
        sim_sum = sum(client_outputs)
        diff = (ref_moe_out - sim_sum).abs()
        print(f"\nSummed sim out[0,:6]: {sim_sum[0, :6]}")
        print(f"max diff:  {diff.max().item():.6f}")
        print(f"mean diff: {diff.mean().item():.8f}")

        # ✅ Kiểm tra xem weights có bị nhân đúp không
        weight_sum_per_token = ref_weights.sum(dim=-1)
        print(f"\nref_weights sum per token (expect 1.0): min={weight_sum_per_token.min():.4f} max={weight_sum_per_token.max():.4f}")

        # ✅ Kiểm tra tổng contribution của tất cả clients
        total_nonzero = sum(
            (o.abs().sum(dim=-1) > 0).sum().item() for o in client_outputs
        )
        print(f"Total nonzero across clients: {total_nonzero} (expect = T*top_k = {flat.shape[0]}*{simulator.num_experts_per_tok}={flat.shape[0]*simulator.num_experts_per_tok})")
