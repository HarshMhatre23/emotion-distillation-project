import time, torch
def measure_latency(model, ids, mask, warmup=10, runs=50, device="cuda"):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup): model(ids, mask)
    if device=="cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs): model(ids, mask)
    if device=="cuda": torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {"latency_ms": (dt/runs)*1000, "throughput": (runs*ids.size(0))/dt, "batch_size": ids.size(0)}
def model_size_mb(model):
    n = sum(p.numel() for p in model.parameters())
    return {"params": n, "fp32_mb": n*4/1024**2, "int8_mb": n/1024**2}
