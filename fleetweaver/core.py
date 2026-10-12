"""Problem generation, original PARCO policy, and independent route validation."""

import random
import numpy as np
import torch
from tensordict import TensorDict
from parco.envs import HCVRPEnv
from parco.models import PARCOPolicy

POLICY_CONFIG = dict(
    embed_dim=64,
    num_encoder_layers=2,
    num_heads=8,
    normalization="rms",
    use_final_norm=True,
    parallel_gated_kwargs={"mlp_activation": "silu", "inner_size_multiple_of": 64},
    context_embedding_kwargs={
        "normalization": "rms",
        "use_final_norm": True,
        "parallel_gated_kwargs": {
            "mlp_activation": "silu",
            "inner_size_multiple_of": 64,
        },
    },
)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_policy():
    import copy

    return PARCOPolicy(**copy.deepcopy(POLICY_CONFIG))


def generate(n, m, count, seed, distribution="uniform"):
    """Use a local RNG: evaluation must never alter the training random stream."""
    g = torch.Generator().manual_seed(seed)
    locs = torch.rand(count, n, 2, generator=g)
    depot = torch.rand(count, 2, generator=g)
    demand = torch.randint(1, 10, (count, n), generator=g).float()
    capacity = torch.randint(20, 41, (count, m), generator=g).float()
    speed = 0.5 + 0.5 * torch.rand(count, m, generator=g)
    if distribution == "clustered":
        centers = 0.15 + 0.7 * torch.rand(count, 3, 2, generator=g)
        ids = torch.randint(0, 3, (count, n), generator=g)
        locs = (
            centers.gather(1, ids[..., None].expand(-1, -1, 2))
            + 0.07 * torch.randn(count, n, 2, generator=g)
        ).clamp(0, 1)
    elif distribution == "ring":
        angle = 2 * np.pi * torch.rand(count, n, generator=g)
        radius = 0.3 + 0.15 * torch.rand(count, n, generator=g)
        locs = torch.stack([0.5 + radius * angle.cos(), 0.5 + radius * angle.sin()], -1)
        depot.fill_(0.5)
    elif distribution == "corner":
        depot = 0.05 * torch.rand(count, 2, generator=g)
    elif distribution == "heavy_demand":
        demand = torch.randint(6, 10, (count, n), generator=g).float()
    elif distribution == "heterogeneous":
        capacity = torch.randint(12, 51, (count, m), generator=g).float()
        speed = 0.3 + 0.9 * torch.rand(count, m, generator=g)
    elif distribution == "stripe":
        locs[..., 1] = 0.45 + 0.1 * locs[..., 1]
    elif distribution != "uniform":
        raise ValueError(distribution)
    return TensorDict(
        dict(
            locs=locs,
            depot=depot,
            demand=demand,
            capacity=capacity,
            speed=speed,
            num_agents=torch.full((count,), m),
        ),
        batch_size=[count],
    )


def instance(td):
    return {
        k: td[k].cpu().numpy() for k in ("locs", "depot", "demand", "capacity", "speed")
    }


def actions_to_routes(actions, m):
    """Original PARCO actions have separate colocated depot IDs and pause actions."""
    routes = []
    for v, seq in enumerate(np.asarray(actions)):
        path = [0]
        previous = v
        for a in seq:
            a = int(a)
            if a == previous:  # conflict-induced pause, not a second service
                continue
            if a < m and a != v:
                raise ValueError("Vehicle selected another vehicle's depot")
            path.append(0 if a < m else a - m + 1)
            previous = a
        if path[-1] != 0:
            path.append(0)
        routes.append(path)
    return routes


def validate_routes(inst, routes):
    """Independent NumPy evaluator; checks unique service, refills, and final return."""
    n = len(inst["locs"])
    if len(routes) != len(inst["speed"]):
        raise ValueError("Wrong fleet size")
    xy = np.vstack([inst["depot"], inst["locs"]]).astype(np.float64)
    visits = []
    times = []
    for v, path in enumerate(routes):
        if len(path) < 1 or path[0] != 0 or path[-1] != 0:
            raise ValueError("Routes must start and end at depot")
        load = 0.0
        dist = 0.0
        for a, b in zip(path, path[1:]):
            if b < 0 or b > n:
                raise ValueError("Unknown node")
            dist += float(np.linalg.norm(xy[a] - xy[b]))
            if b == 0:
                load = 0.0
            else:
                visits.append(b)
                load += float(inst["demand"][b - 1])
                if load > float(inst["capacity"][v]) + 1e-5:
                    raise ValueError("Capacity exceeded")
        times.append(dist / float(inst["speed"][v]))
    if sorted(visits) != list(range(1, n + 1)):
        raise ValueError("Customer missing or served more than once")
    return max(times), times


@torch.inference_mode()
def predict(policy, td, device="cpu", samples=1):
    policy.eval()
    env = HCVRPEnv()
    state = env.reset(td.to(device))
    out = policy(
        state,
        env,
        phase="test",
        decode_type="sampling" if samples > 1 else "greedy",
        num_samples=samples,
        select_best=samples > 1,
        max_steps=10 * td["locs"].shape[1] + 100,
    )
    return out
