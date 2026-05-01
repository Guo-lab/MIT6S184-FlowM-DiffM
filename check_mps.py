"""
Diagnose MPS compatibility for lab_one.ipynb.
Tests three things:
  1. cholesky on MPS
  2. vmap(jacrev(log_prob)) score correctness on MPS
  3. Full Langevin simulation correctness on MPS
"""

import torch


def test_cholesky():
    """Test 1: Can we do cholesky on MPS?"""
    print("=" * 60)
    print("Test 1: cholesky on MPS")
    print("=" * 60)
    try:
        m = torch.eye(3).to("mps") * 5.0
        L = torch.linalg.cholesky(m)
        expected = torch.sqrt(m)
        ok = torch.allclose(L, expected, atol=1e-6)
        print(f"  cholesky: OK")
        print(f"  Result correct: {'YES' if ok else 'NO'}")
        return ok
    except Exception as e:
        print(f"  cholesky: FAILED — {e}")
        return False


def test_score_correctness():
    """Test 2: Is vmap(jacrev(log_prob)) correct on MPS?"""
    print()
    print("=" * 60)
    print("Test 2: vmap(jacrev(log_prob)) correctness on MPS")
    print("=" * 60)

    import torch.distributions as D
    from torch.func import vmap, jacrev

    device_mps = torch.device("mps")

    # Build a simple 2D Gaussian mixture
    means = torch.tensor([[5.0, 0.0], [-3.0, 4.0], [0.0, -5.0]]).to(device_mps)
    covs = torch.diag_embed(torch.ones(3, 2).to(device_mps) * 0.5**2)
    weights = torch.ones(3).to(device_mps) / 3

    dist = D.MixtureSameFamily(
        mixture_distribution=D.Categorical(probs=weights, validate_args=False),
        component_distribution=D.MultivariateNormal(
            loc=means,
            covariance_matrix=covs,
            validate_args=False,
        ),
        validate_args=False,
    )

    def log_density(x):
        return dist.log_prob(x).view(-1, 1)

    def score_fn(x):
        x2 = x.unsqueeze(1)
        s = vmap(jacrev(log_density))(x2)
        return s.squeeze((1, 2, 3))

    # Test points
    test_pts = torch.tensor(
        [
            [0.0, 0.0],  # between modes
            [5.0, 0.0],  # at mode 0 center
            [10.0, 10.0],  # far from all modes
        ]
    ).to(device_mps)

    scores_mps = score_fn(test_pts)

    # Compare with CPU
    device_cpu = torch.device("cpu")
    means_c = means.to(device_cpu)
    covs_c = covs.to(device_cpu)
    weights_c = weights.to(device_cpu)

    dist_c = D.MixtureSameFamily(
        mixture_distribution=D.Categorical(probs=weights_c, validate_args=False),
        component_distribution=D.MultivariateNormal(
            loc=means_c,
            covariance_matrix=covs_c,
            validate_args=False,
        ),
        validate_args=False,
    )

    def log_density_c(x):
        return dist_c.log_prob(x).view(-1, 1)

    def score_fn_c(x):
        x2 = x.unsqueeze(1)
        s = vmap(jacrev(log_density_c))(x2)
        return s.squeeze((1, 2, 3))

    test_pts_c = test_pts.to(device_cpu)
    scores_cpu = score_fn_c(test_pts_c)

    all_ok = True
    for i in range(len(test_pts)):
        mps_val = scores_mps[i]
        cpu_val = scores_cpu[i]
        diff = (mps_val - cpu_val.to(device_mps)).abs().max().item()
        ok = diff < 1e-4
        status = "OK" if ok else "WRONG"
        if not ok:
            all_ok = False
        print(f"  [{test_pts[i, 0].item():.0f}, {test_pts[i, 1].item():.0f}]:")
        print(f"    MPS:  {mps_val.tolist()}")
        print(f"    CPU:  {cpu_val.tolist()}")
        print(f"    diff = {diff:.6f}  [{status}]")

    return all_ok


def test_full_simulation():
    """Test 3: Does Langevin converge correctly on MPS?"""
    print()
    print("=" * 60)
    print("Test 3: Langevin simulation correctness on MPS vs CPU")
    print("=" * 60)

    import torch.distributions as D
    from torch.func import vmap, jacrev

    device = torch.device("mps")
    sigma = 1.0
    theta = 2.0
    var = sigma**2 / (2 * theta)

    # Compare Langevin targeting N(0, var) with OU process
    dist = D.MultivariateNormal(
        torch.zeros(1).to(device), var * torch.eye(1).to(device), validate_args=False
    )

    def log_density(x):
        return dist.log_prob(x).view(-1, 1)

    def score_fn(x):
        x2 = x.unsqueeze(1)
        s = vmap(jacrev(log_density))(x2)
        return s.squeeze((1, 2, 3))

    torch.manual_seed(42)
    noises = torch.randn(500, 1).to(device)
    x0 = torch.randn(1000, 1).to(device) * 3.0
    dt = torch.tensor(0.01)

    x_ou = x0.clone()
    x_lang = x0.clone()

    for step in range(500):
        z = noises[step : step + 1]
        # OU: dX = -theta * X * dt + sigma * dW
        x_ou = x_ou + (-theta * x_ou) * dt + sigma * torch.sqrt(dt) * z
        # Langevin: dX = 0.5*sigma^2 * score * dt + sigma * dW
        x_lang = x_lang + 0.5 * sigma**2 * score_fn(x_lang) * dt + sigma * torch.sqrt(dt) * z

    diff = (x_ou - x_lang).abs().max().item()
    ok = diff < 1e-5
    print(f"  OU vs Langevin max diff: {diff:.10f}")
    print(f"  Identical: {'YES' if ok else 'NO — score function is WRONG on MPS'}")
    print(f"  OU final std:     {x_ou.std().item():.4f}  (expected ~{var**0.5:.4f})")
    print(f"  Langevin final std: {x_lang.std().item():.4f}  (expected ~{var**0.5:.4f})")
    return ok


if __name__ == "__main__":
    print(f"PyTorch {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print()

    r1 = test_cholesky()
    if not r1:
        print()
        print("➜ CONCLUSION: cholesky doesn't work on MPS. Notebook will CRASH.")
        print("  Fix: set device = 'cpu' in the first code cell.")
        print("  Or:  export PYTORCH_ENABLE_MPS_FALLBACK=1")
    else:
        r2 = test_score_correctness()
        r3 = test_full_simulation()
        print()
        if r2 and r3:
            print("➜ CONCLUSION: MPS works correctly! Safe to use.")
        else:
            print("➜ CONCLUSION: MPS has buggy vmap/jacrev. Results will be WRONG.")
