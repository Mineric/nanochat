"""
MLX implementation of Muon optimizer.

Muon - MomentUm Orthogonalized by Newton-schulz
Port of the PyTorch Muon optimizer to MLX for Apple Silicon.

https://kellerjordan.github.io/posts/muon/
"""

import mlx.core as mx
from typing import List, Dict, Any


def zeropower_via_newtonschulz5(G: mx.array, steps: int = 5) -> mx.array:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    Opt to use a quintic iteration whose coefficients are selected to maximize
    the slope at zero. For the purpose of minimizing steps, it turns out to be
    empirically effective to keep increasing the slope at zero even beyond the point
    where the iteration no longer converges all the way to one everywhere on the interval.

    This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to
    hurt model performance at all relative to UV^T, where USV^T = G is the SVD.

    Args:
        G: Gradient tensor (must be 2D)
        steps: Number of Newton-Schulz iterations

    Returns:
        Orthogonalized gradient
    """
    assert G.ndim >= 2, "G must be at least 2D"

    # Quintic iteration coefficients
    a, b, c = (3.4445, -4.7750, 2.0315)

    # Convert to bfloat16 for stability
    X = G.astype(mx.bfloat16)

    # Transpose if needed (make it "tall" matrix)
    transposed = False
    if G.shape[-2] > G.shape[-1]:
        X = X.T
        transposed = True

    # Ensure spectral norm is at most 1
    # Compute Frobenius norm along last two dimensions
    norm = mx.sqrt(mx.sum(X ** 2, axis=(-2, -1), keepdims=True))
    X = X / (norm + 1e-7)

    # Perform the Newton-Schulz iterations
    for _ in range(steps):
        # A = X @ X^T
        A = X @ X.T

        # B = b * A + c * A @ A (quintic computation strategy)
        B = b * A + c * (A @ A)

        # X = a * X + B @ X
        X = a * X + B @ X

    # Transpose back if needed
    if transposed:
        X = X.T

    return X


class MuonMLX:
    """
    MLX implementation of Muon optimizer for Apple Silicon.

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization
    post-processing step, in which each 2D parameter's update is replaced with the
    nearest orthogonal matrix. To efficiently orthogonalize each update, we use a
    Newton-Schulz iteration, which has the advantage that it can be stably run in
    bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully
      connected layer, or any {0,1}-D parameters; those should all be optimized by
      a standard method (e.g., AdamW).
    - Designed for 2D parameters (linear layer weights).

    Arguments:
        params: List of parameters to optimize
        lr: The learning rate used by the internal SGD (default: 0.02)
        momentum: The momentum used by the internal SGD (default: 0.95)
        nesterov: Whether to use Nesterov-style momentum (default: True, recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use (default: 5)
    """

    def __init__(
        self,
        params: List[Any],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5
    ):
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps

        # Store parameters
        self.params = list(params)

        # Group parameters by size (for efficiency)
        # MLX works best when processing same-shaped tensors together
        size_to_params = {}
        for p in self.params:
            if hasattr(p, 'data'):
                param_data = p.data
            else:
                param_data = p

            numel = param_data.size
            if numel not in size_to_params:
                size_to_params[numel] = []
            size_to_params[numel].append(p)

        self.param_groups = [
            {'params': params_list, 'lr': lr}
            for params_list in size_to_params.values()
        ]

        # For compatibility with PyTorch-style interface
        for group in self.param_groups:
            group['initial_lr'] = lr

        # State dictionary to store momentum buffers
        self.state = {}

        print(f"MuonMLX: Initialized with {len(self.params)} parameters")
        print(f"  lr={lr}, momentum={momentum}, nesterov={nesterov}, ns_steps={ns_steps}")

    def zero_grad(self):
        """Zero out gradients."""
        for p in self.params:
            if hasattr(p, 'grad'):
                p.grad = None

    def step(self):
        """
        Perform a single optimization step.

        For each parameter:
        1. Update momentum buffer with current gradient
        2. Apply Nesterov momentum if enabled
        3. Orthogonalize the update via Newton-Schulz
        4. Apply the update with aspect-ratio scaling
        """
        for group in self.param_groups:
            for p in group['params']:
                # Get parameter data and gradient
                if hasattr(p, 'data'):
                    param_data = p.data
                    grad = p.grad if hasattr(p, 'grad') else None
                else:
                    param_data = p
                    grad = getattr(p, 'grad', None)

                if grad is None:
                    continue

                # Get gradient as MLX array
                if hasattr(grad, 'data'):
                    g = grad.data
                else:
                    g = grad

                # Initialize state for this parameter if needed
                param_id = id(p)
                if param_id not in self.state:
                    self.state[param_id] = {
                        'momentum_buffer': mx.zeros_like(g)
                    }

                state = self.state[param_id]
                buf = state['momentum_buffer']

                # Update momentum buffer: buf = momentum * buf + (1 - momentum) * g
                # In PyTorch this is: buf.lerp_(g, 1 - momentum)
                buf = self.momentum * buf + (1 - self.momentum) * g
                state['momentum_buffer'] = buf

                # Apply Nesterov momentum if enabled
                if self.nesterov:
                    # g = momentum * buf + (1 - momentum) * g
                    # In PyTorch: g.lerp_(buf, momentum)
                    g = self.momentum * buf + (1 - self.momentum) * g
                else:
                    g = buf

                # Orthogonalize via Newton-Schulz iteration
                g_ortho = zeropower_via_newtonschulz5(g, steps=self.ns_steps)

                # Apply aspect-ratio scaling
                # Scale by sqrt(max(1, rows/cols)) to account for matrix shape
                if g_ortho.ndim >= 2:
                    aspect_ratio = max(1.0, g_ortho.shape[-2] / g_ortho.shape[-1])
                    scale = mx.sqrt(mx.array(aspect_ratio))
                else:
                    scale = mx.array(1.0)

                # Apply update: param = param - lr * scale * g_ortho
                update = self.lr * scale * g_ortho

                # Update parameter
                if hasattr(p, 'data'):
                    p.data = param_data - update
                else:
                    # For raw arrays, need to handle differently
                    # This is a simplified version - in practice you'd integrate with MLX's optimizer API
                    pass

    def __repr__(self):
        return (f"MuonMLX(lr={self.lr}, momentum={self.momentum}, "
                f"nesterov={self.nesterov}, ns_steps={self.ns_steps})")


class MuonMLXWrapper:
    """
    Wrapper to make MuonMLX compatible with nanochat's training loop.

    This provides a PyTorch-like interface for the MLX Muon optimizer,
    making it a drop-in replacement in the training scripts.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        """
        Initialize MLX Muon optimizer with PyTorch-compatible interface.

        Args:
            params: List of parameters or param groups
            lr: Learning rate
            momentum: Momentum coefficient
            nesterov: Use Nesterov momentum
            ns_steps: Number of Newton-Schulz iterations
        """
        # Handle param_groups (list of dicts) or flat list
        if isinstance(params, list) and len(params) > 0 and isinstance(params[0], dict):
            # param_groups format: [{'params': [...], 'lr': ...}, ...]
            all_params = []
            for group in params:
                all_params.extend(group['params'])
            self.muon = MuonMLX(all_params, lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
            self.param_groups = params
        else:
            # Flat list of parameters
            self.muon = MuonMLX(params, lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
            self.param_groups = [{'params': params, 'lr': lr, 'initial_lr': lr}]

        # Ensure all param groups have initial_lr
        for group in self.param_groups:
            if 'initial_lr' not in group:
                group['initial_lr'] = group.get('lr', lr)

    def zero_grad(self):
        """Zero gradients."""
        self.muon.zero_grad()

    def step(self):
        """Perform optimization step."""
        self.muon.step()

    def state_dict(self):
        """Get optimizer state."""
        return {
            'state': self.muon.state,
            'param_groups': self.param_groups,
        }

    def load_state_dict(self, state_dict):
        """Load optimizer state."""
        self.muon.state = state_dict['state']
        self.param_groups = state_dict['param_groups']

    def __repr__(self):
        return f"MuonMLXWrapper({self.muon})"


# Convenience function
def create_muon_optimizer(params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
    """
    Create MLX Muon optimizer with PyTorch-compatible interface.

    This is the recommended way to create the optimizer for use in nanochat.

    Args:
        params: Parameters to optimize
        lr: Learning rate
        momentum: Momentum coefficient
        nesterov: Use Nesterov momentum
        ns_steps: Number of Newton-Schulz iterations

    Returns:
        MuonMLXWrapper instance
    """
    return MuonMLXWrapper(params, lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)


if __name__ == "__main__":
    """
    Simple test of the MLX Muon optimizer.
    """
    print("Testing MLX Muon optimizer...")

    # Test Newton-Schulz orthogonalization
    print("\n1. Testing Newton-Schulz orthogonalization:")
    G = mx.random.normal((64, 32))
    G_ortho = zeropower_via_newtonschulz5(G, steps=5)

    # Check that result is approximately orthogonal
    # For orthogonal matrix: G_ortho @ G_ortho.T ≈ I
    product = G_ortho @ G_ortho.T
    identity = mx.eye(64)
    error = mx.mean(mx.abs(product - identity))
    print(f"   Input shape: {G.shape}")
    print(f"   Output shape: {G_ortho.shape}")
    print(f"   Orthogonality error: {error:.6f} (should be small)")

    # Test optimizer
    print("\n2. Testing Muon optimizer:")

    # Create dummy parameters
    class DummyParam:
        def __init__(self, shape):
            self.data = mx.random.normal(shape)
            self.grad = mx.random.normal(shape) * 0.01

    params = [
        DummyParam((128, 64)),
        DummyParam((64, 32)),
    ]

    optimizer = create_muon_optimizer(params, lr=0.01, momentum=0.9)

    # Run a few steps
    print(f"   Created optimizer: {optimizer}")
    print(f"   Running 3 optimization steps...")

    for step in range(3):
        old_norm = mx.sqrt(mx.sum(params[0].data ** 2))
        optimizer.step()
        new_norm = mx.sqrt(mx.sum(params[0].data ** 2))
        print(f"   Step {step+1}: param norm changed from {old_norm:.4f} to {new_norm:.4f}")

    print("\n✅ MLX Muon optimizer test complete!")
