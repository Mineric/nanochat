"""
MLX compatibility layer - makes MLX look like PyTorch.

This module wraps MLX to provide a PyTorch-like API, allowing existing PyTorch code
to run on Apple Silicon with minimal changes. Import this module as 'torch' to use MLX
as the backend instead of PyTorch.

Usage:
    import nanochat.mlx_compat as torch
    # Now use 'torch' as you would normally
"""

import sys
import mlx.core as mx
import mlx.nn as mlx_nn
import mlx.optimizers as mlx_optim
from functools import wraps
from typing import Any, Optional, Tuple, Union, List
import numpy as np

# ============================================================================
# Data types
# ============================================================================

float32 = mx.float32
bfloat16 = mx.bfloat16
int32 = mx.int32
int64 = mx.int64
long = mx.int64
bool = mx.bool_

# ============================================================================
# Core Tensor wrapper
# ============================================================================

class Tensor:
    """Wrapper to make MLX arrays behave like PyTorch tensors."""

    def __init__(self, data, requires_grad=False):
        if isinstance(data, Tensor):
            self.data = data.data
        elif isinstance(data, mx.array):
            self.data = data
        elif isinstance(data, (list, tuple, np.ndarray)):
            self.data = mx.array(data)
        else:
            self.data = mx.array([data])

        self.grad = None
        self.requires_grad = requires_grad
        self.dtype = self.data.dtype
        self.device = "mps"  # MLX uses unified memory

    @property
    def shape(self):
        return self.data.shape

    def size(self, dim=None):
        if dim is None:
            return self.data.shape
        if dim < 0:
            dim = len(self.data.shape) + dim
        return self.data.shape[dim]

    def dim(self):
        return len(self.data.shape)

    def ndim(self):
        return len(self.data.shape)

    def numel(self):
        """Return total number of elements."""
        result = 1
        for s in self.data.shape:
            result *= s
        return result

    def view(self, *shape):
        """Reshape tensor."""
        return Tensor(mx.reshape(self.data, shape))

    def reshape(self, *shape):
        """Reshape tensor."""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        return Tensor(mx.reshape(self.data, shape))

    def transpose(self, dim0, dim1):
        """Transpose two dimensions."""
        return Tensor(mx.swapaxes(self.data, dim0, dim1))

    def permute(self, *dims):
        """Permute dimensions."""
        return Tensor(mx.transpose(self.data, dims))

    def contiguous(self):
        """Return contiguous tensor (no-op for MLX)."""
        return self

    def to(self, device=None, dtype=None, non_blocking=False):
        """Move tensor to device/dtype."""
        result = self
        if dtype is not None:
            result = Tensor(self.data.astype(dtype))
        # device is ignored - MLX uses unified memory
        return result

    def cpu(self):
        """Move to CPU (no-op for MLX)."""
        return self

    def cuda(self):
        """Move to CUDA (no-op for MLX)."""
        return self

    def detach(self):
        """Detach from computation graph."""
        return Tensor(self.data)

    def item(self):
        """Get Python scalar."""
        return self.data.item()

    def tolist(self):
        """Convert to Python list."""
        return self.data.tolist()

    def float(self):
        """Convert to float32."""
        return Tensor(self.data.astype(mx.float32))

    def bfloat16(self):
        """Convert to bfloat16."""
        return Tensor(self.data.astype(mx.bfloat16))

    def long(self):
        """Convert to int64."""
        return Tensor(self.data.astype(mx.int64))

    def int(self):
        """Convert to int32."""
        return Tensor(self.data.astype(mx.int32))

    # Indexing
    def __getitem__(self, key):
        return Tensor(self.data[key])

    def __setitem__(self, key, value):
        if isinstance(value, Tensor):
            value = value.data
        self.data[key] = value

    # Math operations
    def __add__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data + other_data)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data - other_data)

    def __rsub__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(other_data - self.data)

    def __mul__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data * other_data)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data / other_data)

    def __rtruediv__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(other_data / self.data)

    def __pow__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data ** other_data)

    def __rpow__(self, other):
        """Right power: other ** self"""
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(other_data ** self.data)

    def __matmul__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data @ other_data)

    def __neg__(self):
        return Tensor(-self.data)

    def __lt__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data < other_data)

    def __le__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data <= other_data)

    def __gt__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data > other_data)

    def __ge__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data >= other_data)

    def __eq__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data == other_data)

    # In-place operations
    def add_(self, other, alpha=1.0):
        """In-place addition."""
        other_data = other.data if isinstance(other, Tensor) else other
        self.data = self.data + alpha * other_data
        return self

    def mul_(self, other):
        """In-place multiplication."""
        other_data = other.data if isinstance(other, Tensor) else other
        self.data = self.data * other_data
        return self

    def div_(self, other):
        """In-place division."""
        other_data = other.data if isinstance(other, Tensor) else other
        self.data = self.data / other_data
        return self

    def sub_(self, other):
        """In-place subtraction."""
        other_data = other.data if isinstance(other, Tensor) else other
        self.data = self.data - other_data
        return self

    def lerp_(self, target, weight):
        """In-place linear interpolation."""
        target_data = target.data if isinstance(target, Tensor) else target
        self.data = self.data * (1 - weight) + target_data * weight
        return self

    def addcmul_(self, tensor1, tensor2, value=1.0):
        """In-place element-wise multiplication and addition."""
        t1_data = tensor1.data if isinstance(tensor1, Tensor) else tensor1
        t2_data = tensor2.data if isinstance(tensor2, Tensor) else tensor2
        self.data = self.data + value * t1_data * t2_data
        return self

    def resize_(self, *shape):
        """Resize tensor in-place."""
        self.data.resize_(shape)
        return self

    # Aggregation operations
    def sum(self, dim=None, keepdim=False):
        """Sum over dimension."""
        if dim is None:
            return Tensor(mx.sum(self.data))
        return Tensor(mx.sum(self.data, axis=dim, keepdims=keepdim))

    def mean(self, dim=None, keepdim=False):
        """Mean over dimension."""
        if dim is None:
            return Tensor(mx.mean(self.data))
        return Tensor(mx.mean(self.data, axis=dim, keepdims=keepdim))

    def max(self, dim=None, keepdim=False):
        """Max over dimension."""
        if dim is None:
            return Tensor(mx.max(self.data))
        result = mx.max(self.data, axis=dim, keepdims=keepdim)
        # PyTorch returns (values, indices), we only return values for now
        return Tensor(result)

    def min(self, dim=None, keepdim=False):
        """Min over dimension."""
        if dim is None:
            return Tensor(mx.min(self.data))
        result = mx.min(self.data, axis=dim, keepdims=keepdim)
        return Tensor(result)

    def sqrt(self):
        """Square root."""
        return Tensor(mx.sqrt(self.data))

    def square(self):
        """Square."""
        return Tensor(self.data ** 2)

    def exp(self):
        """Exponential."""
        return Tensor(mx.exp(self.data))

    def log(self):
        """Natural logarithm."""
        return Tensor(mx.log(self.data))

    def abs(self):
        """Absolute value."""
        return Tensor(mx.abs(self.data))

    def cos(self):
        """Cosine."""
        return Tensor(mx.cos(self.data))

    def sin(self):
        """Sine."""
        return Tensor(mx.sin(self.data))

    def tanh(self):
        """Hyperbolic tangent."""
        return Tensor(mx.tanh(self.data))

    def norm(self, p=2, dim=None, keepdim=False):
        """Compute norm."""
        if dim is None:
            if p == 2:
                return Tensor(mx.sqrt(mx.sum(self.data ** 2)))
            else:
                return Tensor(mx.sum(mx.abs(self.data) ** p) ** (1.0 / p))
        else:
            if p == 2:
                return Tensor(mx.sqrt(mx.sum(self.data ** 2, axis=dim, keepdims=keepdim)))
            else:
                return Tensor(mx.sum(mx.abs(self.data) ** p, axis=dim, keepdims=keepdim) ** (1.0 / p))

    def clamp(self, min=None, max=None):
        """Clamp values."""
        result = self.data
        if min is not None:
            result = mx.maximum(result, min)
        if max is not None:
            result = mx.minimum(result, max)
        return Tensor(result)

    def gather(self, dim, index):
        """Gather values along dimension."""
        index_data = index.data if isinstance(index, Tensor) else index
        return Tensor(mx.take_along_axis(self.data, index_data, axis=dim))

    def __repr__(self):
        return f"Tensor({self.data}, device='{self.device}', dtype={self.dtype})"

# ============================================================================
# Tensor creation functions
# ============================================================================

def tensor(data, dtype=None, device=None, requires_grad=False):
    """Create a tensor."""
    if isinstance(data, Tensor):
        arr = data.data
    else:
        arr = mx.array(data)

    if dtype is not None:
        arr = arr.astype(dtype)

    return Tensor(arr, requires_grad=requires_grad)

def zeros(*shape, dtype=None, device=None):
    """Create tensor filled with zeros."""
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    arr = mx.zeros(shape)
    if dtype:
        arr = arr.astype(dtype)
    return Tensor(arr)

def zeros_like(input, dtype=None, device=None):
    """Create tensor of zeros with same shape as input."""
    input_data = input.data if isinstance(input, Tensor) else input
    arr = mx.zeros_like(input_data)
    if dtype:
        arr = arr.astype(dtype)
    return Tensor(arr)

def ones(*shape, dtype=None, device=None):
    """Create tensor filled with ones."""
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    arr = mx.ones(shape)
    if dtype:
        arr = arr.astype(dtype)
    return Tensor(arr)

def ones_like(input, dtype=None, device=None):
    """Create tensor of ones with same shape as input."""
    input_data = input.data if isinstance(input, Tensor) else input
    arr = mx.ones_like(input_data)
    if dtype:
        arr = arr.astype(dtype)
    return Tensor(arr)

def empty(*shape, dtype=None, device=None, pin_memory=False):
    """Create empty tensor (uninitialized)."""
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    # MLX doesn't have empty, use zeros
    arr = mx.zeros(shape)
    if dtype:
        arr = arr.astype(dtype)
    return Tensor(arr)

def empty_like(input, dtype=None, device=None):
    """Create empty tensor with same shape as input."""
    input_data = input.data if isinstance(input, Tensor) else input
    arr = mx.zeros_like(input_data)
    if dtype:
        arr = arr.astype(dtype)
    return Tensor(arr)

def arange(start, end=None, step=1, dtype=None, device=None):
    """Create range of values."""
    if end is None:
        end = start
        start = 0
    arr = mx.arange(start, end, step)
    if dtype:
        arr = arr.astype(dtype)
    return Tensor(arr)

def linspace(start, end, steps, dtype=None, device=None):
    """Create linearly spaced values."""
    arr = mx.linspace(start, end, steps)
    if dtype:
        arr = arr.astype(dtype)
    return Tensor(arr)

def cat(tensors, dim=0):
    """Concatenate tensors."""
    arrays = [t.data if isinstance(t, Tensor) else t for t in tensors]
    return Tensor(mx.concatenate(arrays, axis=dim))

def stack(tensors, dim=0):
    """Stack tensors."""
    arrays = [t.data if isinstance(t, Tensor) else t for t in tensors]
    return Tensor(mx.stack(arrays, axis=dim))

def where(condition, x, y):
    """Select elements from x or y based on condition."""
    cond_data = condition.data if isinstance(condition, Tensor) else condition
    x_data = x.data if isinstance(x, Tensor) else x
    y_data = y.data if isinstance(y, Tensor) else y
    return Tensor(mx.where(cond_data, x_data, y_data))

def tril(input, diagonal=0):
    """Lower triangular matrix."""
    input_data = input.data if isinstance(input, Tensor) else input
    return Tensor(mx.tril(input_data, k=diagonal))

def triu(input, diagonal=0):
    """Upper triangular matrix."""
    input_data = input.data if isinstance(input, Tensor) else input
    return Tensor(mx.triu(input_data, k=diagonal))

def maximum(input, other):
    """Element-wise maximum."""
    input_data = input.data if isinstance(input, Tensor) else input
    other_data = other.data if isinstance(other, Tensor) else other
    return Tensor(mx.maximum(input_data, other_data))

def minimum(input, other):
    """Element-wise minimum."""
    input_data = input.data if isinstance(input, Tensor) else input
    other_data = other.data if isinstance(other, Tensor) else other
    return Tensor(mx.minimum(input_data, other_data))

def outer(input, vec2):
    """Outer product of two vectors."""
    input_data = input.data if isinstance(input, Tensor) else input
    vec2_data = vec2.data if isinstance(vec2, Tensor) else vec2
    return Tensor(mx.outer(input_data, vec2_data))

def topk(input, k, dim=-1, largest=True, sorted=True):
    """Return k largest/smallest elements."""
    input_data = input.data if isinstance(input, Tensor) else input

    if largest:
        indices = mx.argpartition(-input_data, k-1, axis=dim)[..., :k]
        values = mx.take_along_axis(input_data, indices, axis=dim)
    else:
        indices = mx.argpartition(input_data, k-1, axis=dim)[..., :k]
        values = mx.take_along_axis(input_data, indices, axis=dim)

    return Tensor(values), Tensor(indices)

def argmax(input, dim=None, keepdim=False):
    """Return indices of maximum values."""
    input_data = input.data if isinstance(input, Tensor) else input
    return Tensor(mx.argmax(input_data, axis=dim, keepdims=keepdim))

def argmin(input, dim=None, keepdim=False):
    """Return indices of minimum values."""
    input_data = input.data if isinstance(input, Tensor) else input
    return Tensor(mx.argmin(input_data, axis=dim, keepdims=keepdim))

def multinomial(input, num_samples, replacement=False, generator=None):
    """Sample from multinomial distribution."""
    input_data = input.data if isinstance(input, Tensor) else input

    # Use MLX categorical sampling
    if generator is not None:
        # Use the key from generator if available
        key = getattr(generator, '_key', None)
        if key is not None:
            samples = mx.random.categorical(mx.log(input_data), num_samples=num_samples, key=key)
        else:
            samples = mx.random.categorical(mx.log(input_data), num_samples=num_samples)
    else:
        samples = mx.random.categorical(mx.log(input_data), num_samples=num_samples)

    return Tensor(samples).unsqueeze(-1) if samples.ndim == 1 else Tensor(samples)

# ============================================================================
# Neural network modules
# ============================================================================

class Module:
    """Base module that mimics torch.nn.Module."""

    def __init__(self):
        self._modules = {}
        self._parameters = {}
        self._buffers = {}

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def register_buffer(self, name, tensor, persistent=True):
        """Register a buffer (non-trainable tensor)."""
        self._buffers[name] = tensor
        setattr(self, name, tensor)

    def parameters(self):
        """Return all parameters."""
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            if hasattr(module, 'parameters'):
                yield from module.parameters()

    def named_parameters(self):
        """Return named parameters."""
        for name, param in self._parameters.items():
            yield name, param
        for name, module in self._modules.items():
            if hasattr(module, 'named_parameters'):
                for subname, param in module.named_parameters():
                    yield f"{name}.{subname}", param

    def state_dict(self):
        """Get state dictionary."""
        state = {}
        for name, param in self.named_parameters():
            state[name] = param
        return state

    def load_state_dict(self, state_dict):
        """Load state dictionary."""
        # Simple implementation - would need more sophistication for real use
        for name, param in state_dict.items():
            # Navigate to the right module/parameter
            parts = name.split('.')
            obj = self
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], param)

    def train(self, mode=True):
        """Set training mode."""
        return self

    def eval(self):
        """Set evaluation mode."""
        return self.train(False)

    def to(self, device=None, dtype=None):
        """Move module to device/dtype."""
        # MLX uses unified memory, so this is mostly a no-op
        return self

    def to_empty(self, device=None):
        """Move to empty device (for initialization)."""
        return self

    def zero_grad(self, set_to_none=True):
        """Zero out gradients."""
        for param in self.parameters():
            if hasattr(param, 'grad'):
                if set_to_none:
                    param.grad = None
                else:
                    param.grad = zeros_like(param)

    def apply(self, fn):
        """Apply function to all modules."""
        fn(self)
        for module in self._modules.values():
            if hasattr(module, 'apply'):
                module.apply(fn)
        return self

    def get_device(self):
        """Get device of module."""
        return "mps"

    def estimate_flops(self):
        """Estimate FLOPs (implement in subclass)."""
        raise NotImplementedError

class Linear(Module):
    """Linear layer wrapping MLX."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights
        scale = mx.sqrt(1.0 / in_features)
        self.weight = Tensor(mx.random.uniform(-scale, scale, (out_features, in_features)))

        if bias:
            self.bias = Tensor(mx.zeros((out_features,)))
        else:
            self.bias = None

        self._parameters['weight'] = self.weight
        if bias:
            self._parameters['bias'] = self.bias

    def forward(self, x):
        x_data = x.data if isinstance(x, Tensor) else x
        result = x_data @ self.weight.data.T
        if self.bias is not None:
            result = result + self.bias.data
        return Tensor(result)

class Embedding(Module):
    """Embedding layer wrapping MLX."""

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Initialize embeddings
        self.weight = Tensor(mx.random.normal((num_embeddings, embedding_dim)))
        self._parameters['weight'] = self.weight

    def forward(self, x):
        x_data = x.data if isinstance(x, Tensor) else x
        # MLX take operation for embedding lookup
        return Tensor(self.weight.data[x_data])

class ModuleDict(dict, Module):
    """Dictionary of modules."""

    def __init__(self, modules=None):
        Module.__init__(self)
        dict.__init__(self)
        if modules is not None:
            self.update(modules)

    def __setitem__(self, key, module):
        dict.__setitem__(self, key, module)
        self._modules[key] = module

    def __getattr__(self, key):
        if key.startswith('_'):
            return object.__getattribute__(self, key)
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'ModuleDict' object has no attribute '{key}'")

    def update(self, modules):
        for key, module in modules.items():
            self[key] = module

class ModuleList(list, Module):
    """List of modules."""

    def __init__(self, modules=None):
        Module.__init__(self)
        list.__init__(self)
        if modules is not None:
            self.extend(modules)

    def append(self, module):
        list.append(self, module)
        self._modules[str(len(self) - 1)] = module

    def extend(self, modules):
        for module in modules:
            self.append(module)

    def __iter__(self):
        return list.__iter__(self)

# ============================================================================
# Functional API
# ============================================================================

class functional:
    """Functional operations mimicking torch.nn.functional."""

    @staticmethod
    def rms_norm(x, normalized_shape, eps=1e-5):
        """RMS normalization."""
        x_data = x.data if isinstance(x, Tensor) else x
        # Compute RMS
        variance = mx.mean(x_data ** 2, axis=-1, keepdims=True)
        return Tensor(x_data / mx.sqrt(variance + eps))

    @staticmethod
    def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
        """Layer normalization."""
        x_data = x.data if isinstance(x, Tensor) else x
        mean = mx.mean(x_data, axis=-1, keepdims=True)
        variance = mx.var(x_data, axis=-1, keepdims=True)
        x_norm = (x_data - mean) / mx.sqrt(variance + eps)

        if weight is not None:
            weight_data = weight.data if isinstance(weight, Tensor) else weight
            x_norm = x_norm * weight_data

        if bias is not None:
            bias_data = bias.data if isinstance(bias, Tensor) else bias
            x_norm = x_norm + bias_data

        return Tensor(x_norm)

    @staticmethod
    def relu(x):
        """ReLU activation."""
        x_data = x.data if isinstance(x, Tensor) else x
        return Tensor(mx.maximum(x_data, 0))

    @staticmethod
    def gelu(x):
        """GELU activation."""
        x_data = x.data if isinstance(x, Tensor) else x
        return Tensor(mx.nn.gelu(x_data))

    @staticmethod
    def silu(x):
        """SiLU activation."""
        x_data = x.data if isinstance(x, Tensor) else x
        return Tensor(x_data * mx.sigmoid(x_data))

    @staticmethod
    def softmax(x, dim=-1):
        """Softmax."""
        x_data = x.data if isinstance(x, Tensor) else x
        return Tensor(mx.softmax(x_data, axis=dim))

    @staticmethod
    def log_softmax(x, dim=-1):
        """Log softmax."""
        x_data = x.data if isinstance(x, Tensor) else x
        return Tensor(mx.log(mx.softmax(x_data, axis=dim)))

    @staticmethod
    def cross_entropy(input, target, ignore_index=-1, reduction='mean'):
        """Cross entropy loss."""
        logits = input.data if isinstance(input, Tensor) else input
        targets = target.data if isinstance(target, Tensor) else target

        # Flatten if needed
        if logits.ndim > 2:
            logits_2d = mx.reshape(logits, (-1, logits.shape[-1]))
            targets_1d = mx.reshape(targets, (-1,))
        else:
            logits_2d = logits
            targets_1d = targets

        # Compute cross entropy
        # Create one-hot encoding
        num_classes = logits_2d.shape[-1]
        targets_one_hot = mx.zeros((targets_1d.shape[0], num_classes))

        # MLX cross entropy
        log_probs = mx.log(mx.softmax(logits_2d, axis=-1))

        # Gather the log probabilities of the correct classes
        batch_size = targets_1d.shape[0]
        loss_per_sample = -log_probs[mx.arange(batch_size), targets_1d.astype(mx.int32)]

        # Handle ignore_index
        if ignore_index != -1:
            mask = targets_1d != ignore_index
            loss_per_sample = loss_per_sample * mask

            if reduction == 'mean':
                return Tensor(mx.sum(loss_per_sample) / mx.sum(mask))
            elif reduction == 'sum':
                return Tensor(mx.sum(loss_per_sample))
        else:
            if reduction == 'mean':
                return Tensor(mx.mean(loss_per_sample))
            elif reduction == 'sum':
                return Tensor(mx.sum(loss_per_sample))

        return Tensor(loss_per_sample)

    @staticmethod
    def scaled_dot_product_attention(query, key, value, attn_mask=None, is_causal=False):
        """Scaled dot product attention."""
        q = query.data if isinstance(query, Tensor) else query
        k = key.data if isinstance(key, Tensor) else key
        v = value.data if isinstance(value, Tensor) else value

        # Compute attention scores
        scale = 1.0 / mx.sqrt(mx.array(q.shape[-1], dtype=q.dtype))
        scores = (q @ k.swapaxes(-2, -1)) * scale

        # Apply causal mask if needed
        if is_causal:
            seq_len_q = scores.shape[-2]
            seq_len_k = scores.shape[-1]
            causal_mask = mx.tril(mx.ones((seq_len_q, seq_len_k), dtype=mx.bool_))
            scores = mx.where(causal_mask, scores, -1e9)

        # Apply custom mask if provided
        if attn_mask is not None:
            mask_data = attn_mask.data if isinstance(attn_mask, Tensor) else attn_mask
            # Assuming mask is boolean: True = keep, False = mask
            scores = mx.where(mask_data, scores, -1e9)

        # Softmax and weighted sum
        attn_weights = mx.softmax(scores, axis=-1)
        output = attn_weights @ v

        return Tensor(output)

F = functional()

# ============================================================================
# Optimizers (wrappers around MLX optimizers)
# ============================================================================

class Optimizer:
    """Base optimizer class."""

    def __init__(self, param_groups):
        self.param_groups = param_groups if isinstance(param_groups, list) else [param_groups]

    def zero_grad(self):
        """Zero gradients."""
        for group in self.param_groups:
            for param in group['params']:
                if hasattr(param, 'grad'):
                    param.grad = None

    def step(self):
        """Perform optimization step."""
        raise NotImplementedError

# ============================================================================
# Autocast context (no-op for MLX)
# ============================================================================

class autocast:
    """No-op context manager for MLX (automatic mixed precision)."""

    def __init__(self, device_type, dtype=None, enabled=True):
        self.device_type = device_type
        self.dtype = dtype
        self.enabled = enabled

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

class amp:
    """Mock torch.amp module."""
    autocast = autocast

# ============================================================================
# Device and CUDA mocking
# ============================================================================

class device:
    """Mock device class."""

    def __init__(self, device_str):
        if isinstance(device_str, device):
            self.type = device_str.type
        else:
            self.type = "mps"  # Always MPS for MLX

    def __str__(self):
        return self.type

    def __repr__(self):
        return f"device(type='{self.type}')"

class cuda:
    """Mock CUDA module."""

    @staticmethod
    def is_available():
        return False

    @staticmethod
    def synchronize():
        """Force evaluation of pending operations."""
        mx.eval(mx.array([0]))

    @staticmethod
    def max_memory_allocated():
        """Get peak memory usage."""
        return mx.metal.get_peak_memory()

    @staticmethod
    def manual_seed(seed):
        """Set random seed."""
        mx.random.seed(seed)

    @staticmethod
    def set_device(dev):
        """Set device (no-op for MLX)."""
        pass

class backends:
    """Mock backends module."""

    class mps:
        @staticmethod
        def is_available():
            return True

    class cudnn:
        deterministic = False
        benchmark = False

# ============================================================================
# Distributed training (not supported in MLX)
# ============================================================================

class distributed:
    """Mock distributed module (MLX doesn't support distributed)."""

    @staticmethod
    def is_available():
        return False

    @staticmethod
    def is_initialized():
        return False

    @staticmethod
    def init_process_group(*args, **kwargs):
        raise NotImplementedError("MLX does not support distributed training")

    @staticmethod
    def destroy_process_group():
        pass

    @staticmethod
    def barrier():
        pass

    @staticmethod
    def get_rank():
        return 0

    @staticmethod
    def get_world_size():
        return 1

# ============================================================================
# Random number generation
# ============================================================================

class Generator:
    """Random number generator."""

    def __init__(self, device=None):
        self._key = mx.random.key(0)

    def manual_seed(self, seed):
        """Set random seed."""
        self._key = mx.random.key(seed)
        mx.random.seed(seed)
        return self

def manual_seed(seed):
    """Set random seed."""
    mx.random.seed(seed)

# ============================================================================
# Utilities
# ============================================================================

def set_float32_matmul_precision(mode):
    """Set float32 matmul precision (no-op for MLX)."""
    pass

def compile(model, dynamic=False):
    """Mock compile - MLX auto-compiles via JIT."""
    return model

def inference_mode():
    """No-op decorator for inference mode."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

class no_grad:
    """No-op context manager and decorator for no gradient computation."""
    def __init__(self, func=None):
        self.func = func

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __call__(self, *args, **kwargs):
        if self.func is None:
            # Being used as @no_grad() with parentheses
            if len(args) == 1 and callable(args[0]) and not kwargs:
                # Return a new no_grad instance wrapping the function
                return no_grad(args[0])
            else:
                raise TypeError("no_grad() takes no arguments")
        else:
            # Being used to call the wrapped function
            return self.func(*args, **kwargs)

def enable_grad():
    """No-op context for enabling gradients."""
    class EnableGrad:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    return EnableGrad()

# ============================================================================
# NN utilities
# ============================================================================

class nn:
    """Mock torch.nn module."""
    Module = Module
    Linear = Linear
    Embedding = Embedding
    ModuleDict = ModuleDict
    ModuleList = ModuleList

    class utils:
        @staticmethod
        def clip_grad_norm_(parameters, max_norm, norm_type=2.0):
            """Clip gradient norm (simplified)."""
            # MLX gradients would be handled differently
            # This is a placeholder
            pass

    class init:
        @staticmethod
        def normal_(tensor, mean=0.0, std=1.0):
            """Initialize with normal distribution."""
            if isinstance(tensor, Tensor):
                tensor.data = mx.random.normal(tensor.data.shape, mean=mean, scale=std)
            return tensor

        @staticmethod
        def zeros_(tensor):
            """Initialize with zeros."""
            if isinstance(tensor, Tensor):
                tensor.data = mx.zeros(tensor.data.shape)
            return tensor

        @staticmethod
        def uniform_(tensor, a=0.0, b=1.0):
            """Initialize with uniform distribution."""
            if isinstance(tensor, Tensor):
                tensor.data = mx.random.uniform(a, b, tensor.data.shape)
            return tensor

# ============================================================================
# Export public API
# ============================================================================

# Make this module importable as torch
__all__ = [
    'Tensor', 'tensor', 'zeros', 'zeros_like', 'ones', 'ones_like',
    'empty', 'empty_like', 'arange', 'cat', 'stack', 'where',
    'tril', 'triu', 'topk', 'argmax', 'multinomial',
    'Module', 'Linear', 'Embedding', 'ModuleDict', 'ModuleList',
    'functional', 'F', 'nn', 'autocast', 'amp',
    'device', 'cuda', 'backends', 'distributed',
    'Generator', 'manual_seed', 'compile', 'inference_mode', 'no_grad',
    'float32', 'bfloat16', 'int32', 'int64', 'long',
]
