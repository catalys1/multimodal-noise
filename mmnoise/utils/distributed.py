import torch


def is_distributed():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def world_size():
    if is_distributed():
        return torch.distributed.get_world_size()
    return 1


def rank():
    if is_distributed():
        return torch.distributed.get_rank()
    return 0


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[torch.distributed.get_rank()]
        return grad_out