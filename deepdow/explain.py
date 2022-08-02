"""Collection of tools for explaining trained models."""

import torch


def gradient_wrt_input(
    model,
    target_weights,
    initial_guess,
    n_iter=100,
    mask=None,
    lr=1e-1,
    verbose=True,
    device=None,
    dtype=None,
):
    """Find input tensor such that the model produces an allocation close to the target one.

    Parameters
    ----------
    model : torch.Module
        Network that predicts weight allocation given feature tensor.

    target_weights : torch.Tensor
        Vector of targeted asset weights of shape `(n_assets,)`.

    initial_guess : torch.Tensor
        Initial feature tensor serving as the starting point for the optimization. The shape is
        `(n_channels, lookback, n_assets)` - the sample dimension is not included.

    n_iter : int
        Number of iterations of the gradients descent (or other) algorithm.

    mask : None or torch.Tensor
        If specified, then boolean ``torch.Tensor`` of the same shape as `initial_guess` than
        one can elementwise choose what parts of the inputs to optimize (True) and which
        keep the same as the initial guess (False).

    lr : float
        Learning rate for the optimizer.

    verbose : bool
        If True, then verbosity activated.

    dtype : None or torch.dtype
        Dtype to be used. If specified, casts all used tensors.

    device : None or torch.device
        Device to be used. If specified, casts all used tensors.

    Returns
    -------
    result : torch.Tensor
        Feature tensor of the same shape as `initial_guess` that is mapped by the network (hopefully)
        close to `target_weights`.

    hist : list
        List of losses per iteration.
    """
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32

    x = initial_guess.clone().to(device=device, dtype=dtype)
    x.requires_grad = True

    if mask is None:
        mask = torch.ones_like(x)

    elif torch.is_tensor(mask):
        if mask.shape != x.shape:
            raise ValueError("Inconsistent shape of the mask.")
    else:
        raise TypeError(
            "Incorrect type of the mask, either None or torch.Tensor."
        )

    # casting
    mask = mask.to(dtype=torch.bool, device=device)
    model.to(device=device, dtype=dtype)
    target_weights = target_weights.to(device=device, dtype=dtype)

    optimizer = torch.optim.Adam([x], lr=lr)
    model.train()

    hist = []
    for i in range(n_iter):
        if i % 50 == 0 and verbose:
            msg = (
                "{}-th iteration, loss: {:.4f}".format(i, hist[-1])
                if i != 0
                else "Starting optimization"
            )
            print(msg)

        loss_per_asset = (
            model((x * mask)[None, ...])[0] - target_weights
        ) ** 2
        loss = loss_per_asset.mean()
        hist.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if verbose:
        print("Optimization done, final loss: {:.4f}".format(hist[-1]))

    return x, hist
