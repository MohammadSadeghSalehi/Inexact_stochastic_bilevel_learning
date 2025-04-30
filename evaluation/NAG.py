import torch
import numpy as np
from typing import Callable, Tuple, Union

# Keep the original nmAPG function for reference or comparison if needed,
# but the focus is on implementing NAG.
# def nmAPG(...): ... (your existing code)

import torch
import numpy as np
from typing import Callable, Tuple, Union

def NAG(
    x0: torch.Tensor,
    y: torch.Tensor,
    f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    nabla: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    max_iter: int = 200,
    L_init: float = 1.0,
    tol: float = 1e-4, # Changed tolerance to a slightly stricter value
    rho: float = 0.9, # Factor by which L is multiplied when backtracking (rho < 1, decreases step size)
    delta: float = 0.0001, # Parameter for line search condition
    line_search_max_iter: int = 50, # Added max iterations for inner line search
    verbose: bool = False,
    return_L: bool = False,
) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Nesterov Accelerated Gradient (NAG) method with backtracking line search and restart.

    Minimizes f(x, y) with respect to x using gradients nabla(x, y).
    Handles batch processing across the first dimension. Includes a restart
    mechanism if the line search fails to converge within line_search_max_iter
    or encounters numerical instability (Inf L).

    Args:
        x0: Initial guess for x (Batch_size, ...).
        y: Data tensor (Batch_size, ...).
        f: Callable objective function f(x, y) returning scalar energy per batch item.
           Expected signature: (torch.Tensor, torch.Tensor) -> torch.Tensor (shape Batch_size)
        nabla: Callable gradient function nabla(x, y) returning gradient w.r.t x.
               Expected signature: (torch.Tensor, torch.Tensor) -> torch.Tensor (shape Batch_size, ...)
        max_iter: Maximum number of iterations.
        L_init: Initial estimate for the Lipschitz constant L.
        tol: Relative tolerance for convergence ||x_k - x_{k-1}|| / ||x_k||.
        rho: Factor by which L is multiplied when backtracking (rho < 1, decreases step size).
        delta: Parameter for the Armijo-like line search condition.
        line_search_max_iter: Maximum iterations for the inner backtracking line search loop.
        verbose: Whether to print progress.
        return_L: If True, return the final L tensor instead of iterations count.

    Returns:
        If return_L is False: Tuple containing the final solution x and the number of iterations.
        If return_L is True: Tuple containing the final solution x and the final L tensor.
    """
    x = x0.clone()
    x_old = x0.clone()
    y_extrapolated = x0.clone()

    t = 1.0
    t_old = 0.0

    L = torch.full((x.shape[0], 1, 1, 1), L_init, dtype=x.dtype, device=x.device)

    res = (tol + 1) * torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
    idx = torch.arange(0, x.shape[0], device=x.device)

    grad_y = torch.zeros_like(x)
    energy_y = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

    iterations = max_iter

    if torch.isnan(x).any():
        if verbose:
            print("NaN detected in initial x, returning Inf.")
        return torch.full_like(x, float('inf')), iterations if not return_L else L

    for i in range(max_iter):
        # Store x_k (x at the start of iteration i)
        current_x = x.clone()

        # --- Extrapolation Step (Calculate y_k) ---
        # y_k = x_k + (t_{k-1}-1)/t_k * (x_k - x_{k-1})
        # If a restart happened in the previous iteration (i.e., t_old=1, t=1),
        # then for the current iteration (i), t_old becomes 1 and t becomes 1 (from the end of iter i-1).
        # The extrapolation becomes y_i = x_i + (1-1)/1 * (x_i - x_{i-1}) = x_i.
        # If no restart, t and t_old follow the normal sequence.
        y_extrapolated[idx] = x[idx] + (t_old - 1) / t * (x[idx] - x_old[idx])

        # --- Gradient Calculation at y_k ---
        try:
            energy_y[idx] = f(y_extrapolated[idx], y[idx])
            grad_y[idx] = nabla(y_extrapolated[idx], y[idx])
        except Exception as e:
             print(f"Error in f() or nabla() at iter {i} during extrapolation for idx {idx}: {e}")
             iterations = i
             if return_L: return current_x, L # Return L corresponding to current_x
             else: return current_x, iterations

        # --- Gradient Step from y_k with Backtracking Line Search ---

        # Start search with L/rho (larger step size, smaller L_search) from the previous L
        L_search = L[idx].clone() / rho

        potential_x = torch.zeros_like(x[idx])
        energy_potential_x = torch.zeros_like(energy_y[idx])

        # Flag to signal if line search failed to converge for any active item in this iter
        restart_triggered_in_iter = False

        for ii in range(line_search_max_iter):
            # Proposed step: x_{k+1} = y_k - (1/L) * grad_y_k
            potential_x = y_extrapolated[idx] - grad_y[idx] / L_search

            try:
                energy_potential_x = f(potential_x, y[idx])
            except Exception as e:
                print(f"Error in f(potential_x) inner loop {ii} at iter {i}: {e}")
                # Treat function evaluation error as line search failure and trigger restart
                restart_triggered_in_iter = True
                break # Break inner loop

            # Armijo-like condition: f(potential_x) <= f(y_k) - delta ||potential_x - y_k||^2
            dx = potential_x - y_extrapolated[idx]
            condition = energy_potential_x <= (energy_y[idx] - delta * (dx * dx).sum((1, 2, 3))).view(-1)

            if torch.all(condition):
                break # Line search successful for all active batch items
            else:
                 failed_mask = ~condition
                 failed_indices = idx[failed_mask.nonzero().view(-1)]

                 # Increase L_search for failed items (decrease step size)
                 L_search[failed_mask] = L_search[failed_mask] / rho

                 # Check for numerical instability (L exploding)
                 if torch.isinf(L_search[failed_mask]).any() or (L_search[failed_mask] > 1e10).any():
                     if verbose:
                        print(f"Warning: L_search became Inf or too large in line search at iter {i}, inner iter {ii}. Triggering restart.")
                     # If L_search explodes, trigger restart for the batch
                     restart_triggered_in_iter = True
                     # Use the last potential_x found before L_search exploded
                     break # Break inner loop


        else: # No break occurred in inner loop -> Line search max iter reached for some/all active items
            if verbose: print(f"Warning: Line search max iter ({line_search_max_iter}) reached at iter {i}")
            # --- Trigger Restart ---
            # If the inner loop completed without breaking, it means the line search
            # condition wasn't met for all active items within max_linesearch_iter.
            # This triggers a restart for ALL currently active items in the next iteration.
            restart_triggered_in_iter = True
            if verbose: print(f"  Restart triggered due to line search failure at iter {i}.")

            # Use the potential_x found with the final L_search in the failed line search


        # Use the L found by the line search for the next iteration's starting L_init
        # If line search failed/restarted, L_search will be the large value from the end of line search
        L[idx] = L_search.clone() # Update the main L tracker for active items

        # --- Update x ---
        x[idx] = potential_x # Update active elements of x

        # --- Calculate Residual ---
        res_val = torch.zeros_like(res[idx])
        try:
            dx_res = x[idx] - current_x[idx]
            norm_dx_res = torch.norm(dx_res.flatten(1), p=2, dim=1)
            norm_x_k = torch.norm(current_x[idx].flatten(1), p=2, dim=1)

            safe_norm_x_k = torch.where(norm_x_k == 0, torch.tensor(1e-8, device=x.device, dtype=x.dtype), norm_x_k)
            res_val = norm_dx_res / safe_norm_x_k
            res[idx] = res_val
        except Exception as e:
             print(f"Error calculating residual at iter {i} for idx {idx}: {e}")
             res[idx] = tol + 1


        # --- Update Active Indices & Check Convergence ---
        condition = res[idx] > tol
        new_idx = idx[condition.nonzero().view(-1)]

        if verbose and new_idx.nelement() < idx.nelement():
             converged_count = idx.nelement() - new_idx.nelement()
             if new_idx.nelement() > 0:
                 max_res_active = torch.max(res[new_idx])
                 print(f"Iter {i+1}: {converged_count} converged ({new_idx.nelement()} active). Max res: {max_res_active:.6e}")
             else:
                  print(f"Iter {i+1}: {converged_count} converged ({new_idx.nelement()} active). All converged.")


        if new_idx.nelement() == 0:
            if verbose:
                if idx.nelement() > 0:
                    print(f"All elements converged in iter {i+1} (max residual {torch.max(res):.2e} <= tol {tol:.2e})")
                else:
                    # This case happens if idx was empty at the start of the iteration, which shouldn't normally occur
                    print(f"Convergence reached in iter {i+1} (all residuals <= tol {tol:.2e})")
            iterations = i + 1
            break # Exit main loop successfully

        idx = new_idx # Update active indices for the next iteration

        # --- Update Time Step ---
        # Apply restart: If restart_triggered_in_iter is True, we override the normal t update
        # to set t and t_old for the *next* iteration to 1.0.
        if restart_triggered_in_iter:
             t_old = 1.0
             t = 1.0
             if verbose: print(f"  Resetting t and t_old to 1.0 for restart at end of iter {i}.")
        else:
            # Normal t update
            t_old = t
            t = (1.0 + np.sqrt(1.0 + 4.0 * t_old**2)) / 2.0


        # --- Prepare for Next Iteration ---
        x_old = current_x # x_old for iter i+1 is x_i (current_x)

    # --- End of Loop ---

    if i == max_iter - 1 and idx.nelement() > 0:
       if verbose:
           max_res_val = torch.max(res[idx]) if idx.nelement() > 0 else torch.tensor(0.0)
           print(f"Max iter ({max_iter}) reached, max residual among active: {max_res_val:.6f}")
       iterations = max_iter


    if return_L:
        return x, L
    else:
        return x, iterations

# --- Reconstruct function using NAG ---

def reconstruct_NAG(
    y,
    physics,
    data_fidelity,
    regularizer,
    lamda,
    step_size, # This will be used to set L_init
    max_iter,
    tol,
    x_init=None,
    detach_grads=True, # Controls whether gradients are detached inside f and nabla
    verbose=False,
    return_L=False, # Option to return final L
):
    """
    Reconstructs an image using NAG optimization.

    Args:
        y: Measured data (Batch_size, ...).
        physics: Physics operator (e.g., forward and adjoint).
        data_fidelity: Data fidelity term callable (e.g., L2).
        regularizer: Regularization term callable (e.g., TV).
        lamda: Regularization parameter.
        step_size: Initial step size (used as 1/L_init).
        max_iter: Maximum optimization iterations.
        tol: Convergence tolerance.
        x_init: Optional initial guess for x.
        detach_grads: Whether to detach gradients during energy/grad calculation
                      (often useful to prevent graph building issues if f/nabla
                       are used elsewhere, but might be False for complex custom losses).
        verbose: Whether to print progress.
        return_L: If True, return the final L tensor from NAG.

    Returns:
        If return_L is False: Tuple containing the reconstructed image(s) and the number of iterations.
        If return_L is True: Tuple containing the reconstructed image(s) and the final L tensor.
    """
    if x_init is not None:
        # User-defined initialization or warm start
        # Ensure x_init is a tensor and on the correct device/dtype
        x = torch.as_tensor(x_init, device=y.device, dtype=y.dtype).clone().detach()
    else:
        # Ensure A_dagger returns tensor on correct device and dtype
        # Detach the initial guess as it's not part of the optimization graph
        x = physics.A_dagger(y).to(y.device, y.dtype).detach()

    # Ensure lambda is a tensor or scalar compatible with the function results
    if isinstance(lamda, (int, float)):
        effective_lamda = torch.tensor(lamda, device=y.device, dtype=y.dtype)
    elif isinstance(lamda, torch.Tensor):
        effective_lamda = lamda.to(y.device, y.dtype)
    else:
        raise TypeError("lamda must be a number or a torch.Tensor")

    # Define energy function matching the signature expected by NAG: f(val, y_in)
    # This computes the objective function value.
    def energy(val, y_in):
        # Detach if necessary *before* passing to data_fidelity and regularizer
        val_maybe_detached = val.detach() if detach_grads else val

        # Calculate data fidelity term (assumed to return shape Batch_size)
        df_term = data_fidelity(val_maybe_detached, y_in, physics)

        # Calculate regularization term (assumed to return shape Batch_size)
        reg_term = regularizer.g(val_maybe_detached)

        # Combine terms, ensure consistent shapes for addition
        # Assuming df_term and reg_term output shape (B,) or scalar that broadcasts
        fun = df_term + effective_lamda * reg_term

        # Ensure output shape is (B,)
        return fun.reshape(-1)

    # Define gradient function matching the signature expected by NAG: nabla(val, y_in)
    # This computes the gradient of the objective function w.r.t. val.
    def energy_grad(val, y_in):
         # Ensure gradients can be computed with respect to val
         val_for_grad = val.detach().requires_grad_(True) if not val.requires_grad else val

         # Calculate gradient of data fidelity w.r.t. val
         grad_df = data_fidelity.grad(val_for_grad, y_in, physics)

         # Calculate gradient of regularizer w.r.t. val
         grad_reg = regularizer.grad(val_for_grad)

         # Combine gradients
         grad = grad_df + effective_lamda * grad_reg

         # Detach the final gradient if required
         if detach_grads:
             grad = grad.detach()

         return grad

    # Call the NAG optimizer
    # The step_size is 1/L, so L_init is 1/step_size
    L_init_val = 1.0 / step_size if step_size > 0 else 1.0
    if verbose and step_size <= 0:
        print(f"Warning: step_size ({step_size}) is non-positive. Using L_init=1.0.")

    result = NAG(
        x0=x,
        y=y,
        max_iter=max_iter,
        f=energy,
        nabla=energy_grad,
        L_init=L_init_val,
        tol=tol,
        verbose=verbose,
        return_L=return_L,
    )

    # Unpack result based on return_L
    if return_L:
        rec, L_final = result
        # If NaNs are detected in the result, replace with Inf
        if torch.isnan(rec).any():
             rec = torch.full_like(rec, float('inf'))
        return rec, L_final
    else:
        rec, iterations = result
        # If NaNs are detected in the result, replace with Inf
        if torch.isnan(rec).any():
             rec = torch.full_like(rec, float('inf'))
             print("Warning: NaN detected in reconstruction, returning Inf.")
        return rec, iterations
    