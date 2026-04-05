#' @title Fused Unified Centrality Score Estimation (FUSE)
#' @description A lightweight implementation of FUSE using torch.
#' @import torch
#' @importFrom stats quantile
#' @importFrom utils setTxtProgressBar txtProgressBar
#' @export
NULL

#' Triplet Sampling for Global/Local Anchor-Based Depth Learning
#'
#' Sample three index sets (S0, S1, S2) used in global/local anchor-based depth learning.
#'
#' @param n Total number of data points. Must satisfy n >= 3 for non-empty triplets.
#' @param partitioned Logical. If TRUE, S0 is sampled first, then remaining indices are split deterministically.
#' @param device Torch device.
#' @param seed Random seed for reproducibility.
#' @return A list with S0, S1, S2 as torch tensors.
#' @export
triplet_sampling <- function(n, partitioned = TRUE, device = torch_device("cpu"), seed = NULL) {
  if (!is.null(seed)) {
    torch_manual_seed(seed)
  }

  m <- n %/% 3
  if (m == 0) {
    empty <- torch_empty(0, dtype = torch_long(), device = device)
    return(list(S0 = empty, S1 = empty, S2 = empty))
  }

  # Sample S0
  perm <- torch_randperm(n, device = device)
  S0 <- perm[1:m]
  R <- perm[(m + 1):n]

  if (partitioned) {
    if (R$numel() < 2 * m) {
      empty <- torch_empty(0, dtype = torch_long(), device = device)
      return(list(S0 = S0, S1 = empty, S2 = empty))
    }
    S1 <- R[1:m]
    S2 <- R[(m + 1):(2 * m)]
    return(list(S0 = S0, S1 = S1, S2 = S2))
  } else {
    if (R$numel() < m) {
      empty <- torch_empty(0, dtype = torch_long(), device = device)
      return(list(S0 = S0, S1 = empty, S2 = empty))
    }
    perm_R1 <- torch_randperm(R$numel(), device = device)
    S1 <- R[perm_R1[1:m]]

    perm_R2 <- torch_randperm(R$numel(), device = device)
    S2 <- R[perm_R2[1:m]]

    return(list(S0 = S0, S1 = S1, S2 = S2))
  }
}

#' Multi-Layer Perceptron (MLP)
#'
#' A lightweight MLP with configurable hidden layers.
#'
#' @param in_dim Input dimension.
#' @param hidden Vector of hidden layer sizes.
#' @param out_dim Output dimension.
#' @param dropout Dropout probability.
#' @return An nn_module.
MLP <- nn_module(
  "MLP",
  initialize = function(in_dim, hidden = c(32, 32), out_dim, dropout = 0.0) {
    layers <- list()
    last <- in_dim
    for (h in hidden) {
      layers <- c(layers, list(nn_linear(last, h), nn_gelu()))
      if (dropout > 0) {
        layers <- c(layers, list(nn_dropout(dropout)))
      }
      last <- h
    }
    layers <- c(layers, list(nn_linear(last, out_dim)))
    self$net <- nn_sequential(!!!layers)
  },
  forward = function(x) {
    self$net(x)
  }
)

#' FUSE Model
#'
#' Fused Unified centrality Score Estimation.
#'
#' @param input_dim Input feature dimension.
#' @param hidden Hidden dimensions for encoder.
#' @param dropout Dropout rate.
#' @param temperature_global Temperature for global depth.
#' @return An nn_module.
#' @export
FUSE <- nn_module(
  "FUSE",
  initialize = function(input_dim, hidden = c(32, 32), dropout = 0.0, temperature_global = 1.0) {
    self$encoder <- MLP(input_dim, hidden, if (length(hidden) > 0) hidden[length(hidden)] else input_dim, dropout)
    latent_dim <- if (length(hidden) > 0) hidden[length(hidden)] else input_dim

    self$phi_head <- nn_linear(latent_dim, 1)
    self$d_head <- nn_linear(latent_dim, 1)

    # Initialization
    nn_init_xavier_uniform_(self$phi_head$weight, gain = 0.7)
    nn_init_zeros_(self$phi_head$bias)
    nn_init_xavier_uniform_(self$d_head$weight, gain = 0.7)
    nn_init_zeros_(self$d_head$bias)

    self$Tg <- temperature_global

    # Buffers for normalization
    self$register_buffer("phi_min", torch_tensor(0.0))
    self$register_buffer("phi_max", torch_tensor(1.0))
  },
  forward = function(x) {
    h <- self$encoder(x)
    phi <- self$phi_head(h)$squeeze(-1)
    d <- self$d_head(h)$squeeze(-1)
    list(d = d, phi = phi)
  },
  inference = function(x, t = NULL) {
    res <- self$forward(x)
    d <- res$d
    phi <- res$phi

    g <- torch_sigmoid(d)$unsqueeze(1)

    range_ <- (self$phi_max - self$phi_min)$clamp(min = 1e-8)
    l <- ((phi - self$phi_min) / range_)$clamp(0.0, 1.0)$unsqueeze(1)

    if (is.null(t)) {
      return(list(g = g, l = l))
    }

    if (!torch_is_tensor(t)) {
      t <- torch_tensor(t, device = x$device, dtype = x$dtype)
    }

    if (t$ndim == 0) {
      t <- t$expand(g$size(1))
    } else if (t$ndim == 2 && t$shape[2] == 1) {
      t <- t$squeeze(1)
    } else if (t$ndim != 1) {
      stop("t must be scalar, (B,), or (B,1)")
    }

    f_fuse <- ((1 - t)$unsqueeze(1) * g) + (t$unsqueeze(1) * l)
    f_fuse
  },
  fit = function(X, dissimilarity_matrix = NULL, dissimilarity_fn = NULL,
                 num_epochs = 30, batch_size = 256,
                 anchors_per_pair_global = 64, partitioned = TRUE, allPairs = FALSE,
                 dsm_sigma = 1.0, dsm_resampling = 8,
                 weight_decay = 0.0, optimizer = NULL, lr = 1e-3,
                 device = NULL, grad_clip = NULL, quiet = FALSE, seed = NULL) {

    if (!is.null(seed)) {
      torch_manual_seed(seed)
    }

    device <- if (is.null(device)) self$parameters[[1]]$device else device
    model_dtype <- self$parameters[[1]]$dtype
    X <- X$to(device)

    n <- X$shape[1]
    d <- X$shape[2]

    use_matrix <- !is.null(dissimilarity_matrix)
    if (!use_matrix && is.null(dissimilarity_fn)) {
      stop("Provide either dissimilarity_matrix or dissimilarity_fn.")
    }

    if (use_matrix) {
      dissimilarity_matrix <- dissimilarity_matrix$to(device)
      if (!identical(dissimilarity_matrix$shape, c(n, n))) {
        stop("dissimilarity_matrix must be shape (n, n).")
      }
    }

    sigma <- as.numeric(dsm_sigma)
    inv_sigma2 <- 1.0 / (sigma * sigma)

    if (is.null(optimizer)) {
      optimizer <- optim_adam(self$parameters, lr = lr, weight_decay = weight_decay)
    }

    self$train()
    history <- list()

    epoch_iter <- 1:num_epochs
    if (!quiet) {
      pb <- txtProgressBar(min = 0, max = num_epochs, style = 3)
    }

    for (epoch in epoch_iter) {
      # Global head
      triplets <- triplet_sampling(n, partitioned = partitioned, device = device, seed = seed)
      S0 <- triplets$S0
      S1 <- triplets$S1
      S2 <- triplets$S2

      if (allPairs) {
        if (S1$numel() == 0 || S2$numel() == 0) {
          p1_all <- torch_empty(0, dtype = torch_long(), device = device)
          p2_all <- torch_empty(0, dtype = torch_long(), device = device)
        } else {
          p1_all <- S1$repeat_interleave(S2$numel())
          p2_all <- S2$repeat(S1$numel())
          mask <- p1_all != p2_all
          p1_all <- p1_all[mask]
          p2_all <- p2_all[mask]
        }
      } else {
        L <- min(S1$numel(), S2$numel())
        if (L == 0) {
          p1_all <- torch_empty(0, dtype = torch_long(), device = device)
          p2_all <- torch_empty(0, dtype = torch_long(), device = device)
        } else {
          p1_all <- S1[1:L]
          p2_all <- S2[1:L]
          mask <- p1_all != p2_all
          p1_all <- p1_all[mask]
          p2_all <- p2_all[mask]
        }
      }

      total_pairs <- p1_all$numel()
      if (total_pairs == 0) {
        stop("No valid pairs generated. Dataset too small.")
      }

      steps <- ceiling(total_pairs / batch_size)
      ep_loss_g <- 0.0
      ep_count_g <- 0

      for (step in 1:steps) {
        sl <- (step - 1) * batch_size + 1
        sr <- min(total_pairs, sl + batch_size - 1)
        p1_idx <- p1_all[sl:sr]
        p2_idx <- p2_all[sl:sr]
        B <- p1_idx$numel()
        k0 <- S0$numel()

        if (use_matrix) {
          D1_full <- dissimilarity_matrix[S0]$index_select(2, p1_idx)
          D2_full <- dissimilarity_matrix[S0]$index_select(2, p2_idx)
        } else {
          A_mat <- X[S0]
          P1 <- X[p1_idx]
          P2 <- X[p2_idx]

          D1_full <- torch_zeros(c(k0, B), dtype = X$dtype, device = device)
          D2_full <- torch_zeros(c(k0, B), dtype = X$dtype, device = device)

          for (ai in 1:k0) {
            a <- A_mat[ai]
            for (bi in 1:B) {
              D1_full[ai, bi] <- dissimilarity_fn(a, P1[bi])
              D2_full[ai, bi] <- dissimilarity_fn(a, P2[bi])
            }
          }
        }

        A <- min(anchors_per_pair_global, D1_full$shape[1])
        if (A <= 0) {
          stop("anchors_per_pair_global must be >= 1.")
        }

        if (A < D1_full$shape[1]) {
          scores <- torch_rand(c(k0, B), device = device)
          anc_idx <- scores$topk(A, dim = 1, largest = TRUE, sorted = FALSE)$indices
          D1 <- D1_full$gather(1, anc_idx)
          D2 <- D2_full$gather(1, anc_idx)
        } else {
          D1 <- D1_full
          D2 <- D2_full
        }

        valid <- D1 != D2
        wins <- (D1 < D2) & valid

        wins_sum <- wins$float()$sum(dim = 1)
        valid_cnt <- valid$sum(dim = 1)
        pi_hat <- torch_where(valid_cnt > 0, wins_sum / valid_cnt$clamp(min = 1), torch_full_like(wins_sum, 0.5))

        g1 <- self$forward(X[p1_idx])$d
        g2 <- self$forward(X[p2_idx])$d
        logits <- (g1 - g2) / self$Tg

        loss_global <- nnf_binary_cross_entropy_with_logits(logits, pi_hat)

        optimizer$zero_grad()
        loss_global$backward()

        if (!is.null(grad_clip)) {
          nn_utils_clip_grad_norm_(self$parameters, grad_clip)
        }

        optimizer$step()

        ep_loss_g <- ep_loss_g + as.numeric(loss_global$detach()$cpu())
        ep_count_g <- ep_count_g + 1
      }

      avg_g <- ep_loss_g / max(1, ep_count_g)

      # Local head
      L_gl_sum <- 0.0

      if (dsm_resampling > 0) {
        num_chunks <- ceiling(n / batch_size)

        for (rep in 1:dsm_resampling) {
          L_rep_sum <- 0.0
          count_rep <- 0

          for (kch in 1:num_chunks) {
            sl <- (kch - 1) * batch_size + 1
            sr <- min(n, sl + batch_size)
            idx <- sl:sr

            x_clean <- X[idx]$detach()
            eps <- torch_randn_like(x_clean, dtype = model_dtype) * sigma
            x_tilde <- (x_clean + eps)$clone()$detach()$requires_grad_(TRUE)

            l_pred <- self$forward(x_tilde)$phi
            grad_l <- torch_autograd_grad(l_pred$sum(), x_tilde, create_graph = TRUE)[[1]]

            target <- -eps * inv_sigma2
            L_gl <- nnf_mse_loss(grad_l, target)

            optimizer$zero_grad()
            L_gl$backward()

            if (!is.null(grad_clip)) {
              nn_utils_clip_grad_norm_(self$parameters, grad_clip)
            }

            optimizer$step()

            L_rep_sum <- L_rep_sum + as.numeric(L_gl$detach()$cpu())
            count_rep <- count_rep + 1
          }

          L_gl_sum <- L_gl_sum + (L_rep_sum / max(1, count_rep))
        }
      }

      avg_gradl <- if (dsm_resampling > 0) L_gl_sum / dsm_resampling else 0.0

      history[[epoch]] <- list(
        loss_global = avg_g,
        loss_gradl = avg_gradl,
        loss_total = avg_g + avg_gradl
      )

      if (!quiet) {
        setTxtProgressBar(pb, epoch)
      }
    }

    if (!quiet) {
      close(pb)
    }

    torch_no_grad({
      l_train <- self$forward(X)$phi
      lo <- quantile(as.numeric(l_train$cpu()), 0.01)
      hi <- quantile(as.numeric(l_train$cpu()), 0.99)
      self$phi_min$copy_(torch_tensor(lo))
      self$phi_max$copy_(torch_tensor(hi))
    })
    history
  }
)