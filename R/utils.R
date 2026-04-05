#' @import ggplot2
#' @import umap
#' @import torch
#' @importFrom stats quantile
#' @importFrom grDevices colorRampPalette
#' @importFrom RColorBrewer brewer.pal
NULL

#' Plot Homotopy Contours
#'
#' Plot contour maps of the homotopy centrality function H(x, t) over a 2D grid for multiple t-values.
#'
#' @param hom A model implementing `.inference(x, t)` returning a scalar score.
#' @param X Input data matrix (n x d).
#' @param t_values Vector of t-values for interpolation.
#' @param title Figure title.
#' @param x1lim Manual x1 limits.
#' @param x2lim Manual x2 limits.
#' @param grid_n Number of grid points per axis.
#' @param cmap Colormap name.
#' @param point_size Point size.
#' @param point_edge_alpha Edge alpha.
#' @param point_edge_width Edge width.
#' @param n_levels_filled Number of filled contour levels.
#' @param n_levels_lines Number of contour lines.
#' @param device Torch device.
#' @return A ggplot object or list of plots.
#' @export
plot_homotopy_contours <- function(hom, X, t_values = c(0.0, 0.25, 0.5, 0.75, 1.0),
                                   title = "Homotopy Centrality", x1lim = NULL, x2lim = NULL,
                                   grid_n = 200, cmap = "viridis", point_size = 1,
                                   point_edge_alpha = 0.35, point_edge_width = 0.3,
                                   n_levels_filled = 20, n_levels_lines = 12,
                                   device = NULL) {
  if (ncol(X) < 2) {
    stop("plot_homotopy_contours: requires at least 2D input.")
  }

  dev <- if (is.null(device)) hom$parameters[[1]]$device else torch_device(device)
  dtype <- hom$parameters[[1]]$dtype

  x <- X[, 1:2]
  if (is.null(x1lim)) {
    m <- min(x[, 1])
    M <- max(x[, 1])
    pad <- 0.1 * (M - m + 1e-9)
    x1lim <- c(m - pad, M + pad)
  }
  if (is.null(x2lim)) {
    m <- min(x[, 2])
    M <- max(x[, 2])
    pad <- 0.1 * (M - m + 1e-9)
    x2lim <- c(m - pad, M + pad)
  }

  xs <- seq(x1lim[1], x1lim[2], length.out = grid_n)
  ys <- seq(x2lim[1], x2lim[2], length.out = grid_n)
  grid <- expand.grid(x = xs, y = ys)
  grid2 <- as.matrix(grid)

  input_dim <- if (!is.null(hom$input_dim)) hom$input_dim else ncol(grid2)
  if (input_dim > 2) {
    pad <- matrix(0, nrow(grid2), input_dim - 2)
    grid_full <- cbind(grid2, pad)
  } else {
    grid_full <- grid2
  }

  grid_torch <- torch_tensor(grid_full, device = dev, dtype = dtype)

  plots <- list()
  for (t in t_values) {
    Z <- hom$inference(grid_torch, t)
    Z <- as.numeric(Z$detach()$cpu()$reshape(c(grid_n, grid_n)))

    grid$z <- Z

    p <- ggplot(grid, aes(x = x, y = y, z = z)) +
      geom_contour_filled(bins = n_levels_filled) +
      geom_contour(color = "black", alpha = 0.35, bins = n_levels_lines) +
      geom_point(data = data.frame(x = x[, 1], y = x[, 2]),
                 aes(x = x, y = y), size = point_size, shape = 21,
                 fill = NA, color = "white", alpha = point_edge_alpha, stroke = point_edge_width) +
      scale_fill_viridis_c(option = cmap) +
      labs(title = sprintf("t = %.2f", t), x = "x₁", y = "x₂") +
      theme_minimal() +
      coord_fixed()

    plots[[length(plots) + 1]] <- p
  }

  if (length(plots) == 1) {
    plots[[1]] + ggtitle(title)
  } else {
    # For multiple plots, use patchwork or return list
    plots
  }
}

#' Plot Homotopy UMAP
#'
#' UMAP visualization of labels and homotopy centrality.
#'
#' @param model Model implementing `.inference(x, t)`.
#' @param X Dataset matrix.
#' @param y Class labels.
#' @param t_values Interpolation parameters.
#' @param device Torch device.
#' @param umap_kwargs UMAP parameters.
#' @param batch_size Batch size for evaluation.
#' @param depth_cmap Colormap for depth.
#' @param label_cmap Colormap for labels.
#' @param point_size Point size.
#' @param point_alpha Point alpha.
#' @param normalize_depth Whether to normalize depth.
#' @param title Figure title.
#' @return A ggplot object or list of plots.
#' @export
plot_homotopy_umap <- function(model, X, y = NULL, t_values = c(0.0, 0.33, 0.67, 1.0),
                               device = NULL, umap_kwargs = NULL, batch_size = 8192,
                               depth_cmap = "viridis", label_cmap = NULL, point_size = 1,
                               point_alpha = 0.7, normalize_depth = FALSE,
                               title = "UMAP — Labels + Homotopy Centrality per t") {
  if (is.matrix(X)) {
    X_np <- X
    X_t <- torch_tensor(X_np, dtype = torch_float())
  } else {
    X_t <- X$float()
    X_np <- as.matrix(X_t$detach()$cpu())
  }

  dev <- if (!is.null(device)) torch_device(device) else model$parameters[[1]]$device
  X_t <- X_t$to(dev)
  N <- X_t$shape[1]

  if (is.null(umap_kwargs)) {
    umap_kwargs <- list(n_components = 2, random_state = 42)
  }

  emb <- umap(X_np, config = umap_kwargs)$layout

  depths_per_t <- list()
  for (t in t_values) {
    vals <- c()
    for (i in seq(1, N, by = batch_size)) {
      end_idx <- min(i + batch_size - 1, N)
      xb <- X_t[i:end_idx]
      hb <- as.numeric(model$inference(xb, t)$reshape(-1)$detach()$cpu())
      vals <- c(vals, hb)
    }
    depths_per_t[[length(depths_per_t) + 1]] <- vals
  }

  if (normalize_depth && length(depths_per_t) > 0) {
    allv <- unlist(depths_per_t)
    lo <- min(allv)
    hi <- max(allv)
    rng <- max(hi - lo, 1e-8)
    depths_per_t <- lapply(depths_per_t, function(v) (v - lo) / rng)
  }

  plots <- list()

  # Label panel
  if (!is.null(y)) {
    y_np <- if (torch_is_tensor(y)) as.numeric(y$detach()$cpu()) else as.numeric(y)
    df <- data.frame(UMAP1 = emb[, 1], UMAP2 = emb[, 2], label = as.factor(y_np))
    p_label <- ggplot(df, aes(x = UMAP1, y = UMAP2, color = label)) +
      geom_point(size = point_size, alpha = point_alpha) +
      labs(title = "Labels", x = "UMAP 1", y = "UMAP 2") +
      theme_minimal()
    plots[[length(plots) + 1]] <- p_label
  } else {
    df <- data.frame(UMAP1 = emb[, 1], UMAP2 = emb[, 2])
    p_label <- ggplot(df, aes(x = UMAP1, y = UMAP2)) +
      geom_point(size = point_size, alpha = point_alpha, color = "black") +
      labs(title = "No labels", x = "UMAP 1", y = "UMAP 2") +
      theme_minimal()
    plots[[length(plots) + 1]] <- p_label
  }

  # Depth panels
  for (i in seq_along(t_values)) {
    t <- t_values[i]
    depths <- depths_per_t[[i]]
    df <- data.frame(UMAP1 = emb[, 1], UMAP2 = emb[, 2], depth = depths)
    p <- ggplot(df, aes(x = UMAP1, y = UMAP2, color = depth)) +
      geom_point(size = point_size, alpha = point_alpha) +
      scale_color_viridis_c(option = depth_cmap) +
      labs(title = sprintf("t = %.2f", t), x = "UMAP 1", y = if (i == 1) "UMAP 2" else "") +
      theme_minimal()
    plots[[length(plots) + 1]] <- p
  }

  plots
}