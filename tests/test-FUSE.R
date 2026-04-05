library(testthat)
library(FUSE)
library(torch)

test_check("FUSE")

# Basic test for FUSE model
test_that("FUSE model can be created and trained", {
  model <- FUSE(input_dim = 2, hidden = c(16, 16))

  # Generate simple data
  X <- torch_randn(100, 2)

  # Fit model
  history <- model$fit(X, num_epochs = 2, quiet = TRUE)

  # Check that history is returned
  expect_type(history, "list")
  expect_length(history, 2)

  # Check inference
  scores <- model$inference(X, t = 0.5)
  expect_equal(scores$shape, c(100, 1))
})

# Test triplet sampling
test_that("triplet_sampling works", {
  triplets <- triplet_sampling(100, partitioned = TRUE)
  expect_type(triplets, "list")
  expect_named(triplets, c("S0", "S1", "S2"))
})