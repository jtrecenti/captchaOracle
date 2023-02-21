#' Captcha model using initial model
#'
#' This is the same as the original model, but including an initial
#' model to improve using new data. It only uses the input data, so
#' we do not need to adapt anything from the original function here.
#'
#' @param base_model base model of class `luz_module_fitted`. Defaults to
#'   `NULL`, which indicates no initialization of the model parameters.
#' @param input_dim input image dimensions
#' @param output_ndigits response variable number of digits
#' @param output_vocab_size integer indicating size of the vocabulary
#' @param vocab vocabulary elements
#' @param transform function to transform input data
#' @param dropout vector of two elements indicating dropout hyperparameter
#' @param dense_units integer indicating number of units used in the
#'   dense layer applied after the convolutional layers
#'
#' @importFrom torch nn_module
#'
#' @return object of classes `ORACLE-CNN` and `nn_module`. It works
#'   as a predictive function and as the input to a luz fitting workflow.
#'
#' @export
net_captcha_oracle <- torch::nn_module(

  "ORACLE-CNN",

  initialize = function(base_model = NULL,
                        input_dim,
                        output_ndigits,
                        output_vocab_size,
                        vocab,
                        transform,
                        dropout = c(.25, .25),
                        dense_units = 400) {

    # in_channels, out_channels, kernel_size, stride = 1, padding = 0
    self$batchnorm0 <- torch::nn_batch_norm2d(3)
    self$conv1 <- torch::nn_conv2d(3, 32, 3)
    self$batchnorm1 <- torch::nn_batch_norm2d(32)
    self$conv2 <- torch::nn_conv2d(32, 64, 3)
    self$batchnorm2 <- torch::nn_batch_norm2d(64)
    self$conv3 <- torch::nn_conv2d(64, 64, 3)
    self$batchnorm3 <- torch::nn_batch_norm2d(64)
    self$dropout1 <- torch::nn_dropout2d(dropout[1])
    self$dropout2 <- torch::nn_dropout2d(dropout[2])

    self$fc1 <- torch::nn_linear(
      # must be the same as last convnet
      in_features = prod(calc_dim_conv(input_dim)) * 64,
      out_features = dense_units
    )
    self$batchnorm_dense <- torch::nn_batch_norm1d(dense_units)
    self$fc2 <- torch::nn_linear(
      in_features = dense_units,
      out_features = output_vocab_size * output_ndigits
    )
    self$output_vocab_size <- output_vocab_size
    self$input_dim <- input_dim
    self$output_ndigits <- output_ndigits
    self$vocab <- vocab
    self$transform <- transform

    if (!is.null(base_model)) {

      stopifnot(class(base_model) == "luz_module_fitted")

      operations <- c(
        "conv1", "conv2", "conv3", "fc1", "fc2",
        "batchnorm0", "batchnorm1", "batchnorm2", "batchnorm3"
      )
      for (m in operations) {
        torch::with_no_grad(self[[m]]$weight$copy_(base_model$model[[m]]$weight))
        torch::with_no_grad(self[[m]]$bias$copy_(base_model$model[[m]]$bias))
      }
    }

  },
  forward = function(x) {

    # browser()
    out <- x |>
      # normalize
      self$batchnorm0() |>
      # layer 1
      self$conv1() |>
      torch::nnf_relu() |>
      torch::nnf_max_pool2d(2) |>
      self$batchnorm1() |>

      # layer 2
      self$conv2() |>
      torch::nnf_relu() |>
      torch::nnf_max_pool2d(2) |>
      self$batchnorm2() |>

      # layer 3
      self$conv3() |>
      torch::nnf_relu() |>
      torch::nnf_max_pool2d(2) |>
      self$batchnorm3() |>

      # dense
      torch::torch_flatten(start_dim = 2) |>
      self$dropout1() |>
      self$fc1() |>
      torch::nnf_relu() |>
      self$batchnorm_dense() |>
      self$dropout2() |>
      self$fc2()

    out_view <- out$view(c(
      dim(out)[1],
      self$output_ndigits,
      self$output_vocab_size
    ))

    # list(out_view, x[[2]])
    out_view
  }
)

calc_dim_conv <- function (x) {
  purrr::reduce(1:3, calc_dim_img_one, .init = x)
}

calc_dim_img_one <- function (x, y) {
  floor((x - 2)/2)
}
