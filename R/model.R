#' Modelo de captcha com oraculo
#'
#' É o mesmo modelo que o original, com a diferença que inicializamos
#' o modelo com os pesos do modelo base ajustado no passo anterior.
#'
#' @param base_model modelo base carregado com `luz::load_model()`
#' @param input_dim dimensão de entrada
#' @param output_ndigits quantidade de digitos da resposta
#' @param output_vocab_size tamanho do vocabulario da resposta
#' @param vocab o vocabulario da resposta
#' @param transform função de transformação dos dados
#' @param dropout hiperparâmetro de dropout
#' @param dense_units hiperparâmetro de dense units na camada densa
#'   que vem depois das convoluções
#'
#' @importFrom torch nn_module
#' @export
net_captcha_oracle <- torch::nn_module(

  "ORACLE-CNN",

  initialize = function(base_model,
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

    # browser()

    operations <- c(
      "conv1", "conv2", "conv3", "fc1", "fc2",
      "batchnorm0", "batchnorm1", "batchnorm2", "batchnorm3"
    )
    for (m in operations) {
      torch::with_no_grad(self[[m]]$weight$copy_(base_model$model[[m]]$weight))
      torch::with_no_grad(self[[m]]$bias$copy_(base_model$model[[m]]$bias))
    }

    # self$base_model <- base_model
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
