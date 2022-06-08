#' Loss do oraculo
#'
#' @param weight pesos a serem aplicados nas observações
#' @param reduction função de redução
#'
#' @importFrom torch nn_module
#'
#' @export
oracle_loss <- torch::nn_module(
  "oracle_loss",
  inherit = torch:::nn_weighted_loss,
  initialize = function(weight = NULL, reduction = "mean") {
    super$initialize(weight = weight, reduction = reduction)
  },
  forward = function(input, target) {

    z <- target$z$to(device = "cpu")
    ind_ok <- which(!as.logical(z))
    ind_not_ok <- which(as.logical(z))

    if (length(ind_ok) <= 1) {
      browser()
    }

    # if (length(ind_not_ok) == 0) {
    #   browser()
    # }

    # browser()

    # loss dos casos classificados corretamente
    if (length(ind_ok) > 0) {
      loss_corretos <- myloss(
        input[ind_ok,..,drop=FALSE],
        torch::torch_stack(target$y[ind_ok])$squeeze()
      )
    } else {
      loss_corretos <- 0
    }

    # calculando a loss dos casos classificados errado
    # browser()
    if (length(ind_not_ok) > 0) {
      loss_errados <- nnf_oracle_loss(
        input[ind_not_ok,..,drop=FALSE],
        target$y[ind_not_ok]
      )
    } else {
      loss_errados <- 0
    }

    loss_corretos + loss_errados

  }
)

nnf_oracle_loss <- function(input2, target2) {
  # calculo as probabilidades
  probs <- torch::nnf_softmax(input2, 3)


  # browser()
  # para cada observação, preciso calcular 1-p(obs)
  # produto: probabilidade completa. Probabilidade do erro
  prob_1_menos <- torch::torch_ones(length(target2))
  for (ii in seq_along(target2)) {
    prob_soma <- torch::torch_sum(target2[[ii]] * probs[ii,..,drop=FALSE], 3)
    produto <- torch::torch_prod(prob_soma, 2)
    prob_1_menos[ii] <- prob_1_menos[ii] - torch::torch_sum(produto)
  }

  loss_ind <- -torch::torch_log(prob_1_menos)
  torch::torch_sum(loss_ind)
}

myloss <- function(input, target) {
  log_probs <- torch::nnf_log_softmax(input, 3L)
  loss_ind <- -target * log_probs
  torch::torch_sum(loss_ind)
}
