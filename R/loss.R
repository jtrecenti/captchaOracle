#' Oracle Loss
#'
#' Implements the loss function from the input (probabilities) and target
#' (one-hot encoded label).
#'
#' @param weight pesos a serem aplicados nas observações
#' @param reduction função de redução
#'
#' This is a [torch::nn_module()] with two main methods: initialize and
#' forward. It inherits information from the `nn_weighted_loss` module.
#'
#' The `initialize()` method register model weights (default to `NULL`) and
#' the reduction function, (default to `mean`). The `forward()` method
#' calculates the loss for a minibatch.
#'
#' First, it separates which indices of the minibatch refer to complete
#' labels and which refer to incomplete labels. For the complete labels,
#' we have a different data structure for training and validation datasets:
#' the former is a list of tensors (to address the attempt history) and
#' the latter is a tensor with fixed dimensions. This loss is calculated
#' using a simple crossentropy function.
#'
#' The incomplete loss is calculated only in the training dataset, as the
#' validation data include only complete labels. The loss is calculated
#' using the proposed method in the doctorate thesis, which is to consider
#' the log-probability of not observing the predictions, which is all the
#' information provided by the oracle, similar to the survival analysis
#' framework.
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

    # if (length(ind_not_ok) == 0) {
    #   browser()
    # }

    # browser()
    # calculate loss in correct cases
    if (length(ind_ok) > 0) {
      if (is.list(target$y)) {
        loss_corretos <- myloss(
          input[ind_ok,..,drop=FALSE],
          torch::torch_stack(target$y[ind_ok])$squeeze(2L)
        )
      } else {
        # browser()
        loss_corretos <- myloss(
          input[ind_ok,..,drop=FALSE],
          target$y[ind_ok,..]$squeeze(2L)
        )
      }

    } else {
      loss_corretos <- 0
    }

    # calculate loss in censored cases
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

  # calculate probabilities
  probs <- torch::nnf_softmax(input2, 3)

  # browser()
  # for each obervation, we calculate 1 - P(observed)
  # product: complete probability of the error.
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
