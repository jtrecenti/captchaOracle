#' Captcha accuracy metric with oracle
#'
#' @importFrom luz luz_metric
#' @export
captcha_accuracy_oracle <- luz::luz_metric(
  abbrev = "Captcha Acc",
  initialize = function() {
    self$correct <- 0
    self$total <- 0
  },
  update = function(preds, target) {
    # browser()

    ind_ok <- which(as.logical(!target$z$to(device = "cpu")))

    # print(glue::glue("OK: {length(ind_ok)}"))
    # if (length(ind_ok) == 40L) {
    #   browser()
    # }

    if (length(ind_ok) > 0) {
      pred <- torch::torch_argmax(preds[ind_ok,..,drop=FALSE], dim = 3)

      # bases de treino e teste apresentam estruturas diferentes
      if (is.list(target$y)) {
        tgt <- torch::torch_argmax(torch::torch_stack(target$y[ind_ok])$squeeze(2L), dim = 3)
      } else {
        tgt <- torch::torch_argmax(target$y[ind_ok,..]$squeeze(2L), dim = 3)
      }

      new_correct <- torch::torch_sum(pred == tgt, 2) == dim(pred)[2]
      new_correct <- new_correct$to(dtype = torch::torch_float())$sum()$item()
      self$correct <- self$correct + new_correct
      self$total <- self$total + dim(pred)[1]
    }


  },
  compute = function() {
    self$correct / self$total
  }
)
