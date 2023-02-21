#' Captcha accuracy metric with oracle
#'
#' This object is used to calculate the accuracy of the model in the
#' fitting process.
#'
#' This function is a generator created using [luz::luz_metric()] function.
#' It has a `initialize()` method that sets the total number of instances
#' and total number of correct predictions as zero. For any minibatch, it
#' has an `update()` method that updates the total number of instances and
#' total number of correct predictions with new data. Finally, it has a
#' `compute()` method that calculates accuracy from the total number of
#' correct predictions and total number of instances.
#'
#' Training data has a different structure than validation
#' data: training data includes both complete and incomplete labels
#' from the oracle, but validation includes only complete labels. Accuracy
#' is calculated using only when we have at least one complete label in the
#' minibatch. Accuracy is calculated considering only the complete labels.
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

      # train and validation have different structures
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
