#' Function to download a captcha and classify using the Oracle
#'
#' This function creates annotated datasets from an initial
#' model and functions to access the oracle of a website. It stores the
#' downloaded image and a `log` file containing all the attempts to
#' solve the Captcha.
#'
#' @param path path of the directory to save files
#' @param model model used to predict the label of an image. Defaults to NULL,
#'   which opens a prompt to manual input.
#' @param max_ntry maximum number of attempts. Defaults to 10.
#' @param manual if all the attempts fail, should it opem the prompt for
#'   manual input? Defaults to `TRUE`.
#' @param captcha_access function that downloads a Captcha image and returns
#'   all the information needed to test whether the label is correct.
#' @param captcha_test function that uploads a prediction and tests whether it
#'   is correct, returning `TRUE` or `FALSE`.
#'
#' @return path to the log file.
#'
#' @export
captcha_oracle <- function(path, model = NULL, max_ntry = 10, manual = TRUE,
                           captcha_access, captcha_test) {

  # browser()
  fs::dir_create(path)
  obj <- captcha_access(path)
  f_captcha <- obj$f_captcha
  ntry <- 1
  label <- captcha_candidates(f_captcha, model, n = max_ntry)

  f_log <- paste0(
    dirname(path), "/logs/",
    fs::path_ext_set(basename(f_captcha), ".log")
  )
  fs::dir_create(dirname(f_log))
  acertou <- captcha_test(obj, label[1])

  da_log <- tibble::tibble(
    ntry = ntry,
    label = label[ntry],
    type = "auto",
    result = acertou
  )

  if (acertou) {
    usethis::ui_done("Correct!!!")
    label <- label[ntry]
  } else {
    max_ntry_model <- min(max_ntry, length(label))
    usethis::ui_info("We have {max_ntry_model} candidates...")
  }

  while (!acertou && ntry < max_ntry_model && !is.null(model)) {
    usethis::ui_info("Incorrect. The attempt was: {label[ntry]}")
    ntry <- ntry + 1
    acertou <- captcha_test(obj, label[ntry])
    da_log <- tibble::add_row(
      da_log,
      ntry = ntry,
      label = label[ntry],
      type = "auto",
      result = acertou
    )
    if (acertou) {
      usethis::ui_done("Correct!!!")
      label <- label[ntry]
    }
  }

  if (!acertou && !manual) {
    label <- label[ntry]
  }

  # if tried {max_ntry} times and the model still did not find it
  ntry <- 0
  while (!acertou && ntry < max_ntry && manual) {
    ntry <- ntry + 1
    label <- captcha_label(f_captcha)
    acertou <- captcha_test(obj, label)
    da_log <- tibble::add_row(
      da_log,
      label = label,
      type = "manual",
      result = acertou
    )
  }

  if (acertou == 0) {
    usethis::ui_oops("Incorrect after all the attempts...")
  }

  lab_oracle <- paste0(label, "_", as.character(as.numeric(acertou)))
  captcha::captcha_annotate(f_captcha, lab_oracle, rm_old = TRUE)
  readr::write_csv(da_log, f_log)

}


#' Calculates n captcha candidate labels
#'
#' This function returns a list of candidated using all captcha combinations
#' from some cut value. It uses probabilities returned from the model to get
#' the most likely labels.
#'
#' @param f_captcha captcha file
#' @param model model that generates logits
#' @param cut_value so that we make combinations from real values
#' @param n maximum number of candidates
#'
#' @export
captcha_candidates <- function(f_captcha, model, cut_value = log(.01), n) {
  # browser()
  # from captcha::decrypt
  model$model$eval()
  transformed <- model$model$transform(f_captcha)

  # calculate log-probability
  probs <- as.matrix(torch::nnf_log_softmax(model$model(transformed)[1,..], 2))

  comb_index <- apply(probs > cut_value, 1, which, simplify = FALSE)
  comb <- purrr::map(purrr::cross(comb_index), purrr::flatten_int)
  comb_matrix <- do.call(rbind, comb)
  candidates <- apply(
    comb_matrix,
    MARGIN = 1,
    FUN = function(x) paste(model$model$vocab[x], collapse = "")
  )
  # calculates the log-likelihood of that candidate
  lkl_candidate <- apply(
    comb_matrix,
    MARGIN = 1,
    FUN = function(x) sum(sapply(seq_along(x), function(z) probs[z,x[z]]))
  )
  candidates <- candidates[order(lkl_candidate, decreasing = TRUE)]
  utils::head(candidates, n)
}

captcha_label <- function(cap) {
  cap_ <- captcha::read_captcha(cap)
  plot(cap_)
  ans <- readline("Answer: ")
  ans
}
