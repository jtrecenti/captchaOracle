read_logs <- function(path_logs) {
  fs::dir_ls(path_logs) |>
    purrr::map_dfr(
      readr::read_csv,
      col_types = readr::cols(.default = readr::col_character()),
      show_col_types = FALSE,
      .id = "file"
    ) |>
    dplyr::mutate(file = basename(tools::file_path_sans_ext(file)))
}

#' Collect and bind minibatches with oracle structure
#'
#' This function reimplements the default `collate_fn` function from
#' torch dataloaders to deal with the list-like structure of datasets
#' generated from the [captcha_dataset()] module. Whenever possible,
#' it applies the [torch::torch_stack()] function to bind the tensors.
#'
#' @param mb the minibatch, which is a list of torch tensors.
#'
#' @export
collate_oracle <- function(mb) {
  # browser()
  x <- purrr::map(mb, "x") |> torch::torch_stack()
  y <- purrr::map(mb, "y") |> purrr::map("y")
  z <- purrr::map(mb, "y") |> purrr::map("z") |> torch::torch_stack()
  list(x = x, y = list(y = y, z = z))
}

#' Captcha dataset incorporating incomplete data structure
#'
#' When we use oracles, we have to deal with a different data structure
#' that incorporates incomplete information provided by the websites.
#' First, we need to pass not only the directory of the images but also
#' a `log` file that records all the attempts to solve the Captcha through
#' the website. When there is at least one successful attempt, the image
#' and the label are read normally, as if we were in the complete data
#' framework. However, when all the attempts fail, we need to record all
#' the failed attempts (a list of one-hot encoded labels) for this observation.
#' The number of failed attempts is not fixed, then we have to store the
#' data as a list.
#'
#' @param root (string): root directory of the files
#' @param path_logs path to log files
#' @param transform_image (callable, optional): A function/transform
#'   that takes in an file path and returns an torch tensor prepared
#'   to feed the model.
#' @param transform_label (callable, optional): A function/transform
#'   that takes in the file paths and transform them.
#' @param augmentation (function, optional) If not `NULL`, applies a
#'   function to augment data with randomized preprocessing layers.
#'
#' This is an object of class `dataset_generator` created using
#' [torch::dataset()] function. It has a `initialize()` method that
#' takes directory containing the input images and the log file,
#' then assigns all the information in-memory with the list-like data
#' structure for the response variable. It also assigns a dummy variable
#' that indicates whether the information is incomplete, to facilitate
#' loss and accuracy calculation. It also has a `.getitem()` method that
#' correctly extracts one observation of the dataset in this complex
#' structure, and a `.length()` method that correctly calculates the
#' number of Captchas of the dataset.
#'
#' @importFrom torch dataset
#' @importFrom captcha captcha_transform_image captcha_transform_label
#'
#' @export
captcha_dataset_oracle <- torch::dataset(
  name = "my_captcha",
  initialize = function(root, path_logs,
                        transform_image = captcha::captcha_transform_image,
                        transform_label = captcha::captcha_transform_label,
                        augmentation = NULL) {

    ## create directory and assign
    self$path <- root
    fs::dir_create(root)

    ## global variables to use along the class
    self$path_logs <- path_logs
    usethis::ui_info("Reading oracle logs...")

    usethis::ui_info("Processing...")

    ## build dataset
    files <- fs::dir_ls(root, recurse = TRUE, type = "file")
    self$files <- files

    files_names <- files |>
      basename() |>
      tools::file_path_sans_ext()

    all_letters <- files_names |>
      stringr::str_extract("(?<=_)[0-9a-zA-Z]+") |>
      purrr::map(stringr::str_split, "")

    vocab <- sort(unique(unlist(all_letters)))

    x <- transform_image(files)
    y <- transform_label(all_letters, vocab)
    if (!is.null(self$path_logs)) {
      da_logs <- read_logs(self$path_logs) |>
        dplyr::group_by(file) |>
        # somente os erros
        dplyr::filter(all(result == "FALSE")) |>
        dplyr::ungroup() |>
        dplyr::filter(!is.na(label))
      indices <- files_names |>
        stringr::str_remove("_.*") |>
        purrr::map(~which(da_logs$file == .x))
      oracle <- da_logs$label |>
        purrr::map(stringr::str_split, "") |>
        transform_label(vocab)
    } else {
      indices <- NULL
      oracle <- NULL
    }

    usethis::ui_info("Done!")
    self$data <- x
    self$target <- y
    self$oracle <- oracle
    self$indices <- indices
    self$vocab <- vocab
    self$transform <- transform_image
    self$augmentation <- augmentation

  },

  # check if file exists
  check_exists = function() {
    usethis::ui_stop("not implemented")
  },
  # returns a subset of indexed captchas
  .getitem = function(index) {

    x <- self$data[index,..,drop=TRUE]

    if (!is.null(self$augmentation)) {
      x <- self$augmentation(x)
    }

    ind <- self$indices[[index]]
    z <- length(ind) > 0
    if (z) {
      y <- self$oracle[ind,..,drop=FALSE]
    } else {
      y <- self$target[index,..,drop=FALSE]
    }
    z <- torch::torch_tensor(as.integer(z))

    return(list(x = x, y = list(y = y, z = z)))
  },
  # number of files
  .length = function() {
    length(self$files)
  },
  # active bindings (retrive or modify)
  active = list(
    classes = function(cl) {
      if (missing(cl)) c(letters, 0:9) else cl
    }
  )
)


#' Captcha dataset incorporating online learning
#'
#' This `dataset_generator` object implements an experimental feature to
#' learn how to solve the Captcha automatically accessing the web using
#' an oracle function. It does not need any input data, only the initial
#' model of class `luz_module_fitted` and the `captcha_access_fn` and
#' `captcha_test_fn` functions to access the web.
#'
#' @param root (string): root directory to save new files
#' @param model initial model.
#' @param ntry number of attempts to solve the Captcha. Defaults to 1.
#' @param captcha_access_fn function that downloads a Captcha image and returns
#'   all the information needed to test whether the label is correct.
#' @param captcha_test_fn function that uploads a prediction and tests whether it
#'   is correct, returning `TRUE` or `FALSE`.
#' @param p_new hyperparameter to control the probability of downloading new
#'   data to get a new observation. For example, when `p_new` is `0.2`, the
#'   default, approximately 20% of the times we get an image to compose a
#'   minibatch it is a new image downloaded from the internet, and 80% of the
#'   times it is an already downloaded image.
#'
#' @importFrom torch dataset
#' @importFrom captcha captcha_transform_image captcha_transform_label
#'
#' @description
#' `r lifecycle::badge("experimental")`
#' The difference from this function to [captcha_dataset_oracle()] is that
#' it downloads the new data directly from the internet instead of considering
#' a fixed dataset. To take advantage of the fact that we are downloading a lot
#' of new images (and new information), it is possible to consider a
#' probability to download a new image every time we need to get an
#' observation. This way the downloaded images can be used in more than one
#' minibatch and we will always have the chance to get new information.
#'
#' @export
captcha_dataset_oracle_online <- torch::dataset(
  name = "my_captcha",
  initialize = function(root,
                        model,
                        ntry = 1,
                        captcha_access_fn,
                        captcha_test_fn,
                        p_new = .2) {
    ## create directory and assign
    self$path <- root
    self$ntry <- ntry
    self$model <- model
    self$access <- captcha_access_fn
    self$test <- captcha_test_fn
    self$epoch_size <- 80
    self$p_new <- p_new
    fs::dir_create(root)

  },

  # check if file exists
  check_exists = function() {
    usethis::ui_stop("not implemented")
  },

  # returns a subset of indexed captchas
  .getitem = function(index) {

    has_files <- length(fs::dir_ls(self$path)) > 0
    download_new <- runif(1) < self$p_new || !has_files

    # clean bad files
    fs::dir_ls(self$path, type = "file") |>
      stringr::str_subset("(?<=_)[0-9a-zA-Z]+", negate = TRUE) |>
      fs::file_delete()

    if (download_new) {

      safe_oracle <- purrr::possibly(captcha_oracle, NULL)

      log_data <- NULL

      while (is.null(log_data)) {
        log_data <- safe_oracle(
          self$path,
          manual = FALSE,
          model = self$model,
          max_ntry = self$ntry,
          captcha_access = self$access,
          captcha_test = self$test
        )
      }

      files <- fs::dir_ls(self$path, type = "file") |>
        fs::file_info() |>
        dplyr::arrange(dplyr::desc(birth_time)) |>
        dplyr::pull(path) |>
        dplyr::first()

      files_names <- files |>
        basename() |>
        tools::file_path_sans_ext()

      log_data_errors <- log_data |>
        dplyr::filter(!result) |>
        dplyr::filter(!is.na(label)) |>
        dplyr::mutate(file = files_names)

    } else {

      # browser()
      files <- sample(fs::dir_ls(self$path, type = "file"), 1)

      files_names <- files |>
        basename() |>
        tools::file_path_sans_ext()

      file_log <- paste0(
        dirname(self$path),
        "/logs/",
        stringr::str_remove(files_names, "_.*"),
        ".log"
      )

      if (file.exists(file_log)) {
        log_data_errors <- file_log |>
          readr::read_csv(
            col_types = readr::cols(.default = readr::col_character()),
            show_col_types = FALSE
          ) |>
          dplyr::filter(result == "FALSE") |>
          dplyr::filter(!is.na(label)) |>
          dplyr::mutate(file = files_names)
      }

    }

    vocab <- self$model$model$vocab
    x <- self$model$model$transform(files)
    z <- stringr::str_detect(files_names, "0$")

    if (z) {
      y <- log_data_errors$label |>
        purrr::map(stringr::str_split, "") |>
        captcha::captcha_transform_label(vocab)
    } else {
      y <- files_names |>
        stringr::str_extract("(?<=_)[0-9a-zA-Z]+") |>
        purrr::map(stringr::str_split, "") |>
        captcha::captcha_transform_label(vocab)
    }
    z <- torch::torch_tensor(as.integer(z))

    return(list(x = x$squeeze(1), y = list(y = y, z = z)))
  },
  # number of files
  .length = function() {
    self$epoch_size
  },
  # active bindings (retrive or modify)
  active = list(
    classes = function(cl) {
      if (missing(cl)) c(letters, 0:9) else cl
    }
  )
)




