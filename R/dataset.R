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

#' Função para juntar os dados de um minibatch corretamente
#'
#' @param l lista proveniente de um minibatch
#'
#' @export
collate_oraculo <- function(l) {
  # browser()
  x <- purrr::map(l, "x") |>
    torch::torch_stack()
  y <- purrr::map(l, "y") |>
    purrr::map("y")
  z <- purrr::map(l, "y") |>
    purrr::map("z") |>
    torch::torch_stack()
  list(x = x, y = list(y = y, z = z))
}

#' Captcha dataset do oráculo
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
#' @importFrom torch dataset
#' @importFrom captcha captcha_transform_image captcha_transform_label
#'
#' @export
captcha_dataset_oraculo <- torch::dataset(
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


#' Captcha dataset do oráculo online
#'
#' @param root (string): root directory of the files
#' @param model initial model.
#' @param ntry try n times. Default 1.
#'
#' @importFrom torch dataset
#' @importFrom captcha captcha_transform_image captcha_transform_label
#'
#' @export
captcha_dataset_oraculo_online <- torch::dataset(
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

      safe_oracle <- purrr::possibly(captchaDownload::captcha_oracle, NULL)

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




