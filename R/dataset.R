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
#' @param root (string): root directory of dataset where `captcha.zip`
#'   exists or will be saved to if download is set to `TRUE`
#' @param path_logs path to log files
#' @param captcha (string): name of the captcha, must be one of
#'   [available_captchas()]()
#' @param transform_image (callable, optional): A function/transform
#'   that takes in an file path and returns an torch tensor prepared
#'   to feed the model.
#' @param transform_label (callable, optional): A function/transform
#'   that takes in the file paths and transform them.
#' @param download (bool, optional): If `TRUE`, downloads the dataset
#'   from the internet and puts it in `root`. If dataset is already
#'   downloaded, it is not downloaded again. Defaults to `FALSE`
#' @param in_memory (bool, optional) If `TRUE`, the default, loads
#'   all the files in memory. If `FALSE`, it exports a data generator
#'   function to read batches from disk.
#' @param augmentation (function, optional) If not `NULL`, applies a
#'   function to augment data with randomized preprocessing layers.
#'
#' @importFrom torch dataset
#' @importFrom captcha captcha_transform_image captcha_transform_label
#'
#' @export
captcha_dataset_oraculo <- torch::dataset(
  name = "my_captcha",
  initialize = function(root, captcha, path_logs,
                        transform_image = captcha::captcha_transform_image,
                        transform_label = captcha::captcha_transform_label,
                        download = FALSE,
                        in_memory = TRUE,
                        augmentation = NULL) {

    ## parameter checks
    if (download && missing(captcha)) {
      usethis::ui_stop(c(
        "If download = TRUE, must provide captcha name.",
        "Available names are: {paste(available_captchas(), collapse = ', ')}"
      ))
    }

    ## create directory and assign
    self$path <- root
    fs::dir_create(root)

    ## global variables to use along the class
    self$captcha <- captcha
    self$path_logs <- path_logs
    usethis::ui_info("Reading oracle logs...")

    ## download file from repository
    if (download) {
      self$download(captcha)
    }

    usethis::ui_info("Processing...")

    ## build dataset
    if (in_memory) {
      files <- fs::dir_ls(root, recurse = TRUE, type = "file")
      self$files <- files

      files_names <- files |>
        basename() |>
        tools::file_path_sans_ext()
      all_letters <- files_names |>
        stringr::str_extract("(?<=_)[0-9a-zA-Z]+") |>
        purrr::map(stringr::str_split, "")

      vocab <- sort(unique(unlist(all_letters)))
      ## talvez pegar do base_model e incluir como parametro
      # vocab <- c(vocab, "l")

      # browser()
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


    } else {
      usethis::ui_stop("Not implemented yet.")
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

  # resources = captcha_data_url(),

  # download captcha zip file and unzip it
  download = function(captcha) {
    u <- self$resources[[captcha]]
    dir <- self$path

    ## for testing purposes
    # u <- captcha_data_url()$trt
    # dir <- "~/Downloads/trt"

    ## download
    fs::dir_create(dir)
    filename <- basename(u)
    destpath <- file.path(dir, filename)
    withr::with_options(
      list(timeout = 600),
      utils::download.file(u, destfile = destpath)
    )

    # TODO md5 sum check

    ## unzip and delete original
    zip::unzip(destpath, exdir = dir)
    fs::file_delete(destpath)

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
