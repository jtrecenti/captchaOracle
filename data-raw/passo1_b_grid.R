#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)


# grid de hiperparâmetros do modelo ---------------------------------------

# aqui, combinamos 3 valores de dropout, camadas densas e decaimento da
# taxa de aprendizado, totalizando 27 modelos para cada subset de cada Captcha

hparms <- purrr::cross_df(list(
  dropout = c(.1, .3, .5),
  dense_units = c(100, 200, 300),
  lr_lambda = c(.97, .98, .99)
))

# paths -------------------------------------------------------------------

# exemplo:
# path_data <- "data-raw/simulacao/0500/img/"
path <- args[1]
path_data <- paste0(path, "/img/")

# download and create dataset ---------------------------------------------

captcha_ds <- captcha::captcha_dataset(path_data)

# create train and validation data loaders --------------------------------
set.seed(1)
ids <- seq_along(captcha_ds)
id_train <- sample(ids, .8 * length(captcha_ds))
id_valid <- setdiff(ids, id_train)

captcha_dl_train <- torch::dataloader(
  torch::dataset_subset(captcha_ds, id_train),
  batch_size = 40,
  shuffle = TRUE
)

captcha_dl_valid <- torch::dataloader(
  torch::dataset_subset(captcha_ds, id_valid),
  batch_size = 40
)

# specify model -----------------------------------------------------------

model <- captcha::net_captcha

# run model ---------------------------------------------------------------
readr::write_rds(hparms, paste0(path, "hparms.rds"))

fit_model <- function(hparm, ii, path) {

  usethis::ui_info("Running set {ii}...")
  path_log <- sprintf("%s/model_%02d.log", path, ii)
  path_model <- fs::path_ext_set(path_log, ".pt")

  if (!file.exists(path_model)) {
    fitted <- model |>
      luz::setup(
        loss = torch::nn_multilabel_soft_margin_loss(),
        optimizer = torch::optim_adam,
        metrics = list(captcha::captcha_accuracy())
      ) |>
      luz::set_hparams(
        input_dim = dim(captcha_ds$data)[c(3,4)],
        output_vocab_size = dim(captcha_ds$target)[3],
        output_ndigits = dim(captcha_ds$target)[2],
        vocab = captcha_ds$vocab,
        transform = captcha_ds$transform,
        dropout = c(hparm$dropout, hparm$dropout),
        dense_units = hparm$dense_units
      ) |>
      luz::set_opt_hparams(
        lr = .01
      ) |>
      luz::fit(
        captcha_dl_train,
        valid_data = captcha_dl_valid,
        epochs = 100,
        # weight decay
        callbacks = list(
          luz::luz_callback_lr_scheduler(
            torch::lr_multiplicative,
            lr_lambda = function(x) hparm$lr_lambda
          ),
          luz::luz_callback_early_stopping(
            "valid_captcha acc",
            min_delta = .01,
            patience = 20,
            mode = "max"
          ),
          luz::luz_callback_csv_logger(path_log)
        )
      )

    luz::luz_save(fitted, path_model)
  }

}

# don't stop when the model fails
safe_fit_model <- purrr::possibly(fit_model, NULL)

# run all the models
purrr::iwalk(purrr::transpose(hparms), safe_fit_model, path)
