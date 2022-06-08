path <- "data-raw/simulacao_oraculo/tjmg/00201_10"

fonte <- basename(dirname(path))
n <- stringr::str_extract(basename(path), "[0-9]+")
ntry <- stringr::str_extract(basename(path), "[0-9]+$")
path_base <- stringr::str_glue("data-raw/simulacao/{fonte}")
path_train <- paste0(path, "/train")
path_test <- paste0(dirname(path), "/test")
path_logs <- paste0(path, "/logs")

melhores_modelos <- fs::dir_ls(path_base) |>
  purrr::map_dfr(docfun::encontrar_melhor_modelo)

# modelo base
path_base_model <- melhores_modelos |>
  dplyr::filter(stringr::str_detect(file, n)) |>
  dplyr::pull(file)


base_model <- luz::luz_load(path_base_model)


# checks ------------------------------------------------------------------

invalidos <- fs::dir_ls(path_train, regexp = "_[01]\\.(jpeg|png)$", invert = TRUE)
if (length(invalidos) > 0) {
  usethis::ui_stop("Arquivos inválidos encontrados.")
}

teste_invalidos <- fs::dir_ls(path_test, regexp = "_1\\.(jpeg|png)$", invert = TRUE)
if (length(teste_invalidos) > 0) {
  usethis::ui_stop("Arquivos inválidos encontrados no teste.")
}

teste_1000 <- fs::dir_ls(path_test, regexp = "_1\\.(jpeg|png)$")
if (length(teste_1000) < 1000) {
  usethis::ui_stop("Teste com menos de 1000 casos")
}


# datasets do oraculo -----------------------------------------------------

# essa função que lida com o oráculo.
captcha_ds <- captchaOracle::captcha_dataset_oraculo(
  root = path_train,
  path_logs = path_logs,
  captcha = NULL,
  download = FALSE
)

# captcha_ds$.getitem(1)
# captcha_ds$.getitem(1506)

captcha_dl_train <- torch::dataloader(
  captcha_ds,
  batch_size = 40,
  shuffle = TRUE,
  collate_fn = captchaOraculo::collate_oraculo
)

# captcha_dl_train$.iter()$.next()

# esse aqui são dados novos para teste, classificados automaticamente pelo
# modelo de maxima de acurácia.
test_ds <- captchaOracle::captcha_dataset_oraculo(
  root = path_test,
  path_logs = NULL,
  captcha = NULL,
  download = FALSE
)
captcha_dl_test <- torch::dataloader(
  test_ds,
  batch_size = 40
)

# captcha_dl_test$.iter()$.next()

# model definition --------------------------------------------------------

model <- captchaOracle::net_captcha_oracle

# novas loss --------------------------------------------------------------

# it <- train_dl$.iter()$.next()
# input <- model(it$x$to(device = "cuda"))
# target <- it$y$to(device = "cuda")
# oraculo <- it$oraculo$to(device = "cuda")


# fit ---------------------------------------------------------------------

# hparm deve ser o mesmo que o do base model
hparms <- purrr::cross_df(list(
  dropout = c(.1, .3, .5),
  dense_units = c(100, 200, 300),
  lr_lambda = c(.97, .98, .99)
))

## hparm arbitrário, para testes
# ii <- as.numeric(readr::parse_number(basename(path_base_model)))
# hparm <- hparms[ii,]

fit_model <- function(hparm, ii, path) {
  usethis::ui_info("Modelando com set {ii}...")
  path_log <- sprintf("%s/model_oracle_1_%02d.log", path_oraculo, ii)
  path_model <- fs::path_ext_set(path_log, ".pt")
  fitted <- model |>
    luz::setup(
      loss = captchaOracle::oracle_loss(),
      # loss = torch::nn_multilabel_soft_margin_loss(),
      optimizer = torch::optim_adam,
      metrics = list(captchaOracle::captcha_accuracy_oracle())
    ) |>
    luz::set_hparams(
      base_model = base_model,
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
      data = captcha_dl_train,
      valid_data = captcha_dl_test,
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


safe_fit_model <- purrr::possibly(fit_model, NULL)
# safe_fit_model <- fit_model
purrr::iwalk(purrr::transpose(hparms), safe_fit_model, path)

