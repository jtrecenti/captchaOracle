# Nesse script extra, pegamos o modelo novo, aplicamos nova etapa de download e
# rodamos o modelo do oraculo novamente.

fonte <- "trf5"
path_base <- fs::dir_ls("data-raw/simulacao_oraculo/trf5", regexp = "test|logs", invert = TRUE)
f <- function(path) {
  path |>
    fs::dir_ls(regexp = "\\.log") |>
    purrr::map_dfr(readr::read_csv, show_col_types = FALSE, .id = "file") |>
    dplyr::filter(set == "valid") |>
    dplyr::arrange(dplyr::desc(epoch)) |>
    dplyr::distinct(file, .keep_all = TRUE) |>
    dplyr::slice_max(captcha.acc, n = 1, with_ties = FALSE) |>
    dplyr::mutate(file = fs::path_ext_set(file, ".pt"))
}
melhores_modelos_testar <- path_base |>
  purrr::map_dfr(f) |>
  dplyr::pull(file)

ntry <- c(1)

captcha_nmax <- 2000

captcha_access_fun <- eval(parse(
  text = stringr::str_glue("captchaDownload:::captcha_access_{fonte}")
))
captcha_test_fun <- eval(parse(
  text = stringr::str_glue("captchaDownload:::captcha_test_{fonte}")
))


path_base <- "data-raw/simulacao_oraculo/trf5_novos"
fs::dir_create(path_base)
for (n in ntry) {
  for (m in melhores_modelos_testar) {

    usethis::ui_info("N = {basename(dirname(m))}, ntry = {n} ---------------")

    path <- stringr::str_glue("{path_base}/{basename(dirname(m))}")
    path_oraculo <- stringr::str_replace(path, '/simulacao/', '/simulacao_oraculo/')
    path_new_data <- stringr::str_glue("{path_oraculo}/new_data")
    fs::dir_create(path_new_data)

    path_train <- "data-raw/simulacao_oraculo/trf5/00101_01/train"

    modelo <- luz::luz_load(m)

    # limpa os baixados
    baixados_bad <- fs::dir_ls(
      path_new_data,
      regexp = "_[01]\\.(png|jpeg)",
      invert = TRUE
    )
    fs::file_delete(baixados_bad)
    n_baixados <- length(fs::dir_ls(path_new_data))
    usethis::ui_info("Baixados: {n_baixados}")

    # quantos falta baixar
    # (queremos ter) - (ja temos na base anterior) - (ja baixamos)
    n_download <- captcha_nmax -
      length(fs::dir_ls(path_train)) -
      n_baixados

    usethis::ui_info("baixando {n_download} novos captchas com oraculo...")

    if (n_download > 0) {
      # baixar os dados
      inv <- purrr::rerun(
        .n = n_download,
        purrr::possibly(captchaDownload::captcha_oracle, NULL)(
          path = path_new_data,
          model = modelo,
          max_ntry = n,
          manual = FALSE,
          captcha_access = captcha_access_fun,
          captcha_test = captcha_test_fun
        )
      )
    }


  }
}


# montar base -------------------------------------------------------------


# pasta da base de treino
path_treino_new <- "data-raw/simulacao_oraculo/trf5_novos/00101_01/train"
fs::dir_create(path_treino_new)

# copia arquivos para uma pasta unica
arqs_treino_old <- fs::dir_ls(path_train)
arqs_new_data <- fs::dir_ls(path_new_data)
fs::file_copy(
  c(arqs_treino_old, arqs_new_data),
  path_treino_new
)
# fs::dir_delete(path_new_data)

# copiar os logs
path_logs <- "data-raw/simulacao_oraculo/trf5_novos/00101_01/logs/"
path_logs_old <- "data-raw/simulacao_oraculo/trf5/00101_01/logs/"
fs::file_copy(fs::dir_ls(path_logs_old), path_logs)

# copiar a base de teste
fs::dir_copy(
  "data-raw/simulacao_oraculo/trf5/test/",
  "data-raw/simulacao_oraculo/trf5_novos/"
)

# ajustar novo modelo -----------------------------------------------------

path <- "data-raw/simulacao_oraculo/trf5_novos/00101_01"
fonte <- "trf5"
n <- stringr::str_extract(basename(path), "[0-9]+")
ntry <- stringr::str_extract(basename(path), "[0-9]+$")

path_train <- paste0(path, "/train")
path_test <- paste0(dirname(path), "/test")
path_logs <- paste0(path, "/logs")
path_oraculo <- path


base_model <- modelo

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
  path_logs = path_logs
)

captcha_dl_train <- torch::dataloader(
  captcha_ds,
  batch_size = 40,
  shuffle = TRUE,
  collate_fn = captchaOracle::collate_oraculo
)

# base de teste (validação).
test_ds <- captchaOracle::captcha_dataset_oraculo(
  root = path_test,
  path_logs = NULL
)
captcha_dl_test <- torch::dataloader(
  test_ds,
  batch_size = 40
)

# model definition --------------------------------------------------------

model <- captchaOracle::net_captcha_oracle

# fit ---------------------------------------------------------------------

# hparm deve ter o mesmo dense_units que o base_model
hparms <- purrr::cross_df(list(
  dropout = c(.1, .3, .5),
  dense_units = c(100, 200, 300),
  lr_lambda = c(.97, .98, .99)
))

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
purrr::iwalk(purrr::transpose(hparms), safe_fit_model, path)








