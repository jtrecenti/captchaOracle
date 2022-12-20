#!/usr/bin/env Rscript

# descricao ---------------------------------------------------------------

# aqui, rodamos os modelos a partir das bases geradas no passo 2

# path <- "data-raw/simulacao_oraculo/tjmg/00100_10/"
args <- commandArgs(trailingOnly = TRUE)
path <- args[1]

# lista todas as pastas necessárias para rodar o modelo
fonte <- basename(dirname(path))
n <- stringr::str_extract(basename(path), "[0-9]+")
ntry <- stringr::str_extract(basename(path), "[0-9]+$")
path_base <- stringr::str_glue("data-raw/simulacao/{fonte}")
path_train <- paste0(path, "/train")
path_test <- paste0(dirname(path), "/test")
path_logs <- paste0(path, "/logs")
path_oraculo <- stringr::str_replace(path, "/simulacao/", "/simulacao_oraculo/")

melhores_modelos <- fs::dir_ls(path_base) |>
  # a função encontrar_melhor_modelo pode ser encontrada no
  # script passo1_a_testset.R ou no pacote {docfun}.
  purrr::map_dfr(encontrar_melhor_modelo)

# Carrega o modelo base associado a essa base de dados
path_base_model <- melhores_modelos |>
  dplyr::filter(stringr::str_detect(file, n)) |>
  dplyr::pull(file)

base_model <- luz::luz_load(path_base_model)

# checks ------------------------------------------------------------------

# Os códigos abaixo verificam se está tudo bem nas bases que vamos utilizar.
# Isso é importante porque baixamos as bases da internet e sempre dá
# algum errinho na hora de baixar os dados.

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

# Cria um dataset que lida com a estrutura de dados fornecida pelo oráculo.
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

# Dados novos para validação, classificados pelo
# modelo de máxima de acurácia.
# estamos chamando a base de validação de teste nesse passo.
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

# hparm deve ter o mesmo dense_units que o base_model,
# por isso alguns modelos vão necessariamente falhar.
# Isso reduz o escopo para 9 modelos por chamada da grid.
hparms <- purrr::cross_df(list(
  dropout = c(.1, .3, .5),
  dense_units = c(100, 200, 300),
  lr_lambda = c(.97, .98, .99)
))

## hparm arbitrário, para testes
# ii <- as.numeric(readr::parse_number(basename(path_base_model)))
# hparm <- hparms[ii,]

# função similar à do passo 1, mas adaptada para o caso do oraculo
fit_model <- function(hparm, ii, path_oraculo) {

  usethis::ui_info("Modelando com set {ii}...")
  path_log <- sprintf("%s/model_oracle_1_%02d.log", path_oraculo, ii)
  path_model <- fs::path_ext_set(path_log, ".pt")

  if (!file.exists(path_model)) {

    fitted <- model |>
      luz::setup(
        # nova loss do oráculo
        loss = captchaOracle::oracle_loss(),
        optimizer = torch::optim_adam,
        # nova função de acurácia que leva em conta a estrutura de dados
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

}

# alguns modelos vão falhar, então rodamos com tratamento de erros
safe_fit_model <- purrr::possibly(fit_model, NULL)
purrr::iwalk(purrr::transpose(hparms), safe_fit_model, path)
