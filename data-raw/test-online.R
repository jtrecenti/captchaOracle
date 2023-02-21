library(magrittr)

devtools::load_all()

path <- "data-raw/tjrs"
fs::dir_create(path)

path_train <- paste0(path, "/train")
path_test <- paste0(path, "/test")
# path_oraculo <- "data-raw/tjmg/results"
# path_logs <- paste0(path, "/logs")
path_base_model <- "data-raw/tjrs/model_oracle_1_01_v1.pt"

base_model <- luz::luz_load(path_base_model)
ntry <- 5

captcha_ds <- captcha_dataset_oraculo_online(
  path_train,
  base_model,
  ntry = ntry,
  captcha_access_fn = captchaDownload:::captcha_access_tjrs,
  captcha_test_fn = captchaDownload:::captcha_test_tjrs,
  p_new = .8
)

captcha_dl_train <- torch::dataloader(
  captcha_ds,
  batch_size = 40,
  shuffle = TRUE,
  collate_fn = captchaOracle::collate_oraculo
)

# iteracao <- captcha_dl_train$.iter()$.next()

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
  lr_lambda = c(.999, .98, .99)
))

## hparm arbitrário, para testes
# ii <- as.numeric(readr::parse_number(basename(path_base_model)))
# hparm <- hparms[ii,]

update_model_callback <- luz::luz_callback(
  name = "update_model_callback",
  initialize = function() {

  },
  on_epoch_end = function() {
    # browser()

    acc_inicial <- dplyr::last(captcha_dl_train$dataset$model$records$metrics$valid)[["captcha acc"]]

    n_records <- length(ctx$records$metrics$valid)

    if (n_records > 1) {
      # browser()
      acc_atual <- max(purrr::map_dbl(ctx$records$metrics$valid, "captcha acc"))
    } else {
      acc_atual <- 0
    }

    acc_new <- ctx$records$metrics$valid[[n_records]][["captcha acc"]]

    if (acc_new > acc_inicial && acc_new >= acc_atual) {
      usethis::ui_info("Atualizando modelo...")
      captcha_dl_train$dataset$model$model <- ctx$model
    } else {
      usethis::ui_info("Ainda não faz sentido atualizar o modelo")
    }
  }
)

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
        input_dim = base_model$model$input_dim,
        output_vocab_size = base_model$model$output_vocab_size,
        output_ndigits = base_model$model$output_ndigits,
        vocab = base_model$model$vocab,
        transform = base_model$model$transform,
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
          luz::luz_callback_csv_logger(path_log),

          update_model_callback()
        )
      )

    luz::luz_save(fitted, path_model)
  }

}

# alguns modelos vão falhar, então rodamos com tratamento de erros
# safe_fit_model <- purrr::possibly(fit_model, NULL)
purrr::iwalk(purrr::transpose(hparms)[1], fit_model, path)



# export ------------------------------------------------------------------

da_online <- "data-raw/tjrs/model_oracle_1_01_v1.log" |>
  readr::read_csv(show_col_types = FALSE) |>
  dplyr::filter(set == "valid")

readr::write_rds(da_online, "data-raw/da_online.rds")
