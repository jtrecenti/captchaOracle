# neste passo, queremos criar as bases de dados das simulações do oráculo
# essas bases de dados precisam de acesso a internet, pois literalmente
# vamos aos sites dos entes para obter novos captchas e suas classificações.

#' O que faremos: a partir de um modelo base (e.g. trt com 40% de acurácia)
#' vamos classificar uma nova base, com tamanho suficiente para chegar no
#' modelo ótimo. A nova base será classificada da seguinte forma:
#' 1. Tentamos classificar um captcha `ntry` vezes
#'   - Os valores de ntry são c(1,5,10) quando possível
#' 2. Se o modelo acertar o chute, salvamos o captcha classificado
#' 3. Se o modelo errar o chute, salvamos
#'   - o histórico de chutes
#'   - o último chute.
#' 4. Com base na nova base e histórico de chutes, ajustamos novo modelo.
#'   Para isso, vamos incluir o histórico de chutes na verossimilhança.



# parametros --------------------------------------------------------------

fonte <- "tjmg"

path_base <- stringr::str_glue("data-raw/simulacao/{fonte}")
melhores_modelos <- fs::dir_ls(path_base) |>
  purrr::map_dfr(docfun::encontrar_melhor_modelo)

# vamos testar somente modelos com acurácia menor do que 50%
melhores_modelos_testar <- melhores_modelos |>
  dplyr::filter(captcha.acc < .5, captcha.acc > .1) |>
  dplyr::pull(file)

ntry <- c(1, 5, 10)

## talvez não faça sentido escolher dessa forma
# captcha_nmax <- melhores_modelos |>
#   dplyr::mutate(n = as.numeric(basename(dirname(file)))) |>
#   dplyr::mutate(maximo = max(captcha.acc)) |>
#   dplyr::filter(captcha.acc == maximo) |>
#   dplyr::arrange(n) |>
#   dplyr::slice(1) |>
#   dplyr::pull(n)

captcha_nmax <- 8000

# captcha_access_fun <- eval(parse(
#   text = stringr::str_glue("captchaDownload:::captcha_access_{fonte}")
# ))
# captcha_test_fun <- eval(parse(
#   text = stringr::str_glue("captchaDownload:::captcha_test_{fonte}")
# ))

captcha_access_fun <- purrr::partial(
  captchaDownload:::captcha_access_rcaptcha,
  n_letter = 4
)
captcha_test_fun <- captchaDownload:::captcha_test_rcaptcha


# baixar novos dados com o oráculo ----------------------------------------

# separei a tarefa de montar os dados em duas porque seria muito
# difícil lidar com a parte que acessa a internet e a parte que não acessa
# a internet em um lugar só.


for (n in ntry) {
  for (m in melhores_modelos_testar) {

    usethis::ui_info("N = {basename(dirname(m))}, ntry = {n} ---------------")

    path <- stringr::str_glue("{path_base}/{basename(dirname(m))}_{sprintf('%02d', n)}")
    path_oraculo <- stringr::str_replace(path, '/simulacao/', '/simulacao_oraculo/')
    path_new_data <- stringr::str_glue("{path_oraculo}/new_data")
    fs::dir_create(path_new_data)

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
      floor(as.numeric(basename(dirname(m))) * .8) -
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

# copiando outras coisas que faltam no modelo -----------------------------

for (n in ntry) {
  for (m in melhores_modelos_testar) {

    usethis::ui_info("N = {basename(dirname(m))}, ntry = {n} ---------------")

    # download new data -------------------------------------------------------
    path <- stringr::str_glue("{path_base}/{basename(dirname(m))}_{sprintf('%02d', n)}")
    path_oraculo <- stringr::str_replace(path, '/simulacao/', '/simulacao_oraculo/')
    path_old_data <- stringr::str_glue("{stringr::str_remove(path, '_[0-9]+')}/img")
    path_new_data <- stringr::str_glue("{path_oraculo}/new_data")
    path_old_train <- stringr::str_glue("{path_oraculo}/old_train")
    path_old_valid <- stringr::str_glue("{path_oraculo}/old_valid")
    path_train <- stringr::str_glue("{path_oraculo}/train")


    fs::dir_create(path_train)

    if (length(fs::dir_ls(path_train)) == 0) {

      fs::dir_create(path_old_train)
      fs::dir_create(path_old_valid)

      # Esse aqui são os dados antigos, que foram utilizados para criar o modelo
      usethis::ui_info("Copiando dados antigos...")
      captcha_ds <- captcha::captcha_dataset(
        root = path_old_data,
        captcha = NULL,
        download = FALSE
      )
      set.seed(1)
      ids <- seq_along(captcha_ds)
      # .8 foi definido na etapa 1, e foi feito com set.seed(1)
      id_train <- sample(ids, .8 * length(captcha_ds))
      id_valid <- setdiff(ids, id_train)
      fs::file_copy(captcha_ds$files[id_train], path_old_train)
      fs::file_copy(captcha_ds$files[id_valid], path_old_valid)

      ## aqui estamos copiando os dados da base de treino e os dados
      ## que foram classificados utilizando o oráculo, para montar
      ## a base completa.

      old_train <- fs::dir_ls(path_old_train)
      captcha::classify(
        old_train,
        path = path_train,
        answers = rep(1, length(old_train)),
        rm_old = TRUE
      )
      fs::file_copy(fs::dir_ls(path_new_data), path_train)

      arqs_arrumar <- fs::dir_ls(
        path_train,
        regexp = "_[01]\\.(jpeg|png)$",
        invert = TRUE
      )
      arrumados <- captcha::classify(
        arqs_arrumar,
        answers = rep(1, length(arqs_arrumar)),
        rm_old = TRUE
      )

      fs::dir_delete(path_new_data)
      fs::dir_delete(path_old_train)
      fs::dir_delete(path_old_valid)

    }

    # agora a base tem todas as observações!
    # mas algumas estão marcadas com 0 e outras com 1.

  }
}
