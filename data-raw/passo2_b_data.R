# Neste passo, queremos criar as bases de dados das simulações do oráculo.
# Essas bases de dados precisam de acesso a internet, pois literalmente
# vamos aos sites para obter novos Captchas e suas classificações.

# O que faremos: a partir de um modelo base (e.g. trt com 40% de acurácia)
# vamos classificar uma nova base, com tamanho suficiente para chegar no
# modelo ótimo. A nova base será classificada da seguinte forma:
#
# 1. Tentamos classificar um captcha `ntry` vezes. Os valores de ntry
#    são c(1,5,10) quando possível
# 2. Se o modelo acertar o chute, salvamos o captcha classificado
# 3. Se o modelo errar o chute, salvamos o histórico de chutes
# 4. Com base na nova base e histórico de chutes, ajustamos novo modelo,
#    incluindo o histórico de chutes na verossimilhança.


# parametros --------------------------------------------------------------

# Necessário rodar para todas as fontes de dados
fonte <- "rcaptcha4"

path_base <- stringr::str_glue("data-raw/simulacao/{fonte}")

# seleciona o melhor modelo para cada quantidade de Captchas
melhores_modelos <- fs::dir_ls(path_base) |>
  # a função encontrar_melhor_modelo pode ser encontrada no
  # script passo1_a_testset.R ou no pacote {docfun}.
  purrr::map_dfr(encontrar_melhor_modelo)

# Vamos testar somente modelos com acurácia menor do que 50%
melhores_modelos_testar <- melhores_modelos |>
  dplyr::filter(captcha.acc < .5, captcha.acc > .05) |>
  dplyr::pull(file)

# Todas as combinações de ntry
ntry <- c(1, 5, 10)

# Quantidade de captchas a serem classificados
captcha_nmax <- 8000

# Carrega as funções de acesso e teste do captchaDownload
captcha_access_fun <- eval(parse(
  text = stringr::str_glue("captchaDownload:::captcha_access_{fonte}")
))
captcha_test_fun <- eval(parse(
  text = stringr::str_glue("captchaDownload:::captcha_test_{fonte}")
))

# para o rcaptcha ---------------------------------------------------------

# para o rcaptcha, os códigos são um pouco diferentes porque
# temos o gerador de dados e podemos selecionar a quantidade
# de letras (simulamos com 2 e 4 letras).
captcha_access_fun <- purrr::partial(
  captchaDownload:::captcha_access_rcaptcha,
  n_letter = 4
)
captcha_test_fun <- captchaDownload:::captcha_test_rcaptcha

# baixar novos dados com o oráculo ----------------------------------------

# A tarefa de montar os dados foi separada em dois passos porque seria muito
# difícil lidar com a parte que acessa a internet e a parte que não acessa
# a internet em um lugar só.

# loop em ntry e lista de modelos a testar
for (n in ntry) {

  for (m in melhores_modelos_testar) {

    usethis::ui_info("N = {basename(dirname(m))}, ntry = {n} ---------------")

    # caminhos das pastas a serem utilizadas e criadas
    path <- stringr::str_glue("{path_base}/{basename(dirname(m))}_{sprintf('%02d', n)}")
    path_oraculo <- stringr::str_replace(path, '/simulacao/', '/simulacao_oraculo/')
    path_new_data <- stringr::str_glue("{path_oraculo}/new_data")
    fs::dir_create(path_new_data)

    # carrega o modelo inicial
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

    # quantos falta baixar:
    # (queremos ter) - (ja temos na base anterior) - (ja baixamos)
    n_download <- captcha_nmax -
      floor(as.numeric(basename(dirname(m))) * .8) -
      n_baixados

    usethis::ui_info("baixando {n_download} novos captchas com oraculo...")

    if (n_download > 0) {
      # baixar os dados
      # aqui manual=FALSE pois o modelo de fato pode errar todos os chutes
      # e queremos uma classificação completamente automática.
      inv <- purrr::map(
        .n = seq_len(n_download),
        \(x) purrr::possibly(captchaDownload::captcha_oracle, NULL)(
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

# copiando outros elementos que faltam no modelo --------------------------

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

      # Aqui carregamos os dados antigos, que foram utilizados
      # para ajustar o modelo inicial
      usethis::ui_info("Copiando dados antigos...")
      captcha_ds <- captcha::captcha_dataset(path_old_data)
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
      captcha::captcha_annotate(
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
      arrumados <- captcha::captcha_annotate(
        arqs_arrumar,
        answers = rep(1, length(arqs_arrumar)),
        rm_old = TRUE
      )

      fs::dir_delete(path_new_data)
      fs::dir_delete(path_old_train)
      fs::dir_delete(path_old_valid)

    }

    # agora a base tem todas as observações!
    # Alguns arquivos estão marcados com 0 e outros com 1,
    # identificando se a observação é completa ou parcial.

  }
}
