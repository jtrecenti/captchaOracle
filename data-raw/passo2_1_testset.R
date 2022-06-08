# neste passo, queremos criar as bases teste para as simulações do oráculo
# essas bases de dados precisam pegar o melhor modelo que temos e criar
# uma base de 1000 observações corretamente classificadas.

# é bom que isso seja rodado manualmente, porque
# 1) é só uma vez por modelo
# 2) não pode ter classificações erradas
# 3) isso não é usado em aplicações reais. Serve só para as simulações

# parametros --------------------------------------------------------------

fonte <- "jucesp"
ntry <- 1

# carregar modelo ---------------------------------------------------------

usethis::ui_info("Construindo base de teste...")

path <- stringr::str_glue("data-raw/simulacao/{fonte}") |>
  fs::dir_ls() |>
  sort() |>
  dplyr::last()

path_oraculo <- stringr::str_replace(path, '/simulacao/', '/simulacao_oraculo/')
path_test <- stringr::str_glue("{dirname(path_oraculo)}/test")

fs::dir_create(path_test)

path_melhor_modelo <- path |>
  docfun::encontrar_melhor_modelo() |>
  purrr::pluck("file")

melhor_modelo <- luz::luz_load(path_melhor_modelo)

acc <- melhor_modelo |>
  purrr::pluck("records", "metrics", "valid") |>
  dplyr::last() |>
  purrr::pluck("captcha acc") |>
  scales::percent(accuracy = .001)

usethis::ui_info("Melhor modelo: {path_melhor_modelo}\nAcurácia: {acc}")

# download ----------------------------------------------------------------

(n_atual <- length(fs::dir_ls(path_test)))

captcha_access_fun <- eval(parse(
  text = stringr::str_glue("captchaDownload:::captcha_access_{fonte}")
))
captcha_test_fun <- eval(parse(
  text = stringr::str_glue("captchaDownload:::captcha_test_{fonte}")
))

if (n_atual == 1000) {
  usethis::ui_done("Já foi criada!")
} else {
  inv <- purrr::rerun(
    .n = 1000 - n_atual,
    captchaDownload::captcha_oracle(
      path = path_test,
      model = melhor_modelo,
      max_ntry = ntry,
      manual = TRUE,
      captcha_access = captcha_access_fun,
      captcha_test = captcha_test_fun
    )
  )
}


# verificacoes ------------------------------------------------------------

# deletar classificados errados

arquivos_errados <- fs::dir_ls(path_test, regexp = "_1\\.(png|jpeg)", invert = TRUE)

# caso precise deletar os arquivos
fs::file_delete(arquivos_errados)

# caso precise corrigir os casos antigos
# quando o captcha não aceite dois chutes

arquivos_errados_arrumados <- arquivos_errados |>
  stringr::str_replace("_0(?=\\.jpeg$)", "_1")

fs::file_move(arquivos_errados, arquivos_errados_arrumados)

## não preciso arrumar os logs!!! é a base de teste.
# rstudioapi::navigateToFile(logs_arrumar[1])
