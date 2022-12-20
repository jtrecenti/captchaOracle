# neste passo, queremos criar as bases teste para as simulações do oráculo
# essas bases de dados precisam pegar o melhor modelo que temos e criar
# uma base de 1000 observações corretamente classificadas.

# é bom que isso seja rodado manualmente, porque
#
# 1) é só uma vez por modelo
# 2) não pode ter classificações erradas
# 3) isso não é usado em aplicações reais. Serve só para as simulações

# parametros --------------------------------------------------------------

# alterar aqui para cada fonte
fonte <- "rcaptcha4"

# ntry não é estritamente necessário, mas podemos usar para facilitar a
# classificação caso o melhor modelo não esteja muito bom
ntry <- 1

# funcoes -----------------------------------------------------------------

# A função encontrar_melhor_modelo() é usada para as simulações e foi
# adicionada em um pacote chamado {docfun}, que contém helpers para o
# doutorado, para não poluir o código. A função foi adicionada aqui também
# para tornar o {captchaOracle} autocontido.


#' Encontra melhor modelo da simulação
#'
#' Encontra melhor modelo partir de uma pasta com arquivos de logs. Função a ser
#'   utilizada internamente no meu doutorado.
#'
#' @param path pasta que contém arquivos de log dos resultados dos modelos
#'
#' @export
encontrar_melhor_modelo <- function(path) {
  path |>
    fs::dir_ls(regexp = "model_[0-9]+\\.log") |>
    purrr::map_dfr(readr::read_csv, show_col_types = FALSE, .id = "file") |>
    dplyr::filter(set == "valid") |>
    dplyr::arrange(dplyr::desc(epoch)) |>
    dplyr::distinct(file, .keep_all = TRUE) |>
    dplyr::slice_max(captcha.acc, n = 1, with_ties = FALSE) |>
    dplyr::mutate(file = fs::path_ext_set(file, ".pt"))
}


# carregar modelo ---------------------------------------------------------

usethis::ui_info("Construindo base de teste...")

# obtém o caminho da pasta com o melhor modelo
# por padrão, pega a pasta com mais dados, já que
# é esperado que o modelo funcione melhor nesse caso
path <- stringr::str_glue("data-raw/simulacao/{fonte}") |>
  fs::dir_ls() |>
  sort() |>
  dplyr::last()

# caminho das pastas de simulação
path_oraculo <- stringr::str_replace(path, '/simulacao/', '/simulacao_oraculo/')
path_test <- stringr::str_glue("{dirname(path_oraculo)}/test")

fs::dir_create(path_test)

# seleciona o melhor modelo a partir de uma lista de captchas classificados
# o critério para selecionar o melhor modelo é a acurácia na base de validação.
# o melhor modelo é o que apresenta maior acurácia na última iteração.
path_melhor_modelo <- path |>
  encontrar_melhor_modelo() |>
  purrr::pluck("file")

# carrega o melhor modelo
melhor_modelo <- luz::luz_load(path_melhor_modelo)

# obtém estatísticas do melhor modelo
acc <- melhor_modelo |>
  purrr::pluck("records", "metrics", "valid") |>
  dplyr::last() |>
  purrr::pluck("captcha acc") |>
  scales::percent(accuracy = .001)

# mostra as estatísticas do melhor modelo
usethis::ui_info("Melhor modelo: {path_melhor_modelo}\nAcurácia: {acc}")

# download ----------------------------------------------------------------


(n_atual <- length(fs::dir_ls(path_test)))

# carrega as funções de acesso e teste para download dos dados
captcha_access_fun <- eval(parse(
  text = stringr::str_glue("captchaDownload:::captcha_access_{fonte}")
))
captcha_test_fun <- eval(parse(
  text = stringr::str_glue("captchaDownload:::captcha_test_{fonte}")
))


if (n_atual == 1000) {
  usethis::ui_done("Já foi criada!")
} else {
  # aqui coloca-se manual=TRUE, já que queremos obter a resposta correta
  inv <- purrr::walk(
    seq_len(1000 - n_atual),
    \(x) captchaDownload::captcha_oracle(
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

# não é necessário mexer nos logs da base de teste.

# bases do rcaptcha -------------------------------------------------------

# as bases do rcaptcha são diferentes pois não precisamos nos preocupar
# com as classificações, já que os dados são gerados. Basta chamar
# a função captcha_generate()

progressr::with_progress({
  p <- progressr::progressor(1000)
  for (i in 1:1000) {
    p()
    x <- captcha::captcha_generate(
      TRUE,
      path = glue::glue("data-raw/simulacao_oraculo/{fonte}/test"),
      n_chars = 4, chars = c(1:9, letters)
    )
    captcha::classify(x$file, 1, rm_old = TRUE)
  }
})
