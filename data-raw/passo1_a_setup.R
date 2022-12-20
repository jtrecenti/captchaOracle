# pastas e objetos --------------------------------------------------------

# mudar o captcha aqui
fonte <- "cadesp"

data_path <- glue::glue("data-raw/{fonte}/img")
sim_path <- glue::glue("data-raw/simulacao/{fonte}")

# faz uma amostra da base completa e copia para uma pasta -----------------

copy_captcha_subset <- function(size, files, path) {
  path_subset <- glue::glue("{path}/{sprintf('%05d', size)}/img/")
  if (!fs::dir_exists(path_subset)) {
    fs::dir_create(path_subset)
    fs::file_copy(files[1:size], path_subset)
  }
}

# cria as pastas com os subsets -------------------------------------------

if (fs::dir_exists(data_path)) {
  all_files <- fs::dir_ls(data_path)
  sizes <- round(quantile(seq_along(all_files), 1:10/10, names = FALSE))
  purrr::walk(sizes, copy_captcha_subset, all_files, sim_path)
}

# rodar simulacoes --------------------------------------------------------

chunks <- sort(basename(fs::dir_ls(sim_path)), decreasing = TRUE)
message(paste(chunks, collapse = " "))
purrr::walk(chunks, ~{
  system(glue::glue("Rscript data-raw/passo1_grid.R '{sim_path}/{.x}/'"))
})
