utils::globalVariables(c("super", ".."))

fake_stringr <- function() {
  # this forces a dependency on stringr package, which is used in the
  # dataset function. We need to do this because R CMD CHECK can not
  # detect dependencies inside torch modules.
  stringr::str_detect
}
