calcular_oraculo <- function(x) {
  x |>
    basename() |>
    stringr::str_extract("(?<=_[0-9a-zA-Z]{2,10}_)[0-9](?=\\.)") |>
    as.integer() |>
    torch::torch_tensor()
}

# classify_heuristica <- function(file, model, heuristica) {
#
#   oraculo <- calcular_oraculo(file)
#
#   if (as.numeric(oraculo) == 1) {
#
#     label <- file |>
#       basename() |>
#       stringr::str_extract("(?<=_)[0-9a-zA-Z]+")
#
#     res <- captcha::classify(file, label, rm_old = TRUE)
#
#   } else {
#
#     x_oraculo <- captcha::calcular_x(file, model$parm$input_dim)$unsqueeze(2) |>
#       captcha::valid_transforms() |>
#       torch::as_array() |>
#       torch::torch_tensor()
#
#     model$eval()
#     model$to(device = "cpu")
#     parm <- model$parm
#
#     ans_oraculo <- x_oraculo$to(device = "cpu") |>
#       model() |>
#       heuristica(oraculo) |>
#       torch::torch_max(dim = 3) |>
#       purrr::pluck(2)
#
#     pred_oraculo <- apply(
#       as.matrix(ans_oraculo$to(device = "cpu")), 1, function(x) {
#         paste(model$parm$vocab[x], collapse = "")
#       })
#
#     res <- captcha::classify(file, pred_oraculo, rm_old = TRUE)
#
#   }
#
#   res
#
# }
