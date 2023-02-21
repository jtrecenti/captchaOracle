
<!-- README.md is generated from README.Rmd. Please edit that file -->

# captchaOracle

<!-- badges: start -->

[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
<!-- badges: end -->

This package implements new dataset, loss and model operations in a
`torch`/`luz` framework to address the oracle data structure discussed
in [my doctorate thesis](https://jtrecenti.github.io/doutorado) (in
Portuguese).

The main objects of this package are:

- `captcha_oracle()`: creates annotated datasets from an initial model
  and functions to access the oracle of a website.
- `captcha_dataset_oracle()`: This object is the same as
  `captcha_dataset()` from the `{captcha}` package, but deals with a
  different data structure that incorporates incomplete information
  provided by the websites.
- `net_captcha_oracle()`: This object is the same as `net_captcha()`
  from the `{captcha}` package, but can consider an initial model as a
  parameter in its initialization method.
- `oracle_loss()`: This object implements the oracle loss proposed in
  the doctorate thesis, which incorporates the incomplete information
  provided by the oracle in the websites.
- `captcha_accuracy_oracle()`: calculates the accuracy of the model in
  the fitting process dealing with the new data structure provided by
  `captcha_dataset_oracle()`.

The package also has an experimental feature to learn the parameters
online, by automatically accessing the oracle in the web. It is
implemented with the `captcha_dataset_oracle_online()` function. There
is a script [in this
link](https://github.com/jtrecenti/captchaOracle/blob/main/data-raw/passo_extra_iterativo.R)
providing an example of this experimental feature.

This package is experimental and it was built as a tool to make
simulations for my doctorate thesis. It may have breaking changes before
it is considered as stable.

## License

MIT
