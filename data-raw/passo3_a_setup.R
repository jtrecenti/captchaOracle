# Esse script contém todas as chamada das simulações
# Alguns captchas contém menos chamadas porque os modelos iniciais chegaram em
# acurácias muito altas com poucos dados. Como cortamos os modelos com mais
# de 50% de acurácia, algumas simulações foram descartadas

# pastas de simulação contêm a estrutura
# "data-raw/simulacao_oraculo/{fonte}/{n_train}{n_try}",
# onde fonte é o captcha, n_train é a quantidade de dados de treino do modelo
# inicial e n_try é a quantidade de tentativas.

# tjmg --------------------------------------------------------------------

system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/tjmg/00100_10'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/tjmg/00100_01'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/tjmg/00100_05'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/tjmg/00201_01'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/tjmg/00201_05'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/tjmg/00201_10'", wait = FALSE)

# trf5 --------------------------------------------------------------------

system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/trf5/00101_01'", wait = FALSE)

# trt --------------------------------------------------------------------

system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/trt/0500_01'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/trt/0500_05'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/trt/0500_10'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/trt/0750_01'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/trt/0750_05'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/trt/0750_10'", wait = FALSE)

# esaj --------------------------------------------------------------------

system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/esaj/00901_05/'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/esaj/00901_10/'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/esaj/00901_01/'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/esaj/01201_01/'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/esaj/01201_05/'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/esaj/01201_10/'", wait = FALSE)

# jucesp --------------------------------------------------------------------

system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/jucesp/00801_01/'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/jucesp/01201_01/'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/jucesp/01601_01/'", wait = FALSE)


# tjpe --------------------------------------------------------------------

system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/tjpe/00801_01/'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/tjpe/00801_05/'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/tjpe/00801_10/'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/tjpe/00401_01/'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/tjpe/00401_05/'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/tjpe/00401_10/'", wait = FALSE)

# tjrs --------------------------------------------------------------------

system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/tjrs/00300_01/'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/tjrs/00300_05/'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/tjrs/00300_10/'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/tjrs/00200_01/'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/tjrs/00200_05/'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/tjrs/00200_10/'", wait = FALSE)

# cadesp --------------------------------------------------------------------

system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/cadesp/00583_01/'", wait = FALSE)

# sei ---------------------------------------------------------------------

system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/sei/03001_01/'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/sei/03001_05/'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/sei/03001_10/'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/sei/04001_01/'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/sei/04001_05/'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/sei/04001_10/'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/sei/05000_01/'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/sei/05000_05/'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/sei/05000_10/'", wait = FALSE)

# rfb -------------------------------------------------------------------

system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/rfb/01372_01'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/rfb/01372_03'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/rfb/01372_05'", wait = FALSE)


# rcaptcha2 ---------------------------------------------------------------------

system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/rcaptcha2/01601_10'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/rcaptcha2/02401_01'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/rcaptcha2/02401_05'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/rcaptcha2/02401_10'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/rcaptcha2/00801_01'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/rcaptcha2/00801_05'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/rcaptcha2/00801_10'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/rcaptcha2/01601_01'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/rcaptcha2/01601_05'", wait = FALSE)


# rcaptcha4 ---------------------------------------------------------------

system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/rcaptcha4/04000_01'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/rcaptcha4/04000_05'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/rcaptcha4/04000_10'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/rcaptcha4/03201_01'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/rcaptcha4/03201_05'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/rcaptcha4/03201_10'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/rcaptcha4/04800_01'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/rcaptcha4/04800_05'", wait = FALSE)
system("Rscript data-raw/passo3_simular.R 'data-raw/simulacao_oraculo/rcaptcha4/04800_10'", wait = FALSE)

