function N = logistic_growth(No, r, K, times)
    N = K./(1+(K/No - 1)*exp(-r*times));
end