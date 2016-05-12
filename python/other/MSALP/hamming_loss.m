function loss = hamming_loss(Y,F)

  [dummy label] = max(Y,[],2);
  [dummy est] = max(F,[],2);
  loss = sum(label~=est);