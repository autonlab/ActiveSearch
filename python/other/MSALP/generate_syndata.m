function [X Y] = generate_syndata(noise_level)

  n = 400;
  nc = 4;
  
  nsub = round(n/nc)*ones(nc,1);
  nsub(nc) = nsub(nc) + (n - sum(nsub));

  X = [];
  Y = [];
  fprintf('starting\n');
  for k = 1:nc
    phi = sort(1.5*linspace(0,1,nsub(k))*pi);
    radi = sqrt(pi+phi) - sqrt(pi);
    rot = (k-1)*(2*pi)/nc;
    R = [cos(rot) -sin(rot); sin(rot) cos(rot)];

    Xsub = [];
    Ysub = [];
    for i = 1:nsub(k)
      Xsub(2,i) = radi(i)*cos(phi(i)) + 0.1;
      Xsub(1,i) = radi(i)*sin(phi(i)) + 0.05;
      Xsub(:,i) = R * Xsub(:,i);
      Ysub(i,:) = zeros(1,nc);
      Ysub(i,k) = 1;
    end
    X = [X Xsub];
    Y = [Y; Ysub];
  end

  fprintf('loops done\n')
  ridx = randperm(n);
  n1 = round(n/4);
  if noise_level == 1
    X(:,ridx(1:n1)) = X(:,ridx(1:n1)) + 0.01*randn(2,n1); 
  else
    X(:,ridx(1:n1)) = X(:,ridx(1:n1)) + 0.05*randn(2,n1); 
  end
  X(:,ridx(n1+1:end)) = X(:,ridx(n1+1:end)) + 0.01*randn(2,n-n1);

  fprintf('done\n')
  % figure;
  % hold on;
  % marker = {'o','+','^','x'};
  % col = {[1 0 0], [0.6 0 0], [0.3 0 0], [0 1 1]};
  % for i = 1:nc
  %   plot(X(1,find(Y(:,i))),X(2,find(Y(:,i))),marker{i},'Color',col{i},'markersize',8);
  % end
  % title('Dataset (with true labels)');
  % hold off;
