function [v,d,h] = LaplacianMin(x,L);

v = 0.5*x'*L*x;
if nargout>1
  d = (L+L')*x;
  if nargout>2
     h = L+L'; 
  end
end

