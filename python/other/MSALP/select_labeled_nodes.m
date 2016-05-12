function [lb_idx] = select_labeled_nodes(Y,num_lb)

  lb_idx = [];
  for c = 1:size(Y,2)
    idx = find(Y(:,c) > 0);
    rand_idx = randperm(length(idx));
    lb_idx = [lb_idx; idx(rand_idx(1:num_lb))];
  end

