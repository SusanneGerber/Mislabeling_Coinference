function [ P_opt_Cidx ] = find_P_Cidx( P, Cidx  )

% initialize output variables
N_bootstrap = size(P,4);
P_opt_Cidx = zeros(size(P,2),N_bootstrap);
for idx_bootstrap = 1:N_bootstrap
    % find the P corresponding to Cidx for this bootstrap
    P_opt_Cidx(:,idx_bootstrap) = squeeze(P(Cidx,:,1,idx_bootstrap));
end

end

