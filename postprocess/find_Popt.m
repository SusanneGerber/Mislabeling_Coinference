function [ P_opt ] = find_Popt( P, LogLs_valid  )

% initialize output variables
N_bootstrap = size(LogLs_valid,3);
P_opt = zeros(size(P,2),N_bootstrap);

for idx_bootstrap = 1:N_bootstrap
    % find the best (with respect to LogLs_valid) P for this bootstrap
    [~,i,j] = min_matrix(LogLs_valid(:,:,idx_bootstrap)');
    P_opt(:,idx_bootstrap) = squeeze(P(j,:,i,idx_bootstrap));
end

end

