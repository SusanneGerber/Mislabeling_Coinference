function [ mean_AICc, mean_LogL, mean_LogL_valid ] = compute_mean( AICcs, LogLs, LogLs_valid  )

% initialize output variables
N_bootstrap = size(AICcs,3);
mean_AICc = zeros(size(AICcs,1),size(AICcs,2));
mean_LogL = zeros(size(LogLs,1),size(LogLs,2));
mean_LogL_valid = zeros(size(LogLs_valid,1),size(LogLs_valid,2));

for idx_bootstrap = 1:N_bootstrap
    % update computation of mean value
    mean_AICc = mean_AICc + (1/N_bootstrap)*AICcs(:,:,idx_bootstrap);
    mean_LogL = mean_LogL + (1/N_bootstrap)*LogLs(:,:,idx_bootstrap);
    mean_LogL_valid = mean_LogL_valid + (1/N_bootstrap)*LogLs_valid(:,:,idx_bootstrap);
end

end

