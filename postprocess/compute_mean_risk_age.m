function [ AgeIntervals,mean_risk_age,x_mi,x_pl ] = compute_mean_risk_age( AICcs, P_opt, GLM_function )

N_X = size(P_opt,1);
nnn = size(AICcs,3);
AgeIntervals = [20:10:90]; % compute histogram on those intervals
risk_age = zeros(nnn,length(AgeIntervals));
for idx_bootstrap = 1:nnn
    for j = 1:length(AgeIntervals)
        ppp0 = zeros(1,N_X);
        ppp0(1) = P_opt(1,idx_bootstrap);
        ppp0(N_X) = P_opt(N_X,idx_bootstrap);
        X0 = zeros(1,N_X);X0(1) = 1;
        X0(N_X) = AgeIntervals(j)/100;
        [risk_age(idx_bootstrap,j)] = feval(GLM_function,ppp0,X0);
    end
end
mean_risk_age = mean(risk_age,1);
[x_pl,x_mi] = EmpConfIntArray(mean_risk_age,risk_age,0.95);


end

