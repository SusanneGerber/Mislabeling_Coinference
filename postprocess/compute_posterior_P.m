function compute_posterior_P( mean_P, P_opt )

[x_pl,x_mi] = EmpConfIntArray(mean_P,P_opt',0.95);
figure;
errorbar(1:length(mean_P),mean_P,x_mi,x_pl);

end

