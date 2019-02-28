function compute_posterior_AICc( mean_AICc, C, risk_compute )

[mean_AICc_min,~,~] = min_matrix(mean_AICc);
weight_posterior = zeros(size(mean_AICc,1),size(mean_AICc,2));
for i = 1:size(mean_AICc,1)
    for j = 1:size(mean_AICc,2)
       weight_posterior(i,j) = exp(-0.5*(mean_AICc(i,j)-mean_AICc_min)); 
    end
end
d_risk = diff(risk_compute);
d_C = diff(C);
[v1,v2] = meshgrid(d_risk,d_C);
[N1,N2] = size(v1);
myintegral = sum(sum(weight_posterior(1:N1,1:N2).*v1));
weight_posterior = weight_posterior./myintegral;
prob_zero_hypothesis = sum(weight_posterior(1:N1,1)'.*d_C.*d_risk(1));
Error_Distr = zeros(1,N2);
for j = 1:N2
    Error_Distr(j) = 0;
    for i = 1:N1
        Error_Distr(j) = Error_Distr(j)+weight_posterior(i,j)*d_C(i);
    end
end

figure
hold on
title('A posterior AICc')
plot(risk_compute(1:N2),Error_Distr,'.-');
xlabel('risk')
ylabel('error distr')
hold off
%mesh(Cg,DRg,weight_posterior')

end

