function plot_mean( C, risk_compute, mean_matrix, mytitle, contourlevel )

[Cg,DRg] = meshgrid(C,risk_compute); % prepare meshgrid for plotting
[~,C_idxmin,risk_idxmin] = min_matrix(mean_matrix); % find the minimum

figure;
hold on
title(mytitle)
contourf(Cg,DRg,mean_matrix',contourlevel);
plot(C(C_idxmin),risk_compute(risk_idxmin),'ro')
xlabel('C')
ylabel('risk level')
ylim([0 0.15])
colorbar
hold off

end

