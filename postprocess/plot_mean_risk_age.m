function plot_mean_risk_age( AgeIntervals,mean_risk_age,x_mi,x_pl )

figure;
hold on
errorbar(AgeIntervals,mean_risk_age,x_mi,x_pl)
title('Mean risk age')
xlabel('mean risk age')
ylabel('probability')
hold off

end

