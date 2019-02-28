function plot_P( P, myxlabels, mytitle )


figure
hold on
title(mytitle)
plot(P);
ylabel('value of parameter')

set(gca,'XTickMode','manual')
set(gca,'XTickLabelMode','manual')
set(gca,'XTick',1:numel(myxlabels))
set(gca,'XTickLabel',myxlabels)
set(gca,'XTickLabelRotation',90)
set(gca,'XMinorTick','off')

grid on
hold off

set(gcf, 'Position', [0, 0, 700, 450]) % default size of figure
movegui('center') % move gui to center of screen

end

