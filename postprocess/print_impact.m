function print_impact( Impact_mean, Impact_conf_int, label )

N_X = length(Impact_mean);

[~,ii] = sort(abs(Impact_mean),'descend');
for i = 1:N_X-1
    disp(['i = ' num2str(ii(i)) ', average impact of ' label{ii(i)} ' on risk is ' ...
        num2str(Impact_mean(ii(i))) ' +/- '...
        num2str(Impact_conf_int(ii(i))) ';']);
end


end

