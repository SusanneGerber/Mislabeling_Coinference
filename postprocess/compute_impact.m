function [ Impact, Impact_mean, Impact_conf_int ] = compute_impact( mean_P, X, GLM_function )

N_X = length(mean_P);
N_cohorts = numel(X);

% initialize variables
Impact = cell(1,N_X);
Impact_mean = zeros(1,N_X);
Impact_conf_int = zeros(1,N_X);

for i = 2:N_X
    kkk = 1;
    for n_anneal = 1:1
        ppp0 = mean_P';ppp0(i) = 0;
        ppp1 = mean_P';
        for n = 1:N_cohorts
            for t = 1:size(X{n},1)
                if X{n}(t,i)>0
                    [xx0] = feval(GLM_function,ppp0(1:N_X)',X{n}(t,:));
                    [xx1] = feval(GLM_function,ppp1(1:N_X)',X{n}(t,:));
                    Impact{i}(kkk) = xx1-xx0;
                    kkk = kkk+1;
                end
            end
        end
    end
    Impact_mean(i) = mean(Impact{i});
    Impact_conf_int(i) = 1.96*std(Impact{i});
end


end

