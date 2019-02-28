function [ A, Aci, Az, Aciz ] = compute_auc( mean_P, X, Y, Z, GLM_function )

N_cohorts = numel(Y);
phi = [];
y = [];z = [];
for n = 1:N_cohorts
    for t = 1:size(Y{n},1)
        [xx] = feval(GLM_function,mean_P,X{n}(t,:));
        phi = [phi xx];
        y = [y Y{n}(t)];
        if Z{n}(t) == 4
            z = [z 0.3];
        elseif Z{n}(t) == 5
            z = [z 0.95];
        else
            z = [z 0];
        end
    end
end

figure;
hold on
plot(y,'b');
plot(phi,'r--.');
legend('y','phi');
hold off

[A,Aci] = auc([y' phi'],0.05,'hanley');
[Az,Aciz] = auc([y' z'],0.05,'hanley');

end

