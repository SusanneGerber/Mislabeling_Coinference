function [f,df] = glm_linear(P,Xs)
%GLM_LINEAR linear link function
%   [f,df] = GLM_LINEAR(P,Xs)
%   computes values of linear link function and gradients in point Xs with parameters P
%          f_i(P,r) = min(max(0,P*Xs(i,:)'),1)
%   and df(:,i) stores the gradient of f_i in Xs(i,:).
%
%  INPUT:
%   P - values of model parameters
%   Xs - data, if Xs consists of more points (where each row represents one point)
%        then this functions computes function value and gradient for each point
%
%  OUPUT:
%   f - column vector of function values
%   df - matrix of gradients (stored in columns)
%
%  See also GLM_LINEAR2.
%
% Gerber S., Pospisil L., Fournier D., Torkamani A., Rueda M., Horenko I.
% Published under MIT License, 2017-2018
%

N_points = size(Xs,1); % number of points

% allocate (initialize) output variables
f=zeros(N_points,1);
df=zeros(size(Xs'));

% compute value for each row of Xs (for each data point)
for i=1:N_points
    f(i)=min(max(0,P*Xs(i,:)'),1);
    if and(P*Xs(i,:)'>0,P*Xs(i,:)'<1)
        df(:,i)=Xs(i,:)';
    else
        df=0*Xs(i,:)';
    end
end

end

