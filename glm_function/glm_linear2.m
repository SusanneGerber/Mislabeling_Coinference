function [f,df] = glm_linear2(P,Xs)
%GLM_LINEAR2 linear link function (vectorized implementation)
%   [f,df] = GLM_LINEAR2(P,Xs)
%   computes values of linear link function and gradients in points Xs with parameters P
%          f_i(P,r) = min(max(0,P*Xs(i,:)'),1)
%   and df(:,i) stores the gradient of f_i in Xs(i,:). This implementation
%   is generally faster then GLM_LINEAR (which is based on simple for cycle).
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
%  See also GLM_LINEAR.
%
% Gerber S., Pospisil L., Fournier D., Torkamani A., Rueda M., Horenko I.
% Published under MIT License, 2017-2018
%

temp = Xs*P'; % auxiliary vector
f=min(max(0,temp),1);

df = Xs';
df(:,temp<=0 | temp>=1) = 0;

end

