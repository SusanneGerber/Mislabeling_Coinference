function [f,df] = glm_logit2(P,Xs)
%GLM_LOGIT2 logit function (vectorized implementation)
%   [f,df] = GLM_LOGIT2(P,Xs)
%   computes values of logit function and gradients in points Xs with parameters P
%          f_i(P,r) = 1./(1+exp(-P*Xs(i,:)'))
%   and df(:,i) stores the gradient of f_i in Xs(i,:). This implementation
%   is generally faster then GLM_LOGIT (which is based on simple for cycle).
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
%  See also GLM_LOGIT.
%
% Gerber S., Pospisil L., Fournier D., Torkamani A., Rueda M., Horenko I.
% Published under MIT License, 2017-2018
%


temp = exp(-Xs*P'); % auxiliary vector
f = 1./(1+temp);

% multiply each column of Xs by respective constants in (f.^2).*temp vector
df = bsxfun(@times,Xs,(f.^2).*temp)';

end

