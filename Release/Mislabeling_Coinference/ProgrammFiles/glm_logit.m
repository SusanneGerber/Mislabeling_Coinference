function [f,df] = glm_logit(P,Xs)
%GLM_LOGIT logit function
%   [f,df] = GLM_LOGIT(P,Xs)
%   computes values of logit function and gradients in points Xs with parameters P
%          f_i(P,r) = 1./(1+exp(-P*Xs(i,:)'))
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
%  See also GLM_LOGIT2.
%
% Gerber S., Pospisil L., Fournier D., Torkamani A., Rueda M., Horenko I.
% Published under MIT License, 2017-2018
%

N_points = size(Xs,1); % number of datapoints is the number of rows

% allocate (initialize) output variables
f=zeros(N_points,1);
df=zeros(size(Xs'));

% compute value for each row of Xs (for each data point)
for i=1:N_points
    f(i)=1./(1+exp(-P*Xs(i,:)'));
    df(:,i)=-1/(1+exp(-P*Xs(i,:)'))^2*(exp(-P*Xs(i,:)'))*(-Xs(i,:)'); 
end

end

