function [val, x, y] = min_matrix(A)
%MIN_MATRIX find minimal element in given matrix
%   [val,x,y] = MIN_MATRIX(A)
%
%  INPUT:
%   A - general real matrix
%
%  OUPUT:
%   value - value of minimal element
%   x - the index of row with minimal value
%   y - the index of column with minimal value
%       i.e., value = A(x,y)
%
% Gerber S., Pospisil L., Fournier D., Torkamani A., Rueda M., Horenko I.
% Published under MIT License, 2017-2018
%

    [v, x1] = min(A); 
    [val, y] = min(v); 
    x = x1(y); 
end
