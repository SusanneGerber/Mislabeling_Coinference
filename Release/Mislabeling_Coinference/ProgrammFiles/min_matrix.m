            function [val, x, y] = min_matrix(A)
                [v, x1] = min(A); 
                [val, y] = min(v); 
                x = x1(y); 
            end
