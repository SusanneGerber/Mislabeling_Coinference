function [val_out, i_out, j_out,l_out] = min_matrix3D(A)

L=size(A,3);

 

for l=1:L

    [val(l), i(l), j(l)] = min_matrix(A(:,:,l));

end

[val_out,l_out]=min(val);

i_out=i(l_out);j_out=j(l_out);