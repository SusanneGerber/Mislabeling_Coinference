function [x_pl,x_mi]=EmpConfIntArray(x,x_unc,CI);
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
DIM=length(x);
for d=1:DIM
    dist_pl=[];
    dist_mi=[];
    for n=1:size(x_unc,1) 
        dist=x_unc(n,d)-x(d);
        if dist>0
            dist_pl=[dist_pl dist];
        else
            dist_mi=[dist_mi -dist];
        end
    end
    if length(dist_pl)>0
        x_pl(d)=quantile(dist_pl,CI);
    else
        x_pl(d)=0;
    end
    if length(dist_mi)>0
        x_mi(d)=quantile(dist_mi,CI);
    else
        x_mi(d)=0;
    end
end

