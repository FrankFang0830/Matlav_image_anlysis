function [fv]=getSiftFv(sift, A, gmm)
kd = 16;  nc = 32;
% fisher vec
dsift_fv = zeros(1, kd*nc);
% 2*3=6 gmm matrix on this file and pick one to calculate
fv = vl_fisher(A(1:kd, :) * double(sift), gmm(1, 1).m, gmm(1, 1).cov,  gmm(1, 1).p);
dsift_fv(1, :) = fv(1:kd * nc);
fprintf('\n %d sift ', 1)
end

