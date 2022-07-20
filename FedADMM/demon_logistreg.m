clc; clear; close all;
addpath(genpath(pwd));

dat     = load('toxicity.mat'); 
lab     = load('toxicityclass.mat'); 
A       = Normalization(dat.X,3); 
b       = lab.y;
b(b~=1) = 0;
[d,n]   = size(A);  

I       = randperm(d);
A       = A(I,:);  % shuffle samples
b       = b(I,:);  
m       = 64;     % divide smaples into m groups
while 1
    idx = unique([randperm(d-2,m-1)+1 d]);
    di  = idx-[0 idx(1:end-1)]; 
    if min(di)>0.1*d/m; break; end
end

k0    = 10;
out   = FedADMMLog(di,n,A,b,k0) 
plotobj(out.objx)