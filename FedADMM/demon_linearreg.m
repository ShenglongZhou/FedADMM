clc; clear; close all;
addpath(genpath(pwd));
 
m     = 64;
n     = 100; 
di    = randi([50 150],1,m); 
[A,b] = GenerateData(m,n,di,1/3,1/3);  
I     = randperm(sum(di));
A     = A(I,:);  % shuffle samples
b     = b(I,:); 

k0    = 10;
out   = FedADMMLin(di,n,A,b,k0) 
plotobj(out.objx)

