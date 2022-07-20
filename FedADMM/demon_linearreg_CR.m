clc; clear; close all;
addpath(genpath(pwd));
 
m     = 64;
n     = 100; 
di    = randi([50 150],1,m); 
[A,b] = GenerateData(m,n,di,1/3,1/3);  
I     = randperm(sum(di));
A     = A(I,:);  % shuffle samples
b     = b(I,:); 

k0         = [1 5 10 15 20];
out        = cell(1,nnz(k0)); 
for i      = 1:nnz(k0)
    out{i} = FedADMMLin(di,n,A,b,k0(i));  
end

figure('Renderer', 'painters', 'Position',[800 200 504 460]);
axes('Position', [0.115 0.11 0.86 0.85] ); 
styles  = {'--','-.','-','-.','--'};
colors = {'#173f5f','#20639b','#3caea3','#f6d55c','#ed553b'};
for i = 1 : nnz(k0)
    mi = length(1:k0(i):out{i}.iter);
    pl = semilogx(1:mi,out{i}.objx(1:k0(i):end),styles{i}); hold on
    pl.Color     = colors{i};
    pl.LineWidth = 2; 
end
ylabel('Objective'); xlabel('CR'); grid on
legend('k_0=1',  'k_0=5',  'k_0=10', 'k_0=15', 'k_0=20');

