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


k0         = [1 5 10 15 20];
out        = cell(1,nnz(k0)); 
for i      = 1:nnz(k0)
    out{i} = FedADMMLog(di,n,A,b,k0(i));  
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

