clc; clear; close all;
addpath(genpath(pwd));

Prob = {'LinReg','LogReg'};
prob  = Prob{1};

m     = 100;
switch prob 
    case 'LinReg' 
        n       = 100;
        var1    = 1/3;
        var2    = 1/3;
        pars.r0 = 0.5; %Increase this value if you find the solver diverges
    case 'LogReg'
        var1    = load('toxicity.mat').X; 
        var2    = load('toxicityclass.mat').y; 
        n       = size(var1,2); 
        pars.r0 = 0.05; %Increase this value if you find the solver diverges

end 
[A,b,dim,n] = DataGeneration(prob,m,n,var1,var2); 
 
k0         = [1 5 10 15 20];
out        = cell(1,nnz(k0)); 
for i      = 1:nnz(k0)
    out{i} = FedADMM(dim,n,A,b,k0(i),prob,pars);  
end

figure('Renderer', 'painters', 'Position',[800 200 504 460]);
axes('Position', [0.115 0.11 0.86 0.85] ); 
styles  = {'--','-.','-','-.','--'}; 
colors = {'#173f5f','#20639b','#3caea3','#f6d55c','#ed553b'};
for i = 1 : nnz(k0)
    mi = length(1:k0(i):out{i}.iter);
    pl = semilogx(1:mi,out{i}.OBJ(1:k0(i):end),styles{i}); hold on
    pl.Color     = colors{i};
    pl.LineWidth = 2; 
    leg{i} = strcat('k_0=', num2str(k0(i)));
end
ylabel('Objective'); xlabel('CR'); legend(leg); grid on


