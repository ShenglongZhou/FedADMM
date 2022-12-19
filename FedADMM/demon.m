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

k0          = 10;
pars.rho    = 0.5;
out         = FedADMM(dim,n,A,b,k0,prob,pars);
PlotObj(out.OBJ,k0)
