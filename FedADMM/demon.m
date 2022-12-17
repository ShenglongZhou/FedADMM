clc; clear; close all;
addpath(genpath(pwd));

 
Prob = {'LinReg','LogReg'};
prob  = Prob{1};

m     = 100;
switch prob 
    case 'LinReg' 
        n     = 100;
        var1  = 1/3;
        var2  = 1/3;
    case 'LogReg'
        var1  = load('toxicity.mat').X; 
        var2  = load('toxicityclass.mat').y; 
        n     = size(var1,2);
end 

[A,b,dim,n] = DataGeneration(prob,m,n,var1,var2); 
out         = FedADMM(dim,n,A,b,10,prob);
