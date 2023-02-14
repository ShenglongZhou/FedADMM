function out = FedADMM(dim,n,A,b,k0,prob,pars)
% This solver solves linear regression problem in the following form:
% This solver solves an optimization problem in the following form:
%
%         min_{xi,x}  (1/m) sum_{i=1}^m fi(xi) 
%            s.t.       xi = x, i=1,2,...,m
%
% where xi\in\R^n and x\in\R^
%
% ------------------------- Linear regression (LinReg)---------------------
% For every i = 1,2,...,m
%
%           fi(xi) = (1/2/di)||Ai*xi-bi||^2   
%
% where (Ai,bi) is the data for node/client i
%       Ai\in\R^{di-by-n} the measurement matrix
%       bi\in\R^{di-by-1} the observation vector 
%
% -------------------- Logistic regression (LogReg)------------------------
% For every i = 1,2,...,m
%
%     fi(xi) = (1/di)*sum_{j=1}^{di}[ ln(1+exp(<ai_j,xi>)-bi_j*<ai_j,xi> ]
%            + (mu/2/di)*||xi||^2 
%
% where (ai_1,...,ai_{di}),(bi_1,...,bi_{di}) are the data for client i
%       ai_j\in\R^{n-by-1}, j=1,...,di
%       bi_j\in\R^{1},      j=1,...,di
%       mu>0, (default value: 0.001)
%
% =========================================================================
% Inputs:
%   dim     : A 1-by-m row vector, dim = (d1, d2, ..., dm)        (REQUIRED)
%             di is the number of rows of Ai, i=1,2,...,m
%             Let d = d1 + d2 + ... + dm
%   n       : Dimension of solution x                             (REQUIRED)
%   A       : A=[A1; A2; ...; Am]\in\R^{d-by-n}                   (REQUIRED)
%   b       : b=[b1; b2; ...; bm]\in\R^{d-by-1}                   (REQUIRED)
%   k0      : A positive integer controlling communication rounds (REQUIRED)
%             The larger k0 is the fewer communication rounds are
%   prob    : must be one of {'LinReg','LogReg'}                  (REQUIRED)
%   pars  :   All parameters are OPTIONAL             
%             pars.rho   --  Participation rate (default: 0.5)
%             pars.r0    --  A positive scalar  (default: 0.5 for LinReg 0.05 for LogReg) 
%                            NOTE: Increase this value if you find the solver diverges   
%             pars.tol   --  Tolerance of the halting condition (default,1e-7)
%             pars.maxit --  Maximum number of iterations (default,1000*k0) 
% =========================================================================
% Outputs:
%     out.sol:      The solution x
%     out.obj:      Objective function value at out.sol
%     out.time:     CPU time
%     out.iter:     Number of iterations 
%     out.cr:       Total number of communication rounds
% =========================================================================
% Written by Shenglong Zhou on 12Dec2022 based on the algorithm proposed in
%     Shenglong Zhou & Geoffrey Ye Li, Federated Learning via Inexact ADMM, 
%     IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023.
% Send your comments and suggestions to <<< slzhou2021@163.com >>>                                  
% WARNING: Accuracy may not be guaranteed!!!!!  
% =========================================================================
warning off; rng('default'); 

t0  = tic;
if     nargin < 6
       disp(' No enough inputs. No problems will be solverd!'); return;
elseif nargin < 7 
       pars   = [];  
end

d = sum(dim);
if size(A,1)~=d  
    fprintf(' Dimensions are not consistent !!! \n No problems will be solved!!!\n'); 
    return;
end


[m,tol,eps,m0,maxit,sigmai,sigma,wri,wrsig,funf,grad] ...
         = set_parameters(dim,n,A,b,k0,prob,pars);
Fnorm    = @(x)norm(x,'fro')^2;
OBJ      = zeros(1,maxit);
ERROR    = zeros(1,maxit); 
W        = zeros(n,m); 
Pi       = W;
Z        = sigmai.*W + Pi;      

fprintf(' Start to run the solver -- FedADMM \n');
fprintf(' -------------------------------------------------------------\n');
fprintf('                          Iter    f(w)      Error      Time  \n');  
fprintf(' -------------------------------------------------------------\n');

 
% main body ------------------------------------------------ 
for iter = 0 : maxit
       
    if  mod(iter, k0)==0              
        w       = sum(Z,2)/sigma;  
        [fw,gW] = funf(w);  
        err     = Fnorm(mean(gW,2));  
        
        M       = randperm(m);
        M       = sort(M(1:m0));  
    end
    
    if iter == 0
        tol = min(tol,0.2*err);
    end 
    
    OBJ(iter+1)   = fw;
    ERROR(iter+1) = err;  
    if mod(iter, k0)==0    
    fprintf(' Communication at iter = %4d %9.4f   %9.3e  %6.3fsec\n',...
              iter, fw, err, toc(t0)); 
    end         
    
    if err < tol && mod(iter,k0)==0; break;  end 
    
    for i   = 1:m
        if  ismember(i,M)
            eps(i)     = 0.95*eps(i);
            syPi       = sigmai(i)*w-Pi(:,i); 
            gXi        = gW(:,i);
            Xi         = w;
            for t      = 1 : 3              
                Xi     = (wri(i)*Xi+syPi-gXi )/wrsig(i);   
                sX     = sigmai(i)*Xi;    
                if Fnorm(syPi-sX-gXi)<eps(i)*n, break, end 
                gXi    = grad{i}(Xi);
            end         
            W(:,i)     = Xi;
            Pi(:,i)    = sX - syPi;
            Z(:,i)     = sX + Pi(:,i);      
        end
    end
end

out.sol    = w;
out.obj    = fw;
out.OBJ    = OBJ(1:iter+1); 
out.ERROR  = ERROR(1:iter+1);  
out.iter   = iter+1;
out.time   = toc(t0);  
out.comrnd = ceil(iter/k0);
fprintf(' -------------------------------------------------------------\n');
fprintf(' Objective:     %10.3f\n',fw); 
fprintf(' Iteration:     %10d\n',iter);
fprintf(' Error:         %10.2e\n',err);
fprintf(' Time:          %7.3fsec\n',out.time);
fprintf(' CR:            %10d\n',out.comrnd);
fprintf(' -------------------------------------------------------------\n');
end

%Set parameters --------------------------------------------------------------------------
function [m,tol,eps,m0,maxit,sigmai,sigma,wri,wrsig,funf,grad] = set_parameters(dim,n,A,b,k0,prob,pars) 
   
    m       = length(dim);
    d       = sum(dim);
    maxit   = 1e3*k0;    
    
    eps     = k0^2*ones(m,1);
    m0      = ceil(0.5*m);
    
    if isequal(prob,'LogReg') 
       r0   = 0.05 + 0.01*(d/n>=1e3);
       tol  = 5e-7*n/m/d; 
    else
       r0   = 0.5/log10(9+k0);
       tol  = 2e-3*n/m/d; 
    end
    if isfield(pars,'r0');    r0    = pars.r0;          end 
    if isfield(pars,'eps');   eps   = pars.eps;         end
    if isfield(pars,'maxit'); maxit = pars.maxit;       end
    if isfield(pars,'rho');   m0    = ceil(pars.rho*m); end  
    if isfield(pars,'tol');   tol   = pars.tol;         end
    
    I       = zeros(m+1,1);
    I(1)    = 0;
    for i   = 1 : m  
        I(i+1) = I(i)+dim(i);
    end

    ri      = zeros(1,m);
    Ai      = cell(1,m);  
    bi      = cell(1,m);
    for i   = 1 : m 
        indj   = I(i)+1:I(i+1);
        sA     = A(indj,:);  
        sb     = b(indj);
        Ai{i}  = sA;  
        bi{i}  = sb;
        if  dim(i) >= n
           ri(i) = eigs(sA'*sA,1)/dim(i); 
        else
           ri(i) = eigs(sA*sA',1)/dim(i);
        end     
    end
    
    if isequal(prob,'LogReg') 
       ri    = ri/5+1e-3./dim; 
    end
    diw      = 1./dim/m;
    wri      = ri/m;
    sigmai   = r0*wri;
    sigma    = sum(sigmai);   
    wrsig    = wri+sigmai; 
    grad     = cell(1,m);  
    switch prob
        case 'LinReg'    
            funf     = @(X)funcLinear(X,Ai,bi,m,n,diw);            
            for i    = 1:m
                grad{i} = @(v)gradLinearClienti(v,Ai{i},bi{i},diw(i)); 
            end 
        case 'LogReg'
            funf     = @(x)funcLogist(x,Ai,bi,m,n,diw,1e-3);  
            for i    = 1:m
                grad{i} = @(v)gradLogistClienti(v,Ai{i},bi{i},diw(i),1e-3); 
            end     
        otherwise
            fprintf( ' ''prob'' is incorrect !!!\n ''porb'' must be one of {''LinReg'',''LogReg''}\n')
    end
    
        
end

%--------------------------------------------------------------------------
function  [objX,gradX]  = funcLinear(x,Ai,bi,m,n,dim) 
     
    objX     = 0; 
    gradX    = zeros(n,m);
    for i    = 1:m  
        tmp  = Ai{i}*x-bi{i};
        objX = objX  + norm( tmp )^2*dim(i); 
        if nargout   == 2
           gradX(:,i) = (tmp'* Ai{i} )'*dim(i);
        end
    end
    objX = objX/2;
end

%--------------------------------------------------------------------------
function  gradj = gradLinearClienti(x,Ai,bi,di)  
          gradj = ((Ai*x-bi)'* Ai)'*di;
end


%--------------------------------------------------------------------------
function  [objX,gradX]  = funcLogist(x,Ai,bi,m,n,dim,lam) 
     
    objX   = 0; 
    gradX  = zeros(n,m);
    for i  = 1:m
        Ax   = Ai{i}*x;  
        eAx  = 1 + exp(Ax);
        objX = objX +  (sum( log(eAx)-bi{i}.*Ax ) + (lam/2)*norm(x,'fro')^2)*dim(i); 
        if nargout   == 2 
           gradX(:,i) =   ( ((1-bi{i}-1./eAx)'*Ai{i})'+lam*x)*dim(i);
        end
    end
end


%--------------------------------------------------------------------------
function  gXj  = gradLogistClienti(x,Ai,bi,di,lam) 
          Ax   = Ai*x;  
          eAx  = 1 + exp(Ax); 
          gXj  = ( ((1-bi-1./eAx)'*Ai)'+lam*x)*di;
end

