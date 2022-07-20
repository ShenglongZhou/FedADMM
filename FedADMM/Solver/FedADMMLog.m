function out = FedADMMLog(di,n,A,b,k0,pars)
% This solver solves logistic regression problem in the following form:
%
%         min_{x_i,x\in\R^n}  sum_{i=1}^m f_i(x_i;(A_i,b_i))  
%            s.t.             x_i=x, i=1,2,...,m
%
% where 
%      f_i(x;(A_i,b_i)) = (mu/2)*||x||^2 
%                       + sum_{j=1}^{d_i} (log(1+exp(<x,(a^i)_j>))-(b^i)_j*<x,(a^i)_j> )
%      (A_i,b_i) is the data for node/client i
%      A_i = [(a^i)_1, (a^i)_2, ..., (a^i)_{d_i}]^T \in\R^{d_i-by-n} 
%      b_i = [(b^i)_1, (b^i)_2, ..., (b^i)_{d_i}]^T \in\R^{d_i-by-1} 
% =========================================================================
% Inputs:
%   di      : A 1-by-m row vector, di = (d_1, d_2, ..., d_m)      (REQUIRED)
%             d_i is the number of rows of A_i
%             Let d = d_1+d_2+...+d_m
%   n       : Dimension of solution x                             (REQUIRED)
%   A       : A=[A_1; A_2; ...; A_m]\in\R^{d-by-n}                (REQUIRED)
%   b       : b=[b_1; b_2; ...; b_m]\in\R^{d-by-1}                (REQUIRED)
%   k0      : A positive integer controlling communication rounds (REQUIRED)
%             The larger k0 is the fewer communication rounds are
%   pars  :   All parameters are OPTIONAL                                                     
%             pars.r0    --  A positive scalar, (default:  max(0.1,8*log(d)/n)) 
%                            NOTE: Incease this value if you find the solver diverges 
%             pars.mu    --  A positive regularization parameter, (default: 0.01) 
%             pars.tol   --  Tolerance of the halting condition (default,1e-7*sqrt(n*d))
%             pars.maxit --  Maximum number of iterations (default,50000) 
% =========================================================================
% Outputs:
%     out.sol:      The solution x
%     out.obj:      Objective function value at out.sol
%     out.acc:      Classification accuracy
%     out.time:     CPU time
%     out.iter:     Number of iterations 
%     out.comround: Total number of communication rounds
% =========================================================================
% Written by Shenglong Zhou on 10/04/2022 based on the algorithm proposed in
%     Shenglong Zhou and Geoffrey Ye Li,
%     Federated Learning via Inexact ADMM,
%     arXiv:2204.10607, 2022    
% Send your comments and suggestions to <<< slzhou2021@163.com >>>                                  
% WARNING: Accuracy may not be guaranteed!!!!!  
% =========================================================================

warning off; rng('shuffle'); %rng(1);

t0  = tic;
if  nargin < 5
    disp(' No enough inputs. No problems will be solverd!'); return;
elseif nargin<6 
    pars   = [];
end

[m,r0,lam,tol,eps,smprate,maxit] = set_parameters(di,n,k0,pars); 
I      = zeros(m+1,1);
I(1)   = 0;
for i  = 1 : m  
    I(i+1) = I(i)+di(i);
end


ri      = zeros(1,m);
Ai      = cell(1,m);
bi      = cell(1,m); 
for j   = 1 : m 
    indj  = I(j)+1:I(j+1);
    sA    = A(indj,:);  
    sb    = b(indj);
    Ai{j} = sA;  
    bi{j} = sb;
    if di(j) >= n
       ri(j) = (eigs(sA'*sA,1)/4+lam)/di(j);   
    else
       ri(j) = (eigs(sA*sA',1)/4+lam)/di(j); 
    end
end
diw    = 1./di/m; 
wri    = ri/m;
sigmai = r0*wri;
sigma  = sum(sigmai); 
wrsig  = wri+sigmai;  

Fnorm     = @(x)norm(x,'fro')^2;
fun       = @(X)func(X,Ai,bi,m,n,diw,lam,Fnorm); 
gradj     = @(v,j)gradXj(v,j,Ai,bi,diw,lam); 
objx      = zeros(1,maxit); 
errx      = zeros(1,maxit);
X         = zeros(n,m);     
Pi        = X;
Z         = sigmai.*X + Pi;  
m0        = ceil(smprate*m); 

fprintf(' Start to run the solver -- FediADMM \n');
fprintf(' -------------------------------------------------------------\n');
fprintf('                          Iter    f(y)      Error      Time  \n');  
fprintf(' -------------------------------------------------------------\n');

% main body --------------------------------------------------
for iter = 0 : maxit
    
    if mod(iter, k0)==0     
       x       = sum(Z,2)/sigma;  
       [fx,gY] = fun(x*ones(1,m)); 
       err     = Fnorm(sum(gY,2));     
       M       = randperm(m);
       M       = sort(M(1:m0));   
    end
    
    objx(iter+1) = fx; 
    errx(iter+1) = err; 
    if mod(iter, k0)==0    
    fprintf(' Communication at iter = %4d %9.4f   %9.3e  %6.3fsec\n',...
              iter, fx, err, toc(t0)); 
    end       
    if err < tol && mod(iter,k0)==0 && iter>1; break;  end      
    
    for j      = 1 : m       
        if  ismember(j,M) 
            eps(j)     = 0.95*eps(j);
            syPi       = sigmai(j)*x-Pi(:,j); 
            gXj        = gY(:,j);
            Xj         = x;
            for t      = 1 : 5              
                Xj     = (wri(j)*Xj+syPi-gXj )/wrsig(j);   
                sX     = sigmai(j)*Xj;    
                if Fnorm(syPi-sX-gXj)<eps(j), break, end 
                gXj    = gradj(Xj,j);
            end
            X(:,j)     = Xj;
            Pi(:,j)    = sX - syPi;
            Z(:,j)     = sX + Pi(:,j); 
        end
    end 
             
end

out.x      = x;
out.obj    = fx;
out.acc    = 1-nnz(b-max(0,sign(A*x)))/length(b); 
out.objx   = objx(1:iter+1); 
out.errx   = errx(1:iter+1);  
out.iter   = iter+1;
out.time   = toc(t0);
out.comrnd = ceil(iter/k0)+1; 

fprintf(' -----------------------------------------------------------\n');

end

%--------------------------------------------------------------------------
function [m,r0,lam,tol,eps,smprate,maxit] = set_parameters(di,n,k0,pars) 
    m        = length(di);
    maxit    = 1e4;
    tol      = 1e-5*n/sum(di);   
    lam      = 1e-3;  
    smprate  = 0.5;
    r0       = 0.1/log10(9+k0);
    eps      = k0^2*ones(m,1); 
    if isfield(pars,'smprate'); smprate = pars.smprate; end 
    if isfield(pars,'r0');      r0      = pars.r0;      end 
    if isfield(pars,'tol');     tol     = pars.tol;     end
    if isfield(pars,'eps');     eps     = pars.eps;     end
    if isfield(pars,'maxit');   maxit   = pars.maxit;   end
end

%--------------------------------------------------------------------------
function  [objX,gradX]  = func(X,Ai,bi,m,n,w,lam,Fnorm) 
     
    objX   = 0; 
    gradX  = zeros(n,m);
    for i  = 1:m
        Ax   = Ai{i}*X(:,i);  
        eAx  = 1 + exp(Ax);
        objX = objX + w(i)* (sum( log(eAx)-bi{i}.*Ax ) + (lam/2)*Fnorm(X(:,i))); 
        if nargout   == 2 
           gradX(:,i) =  w(i)*( ((1-bi{i}-1./eAx)'*Ai{i})'+lam*X(:,i));
        end
    end
     
end


%--------------------------------------------------------------------------
function  gXj  = gradXj(x,j,Ai,bi,w,lam) 
          Ax   = Ai{j}*x;  
          eAx  = 1 + exp(Ax); 
          gXj  =  w(j)*( ((1-bi{j}-1./eAx)'*Ai{j})'+lam*x);
end