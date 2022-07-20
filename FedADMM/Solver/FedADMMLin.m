function out = FedADMMLin(di,n,A,b,k0,pars)
% This solver solves linear regression problem in the following form:
%
%         min_{x_i,x\in\R^n}  sum_{i=1}^m 0.5||A_ix_i-b_i||^2  
%            s.t.             x_i=x, i=1,2,...,m
%
% where (A_i,b_i) is the data for node/client i
%       A_i\in\R^{d_i-by-n} the measurement matrix
%       b_i\in\R^{d_i-by-1} the observation vector 
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
%   inexact : A binary integer in {0,1}                           (REQUIRED)
%             = 0 for CEADMM; = 1 for ICEADMM (default)
%   pars  :   All parameters are OPTIONAL                                                     
%             pars.r0    --  A scalar in (0,1)
%                            NOTE: Incease this value if you find the solver diverges
%                           (default: = 0.1 for CEADMM; = 0.2 for ICEADMM) 
%             pars.tol   --  Tolerance of the halting condition (default,1e-7*sqrt(n*d))
%             pars.maxit --  Maximum number of iterations (default,50000) 
% =========================================================================
% Outputs:
%     out.sol:      The solution x
%     out.obj:      Objective function value at out.sol
%     out.time:     CPU time
%     out.iter:     Number of iterations 
%     out.comround: Total number of communication rounds
% =========================================================================
% Written by Shenglong Zhou on 20/04/2022 based on the algorithm proposed in
%     Shenglong Zhou and Geoffrey Ye Li,
%     Federated Learning via Inexact ADMM,
%     arXiv:2204.10607, 2022    	
% Send your comments and suggestions to <<< slzhou2021@163.com >>>                                  
% WARNING: Accuracy may not be guaranteed!!!!!  
% =========================================================================
warning off; rng('shuffle'); 

t0  = tic;
if  nargin < 5
    disp(' No enough inputs. No problems will be solverd!'); return;
elseif nargin<6 
    pars   = [];  
end

[m,r0,tol,eps,smprate,maxit] = set_parameters(di,n,k0,pars); 
I      = zeros(m+1,1);
I(1)   = 0;
for j  = 1 : m  
    I(j+1) = I(j)+di(j);
end

ri      = zeros(1,m);
Ai      = cell(1,m);
bi      = cell(1,m);
for j   = 1 : m 
    indj   = I(j)+1:I(j+1);
    sA     = A(indj,:);  
    sb     = b(indj);
    Ai{j}  = sA;  
    bi{j}  = sb;
    if  di(j) >= n
       ri(j) = eigs(sA'*sA,1)/di(j); 
    else
       ri(j) = eigs(sA*sA',1)/di(j);
    end     
end
diw      = 1./di/m;
wri      = ri/m;
sigmai   = r0*wri;
sigma    = sum(sigmai);   
wrsig    = wri+sigmai;  

Fnorm    = @(x)norm(x,'fro')^2;
fun      = @(X)funcX(X,Ai,bi,m,n,diw); 
grad     = cell(1,m);
for j    = 1:m
    grad{j} = @(v)gradXj(v,Ai{j},bi{j},diw(j)); 
end
objx     = zeros(1,maxit);
errx     = zeros(1,maxit); 
X        = zeros(n,m); 
Pi       = X;
Z        = sigmai.*X + Pi;         
m0       = ceil(smprate*m);

fprintf(' Start to run the solver -- FedADMM \n');
fprintf(' -----------------------------------------------------------\n');
fprintf('                          Iter    f(y)      Error     Time  \n');  
fprintf(' -----------------------------------------------------------\n');
 
% main body ------------------------------------------------ 
for iter = 0 : maxit
       
    if  mod(iter, k0)==0              
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
    
    if err < tol && mod(iter,1)==0; break;  end
    
    for j = 1:m
        if  ismember(j,M)
            eps(j)     = 0.95*eps(j);
            syPi       = sigmai(j)*x-Pi(:,j); 
            gXj        = gY(:,j);
            Xj         = x;
            for t      = 1 : 5              
                Xj     = (wri(j)*Xj+syPi-gXj )/wrsig(j);   
                sX     = sigmai(j)*Xj;    
                if Fnorm(syPi-sX-gXj)<eps(j), break, end 
                gXj    = grad{j}(Xj);
            end
            X(:,j)     = Xj;
            Pi(:,j)    = sX - syPi;
            Z(:,j)     = sX + Pi(:,j);      
        end
    end
end

out.sol    = x;
out.obj    = fx;
out.objx   = objx(1:iter+1); 
out.errx   = errx(1:iter+1);  
out.iter   = iter+1;
out.time   = toc(t0);  
out.comrnd = ceil(iter/k0);
fprintf(' -----------------------------------------------------------\n');

end

%--------------------------------------------------------------------------
function [m,r0,tol,eps,smprate,maxit] = set_parameters(di,n,k0,pars) 
    m       = length(di);
    maxit   = 1e4;
    tol     = 5e-5*n/sum(di);  
    eps     = k0^2*ones(m,1);
    r0      = 0.2/log10(9+k0);
    smprate = 0.5;
    if isfield(pars,'r0');      r0      = pars.r0;      end 
    if isfield(pars,'tol');     tol     = pars.tol;     end
    if isfield(pars,'eps');     eps     = pars.eps;    end
    if isfield(pars,'maxit');   maxit   = pars.maxit;   end
    if isfield(pars,'smprate'); smprate = pars.smprate; end  
end

%--------------------------------------------------------------------------
function  [objX,gradX]  = funcX(X,Ai,bi,m,n,w) 
     
    objX     = 0; 
    gradX    = zeros(n,m);
    for j    = 1:m  
        tmp  = Ai{j}*X(:,j)-bi{j};
        objX = objX  + norm( tmp )^2*w(j); 
        if nargout   == 2
           gradX(:,j) =  w(j)*(tmp'* Ai{j} )';
        end
    end
    objX = objX/2;
end

%--------------------------------------------------------------------------
function  gradj = gradXj(x,Aj,bj,wj)  
          gradj = wj*((Aj*x-bj)'* Aj)';
end

