function NX = normalization( X, normal_type)
% This file aims at normalize the input matrix X;
% Inputs:
%       X           -- an (m x n) order matrix that will be normalized;
%       normal_type -- type of normalization
%                      =0; no normalization, i.e., NX=X;
%                      =1; sample(row)-wise and then feature(column)-wise
%                          normalization. This is better for the case when n<1000;
%                      =2; feature(column)-wisely scaled to [-1,1], typically for logistic regression problem
%                      =3; feature(column)-wisely scaled to unit norm columns, typically for CS problem
% Outputs:
%       NX           --  normalized m x n order matrix
% written by Shenglong Zhou, 13/10/2018

t0   = tic;
if normal_type==0                              % No normalization
    NX = X; 
    
elseif normal_type==1
    C    = bsxfun(@minus, X, mean(X,2));
    Yrow = bsxfun(@rdivide, C, std(X,0,2));    % Sample-wise  normalization
    Y    = Yrow';
    D    = bsxfun(@minus, Y, mean(Y,2));
    Ycol = bsxfun(@rdivide, D, std(Y,0,2));    % Feature-wise normalization
    NX   = Ycol';
    if max(max(isnan(NX))) 
        nX  = 1./sqrt(sum(X.*X));
        lX  = length(nX);
        NX  = X*sparse(1:lX,1:lX, nX,lX,lX);
    end
else
    if normal_type==2     
    nX = 1./max(abs(X),[],1);          % Feature-wisely scaled to [-1,1],  
    else                                  % typically for logistic regression problem.
    nX = 1./sqrt(sum(X.*X));              % Feature-wisely scaled to has unit norm columns, 
    end                                   % typically for CS problem.
    
    lX = length(nX);
    if lX <= 10000
        NX  = X*sparse(1:lX,1:lX, nX,lX,lX);             
    else                                  % If lX is too large, seperate X into        
        k  = 5e3;                         % sveral smaller sub-matrices. 
        if nnz(X)/lX/lX<1e-4; k = 1e5; end         
        K      = ceil(lX/k); 
        for i  = 1:K-1
        T      = ((i-1)*k+1):(i*k);
        X(:,T) =  X(:,T)*sparse(1:k,1:k,nX(T),k,k); 
        end
        T      = ((K-1)*k+1):lX;
        k0     = length(T);
 
        X(:,T) = sparse(X(:,T))*sparse(1:k0,1:k0,nX(T),k0,k0);
        NX     = X;
    end
end
NX(isnan(NX))=0;
fprintf(' Nomorlization used %2.4f seconds.\n',toc(t0)); 

end