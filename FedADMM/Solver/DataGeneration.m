function [A,b,dim,n] = DataGeneration(prob,m,n,var1,var2)

rng('shuffle'); 

switch prob
    case 'LinReg'
        dim  = randi([50 150],1,m);
        d    = sum(dim); 
        A    = randn(d,n);  
        b    = randn(d,1);  
        if  var1+var2 > 0
            T1        = ceil(var1*m); % student t distribution 
            A(1:T1,:) = trnd(5,ceil(var1*m),n);
            b(1:T1)   = trnd(5,ceil(var1*m),1);
            T2        =  ceil(var2*m); % uniform distribution 
            A((T1+1):(T1+T2),:) = -5+10*rand(T2,n);
            b((T1+1):(T1+T2))   = -5+10*rand(T2,1);
            I       = randperm(d);
            A       = A(I,:);  % randomize samples
            b       = b(I,:);  
        end
    case 'LogReg' 
        var2(var2==-1)= 0;
        A       = Normalization(var1,3); 
        b       = var2;
        [d,n]   = size(A); 
        I       = randperm(d);
        A       = A(I,:);  % randomize samples
        b       = b(I,:);  
        while 1
            idx = unique([randperm(d-2,m-1)+1 d]);
            dim  = idx-[0 idx(1:end-1)]; 
            if min(dim)>0.025*d/m; break; end
        end 
    otherwise; fprintf( ' ''prob'' is incorrect !!!\n ''porb'' must be one of {''LinReg'',''LogReg''}\n')
end
end
