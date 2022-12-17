function [A,b,dim,n] = DataGeneration(prob,m,n,var1,var2)

rng('shuffle'); 

switch prob
    case 'LinReg'
        dim  = randi([50 150],1,m);
        d    = sum(dim);  
        A    = randn(d,n);
        b    = randn(d,1); 
        rate = var1+var2; 
        if  rate > 0
            loc  = [1 zeros(1,m-1)]; 
            for i = 1 : m-1
                loc(i+1)=loc(i)+dim(i);
            end  
            T0  = randperm(m);

            T   = T0(1:ceil(var1*m));
            for j   = 1:ceil(var1*m)   % student t distribution
                ind = loc(T(j)):loc(T(j))+dim(j)-1;
                A(ind,:) = trnd(5,dim(j),n);
                b(ind)   = trnd(5,dim(j),1);
            end

            T   = T0(ceil(var1*m)+1:ceil(rate*m));
            for j   = 1:ceil(rate*m)-ceil(var1*m)   % uniform distribution
                ind = loc(T(j)):loc(T(j))+dim(j)-1;
                A(ind,:) = -5+10*rand(dim(j),n);
                b(ind)   = -5+10*rand(dim(j),1);
            end
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

