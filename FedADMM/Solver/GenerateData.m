function [A,b] = GenerateData(m,n,di,rate1,rate2)
    rng('shuffle'); 
    rate = rate1+rate2;
    d    = sum(di);
	A    = randn(d,n);
	b    = randn(d,1);  
	if  rate > 0
        loc  = [1 zeros(1,m-1)]; 
        for i = 1 : m-1
            loc(i+1)=loc(i)+di(i);
        end  
        T0  = randperm(m);
        
        T   = T0(1:ceil(rate1*m));
        for j   = 1:ceil(rate1*m)   % student t distribution
            ind = loc(T(j)):loc(T(j))+di(j)-1;
            A(ind,:) = trnd(5,di(j),n);
            b(ind)   = trnd(5,di(j),1);
        end

        T   = T0(ceil(rate1*m)+1:ceil(rate*m));
        for j   = 1:ceil(rate*m)-ceil(rate1*m)   % uniform distribution
            ind = loc(T(j)):loc(T(j))+di(j)-1;
            A(ind,:) = -5+10*rand(di(j),n);
            b(ind)   = -5+10*rand(di(j),1);
        end
  end
end

