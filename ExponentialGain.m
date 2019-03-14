function [intCorFactor] = ExponentialGain(N, NA, lamda, gain)
x =0:N-1;
y = NA*exp(-lamda*x);
y= diff(y);
y(length(y)+1) = y(length(y)); 
y = -min(y) + y + 1;
intCorFactor = min(y) + gain*(y-min(y))/(max(y)-min(y));
end

