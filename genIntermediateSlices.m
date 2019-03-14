function intVolume= genIntermediateSlices(volume, times)

[dx, dy, dz] = size(volume);

newDz = dz * times -times + 1;
intVolume = zeros(dx,dy,newDz);
for i = 1: dz
    intVolume(:,:,i*times-times +1) = volume(:,:,i);
end

for i = 1:dz-1
    for j = 2:times
        intVolume(:,:,i*times-times +j) = intVolume(:,:,i*times-times +1) *(1-(j-1)/times) + intVolume(:,:,i*times +1) * (j-1)/times; 
    end
end