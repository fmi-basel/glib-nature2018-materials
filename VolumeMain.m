clear all; close all; clc

fp = ['.' filesep 'Data' filesep];
outDir = ['.' filesep 'Output' filesep]; % directory of output file

%Type of processing
lowResolution = 0; % 1: yes (xyz have same resolution, 0: no - (generates additional z slices) 
%Filtering parameters
if (lowResolution == 1)
    hsize=5; %Kernel size for gaussian blur
    sigma=3; %Sigma for gaussian blur
    smoothcontour = 15; % number of points used in smoothing
else
    hsize=5; %Kernel size for gaussian blur
    sigma=2.5; %Sigma for gaussian blur
    smoothcontour = 15; %number of points used in smoothing
end

% Parameters
scale = 2;
search_str = '*.tif*';
name_reg = [11:15];
pixel_xy = 0.26;
pixel_z = 2;


% Get image dimension: Assuming all crops have similar sizes
image_names = dir([fp filesep search_str ]);
info = imfinfo([fp filesep image_names(1).name]);
y_dim = info(1).Width;
x_dim = info(1).Height;
NzOrig = size(info, 1);

for i = 1: numel(image_names)
    tpnames{i} = image_names(i).name(name_reg);
end


DIMENSIONSX = round(x_dim/scale);
DIMENSIONSY = round(y_dim/scale);

%Size of voxels in x, y and z
voxelSizeX = pixel_xy*scale ;
voxelSizeY = pixel_xy*scale ;
voxelSizeZ = pixel_z;
nBlobs = 1; %Maximum number of object to be analyzed

tpnames= unique(tpnames);
Nt=length(tpnames); % number of timepoint

tinit = 1; % Set higher than one if you do not want to process some timepoints before tinit
tfin = Nt; % Set lower than Nt if you do not wish to process some time points after tfin
zinit = 1; % Set higher than one if you do not want to process some lower slices

if lowResolution == 0
    zFactor = round(voxelSizeZ/voxelSizeX);
    scalingfactor = 1;
    landStackOrig=zeros(DIMENSIONSX,DIMENSIONSY,NzOrig-zinit+1);
    landStack=zeros(DIMENSIONSX, DIMENSIONSY,(NzOrig-zinit+1)*zFactor - zFactor +1);

else
    zFactor = 1;
    scalingfactor = (voxelSizeX/voxelSizeZ);
    landStackOrig=zeros(DIMENSIONSX,DIMENSIONSY,NzOrig-zinit+1);
    landStack=zeros(ceil(DIMENSIONSX*scalingfactor),ceil(DIMENSIONSX*scalingfactor),NzOrig-zinit+1);
end

tStart = 1;
tpoint = tStart;
while tpoint <=tfin
   
    info = imfinfo([fp image_names(tpoint).name]);
    
    for zplane= 1: numel(info)
        stacklsmoriginal = imread([fp image_names(tpoint).name], zplane);
        stack = imresize(stacklsmoriginal,[DIMENSIONSX DIMENSIONSY]);
        landStackOrig(:,:,zplane-zinit+1) =stack;
            
        if lowResolution == 1 %Generate low resolution stacks
            landStack(:,:,zplane-zinit+1) = imresize(landStackOrig(:,:,zplane-zinit+1),scalingfactor,'bicubic');
        end
        
        clear stack
        clear stacklsmoriginal
    end
    
    
    disp(sprintf('tpoint = %g', tpoint));
    
    if lowResolution == 0 %Generate intermediate z slices 
        landStack = genIntermediateSlices(landStackOrig, zFactor);
    end
    clear landStackOrig;
    
    if tpoint == tStart
        if lowResolution == 1 %Equal pixel size in x,y and z
            voxelSizeX = voxelSizeZ;
            voxelSizeY = voxelSizeZ;
        else
            voxelSizeZ = voxelSizeZ/zFactor; % voxelsize in z gets closer to that in xy
        end
        [dx dy Nz] = size(landStack); % getting size of the volume to be processed further

        corIndex = ExponentialGain(Nz,  2, 0.2, 0.4);
    end
    
    for i = 1: Nz
        landStack(:,:,i) = landStack(:,:,i)* corIndex(i);
       
    end
    
    voxelSize = voxelSizeX * voxelSizeY * voxelSizeZ;
   
    %Apply gussian filter in 3D
    landStack = imgaussian(landStack,sigma,hsize);
    
    %% thresholding
    
    orRegion = zeros(dx,dy,Nz);
    orRegion(landStack(:,:,:) >=200) = 1;
        
    for i=1:Nz
        orRegion(:,:,i) = imfill(orRegion(:,:,i), 'holes');
    end
        
    threeDLabel = bwconncomp(orRegion);
    numPixels = cellfun(@numel,threeDLabel.PixelIdxList);
    
    orRegion(:,:,:) = 0;
    for i = 1:length(numPixels)
        if numPixels(i) * voxelSizeX * voxelSizeY >= 20
            orRegion(threeDLabel.PixelIdxList{i}) = 1;
        end
    end
 
    blobIdx = 1;
    
    cellIdx = 1;
    
    while blobIdx <= min(length(numPixels), nBlobs)
        orRegion(:,:,:) = 0;
        [biggest,idx] = max(numPixels);
        
        if biggest * voxelSize < 20 % Minimal Organoid Volume
            break;
        end
        
        orRegion(threeDLabel.PixelIdxList{idx}) = 1;
        numPixels(idx) = 0;
        blobIdx = blobIdx + 1;
    end
    
    %% 3 way filling of thresholded image
    
    for zplane = 1: Nz
        orRegion(:,:,zplane)=imdilate(orRegion(:,:,zplane),strel('disk',2,0));
        orRegion(:,:,zplane)=imfill(orRegion(:,:,zplane),'holes');
        for i=1:2
            orRegion(:,:,zplane)=imerode(orRegion(:,:,zplane),strel('diamond',1));
        end
    end
    
orRegionZ = max(orRegion(:,:,:),[],3);
region3D = regionprops3(orRegion);
majoraxis = region3D.MajorAxisLength * voxelSizeX;
CentX = region3D.Centroid(1);
CentY = region3D.Centroid(2);
CentZ = region3D.Centroid(3);

detArea = sum(sum(orRegionZ)) * (voxelSizeX * voxelSizeY);
detVolume = sum(sum(sum(orRegion))) * voxelSize;

disp(sprintf('Detected Volume = %g', detVolume));
disp(sprintf('Deteceted Area = %g', detArea));


if tpoint == 1
    xlswrite([outDir 'Volume'  '.xls'], {'Total_Area'}, 1, 'A1');
    xlswrite([outDir 'Volume'  '.xls'], {'Total_Volume'}, 1, 'B1');
    xlswrite([outDir 'Volume'  '.xls'], {'majoraxis'}, 1, 'C1');
   
end

letter = ('A' + 5*(cellIdx-1));
xlswrite([outDir 'Volume'  '.xls'], detArea,  1, strcat(letter, num2str(tpoint+1)));
letter = ('B' + 5*(cellIdx-1));
xlswrite([outDir 'Volume'  '.xls'], detVolume,  1, strcat(letter, num2str(tpoint+1)));
letter = ('C' + 5*(cellIdx-1));
xlswrite([outDir 'Volume'  '.xls'], majoraxis,  1, strcat(letter, num2str(tpoint+1)));
letter = ('D' + 5*(cellIdx-1));


hVol = figure('Name', strcat('Reconstructed 3D volume - timepoint: ', num2str(tpoint)));
[X,Y,Z] = meshgrid((1:dy)*voxelSizeY,(1:dx)*voxelSizeX,(1:Nz)*voxelSizeZ);
isosurface(X,Y,Z,orRegion,0.9);
alpha(0.5)

hold on
lbl = strcat('\mu','m');
xlabel(lbl);
ylabel(lbl)
zlabel(lbl);
daspect([1 1 1]);
axis([CentX*voxelSizeX-120 CentX*voxelSizeX+120, CentY*voxelSizeY-120 CentY*voxelSizeY+120, CentZ*voxelSizeZ-120 CentZ*voxelSizeZ+120])
savefile = [outDir 'timepoint_' num2str(tpoint)  '.png'];
print(hVol,'-dpng',savefile,'-r400');
tpoint = tpoint + 1;
clear hVol
close all
end

    
