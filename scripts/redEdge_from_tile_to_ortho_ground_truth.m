%Copyright (C) 2018, by Inkyu Sa, enddl22@gmail.com

%This is free software: you can redistribute it and/or modify
%it under the terms of the GNU Lesser General Public License as published by
%the Free Software Foundation, either version 3 of the License, or
%(at your option) any later version.
 
%This software package is distributed in the hope that it will be useful,
%but WITHOUT ANY WARRANTY; without even the implied warranty of
%MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%GNU Lesser General Public License for more details.

%You should have received a copy of the GNU Leser General Public License.
%If not, see <http://www.gnu.org/licenses/>.

clear all;
%====================
% Define valuables
%====================
% This valuable selects the corresponding dataset and 
% it is only what you need to change in order to generate 
% an orthomosaic map from tiles.
nDataset=1%2,3,4,5 
if nDataset==1
    dataSet='000';
    tileRow=17;
    tileCol=13;
    rowPadding=266;
    colPadding=245;
elseif nDataset==2
    dataSet='001';
    tileRow=16;
    tileCol=11;
    rowPadding=186;
    colPadding=413;
elseif nDataset==3
    dataSet='002';
    tileRow=18;
    tileCol=14;
    rowPadding=75;
    colPadding=317;
elseif nDataset==4
    dataSet='003';
    tileRow=17;
    tileCol=12;
    rowPadding=125;
    colPadding=290;
elseif nDataset==5
    dataSet='004';
    tileRow=13;
    tileCol=9;
    rowPadding=174;
    colPadding=1;
end
imgWidth=480;
imgHeight=360;
plantColor=[0 1 0]; %red
weedColor=[1 0 0]; %green
map=[plantColor;weedColor];

%====================
% Set up paths
%====================
inputDirPath=['../Tiles/RedEdge/' dataSet '/groundtruth'];
dirInfo=dir([inputDirPath '/*.png']);
outputDirPath='../orthoSegResults';
if ~exist(outputDirPath)
    display('Output directory not exist, create it\n');
    mkdir(outputDirPath)
end

%Cell2Mat
tempCnt=1;
probImg=zeros(imgHeight,imgWidth,3);
for i=1:tileRow
    for j=1:tileCol
        idx=sprintf('%04d',tempCnt-1);
        fileName=[dataSet '_frame' idx '_GroundTruth_iMap.png'];
        image_stitch_iMap=imread(fullfile(inputDirPath,fileName));
        %Note that crop (index as 1 contains 10000 for some reasons. so
        %find them and replace as 1).
        image_stitch_iMap(find(image_stitch_iMap==10000))=1;
        image_stitch=label2rgb(image_stitch_iMap,map,[0,0,0]); %label img, map, bg color
        tempCnt=tempCnt+1;
        imgcellArr{i,j}=uint8(image_stitch);
    end
end
weedMap=cell2mat(imgcellArr);
weedMapCrop=weedMap(1:size(weedMap,1)-rowPadding,1:size(weedMap,2)-colPadding,:);

% Construct TIFF image...
t = Tiff([outputDirPath '/' dataSet '.tif'],'w'); 

% ...with these custom parameters...
tagstruct = struct(...
    'ImageLength'        , size(weedMapCrop,1),...
    'ImageWidth'         , size(weedMapCrop,2),...
    'Compression'        , Tiff.Compression.None,...
    'Photometric'        , Tiff.Photometric.MinIsBlack,...
    'BitsPerSample'      , 8,... % 8 bytes / double
    'SamplesPerPixel'    , 3,...
    'PlanarConfiguration', Tiff.PlanarConfiguration.Chunky);

t.setTag(tagstruct); 

% ...and write it to disk.
t.write(weedMapCrop); 
t.close();
str=sprintf('Output file generated at%s',[outputDirPath '/' dataSet '.tif']);
disp(str);
