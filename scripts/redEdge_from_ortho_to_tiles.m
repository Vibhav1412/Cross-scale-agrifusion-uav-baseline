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
close all;

dataSet='002'; %'000' '001' '002'
drawOn =0;
dataType={'NDVI' 'CIR' 'B' 'G' 'R' 'NIR' 'RE' 'RGB'}; %'CIR','RGB' ,'NIR',
% SegNet input size w/o resizing.
blockSizeR = 360; % Rows in block. height
blockSizeC = 480; % Columns in block. width

for k=1:length(dataType)
    str=sprintf('Fetching tiles from %s channel',dataType{k});
    disp(str);
    dataPathRGB=fullfile('../Orthomosaic/RedEdge',dataSet,'composite-png',[dataType{k} '.png']);
    im=imread(dataPathRGB);
    storePath=fullfile('../Orthomosaic/RedEdge',dataSet,'tile',dataType{k});
    if ~exist(storePath)
        mkdir(storePath);
        disp('creating a folder to store tiles');
    end
    [rows,columns,numberOfColorBands] = size(im);
    % The first way to divide an image up into blocks is by using mat2cell().
    % Figure out the size of each block in rows. 
    % Most will be blockSizeR but there may be a remainder amount of less than that.
    %wholeBlockRows = floor(rows / blockSizeR);
    wholeBlockRows = ceil(rows / blockSizeR);
    blockVectorR = blockSizeR * ones(1, wholeBlockRows);
    
    % Figure out the size of each block in columns. 
    wholeBlockCols = ceil(columns / blockSizeC);
    blockVectorC = blockSizeC * ones(1, wholeBlockCols);

    %padding to image.
    padR=rem(rows, blockSizeR);
    padC=rem(columns, blockSizeC);
    padSz=[blockSizeR-padR blockSizeC-padC];   
    im = padarray(im,padSz,'post');

    % Create the cell array, ca.  
    % Each cell (except for the remainder cells at the end of the image)
    % in the array contains a blockSizeR by blockSizeC by 3 color array.
    % This line is where the image is actually divided up into blocks.
    if numberOfColorBands > 1
        % It's a color image.
        ca = mat2cell(im, blockVectorR, blockVectorC, numberOfColorBands);
    else
        ca = mat2cell(im, blockVectorR, blockVectorC);
    end
    % Now display all the blocks.
    %!!!!!!!!!! Tiled images start from 0 not 1
    plotIndex = 0;
    % Since deep labelTool also starts from 0 not 1
    numPlotsR = size(ca, 1);
    numPlotsC = size(ca, 2);
    for r = 1 : numPlotsR
        for c = 1 : numPlotsC
            fName=sprintf('frame%.4d.png',plotIndex);
            imwrite(ca{r,c},[storePath '/' fName]);
            if drawOn
                % Specify the location for display of the image.
                subplot(numPlotsR, numPlotsC, plotIndex+1,[0.0001,0.0001]);
                rgbBlock = ca{r,c};
                imshow(rgbBlock); % Could call imshow(ca{r,c}) if you wanted to.
                [rowsB,columnsB,numberOfColorBandsB] = size(rgbBlock);
                % Make the caption the block number.
                caption = sprintf('#%d',plotIndex);
                title(caption);
                drawnow;
                % Increment the subplot to the next location.
            end
            plotIndex = plotIndex + 1;
        end
    end
end