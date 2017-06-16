function visualize_filter_patch(net, imdb, modelName, layer)

%{
net = load(fullfile('data', 'bcnn-train_vdm', 'cars-seed-01', ...
        'fine-tuned-model', 'fine-tuned-netb-imagenet-vgg-verydeep-16'));
%}

if isequal(modelName, 'net')
    net.layers = net.layers(1:layer); % chop off layers at conv5 level
end

scale = 2;
K = 20; % top k patches for each filter
imgOutDirPath = sprintf('data/bcnn_vis/patches/%s-patches', modelName);
vl_xmkdir(imgOutDirPath);

net = vl_simplenn_move(net, 'gpu') ;

%   Sort by highest response for conv5 filters
filterData = get_filter_patches(net, imdb, scale); 
sortedIndex = cell(1, size(filterData.responses, 2));
for i = 1:size(filterData.responses, 2)
    [~, idx] = sort(filterData.responses(:, i), 'descend');
    sortedIndex{i} = idx;
end
filterData.sortedIndex = sortedIndex;
     


%   Writing top-K patch images to disk
% filterPatches = cell(1, K);
for i = 1:size(filterData.patches, 2)
	filterDirPath = fullfile(imgOutDirPath, num2str(i));
	mkdir(filterDirPath);
	idx = filterData.sortedIndex{i};
	for j = 1:K
		im = filterData.patches{idx(j), i};
		imName = sprintf('%s.jpg', num2str(j));
		imwrite(im, fullfile(filterDirPath, imName ));
        filterPatches{j} = im;
    end
end
fprintf('\n Filter patches written to: %s.\n', filterDirPath);


%  Writing top-K patches visualization to HTML file
list_patches_html(imgOutDirPath);
disp('Visualizations written to HTML file');



                                
% -------------------------------------------------------------------------
%                                   Get highest-response patch for filters
% -------------------------------------------------------------------------                                
function filterData = get_filter_patches(net, imdb, scale)
% -------------------------------------------------------------------------
SUBSAMPLE = 1500; % 3000; % subset of all CUB images
% padVal = 100; % padding around original colour image

% calculate geometrical parameters
info = vl_simplenn_display(net) ;
x=1 ;
for l=numel(net.layers):-1:1
    x=(x-1)*info.stride(2,l)-info.pad(2,l)+1 ;
end
offset = round(x + info.receptiveField(end)/2 - 0.5);
stride = prod(info.stride(1,:)) ;
border = round(info.receptiveField(end)/2+1) ;
rfSize = info.receptiveField(end);
averageColour = mean(mean(net.normalization.averageImage,1),2) ;
imageSize = net.normalization.imageSize;

valFileName = imdb.images.name(imdb.images.set==3); % unseen set of images
valFileName = valFileName(randperm(length(valFileName), SUBSAMPLE));

numFilters = length(net.layers{end-1}.biases);
filterPatches = cell(length(valFileName), numFilters); % numImg x numFilter
filterResponses = zeros(length(valFileName), numFilters);

for i = 1:length(valFileName)
    fprintf('\nImg : %d / %d', i, length(valFileName));
    
    imgPath = fullfile(imdb.imageDir, valFileName{i});
    im = imread(imgPath);
    
    im_crop = imresize(single(im), imageSize([2 1]), 'bilinear');
    im_colour = imresize(im, imageSize([2 1]), 'bilinear'); % RGB resized
    
    im_crop = imresize(im_crop, scale);
    im_colour = imresize(im_colour, scale);
    
    im_resized = bsxfun(@minus, im_crop, averageColour) ; 
    im_resized = gpuArray(im_resized);
    
    res = vl_simplenn(net, im_resized);
    
    w = size(res(end).x,2) ;
    h = size(res(end).x,1) ;
    
    % response maps  
    filterMaps = gather(res(end).x);
    mapSize = size(filterMaps,1);
    
    % max response locations for each map
    vecMap = reshape(filterMaps, mapSize*mapSize, numFilters);
    [maxResVal, maxPos] = max(vecMap); % col-max of (numPixels x numFilters)
    [row, col] = ind2sub([mapSize mapSize],maxPos);    
    
    % extract crops for i-th image for all filters
    % im_colour = padarray(im_colour, [padVal padVal],'both');
    for j = 1:numFilters
        r = row(j);
        c = col(j);
        
        % map filter response to original image coordinates
        xmin = offset + (c-1)*stride - floor(rfSize/2);
        ymin = offset + (r-1)*stride - floor(rfSize/2);
        rect = [xmin ymin rfSize rfSize];
        
        imcropped = imcrop(im_colour, rect);
        assert(~isempty(imcropped));
        if(isempty(find(vecMap(:,j), 1)))
            filterPatches{i, j} = zeros(size(imcropped));
            filterResponses(i, j) = maxResVal(j);
        else
            filterPatches{i, j} = imcropped;
            filterResponses(i, j) = maxResVal(j);
        end
    end  
    %{
    figure(1)
    imagesc(filterMaps(:,:,2)); axis off
    figure(2)
    imagesc(im_colour);axis off
    figure(3)
    imagesc(filterPatches{i, 2}); axis off
    pause
    %}
end
filterData.patches = filterPatches;
filterData.responses = filterResponses;


% -------------------------------------------------------------------------
function list_patches_html(imgOutDirPath)
% -------------------------------------------------------------------------
% parentFolderName = 'netb-vgg-vdm-patches';
K = 20;

[~, parentFolderName] = fileparts(imgOutDirPath);
htmlOutPath = sprintf('%s.html', imgOutDirPath);

% fid = fopen('data/bcnn-train_vis/patches/test_matlab.html', 'w');
fid = fopen(htmlOutPath, 'w');

fprintf(fid, '\n<head>');
fprintf(fid, '\n<title>Images</title>');
fprintf(fid, '\n</head>');
fprintf(fid, '\n<body>');
fprintf(fid, '\n<p>Top 20 patches for each filter:</p>');

% looping over filters
for i = 1:512
    fprintf(fid, '\n<br/><b>Filter %d</b><br/>', i);
    
    % over images
    for j = 1:K
        imagePath = sprintf('%s/%s',parentFolderName, num2str(i));
        fprintf(fid, '\n<a href="%s/%d.jpg"><img src="%s/%d.jpg" height="80" width="80" /></a>', ...
            imagePath, j, imagePath, j);
    end
    fprintf(fid, '\n<hr>');
end
        
fprintf(fid, '\n</body>');
fprintf(fid, '\n</html>');


% -------------------------------------------------------------------------
function show_filter_patches(patches, filterNum)
% -------------------------------------------------------------------------
imarray = cat(4, patches{:,filterNum});
vl_imarraysc(imarray);