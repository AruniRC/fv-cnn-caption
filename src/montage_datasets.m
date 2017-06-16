function montage_datasets(sizes)

numImages = prod(sizes);
imageSize = [96 128]*4;

% CUB dataset
imdb = cub_get_database('data/cub', false);
rp = randperm(length(imdb.images.id));
imarray = zeros([imageSize 3 numImages],'uint8');
for i = 1:numImages,
    im = imread(fullfile(imdb.imageDir, imdb.images.name{rp(i)}));
    imarray(:,:,:,i) = imresize(im, imageSize);
end
figure(1); clf;
montage(imarray, 'Size', sizes);


% Aircraft dataset
imdb = aircraft_get_database('data/fgvc-aircraft-2013b', 'variant');
rp = randperm(length(imdb.images.id));
imarray = zeros([imageSize 3 numImages],'uint8');
for i = 1:numImages,
    im = imread(fullfile(imdb.imageDir, imdb.images.name{rp(i)}));
    imarray(:,:,:,i) = imresize(im, imageSize);
end
figure(2); clf;
montage(imarray, 'Size', sizes);

% Stanford cars
imdb = cars_get_database('data/cars', false);
rp = randperm(length(imdb.images.id));
imarray = zeros([imageSize 3 numImages],'uint8');
for i = 1:numImages,
    im = imread(fullfile(imdb.imageDir, imdb.images.name{rp(i)}));
    imarray(:,:,:,i) = imresize(im, imageSize);
end
figure(3); clf;
montage(imarray, 'Size', sizes);
