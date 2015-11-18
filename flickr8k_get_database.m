function imdb = flickr8k_get_database(flickrDir, varargin)
% Get imdb struct for Flickr8k database.

opts.seed = 0 ;
opts.imgDir = 'Flickr8k_Dataset' ;
opts.txtDir = 'Flickr8k_text' ;
opts = vl_argparse(opts, varargin) ;
rng(opts.seed) ;

imgDir = fullfile(flickrDir, opts.imgDir);
txtDir = fullfile(flickrDir, opts.txtDir);

dirListing = dir(fullfile(imgDir, '*.jpg'));


% Images and class
imdb.imageDir = imgDir;
imdb.images.name = {dirListing.name}';
imdb.images.id = (1:numel(imdb.images.name));
imdb.images.set = zeros(1, numel(imdb.images.name));
imdb.classes.name = {};
imdb.images.label = [] ;

% Splits for train, dev and test
imdb.sets = {'train', 'val', 'test'};
trainFiles = readFlickrSplit(fullfile(txtDir, 'Flickr_8k.trainImages.txt'));
valFiles = readFlickrSplit(fullfile(txtDir, 'Flickr_8k.devImages.txt'));
testFiles = readFlickrSplit(fullfile(txtDir, 'Flickr_8k.testImages.txt'));
[~, train] = ismember(trainFiles, imdb.images.name);
[~, val] = ismember(valFiles, imdb.images.name);
[~, test] = ismember(testFiles, imdb.images.name);
imdb.images.set(train) = 1;
imdb.images.set(val) = 2;
imdb.images.set(test) = 3;

% make this compatible with the OS imdb
imdb.meta.classes = {} ;
imdb.meta.inUse = [] ;



function filenames = readFlickrSplit(filePath)
fid = fopen(filePath, 'r');
filelist = textscan(fid, '%s', 'Delimiter', '\n');
fclose(fid);
filenames = filelist{1}';
