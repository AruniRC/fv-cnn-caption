function imdb = facescrub_get_database(facescrubDir, varargin)
% Set the random seed generator
opts.seed = 0 ;
opts.datasetSize = inf;
opts = vl_argparse(opts, varargin) ;
rng(opts.seed) ;

imdb.imageDir = fullfile(facescrubDir);

fid = fopen(fullfile(facescrubDir, 'filelist.txt'));
filelist = textscan(fid, '%s', 'Delimiter', '\n');
fclose(fid);

% Images and class
imdb.images.name = filelist{1}';
imdb.images.id = 1:length(imdb.images.name);
class = cellfun(@(x) getClass(x), imdb.images.name, 'UniformOutput', false);

% Class names
classNames = unique(class);
imdb.classes.name = classNames;
[~, imdb.images.label] = ismember(class, classNames);

% No standard image splits are provided for this dataset
imdb.sets = {'train', 'val', 'test'};
imdb.images.set = zeros(1,length(imdb.images.id));
for c = 1:length(imdb.classes.name), 
    isclass = find(imdb.images.label == c);
    
    % split equally into train, val, test
    order = randperm(length(isclass));
    subsetSize = ceil(length(order)/3);
    train = isclass(order(1:subsetSize));
    val = isclass(order(subsetSize+1:2*subsetSize));
    test  = isclass(order(2*subsetSize+1:end));
    
    imdb.images.set(train) = 1;
    imdb.images.set(val) = 2;
    imdb.images.set(test) = 3;
end

% make this compatible with the OS imdb
imdb.meta.classes = imdb.classes.name ;
imdb.meta.inUse = true(1,numel(imdb.meta.classes)) ;


% helper function to get class
function class = getClass(fileName)
[pathstr, ~] = fileparts(fileName);
[~, class] = fileparts(pathstr);