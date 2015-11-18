function [opts, imdb] = model_setup(varargin)
setup ;

opts.seed = 1 ;
opts.batchSize = 128 ;
opts.useGpu = true ;
opts.regionBorder = 0.05 ;
opts.numDCNNWords = 64 ;
opts.numDSIFTWords = 256 ;
opts.numSamplesPerWord = 1000 ;
opts.printDatasetInfo = true ;
opts.excludeDifficult = true ;
opts.datasetSize = inf;
opts.encoders = {struct('type', 'rcnn', 'opts', {})} ;
opts.dataset = 'lfw' ;
opts.facescrubDir = 'data/facescrub' ;
opts.mitDir = 'data/mit_indoor';
opts.cubDir = 'data/cub';
opts.dogDir = 'data/stanford_dogs';
opts.aircraftDir = 'data/fgvc-aircraft-2013b';
opts.modelnetDir = 'data/modelnet40toon';
% opts.radarDir = '/home/arunirc/Dropbox/NARRstyle/radar/data';
opts.radarDir = '/scratch3/arunirc/radar_project/radar_data/data';
opts.flickrDir = 'data/Flickr8k';
opts.suffix = 'baseline' ;
opts.prefix = 'v1' ;
opts.model  = 'imagenet-vgg-m.mat';
opts.modela = 'imagenet-vgg-m.mat';
opts.modelb = 'imagenet-vgg-s.mat';
opts.layer  = 14;
opts.layera = 14;
opts.layerb = 14;
[opts, varargin] = vl_argparse(opts,varargin) ;

opts.expDir = sprintf('data/%s/%s-seed-%02d', opts.prefix, opts.dataset, opts.seed) ;
opts.imdbDir = fullfile(opts.expDir, 'imdb') ;
opts.resultPath = fullfile(opts.expDir, sprintf('result-%s.mat', opts.suffix)) ;

opts = vl_argparse(opts,varargin) ;

if nargout <= 1, return ; end

% Setup GPU if needed
if opts.useGpu
  gpuDevice(1) ;
end

% -------------------------------------------------------------------------
%                                                            Setup encoders
% -------------------------------------------------------------------------

models = {} ;
for i = 1:numel(opts.encoders)
  if isstruct(opts.encoders{i})
    name = opts.encoders{i}.name ;
    opts.encoders{i}.path = fullfile(opts.expDir, [name '-encoder.mat']) ;
    opts.encoders{i}.codePath = fullfile(opts.expDir, [name '-codes.mat']) ;
    models = horzcat(models, get_cnn_model_from_encoder_opts(opts.encoders{i})) ;
  else
    for j = 1:numel(opts.encoders{i})
      name = opts.encoders{i}{j}.name ;
      opts.encoders{i}{j}.path = fullfile(opts.expDir, [name '-encoder.mat']) ;
      opts.encoders{i}{j}.codePath = fullfile(opts.expDir, [name '-codes.mat']) ;
      models = horzcat(models, get_cnn_model_from_encoder_opts(opts.encoders{i}{j})) ;
    end
  end
end

% -------------------------------------------------------------------------
%                                                       Download CNN models
% -------------------------------------------------------------------------

for i = 1:numel(models)
  if ~exist(fullfile('data/models', models{i}))
    fprintf('downloading model %s\n', models{i}) ;
    vl_xmkdir('data/models') ;
    urlwrite(fullfile('http://www.vlfeat.org/matconvnet/models', models{i}),...
      fullfile('data/models', models{i})) ;
  end
end

% -------------------------------------------------------------------------
%                                                              Load dataset
% -------------------------------------------------------------------------

vl_xmkdir(opts.expDir) ;
vl_xmkdir(opts.imdbDir) ;

imdbPath = fullfile(opts.imdbDir, sprintf('imdb-seed-%d.mat', opts.seed)) ;
if exist(imdbPath)
  imdb = load(imdbPath) ;
  return ;
end

switch opts.dataset
    case 'cubcrop'
        imdb = cub_get_database(opts.cubDir, true);
    case 'cub'
        imdb = cub_get_database(opts.cubDir, false);
    case 'dogcrop'
        imdb = stanford_dogs_get_database(opts.dogDir, true);
    case 'dog'
        imdb = stanford_dogs_get_database(opts.dogDir, false);
    case 'mitindoor'
        imdb = mit_indoor_get_database(opts.mitDir);
    case 'facescrub'
        imdb = facescrub_get_database(opts.facescrubDir) ;
    case 'aircraft-variant'
        imdb = aircraft_get_database(opts.aircraftDir, 'variant');
    case 'aircraft-model'
        imdb = aircraft_get_database(opts.aircraftDir, 'model');
    case 'aircraft-family'
        imdb = aircraft_get_database(opts.aircraftDir, 'family');
    case 'modelnet'
        imdb = modelnet_get_database(opts.modelnetDir);
    case 'flickr8k'
        imdb = flickr8k_get_database(opts.flickrDir);
    case 'radar-sweep01'
        imdb = radar_get_database(true, 'radarPath', opts.radarDir);
    case 'radar-sweep02'
        imdb = radar_get_database(true, 'radarPath', opts.radarDir, ...
                'sweep', 2);
    case 'radar-sweep03'
        imdb = radar_get_database(true, 'radarPath', opts.radarDir, ...
                'sweep', 3);
    case 'radar-sweeps'
        imdb = radar_get_database(true, 'radarPath', opts.radarDir, ...
                'sweep', [1 2 3]);
    case 'radar-sw-sweep01'
        imdb = radar_get_database(true, 'radarPath', opts.radarDir, ...
                'mode', 'sw', 'sweep', 1);
    case 'radar-vr-sweep01'
        imdb = radar_get_database(true, 'radarPath', opts.radarDir, ...
                'mode', 'vr', 'sweep', 1); 
    case 'radar-sw-sweep02'
        imdb = radar_get_database(true, 'radarPath', opts.radarDir, ...
                'mode', 'sw', 'sweep', 2);
    case 'radar-vr-sweep02'
        imdb = radar_get_database(true, 'radarPath', opts.radarDir, ...
                'mode', 'vr', 'sweep', 2); 
    case 'radar-sw-sweep03'
        imdb = radar_get_database(true, 'radarPath', opts.radarDir, ...
                'mode', 'sw', 'sweep', 3);
    case 'radar-vr-sweep03'
        imdb = radar_get_database(true, 'radarPath', opts.radarDir, ...
                'mode', 'vr', 'sweep', 3); 
    case 'radar-sw-sweeps'
        imdb = radar_get_database(true, 'radarPath', opts.radarDir, ...
                'mode', 'sw', 'sweep', [1 2 3]);    
    case 'radar-vr-sweeps'
        imdb = radar_get_database(true, 'radarPath', opts.radarDir, ...
                'mode', 'vr', 'sweep', [1 2 3]);
    otherwise
        error('Unknown dataset %s', opts.dataset) ;
end

save(imdbPath, '-struct', 'imdb') ;

if opts.printDatasetInfo
  print_dataset_info(imdb) ;
end

% -------------------------------------------------------------------------
function model = get_cnn_model_from_encoder_opts(encoder)
% -------------------------------------------------------------------------
p = find(strcmp('model', encoder.opts)) ;
if ~isempty(p)
  [~,m,e] = fileparts(encoder.opts{p+1}) ;
  model = {[m e]} ;
else
  model = {} ;
end

% bilinear cnn models
p = find(strcmp('modela', encoder.opts)) ;
if ~isempty(p)
  [~,m,e] = fileparts(encoder.opts{p+1}) ;
  model = horzcat(model,{[m e]}) ;
end
p = find(strcmp('modelb', encoder.opts)) ;
if ~isempty(p)
  [~,m,e] = fileparts(encoder.opts{p+1}) ;
  model = horzcat(model,{[m e]}) ;
end


