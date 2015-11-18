function model_train(varargin)

[opts, imdb] = model_setup(varargin{:}) ;

% -------------------------------------------------------------------------
%                                          Train encoders and compute codes
% -------------------------------------------------------------------------

if ~exist(opts.resultPath)
  psi = {} ;
  for i = 1:numel(opts.encoders)
    if exist(opts.encoders{i}.codePath)
      load(opts.encoders{i}.codePath, 'code', 'area') ;
    else
      if exist(opts.encoders{i}.path)
        encoder = load(opts.encoders{i}.path) ;
        if isfield(encoder, 'net')
            if opts.useGpu
              encoder.net = vl_simplenn_move(encoder.net, 'gpu') ;
              encoder.net.useGpu = true ;
            else
              encoder.net = vl_simplenn_move(encoder.net, 'cpu') ;
              encoder.net.useGpu = false ;
            end
        end
        if isfield(encoder, 'neta')
            if opts.useGpu
              encoder.neta = vl_simplenn_move(encoder.neta, 'gpu') ;
              encoder.netb = vl_simplenn_move(encoder.netb, 'gpu') ;
              encoder.neta.useGpu = true ;
              encoder.netb.useGpu = true ;
            else
              encoder.neta = vl_simplenn_move(encoder.neta, 'cpu') ;
              encoder.netb = vl_simplenn_move(encoder.netb, 'cpu') ;
              encoder.neta.useGpu = false ;
              encoder.netb.useGpu = false ;
            end
        end
      else
        train = find(ismember(imdb.images.set, [1 2])) ;
        train = vl_colsubset(train, 1000, 'uniform') ;
        encoder = encoder_train_from_images(...
          imdb, imdb.images.id(train), ...
          opts.encoders{i}.opts{:}, ...
          'useGpu', opts.useGpu) ;
        encoder_save(encoder, opts.encoders{i}.path) ;
      end
      code = encoder_extract_for_images(encoder, imdb, imdb.images.id) ;
      savefast(opts.encoders{i}.codePath, 'code') ;
    end
    psi{i} = code ;
    clear code ;
  end
  % psi = cat(1, psi{:}) ;
  psi = cell2mat(psi);
end

% -------------------------------------------------------------------------
%                                                            Train and test
% -------------------------------------------------------------------------

if exist(opts.resultPath)
  info = load(opts.resultPath) ;
else
  info = traintest(opts, imdb, psi) ;
  save(opts.resultPath, '-struct', 'info') ;
  vl_printsize(1) ;
  [a,b,c] = fileparts(opts.resultPath) ;
  print('-dpdf', fullfile(a, [b '.pdf'])) ;
end

str = {} ;
str{end+1} = sprintf('data: %s', opts.expDir) ;
str{end+1} = sprintf(' setup: %10s', opts.suffix) ;
str{end+1} = sprintf(' mAP: %.1f', info.test.map*100) ;
if isfield(info.test, 'acc')
  str{end+1} = sprintf(' acc: %6.1f ', info.test.acc*100);
end
str{end+1} = sprintf('\n') ;
str = cat(2, str{:}) ;
fprintf('%s', str) ;

[a,b,c] = fileparts(opts.resultPath) ;
txtPath = fullfile(a, [b '.txt']) ;
f=fopen(txtPath, 'w') ;
fprintf(f, '%s', str) ;
fclose(f) ;


% -------------------------------------------------------------------------
function info = traintest(opts, imdb, psi)
% -------------------------------------------------------------------------

% Train using verification or not
verificationTask = isfield(imdb, 'pairs');

if verificationTask, 
    train = ismember(imdb.pairs.set, [1 2]) ;
    test = ismember(imdb.pairs.set, 3) ;
else % classification task
    multiLabel = (size(imdb.images.label,1) > 1) ; % e.g. PASCAL VOC cls
    train = ismember(imdb.images.set, [1 2]) ;
    test = ismember(imdb.images.set, 3) ;
    info.classes = find(imdb.meta.inUse) ;
    
    % Train classifiers
    C = 1 ;
    w = {} ;
    b = {} ;
    
    for c=1:numel(info.classes)
      if ~multiLabel
        y = 2*(imdb.images.label == info.classes(c)) - 1 ;
      else
        y = imdb.images.label(c,:) ;
      end
      np = sum(y(train) > 0) ;
      nn = sum(y(train) < 0) ;
      n = np + nn ;

      [w{c},b{c}] = vl_svmtrain(psi(:,train & y ~= 0), y(train & y ~= 0), 1/(n* C), ...
        'epsilon', 0.001, 'verbose', 'biasMultiplier', 1, ...
        'maxNumIterations', n * 200) ;

      pred = w{c}'*psi + b{c} ;

      % try cheap calibration
      mp = median(pred(train & y > 0)) ;
      mn = median(pred(train & y < 0)) ;
      b{c} = (b{c} - mn) / (mp - mn) ;
      w{c} = w{c} / (mp - mn) ;
      pred = w{c}'*psi + b{c} ;

      scores{c} = pred ;

      [~,~,i]= vl_pr(y(train), pred(train)) ; ap(c) = i.ap ; ap11(c) = i.ap_interp_11 ;
      [~,~,i]= vl_pr(y(test), pred(test)) ; tap(c) = i.ap ; tap11(c) = i.ap_interp_11 ;
      [~,~,i]= vl_pr(y(train), pred(train), 'normalizeprior', 0.01) ; nap(c) = i.ap ;
      [~,~,i]= vl_pr(y(test), pred(test), 'normalizeprior', 0.01) ; tnap(c) = i.ap ;
    end
    
    % Book keeping
    info.w = cat(2,w{:}) ;
    info.b = cat(2,b{:}) ;
    info.scores = cat(1, scores{:}) ;
    info.train.ap = ap ;
    info.train.ap11 = ap11 ;
    info.train.nap = nap ;
    info.train.map = mean(ap) ;
    info.train.map11 = mean(ap11) ;
    info.train.mnap = mean(nap) ;
    info.test.ap = tap ;
    info.test.ap11 = tap11 ;
    info.test.nap = tnap ;
    info.test.map = mean(tap) ;
    info.test.map11 = mean(tap11) ;
    info.test.mnap = mean(tnap) ;
    clear ap nap tap tnap scores ;
    fprintf('mAP train: %.1f, test: %.1f\n', ...
      mean(info.train.ap)*100, ...
      mean(info.test.ap)*100);

    % Compute predictions, confusion and accuracy
    [~,preds] = max(info.scores,[],1) ;
    [~,gts] = ismember(imdb.images.label, info.classes) ;
    [info.train.confusion, info.train.acc] = compute_confusion(numel(info.classes), gts(train), preds(train)) ;
    [info.test.confusion, info.test.acc] = compute_confusion(numel(info.classes), gts(test), preds(test)) ;
end

% -------------------------------------------------------------------------
function code = encoder_extract_for_images(encoder, imdb, imageIds, varargin)
% -------------------------------------------------------------------------
opts.batchSize = 128 ;
opts.maxNumLocalDescriptorsReturned = 500 ;
opts.concatenateCode = true;
opts = vl_argparse(opts, varargin) ;

[~,imageSel] = ismember(imageIds, imdb.images.id) ;
imageIds = unique(imdb.images.id(imageSel)) ;
n = numel(imageIds) ;

% prepare batches
n = ceil(numel(imageIds)/opts.batchSize) ;
batches = mat2cell(1:numel(imageIds), 1, [opts.batchSize * ones(1, n-1), numel(imageIds) - opts.batchSize*(n-1)]) ;
batchResults = cell(1, numel(batches)) ;

% just use as many workers as are already available
numWorkers = matlabpool('size') ;
%parfor (b = 1:numel(batches), numWorkers)
for b = numel(batches):-1:1
  batchResults{b} = get_batch_results(imdb, imageIds, batches{b}, ...
                        encoder, opts.maxNumLocalDescriptorsReturned) ;
end

code = cell(size(imageIds)) ;
for b = 1:numel(batches)
  m = numel(batches{b});
  for j = 1:m
      k = batches{b}(j) ;
      code{k} = batchResults{b}.code{j};
  end
end
if opts.concatenateCode
   code = cat(2, code{:}) ;
end

% code is either:
% - a cell array, each cell containing an array of local features for a
%   segment
% - an array of FV descriptors, one per segment

% -------------------------------------------------------------------------
function result = get_batch_results(imdb, imageIds, batch, encoder, maxn)
% -------------------------------------------------------------------------
m = numel(batch) ;
im = cell(1, m) ;
task = getCurrentTask() ;
if ~isempty(task), tid = task.ID ; else tid = 1 ; end
for i = 1:m
  fprintf('Task: %03d: encoder: extract features: image %d of %d\n', tid, batch(i), numel(imageIds)) ;
  im{i} = imread(fullfile(imdb.imageDir, imdb.images.name{imdb.images.id == imageIds(batch(i))}));
  if size(im{i}, 3) == 1, im{i} = repmat(im{i}, [1 1 3]);, end; %grayscale image
end

if ~isfield(encoder, 'numSpatialSubdivisions')
  encoder.numSpatialSubdivisions = 1 ;
end
switch encoder.type
  case 'rcnn'
    code_ = get_rcnn_features(encoder.net, ...
      im, ...
      'regionBorder', encoder.regionBorder) ;
  case 'dcnn'
    gmm = [] ;
    if isfield(encoder, 'covariances'), gmm = encoder ; end
    code_ = get_dcnn_features(encoder.net, ...
      im, ...
      'encoder', gmm, ...
      'numSpatialSubdivisions', encoder.numSpatialSubdivisions, ...
      'maxNumLocalDescriptorsReturned', maxn) ;
  case 'dsift'
    gmm = [] ;
    if isfield(encoder, 'covariances'), gmm = encoder ; end
    code_ = get_dcnn_features([], im, ...
      'useSIFT', true, ...
      'encoder', gmm, ...
      'numSpatialSubdivisions', encoder.numSpatialSubdivisions, ...
      'maxNumLocalDescriptorsReturned', maxn) ;
   case 'bcnn'
       code_ = get_bcnn_features(encoder.neta, encoder.netb,...
         im, ...
        'regionBorder', encoder.regionBorder);
end
result.code = code_ ;

% -------------------------------------------------------------------------
function encoder = encoder_train_from_images(imdb, imageIds, varargin)
% -------------------------------------------------------------------------
opts.type = 'rcnn' ;
opts.model = '' ;
opts.modela = '';
opts.modelb = '';
opts.layer = 0 ;
opts.layera = 0 ;
opts.layerb = 0 ;
opts.useGpu = false ;
opts.regionBorder = 0.05 ;
opts.numPcaDimensions = +inf ;
opts.numSamplesPerWord = 1000 ;
opts.whitening = false ;
opts.whiteningRegul = 0 ;
opts.renormalize = false ;
opts.numWords = 64 ;
opts.numSpatialSubdivisions = 1 ;
opts = vl_argparse(opts, varargin) ;

encoder.type = opts.type ;
encoder.regionBorder = opts.regionBorder ;
switch opts.type
  case {'dcnn', 'dsift'}
    encoder.numWords = opts.numWords ;
    encoder.renormalize = opts.renormalize ;
    encoder.numSpatialSubdivisions = opts.numSpatialSubdivisions ;
end

switch opts.type
    case {'rcnn', 'dcnn'}
        encoder.net = load(opts.model) ;
        encoder.net.layers = encoder.net.layers(1:opts.layer) ;
        if opts.useGpu
            encoder.net = vl_simplenn_move(encoder.net, 'gpu') ;
            encoder.net.useGpu = true ;
        else
            encoder.net = vl_simplenn_move(encoder.net, 'cpu') ;
            encoder.net.useGpu = false ;
        end
   case 'bcnn'
       encoder.neta = load(opts.modela);
       encoder.neta.layers = encoder.neta.layers(1:opts.layera);
       encoder.netb = load(opts.modelb);
       encoder.netb.layers = encoder.netb.layers(1:opts.layerb);
       if opts.useGpu, 
           encoder.neta = vl_simplenn_move(encoder.neta, 'gpu');
           encoder.netb = vl_simplenn_move(encoder.netb, 'gpu');
           encoder.neta.useGpu = true;
           encoder.netb.useGpu = true;
       else
           encoder.neta = vl_simplenn_move(encoder.neta, 'cpu');
           encoder.netb = vl_simplenn_move(encoder.netb, 'cpu');
           encoder.neta.useGpu = false;
           encoder.netb.useGpu = false;
       end           
end

switch opts.type
  case {'rcnn', 'bcnn'}
    return ;
end

% Step 0: sample descriptors
fprintf('%s: getting local descriptors to train GMM\n', mfilename) ;
code = encoder_extract_for_images(encoder, imdb, imageIds, 'concatenateCode', false) ;
descrs = cell(1, numel(code)) ;
numImages = numel(code);
numDescrsPerImage = floor(encoder.numWords * opts.numSamplesPerWord / numImages);
for i=1:numel(code)
  descrs{i} = vl_colsubset(code{i}, numDescrsPerImage) ;
end
descrs = cat(2, descrs{:}) ;
fprintf('%s: obtained %d local descriptors to train GMM\n', ...
  mfilename, size(descrs,2)) ;


% Step 1 (optional): learn PCA projection
if opts.numPcaDimensions < inf || opts.whitening
  fprintf('%s: learning PCA rotation/projection\n', mfilename) ;
  encoder.projectionCenter = mean(descrs,2) ;
  x = bsxfun(@minus, descrs, encoder.projectionCenter) ;
  X = x*x' / size(x,2) ;
  [V,D] = eig(X) ;
  d = diag(D) ;
  [d,perm] = sort(d,'descend') ;
  d = d + opts.whiteningRegul * max(d) ;
  m = min(opts.numPcaDimensions, size(descrs,1)) ;
  V = V(:,perm) ;
  if opts.whitening
    encoder.projection = diag(1./sqrt(d(1:m))) * V(:,1:m)' ;
  else
    encoder.projection = V(:,1:m)' ;
  end
  clear X V D d ;
else
  encoder.projection = 1 ;
  encoder.projectionCenter = 0 ;
end
descrs = encoder.projection * bsxfun(@minus, descrs, encoder.projectionCenter) ;
if encoder.renormalize
  descrs = bsxfun(@times, descrs, 1./max(1e-12, sqrt(sum(descrs.^2)))) ;
end

% Step 2: train GMM
v = var(descrs')' ;
[encoder.means, encoder.covariances, encoder.priors] = ...
  vl_gmm(descrs, opts.numWords, 'verbose', ...
  'Initialization', 'kmeans', ...
  'CovarianceBound', double(max(v)*0.0001), ...
  'NumRepetitions', 1) ;
