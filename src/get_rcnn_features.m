function code = get_rcnn_features(net, im, varargin)
% GET_RCNN_FEATURES
%    This function gets the fc7 features for an image region,
%    extracted from the provided mask.

opts.batchSize = 8 ;
opts.regionBorder = 0.05;
opts = vl_argparse(opts, varargin) ;

if ~iscell(im)
  im = {im} ;
end

res = [] ;
cache = struct() ;
resetCache() ;

    % for each image
    function resetCache()
        cache.images = cell(1,opts.batchSize) ;
        cache.indexes = zeros(1, opts.batchSize) ;
        cache.numCached = 0 ;
    end

    function flushCache()
        if cache.numCached == 0, return ; end
        images = cat(4, cache.images{:}) ;
        images = bsxfun(@minus, images, net.normalization.averageImage) ;
        if net.useGpu
            images = gpuArray(images) ;
        end
        res = vl_simplenn(net, images, ...
                        [], res, ...
                        'conserveMemory', true, ...
                        'sync', true) ;
        code_ = squeeze(gather(res(end).x)) ;
        code_ = bsxfun(@times, 1./sqrt(sum(code_.^2)), code_) ;
        for q=1:cache.numCached
            code{cache.indexes(q)} = code_(:,q) ;
        end
        resetCache() ;
    end

    function appendCache(i,im)
        cache.numCached = cache.numCached + 1 ;
        cache.images{cache.numCached} = im ;
        cache.indexes(cache.numCached) = i;
        if cache.numCached >= opts.batchSize
            flushCache() ;
        end
    end

    code = {} ;
    for k=1:numel(im)
        appendCache(k, getImage(opts, single(im{k}), net.normalization.imageSize(1)));
    end
    flushCache() ;
end

% -------------------------------------------------------------------------
function reg = getImage(opts, im, regionSize)
% -------------------------------------------------------------------------
    reg = imresize(im, [regionSize, regionSize], 'bicubic') ;
end
