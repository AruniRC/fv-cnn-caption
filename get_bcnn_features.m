function code = get_bcnn_features(neta, netb, im, varargin)
% GET_BCNN_FEATURES  Get bilinear cnn features for an image
%   This function extracts the binlinear combination of CNN features
%   extracted from two different networks.

opts.crop = true ;
%opts.scales = 2.^(1.5:-.5:-3); % try a bunch of scales
opts.scales = 2;
opts.encoder = [] ;
opts.regionBorder = 0.05;
opts = vl_argparse(opts, varargin) ;

% get parameters of the network
info = vl_simplenn_display(neta) ;
borderA = round(info.receptiveField(end)/2+1) ;
averageColourA = mean(mean(neta.normalization.averageImage,1),2) ;
imageSizeA = neta.normalization.imageSize;

info = vl_simplenn_display(netb) ;
borderB = round(info.receptiveField(end)/2+1) ;
averageColourB = mean(mean(netb.normalization.averageImage,1),2) ;
imageSizeB = netb.normalization.imageSize;

assert(all(imageSizeA == imageSizeB));

if ~iscell(im)
  im = {im} ;
end

code = cell(1, numel(im));
% for each image
for k=1:numel(im)
    im_cropped = imresize(single(im{k}), imageSizeA([2 1]), 'bilinear');
    crop_h = size(im_cropped,1) ;
    crop_w = size(im_cropped,2) ;
    resA = [] ;
    resB = [] ;
    psi = cell(1, numel(opts.scales));
    % for each scale
    for s=1:numel(opts.scales)
        if min(crop_h,crop_w) * opts.scales(s) < max(borderA, borderB), continue ; end
        if sqrt(crop_h*crop_w) * opts.scales(s) > 1024, continue ; end

        % resize the cropped image and extract features everywhere
        im_resized = imresize(im_cropped, opts.scales(s)) ;
        im_resizedA = bsxfun(@minus, im_resized, averageColourA) ;
        im_resizedB = bsxfun(@minus, im_resized, averageColourB) ;
        if neta.useGpu
            im_resizedA = gpuArray(im_resizedA) ;
            im_resizedB = gpuArray(im_resizedB) ;
        end
        resA = vl_simplenn(neta, im_resizedA, [], resA, ...
                            'conserveMemory', true, 'sync', true);
        resB = vl_simplenn(netb, im_resizedB, [], resB, ...
                            'conserveMemory', true, 'sync', true);
        A = gather(resA(end).x);
        B = gather(resB(end).x);

        psi{s} = bilinear_pool(A,B);
        feat_dim = max(cellfun(@length,psi));
        code{k} = zeros(feat_dim, 1);
    end
    % pool across scales
    for s=1:numel(opts.scales),
        if ~isempty(psi{s}),
            code{k} = code{k} + psi{s};
        end
    end
    assert(~isempty(code{k}));
end

%
% square-root and l2 normalize (like: Improved Fisher?)
for k=1:numel(im),
    code{k} = sign(code{k}).*sqrt(abs(code{k}));
    code{k} = code{k}./(norm(code{k})+eps);
end

function psi = bilinear_pool(A, B)
w1 = size(A,2) ;
h1 = size(A,1) ;
w2 = size(B,2) ;
h2 = size(B,1) ;

%figure(1); clf;
%montage(reshape(A, [h1 w1 1 size(A,3)]));
%figure(2); clf;
%montage(reshape(B, [h2 w2 1 size(B,3)]));
%pause;

if w1*h1 <= w2*h2,
    %downsample B
    B = array_resize(B, w1, h1);
    A = reshape(A, [w1*h1 size(A,3)]);
else
    %downsample A
    A = array_resize(A, w2, h2);
    B = reshape(B, [w2*h2 size(B,3)]);
end

% bilinear pool
psi = A'*B;
psi = psi(:);

function Ar = array_resize(A, w, h)
numChannels = size(A, 3);
indw = round(linspace(1,size(A,2),w));
indh = round(linspace(1,size(A,1),h));
Ar = zeros(w*h, numChannels, 'single');
for i = 1:numChannels,
    Ai = A(indh,indw,i);
    Ar(:,i) = Ai(:);
end
