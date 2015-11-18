function y = vl_nnbilinear(x)
% VL_NNBILINEAR  bilinear features at each location

gpuMode = isa(x, 'gpuArray');

[h, w, ch, bs] = size(x);
if gpuMode
    y = gpuArray(zeros([h, w, ch*ch, bs], 'single'));
else
    y = zeros([h, w, ch*ch, bs], 'single');
end

for b = 1:bs,
    for yy = 1:h,
        for xx = 1:w,
            a = squeeze(x(yy,xx,:,b));
            y(yy,xx,:, b) = reshape(a*a', [1 ch*ch]);
        end
    end
end