function y = vl_l2norm(x)
% VL_L2NORM computes l2 normalization at each location

gpuMode = isa(x, 'gpuArray');

[h, w, ch, bs] = size(x);
if gpuMode
    y = gpuArray(zeros([h, w, ch, bs], 'single'));
else
    y = zeros([h, w, ch, bs], 'single');
end
for b = 1:bs,
    for yy = 1:h,
        for xx = 1:w,
            A = squeeze(x(yy,xx,:,b));
            y(yy,xx,:, b) = A./sqrt(A'*A+eps);
        end
    end
end