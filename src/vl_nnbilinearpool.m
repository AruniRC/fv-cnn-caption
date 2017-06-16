function y = vl_nnbilinearpool(x)
% VL_NNBILINEARPOOL pools bilinear feature across all locations

gpuMode = isa(x, 'gpuArray');

[h, w, ch, bs] = size(x);
if gpuMode
    y = gpuArray(zeros([1, 1, ch*ch, bs], 'single'));
else
    y = zeros([1, 1, ch*ch, bs], 'single');
end

for b = 1:bs,
    a = reshape(x(:,:,:,b), [h*w, ch]);
    y(1,1,:, b) = reshape(a'*a, [1 ch*ch]);
end
