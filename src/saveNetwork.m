% -------------------------------------------------------------------------
function saveNetwork(fileName, net, ignoreDropout)
% -------------------------------------------------------------------------
if nargin < 3,
   ignoreDropout=false;
end

layers = net.layers;

% Replace the last layer with softmax
layers{end}.type = 'softmax';
layers{end}.name = 'prob';

% Remove fields corresponding to training parameters
ignoreFields = {'filtersMomentum', ...
                'biasesMomentum',...
                'filtersLearningRate',...
                'biasesLearningRate',...
                'filtersWeightDecay',...
                'biasesWeightDecay',...
                'class'};
for i = 1:length(layers),
    layers{i} = rmfield(layers{i}, ignoreFields(isfield(layers{i}, ignoreFields)));
end
classes = net.classes;
normalization = net.normalization;

% Remove dropout layers from the final model
if ignoreDropout
  dropoutLayer = cellfun(@(x) strcmp(x.type, 'dropout'), layers);
  layers = layers(~dropoutLayer);
end

save(fileName, 'layers', 'classes', 'normalization');
