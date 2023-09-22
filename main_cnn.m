close all
clear
clc

% From:
% https://www.mathworks.com/help/deeplearning/ug/
% create-simple-deep-learning-network-for-classification.html

% Load the digit sample data as an image datastore. imageDatastore automatically
% labels the images based on folder names and stores the data as an ImageDatastore
% object.

% An image datastore enables you to store large image data, including data that
% does not fit in memory,
% and efficiently read batches of images during training of a convolutional neural
% network.

digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');

imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% Display some of the images in the datastore.
figure;
perm = randperm(10000,20);
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
end

% Calculate the number of images in each category. labelCount is a table that
% contains the labels and the number of images having each label.
% The datastore contains 1000 images for each of the digits 0-9, for a total of
% 10000 images.
% You can specify the number of classes in the last fully connected layer of your
% neural network as the OutputSize argument.
% labelCount = countEachLabel(imds)
% You must specify the size of the images in the input layer of the neural network.
% Check the size of the first image in digitData.
% Each image is 28-by-28-by-1 pixels.

img = readimage(imds,1);
size(img)

% Divide the data into training and validation data sets,
% so that each category in the training set contains 750 images, and the validation
% set contains the remaining images from each label. splitEachLabel splits the
% datastore digitData into two new datastores, trainDigitData and valDigitData.

numTrainFiles = 750;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');

% Define the convolutional neural network architecture.
layers = [
    imageInputLayer([28 28 1])
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

% Display the network
analyzeNetwork(layers);
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
        
% https://www.mathworks.com/help/deeplearning/ref/trainnetwork.html
net = trainNetwork(imdsTrain,layers,options);

% Predict the labels of the validation data using the trained neural network, and
% calculate the final validation accuracy.
% Accuracy is the fraction of labels that the neural network predicts correctly.
% In this case, more than 99% of the predicted labels match the true labels of the
% validation set.
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

CM = zeros(10, 10);

for i=1:2500
    x = grp2idx(YPred(i));
    y = grp2idx(YValidation(i));
    CM(x, y)=CM(x, y)+1;
end 

accuracy = sum(YPred == YValidation)/numel(YValidation);




