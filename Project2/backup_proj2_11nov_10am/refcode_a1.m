%% Read data and convert to matrix form
%fid = fopen('Querylevelnorm.txt','rt');
fid = fopen('Querylevelnorm.txt','rt');
%Read data row-wise
wholerowread = textscan(fid,'%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s',69623);
% Read relevance
Y = wholerowread{1,1};
%Read features
parameters = wholerowread(:,3:48);
%Convert features to vector
X = [parameters{:}];

%Create matrix and initialize to all zeros
X_new = zeros(69623,46);
[rowSize,columnSize] = size(X);
for k=1:rowSize
    for j=1:columnSize
    %For X_new enter matrix values from position 3 to 48(features)
        strings = strsplit(X{k,j},':');
        X_new(k,j) = str2double(strings(2));
    end    
end
Y_new = str2double(Y);
%%Divide data into random training set and apply random permutation to
% XVectors
%% Divide data into training, test and validation
noTrainDocs = floor(0.8 * length(X_new));
noValidationDocs = floor(0.9 * length(X_new));

X_rand = randperm(length(X_new));
X_rand = (X_rand).';

trainingIndexes = X_rand(1:noTrainDocs);
X_training = X_new(trainingIndexes,:);
Y_training = Y_new(trainingIndexes,:);

validationIndexes = X_rand(noTrainDocs+1:noValidationDocs);
X_validation = X_new(validationIndexes,:);
Y_validation = Y_new(validationIndexes,:);

testIndexes = X_rand(noValidationDocs+1:end);
X_test = X_new(testIndexes,:);
Y_test = Y_new(testIndexes,:);

trainInd1 = trainingIndexes;
validInd1 = validationIndexes;
noValidationDocs = noValidationDocs-noTrainDocs; 
%%
fclose(fid);