%%**********************************Loading the Data******************************%
%loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing the raw MNIST images
filename = '/Users/Sneha/Documents/UB/ML/Project3_digit recognition/imgFile.idx1-ubyte';
fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);
magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);
trainNumImages = fread(fp, 1, 'int32', 0, 'ieee-be');
trainNumRows = fread(fp, 1, 'int32', 0, 'ieee-be');
trainNumCols = fread(fp, 1, 'int32', 0, 'ieee-be');
trainImages = fread(fp, inf, 'unsigned char');
trainImages = reshape(trainImages, trainNumCols, trainNumRows, trainNumImages);
trainImages = permute(trainImages,[1 2 3]);
fclose(fp);
% Reshape to #pixels x #examples
trainImages = reshape(trainImages, size(trainImages, 1) * size(trainImages, 2), size(trainImages, 3));
% Convert to double and rescale to [0,1]
trainImages = double(trainImages) / 255;

%%
filename = '/Users/Sneha/Documents/UB/ML/Project3_digit recognition/t10k-images.idx3-ubyte';
fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);
magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);
validNumImages = fread(fp, 1, 'int32', 0, 'ieee-be');
validNumRows = fread(fp, 1, 'int32', 0, 'ieee-be');
validNumCols = fread(fp, 1, 'int32', 0, 'ieee-be');
validImages = fread(fp, inf, 'unsigned char');
validImages = reshape(validImages, validNumCols, validNumRows, validNumImages);
validImages = permute(validImages,[1 2 3]);
fclose(fp);
% Reshape to #pixels x #examples
validImages = reshape(validImages, size(validImages, 1) * size(validImages, 2), size(validImages, 3));
% Convert to double and rescale to [0,1]
validImages = double(validImages) / 255;


%%
%loadMNISTLabels returns a [number of MNIST images]x1 matrix containing the labels for the MNIST images
filename = '/Users/Sneha/Documents/UB/ML/Project3_digit recognition/labelFile.idx1-ubyte';
fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);
magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename, '']);
trainNumLabels = fread(fp, 1, 'int32', 0, 'ieee-be');
trainLabels = fread(fp, inf, 'unsigned char');
assert(size(trainLabels,1) == trainNumLabels, 'Mismatch in label count');
fclose(fp);

maintrainLabels = zeros(trainNumImages,10);
cntr = 1;
while cntr <= trainNumImages
    i = trainLabels(cntr,1);
    maintrainLabels(cntr,(i+1)) = 1;
    cntr = cntr + 1;
end  
%%
filename = '/Users/Sneha/Documents/UB/ML/Project3_digit recognition/t10k-labels.idx1-ubyte';
fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);
magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename, '']);
validNumLabels = fread(fp, 1, 'int32', 0, 'ieee-be');
validLabels = fread(fp, inf, 'unsigned char');
assert(size(validLabels,1) == validNumLabels, 'Mismatch in label count');
fclose(fp);

mainvalidLabels = zeros(validNumImages,10);
cntr = 1;
while cntr <= validNumImages
    i = validLabels(cntr,1);
    mainvalidLabels(cntr,(i+1)) = 1;
    cntr = cntr + 1;
end

%% Perform One of K validation on the training dataset

Wlr = ones(784,10);
ak = zeros(1,10);
yk = zeros(60000,10);
eta = 0.01;
% 'ones' is added for bias
for i=1:60000
        ak = (Wlr.' * trainImages(:,i)).' + ones(1,10); 
        Max_Elevalue = max(ak);
        yk(i,:) = exp(ak/Max_Elevalue)/sum(exp(ak/Max_Elevalue));
        Max_Indexvalue = find(yk(i,:)==max(yk(i,:)));
         for l = 1:10
              if(l==Max_Indexvalue)
                     yk(i,l) = 0.999999;
              else
                    yk(i,l) = 0.0001;
              end
         end
    Wlr = Wlr - (eta * trainImages(:,i) * (yk(i,:) - maintrainLabels(i,:)) );
end
%%
correct = 0;
yk2 = zeros(10000,10);
blr = ones(1,10);
for i=1:10000
        ak = (Wlr.' * validImages(:,i)).' + ones(1,10); % ones for bias factor
        Max_Elevalue = max(ak);
        yk2(i,:) = exp(ak/Max_Elevalue)/sum(exp(ak/Max_Elevalue));
        Max_Indexvalue = find(yk2(i,:)==max(yk2(i,:)));
        for l = 1:10
            if(l==Max_Indexvalue)
                yk2(i,l) = 1;
            else
                yk2(i,l) = 0;
            end
        end
%% Calculate correct calculation rate.    
        if(yk2(i,:)==mainvalidLabels(i,:))
             correct = correct + 1;
        end    
        w1 = w1 - (0.5 * trainImages(:,i) * (yk(i,:) - maintrainLabels(i,:)) );
end  

%% Classification based on neural network.
%% Load images
clc
% Training data
trainimg = loadMNISTImages('/Users/Sneha/Documents/UB/ML/Project3_digit recognition/imgFile.idx1-ubyte');
trainlbl = loadMNISTLabels('/Users/Sneha/Documents/UB/ML/Project3_digit recognition/labelFile.idx1-ubyte');
% Testing data
testimg = loadMNISTImages('/Users/Sneha/Documents/UB/ML/Project3_digit recognition/t10k-images.idx3-ubyte');
testlbl = loadMNISTLabels('/Users/Sneha/Documents/UB/ML/Project3_digit recognition/t10k-labels.idx1-ubyte');

% Define number of training images
% Define number of validation images
trainNumImages = 60000;
validNumImages = 10000;
% Define count of training data
N = 60000;
K = 10;
Jarray = [300];
D = 784;

%% Follow loop for size(Jarray) = 300 
for J_size = 1: size(Jarray , 2)
         J = Jarray(J_size);
% Define Activation function sigmoid
         h = 'sigmoid';
   sigmoid = @(a) 1.0./(1.0 + exp(-a));

%% Training dataset
% Perform regression on this dataset
% Initialize classification rate value
eta_arr = [0.01];

for qr = 1: size(eta_arr , 2)
    eta = eta_arr(qr);
    fprintf('eta : %d ',eta);
    plot(eta, eta_arr);
 
%% Random generation of Wnn and bnn for training dataset.
Wnn1 = randn(D,J);
Wnn2 = randn(J,K);
%
bnn1 =  zeros(1,J);
bnn2 =  zeros(1,K);

%% Initialize training data with padding all training data with 0
% Add training data to t_train_k 
t_train_k = zeros(N,K);       
for n = 1:N
    t_train_k(n,trainlbl(n)+1)=1;         
end

%%
% Check misclassification rate on the model by varying epochs constraint 
for epochs = 1:30
 for Tau_value = 1: N
         Arr_j = zeros(1,J);
         Arr_k = zeros(1,K);
    
 z = zeros(1,J);
 y = zeros(1,K);
 
%% Calculate Arr_j value for J values using Wnn and bnn value for training
% dataset
    for j= 1:J
        Arr_j(j) = (Wnn1(:,j)'* trainimg(:,Tau_value)) + bnn1(j); 
    end
    z = sigmoid(Arr_j);
    
%% Calculate Arr_k value for k values using Wnn and bnn values and the
% sigmoid activation function
    for k = 1:K 
        Arr_k(k) = (Wnn2(:,k)'* z') + bnn2(k);
    end

%% Calculate y values for the training set.
Next_y = sum(exp(Arr_k));
    for k = 1:K 
        y(k) = exp(Arr_k(k))./(Next_y);
    end
    
esc_k = y - t_train_k(Tau_value,:);
    for j=1:J
        for k=1:10
            err1(k) = Wnn2(j,k).*esc_k(k);
        end
         esc_j(j) = (sigmoid(z(j)).*(1 - sigmoid(z(j)))).*(sum(err1)) ;
    end
    
%% updating weights for the model
    for j=1:J
        Wnn1(:,j) = Wnn1(:,j) - eta.*(esc_j(j).*trainimg(:,Tau_value));
    end
    
    for k=1:K
        Wnn2(:,k) = Wnn2(:,k) - (eta.*(esc_k(k).*z))';
    end
 end

 
end
%% Testing dataset
% perform regression on this dataset

N1 = 10000;

right_counter=0;
wrong_counter=0;

for Tau_value = 1:N1
    Arr_j = zeros(1,J);
    Arr_k = zeros(1,K);
    
    z = zeros(1,J);
    y = zeros(1,K);
    
    for j= 1:J
        Arr_j(j) = (Wnn1(:,j)'* testimg(:,Tau_value)) + bnn1(j); 
        z(j) = sigmoid(Arr_j(j));                
    end
    
    for k = 1:K 
        Arr_k(k) = (Wnn2(:,k)'* z') + bnn2(k);
    end
%     ak_max = max(Arr_k);
    Next_y = sum(exp(Arr_k));
    for k = 1:K 
        y(k) = exp(Arr_k(k))./(Next_y);
    end

        [val IDx]=max(y);
	if(IDx==testlbl(Tau_value,1)+1)
		right_counter=right_counter+1;
	else
		wrong_counter=wrong_counter+1;
    end
end
%% Print values of classification rate and J 
fprintf('eta : %d ',eta);
fprintf('\tJ : %d ',J);
%% Print right and wrong counters
fprintf('\t Right : %d ',right_counter);
fprintf('\t Wrong : %d \n',wrong_counter);
end

end

%% Convolutional neural network Taken from github repo mentioned in project requirement.
%% PLease note. download full cnn folder for data and functions to run this section of the code.
% ******requires CNN folder downloaded from github to run********
%Convolutional neural network for handwriten digits recognition: training
%and simulation.
%(c)Mikhail Sirotenko, 2009.
%This program implements the convolutional neural network for MNIST handwriten 
%digits recognition, created by Yann LeCun. CNN class allows to make your
%own convolutional neural net, defining arbitrary structure and parameters.
%It is assumed that MNIST database is located in './MNIST' directory.
%References:
%URL: http://web.mit.edu/jvb/www/cv.html

clear;
clc;
%Load the digits into workspace
[I,labels,I_test,labels_test] = readMNIST(1000); 
%%

%Define the structure according to [2]
%Total number of layers
numLayers = 8; 
%Number of subsampling layers
numSLayers = 3; 
%Number of convolutional layers
numCLayers = 3; 
%Number of fully-connected layers
numFLayers = 2;
%Number of input images (simultaneously processed). Need for future
%releases, now only 1 is possible
numInputs = 1; 
%Image width
InputWidth = 32; 
%Image height
InputHeight = 32;
%Number of outputs
numOutputs = 10; 
%Create an empty convolutional neural network with deined structure
sinet = cnn(numLayers,numFLayers,numInputs,InputWidth,InputHeight,numOutputs);

%Now define the network parameters


%Due to implementation specifics layers are always in pairs. First must be
%subsampling and last (before fulli connected) is convolutional layer.
%That's why here first layer is dummy.
sinet.SLayer{1}.SRate = 1;
sinet.SLayer{1}.WS{1} = ones(size(sinet.SLayer{1}.WS{1}));
sinet.SLayer{1}.BS{1} = zeros(size(sinet.SLayer{1}.BS{1}));
sinet.SLayer{1}.TransfFunc = 'purelin';
%Weights 1
%Biases 1


%Second layer - 6 convolution kernels with 5x5 size 
sinet.CLayer{2}.numKernels = 6;
sinet.CLayer{2}.KernWidth = 5;
sinet.CLayer{2}.KernHeight = 5;
%Weights 150
%Biases 6

%Third layer
%Subsampling rate
sinet.SLayer{3}.SRate = 2;
%Weights 6
%Biases 6

%Forth layer - 16 kernels with 5x5 size 
sinet.CLayer{4}.numKernels = 16;
sinet.CLayer{4}.KernWidth = 5;
sinet.CLayer{4}.KernHeight = 5;
%Weights 150
%Biases 6

%Fifth layer
%Subsampling rate
sinet.SLayer{5}.SRate = 2;
%Weights 6
%Biases 6

%Sixth layer - outputs 120 feature maps 1x1 size
sinet.CLayer{6}.numKernels = 120;
sinet.CLayer{6}.KernWidth = 5;
sinet.CLayer{6}.KernHeight = 5;
%Weights 3000
%Смещений 120

%Seventh layer - fully connected, 84 neurons
sinet.FLayer{7}.numNeurons = 84;
%Weights 10080
%Biases 84

%Eight layer - fully connected, 10 output neurons
sinet.FLayer{8}.numNeurons = 10;
%Weights 840
%Biases 10

%Initialize the network
sinet = init(sinet);

%According to [2] the generalisation is better if there's unsimmetry in
%layers connections. Yann LeCun uses this kind of connection map:
sinet.CLayer{4}.ConMap = ...
[1 0 0 0 1 1 1 0 0 1 1 1 1 0 1 1;
 1 1 0 0 0 1 1 1 0 0 1 1 1 1 0 1;
 1 1 1 0 0 0 1 1 1 0 0 1 0 1 1 1;
 0 1 1 1 0 0 1 1 1 1 0 0 1 0 1 1;
 0 0 1 1 1 0 0 1 1 1 1 0 1 1 0 1; 
 0 0 0 1 1 1 0 0 1 1 1 1 0 1 1 1; 
]';
%but some papers proposes to randomly generate the connection map. So you
%can try it:
%sinet.CLayer{6}.ConMap = round(rand(size(sinet.CLayer{6}.ConMap))-0.1);
sinet.SLayer{1}.WS{1} = ones(size(sinet.SLayer{1}.WS{1}));
sinet.SLayer{1}.BS{1} = zeros(size(sinet.SLayer{1}.BS{1}));
%In my impementation output layer is ordinary tansig layer as opposed to
%[1,2], but I plan to implement the radial basis output layer

%sinet.FLayer{8}.TransfFunc = 'radbas';


%%
%Now the final preparations
%Number of epochs
sinet.epochs = 3;
%Mu coefficient for stochastic Levenberg-Markvardt
sinet.mu = 0.001;
%Training coefficient
%sinet.teta =  [50 50 20 20 20 10 10 10 5 5 5 5 1]/100000;
sinet.teta =  0.0005;
%0 - Hessian running estimate is calculated every iteration
%1 - Hessian approximation is recalculated every cnet.Hrecomp iterations
%2 - No Hessian calculations are made. Pure stochastic gradient
sinet.HcalcMode = 0;    
sinet.Hrecalc = 300; %Number of iterations to passs for Hessian recalculation
sinet.HrecalcSamplesNum = 50; %Number of samples for Hessian recalculation

%Teta decrease coefficient
sinet.teta_dec = 0.4;

%Images preprocessing. Resulting images have 0 mean and 1 standard
%deviation. Go inside the preproc_data for details
[Ip, labtrn] = preproc_data(I,1000,labels,0);
[I_testp, labtst] = preproc_data(I_test,100,labels_test,0);
%Actualy training
sinet = train(sinet,Ip,labtrn,I_testp,labtst);