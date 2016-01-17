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