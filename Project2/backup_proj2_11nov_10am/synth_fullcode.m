%%
%Create matrix and initialize to all zeros

load('synthetic.mat');
noTrainSyn = floor(0.9 * length(x));
noValidationSyn= size(x,2)- noTrainSyn;
X = x.';
Y = t;

%% Divide data into training, test and validation

X_rand_syn = randperm(length(X));
X_rand_syn = (X_rand_syn).';
trainingIndexes_syn = X_rand_syn(1:noTrainSyn);
validationIndexes_syn = X_rand_syn(noTrainSyn+1:end);
%%

X_training_syn = X(trainingIndexes_syn,:);
Y_training_syn = Y(trainingIndexes_syn,:);
trainInd2 = trainingIndexes_syn;
%% 

M2= 5;
mu2 = zeros(10,M2);

Xforclosed_syn = randperm(noTrainSyn,M2);
Xforclosed_syn = (Xforclosed_syn).';
% populate mu2 matrix
for i= 1 : M2
    mu2(:,i) = X_training_syn(Xforclosed_syn(i),:); 
end
%% populate sigma_syn
sigma_syn = zeros(10:10);
sigma_syn = var(X_training_syn)*0.1;
for i=1:10
     if sigma_syn(1,i) < 0.0001 
         sigma_syn(1,i) = 0.01;
     end
 end
 
 sigma_syn = diag(sigma_syn);
 D= 10;
 Sigma2= zeros(D,D);
 
 for i= 1:M2
     Sigma2(:,:,i) = sigma_syn;
 end
 

%% Calculate design matrix for training set
phi_syn = zeros(noTrainSyn,M2);
phi_syn(:,1)= 1;
   
for j= 2 : M2

   for i = 1 : noTrainSyn
   a1= inv(Sigma2(:,:,j));   
   b1= (X_training_syn(i,:).'-mu2(:,j)).';
   c1= (X_training_syn(i,:).'-mu2(:,j));
   d1= -0.5 * b1 * a1 * c1;
   phi_syn(i,j) = exp(d1);
   end
   
end

%%
lambda2= 0.2;

w2 = inv( lambda2*eye(M2,M2)+ phi_syn.'*phi_syn)*phi_syn.'*Y_training_syn;

%% Root mean square error for training set

Err2= 0.5 * ((Y_training_syn-(phi_syn*w2)).')*(Y_training_syn-(phi_syn*w2));
trainPer2 = sqrt((2*Err2)/noTrainSyn);


%%
%Validation data calculations
X_validation_syn = X(validationIndexes_syn,:);
Y_validation_syn = Y(validationIndexes_syn,:);
validInd2 = validationIndexes_syn;

%% Calculate design matrix for training set
phi_syn_valid = zeros(noValidationSyn,M2);
phi_syn_valid(:,1)= 1;
   
for j= 2 : M2

   for i = 1 : noValidationSyn
   a1= inv(Sigma2(:,:,j));   
   b1= (X_training_syn(i,:).'-mu2(:,j)).';
   c1= (X_training_syn(i,:).'-mu2(:,j));
   d1= -0.5 * b1 * a1 * c1;
   phi_syn_valid(i,j) = exp(d1);
   end
   
end

%% Root mean square error for validation set

Err2syn= 0.5 * ((Y_validation_syn-(phi_syn_valid*w2)).')*(Y_validation_syn-(phi_syn_valid*w2));
%%
validPer2 = sqrt((2*Err2syn)/noValidationSyn);
