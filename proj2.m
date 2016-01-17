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
%%
%%Create matrix and initialize to all zeros
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

%% create va;idation data
validationIndexes = X_rand(noTrainDocs+1:noValidationDocs);
X_validation = X_new(validationIndexes,:);
Y_validation = Y_new(validationIndexes,:);
%% create test data
testIndexes = X_rand(noValidationDocs+1:end);
X_test = X_new(testIndexes,:);
Y_test = Y_new(testIndexes,:);

trainInd1 = trainingIndexes;
validInd1 = validationIndexes;
noValidationDocs = noValidationDocs-noTrainDocs; 
%%
fclose(fid);

%%calculate mu1
M1= 5;
mu1 = zeros(46,M1);

Xforclosed = randperm(noTrainDocs,M1);
Xforclosed = (Xforclosed).';
% populate mu1 matrix
for i= 1 : M1
    mu1(:,i) = X_training(Xforclosed(i),:); 
end
%%
%plot(mu1);
%%
% populate sigma
sigma = zeros(46:46);
sigma = var(X_training)*0.1;
for i=1:46
     if sigma(1,i) < 0.0001 
         sigma(1,i) = 0.01;
     end
 end
 
 sigma = diag(sigma);
 D= 46;
 Sigma1= zeros(D,D);
 
 for i= 1:M1
     Sigma1(:,:,i) = sigma;
 end
 %%
 %% Calculate design matrix for training set
phi = zeros(noTrainDocs,M1);
phi(:,1)= 1;

   
for j= 2 : M1

   for i = 1 : noTrainDocs
   a= inv(Sigma1(:,:,j));   
   b= (X_training(i,:).'-mu1(:,j)).';
   c= (X_training(i,:).'-mu1(:,j));
   d= -0.5 * b * a * c;
   phi(i,j) = exp(d);
   end
   
end

%%
lambda1= 0.2;

w1 = inv( lambda1*eye(M1,M1)+ phi.'*phi)*phi.'*Y_training;

%%
%plot(w1,lambda1);
%% Root mean square error for training set

Err= 0.5 * ((Y_training-(phi*w1)).')*(Y_training-(phi*w1));
trainPer1 = sqrt((2*Err)/noTrainDocs);

%% For validation set

%calculate design matrix
phi_valid = zeros(noValidationDocs,M1);
phi_valid(:,1)= 1;

for j= 2 : M1

   for i = 1 : noValidationDocs
   a= inv(Sigma1(:,:,j));   
   b= (X_validation(i,:).'-mu1(:,j)).';
   c= (X_validation(i,:).'-mu1(:,j));
   d= -0.5 * b * a * c;
   phi_valid(i,j) = exp(d);
   end
   
end
%%
lambda1= 0.2;

%w1_valid = inv( lambda1*eye(M1,M1)+ phi_valid.'*phi_valid)*phi_valid.'*Y_validation;

%% Root mean square error for training set

Err_valid= 0.5 * ((Y_validation-(phi_valid*w1)).')*(Y_validation-(phi_valid*w1));
validPer1 = sqrt((2*Err_valid)/noValidationDocs);

%%
%plot(lambda1,validPer1);
%%
w02 = w02.';
%%%%
%Create matrix and initialize to all zeros

load('synthetic.mat');
noTrainSyn = floor(0.9 * length(x));
noValidationSyn= size(x,2)- noTrainSyn;
X = x.';
Y = t;

%%
plot(X,Y);
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
%%
%%
% Stochastic gradient models
eta1= zeros(1,noTrainDocs);
eta1(1,:)=1; 
dw1=zeros(M1,noTrainDocs);
w01= 100*rand(1,M1);
%%
prevw= w01;
%%
for i= 1: noTrainDocs
deltaed = -(Y_training(i,:)- (prevw)*phi(i,:).')*phi(i,:);
deltae = deltaed + (lambda1* prevw);
deltaw = (-1*eta1(1,i)) * deltae;
dw1(:,i) = deltaw;
prevw = prevw+ deltaw;
end
%%
w01= w01.';

%%
%synthetic data
% Stochastic gradient models
eta2= zeros(1,noTrainSyn);
eta2(1,:)=1; 
dw2=zeros(M2,noTrainSyn);
w02= 100*rand(1,M2);
%%
prevw2= w02;
%%
for i= 1: noTrainSyn
deltaed = -(Y_training_syn(i,:)- (prevw2)*phi_syn(i,:).')*phi_syn(i,:);
deltae = deltaed + (lambda2* prevw2);
deltaw = (-1*eta2(1,i)) * deltae;
dw2(:,i) = deltaw;
prevw2 = prevw2+ deltaw;
end

%%
 yplot=Y;
 xplot1=X(:,1);
 polyval(xplot1,yplot,M2);
xplot2=X(:,2);
xplot3=X(:,3);
xplot4=X(:,4);
xplot5=X(:,5);
xplot6=X(:,6);
xplot7=X(:,7);
xplot8=X(:,8);
xplot9=X(:,9);
yplottemp=smooth(xplot1,yplot);
plot(xplot1,yplottemp,'*'); grid on;
% wplot1=w(:,1);
cftool;
%% hyper tuning
%%Create matrix and initialize to all zeros

load('synthetic.mat');
noTrainSyn = floor(0.9 * length(x));
noValidationSyn= size(x,2)- noTrainSyn;
X = x.';
Y = t;
minvalidper=1000;
%%
%plot(X,Y);
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
for M2 1:9

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
%lambda2= 0.2;
for lambda = 0.2:0.9
    
w2 = inv( lambda2*eye(M2,M2)+ phi_syn.'*phi_syn)*phi_syn.'*Y_training_syn;
end
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

if minvalidper > validper2
   minmu2= mu2;
   minSigma2= Sigma2;
   minphi2=phi_syn_valid;
   
end
end
%%

