%% 
M1= 5;
mu1 = zeros(46,M1);

Xforclosed = randperm(noTrainDocs,M1);
Xforclosed = (Xforclosed).';
% populate mu1 matrix
for i= 1 : M1
    mu1(:,i) = X_training(Xforclosed(i),:); 
end
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