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
