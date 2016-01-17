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