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
w02 = w02.';