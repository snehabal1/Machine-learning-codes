%%
 yplot=Y;
 xplot1=X(:,1);
 polyval(xplot1,yplot,M2);
% xplot2=X(:,2);
% xplot3=X(:,3);
% xplot4=X(:,4);
% xplot5=X(:,5);
% xplot6=X(:,6);
% xplot7=X(:,7);
% xplot8=X(:,8);
% xplot9=X(:,9);
yplottemp=smooth(xplot1,yplot);
plot(xplot1,yplottemp,'*'); grid on;
% wplot1=w(:,1);
cftool;
