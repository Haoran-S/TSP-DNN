% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % MATLAB code to generate the model for IMAC case.
% % % To get results for other sections, slightly modification may apply.
% % % Code has been tested successfully on MATLAB 2016b platform.
% % %
% % % References:
% % % [1] Haoran Sun, Xiangyi Chen, Qingjiang Shi, Mingyi Hong, Xiao Fu, and Nicholas D. Sidiropoulos.
% % % "Learning to optimize: Training deep neural networks for interference management."
% % % IEEE Transactions on Signal Processing 66, no. 20 (2018): 5438-5453.
% % % version 1.0 -- February 2017.
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %


%% locations of BSs
Rcell = Cell.Rcell;       % radius of each cell
Nbs   = Cell.Ncell;    % total number of cells in the scenario

Cell.Position	= zeros(Nbs,2);
% if (Nbs==1)
%     error('Only one cell?');
% end
if (Nbs>1)
    theta                   = (0:Nbs-2)'*pi/3;
    Cell.Position(2:end,1:2)	= sqrt(3)*Rcell*[cos(theta) sin(theta)];
end
if (Nbs>7)
    theta = -pi/6:pi/3:5/3*pi;
    x     = 3*Rcell*cos(theta);
    y     = 3*Rcell*sin(theta);
    theta = 0:pi/3:5/3*pi;
%     [x; 2*sqrt(3)*Rcell*cos(theta)];
     x     = reshape([x; 2*sqrt(3)*Rcell*cos(theta)],numel([x; 2*sqrt(3)*Rcell*cos(theta)]),1);
     y     = reshape([y; 2*sqrt(3)*Rcell*sin(theta)],numel([y; 2*sqrt(3)*Rcell*sin(theta)]),1);
    if Nbs>19
        Cell.Position(8:19,1:2)  = [x y];
    else
        Cell.Position(8:Nbs,1:2) = [x(1:(Nbs-7)) y(1:(Nbs-7))];
    end
end
if (Nbs>19) && (Nbs<38)
    theta  = -asin(3/sqrt(21)):pi/3:5/3*pi;
    x1     =  sqrt(21)*Rcell*cos(theta);
    y1     =  sqrt(21)*Rcell*sin(theta);
    theta  = -asin(3/2/sqrt(21)):pi/3:5/3*pi;
    x2     =  sqrt(21)*Rcell*cos(theta);
    y2     =  sqrt(21)*Rcell*sin(theta);
    theta  =  0:pi/3:5/3*pi;
    x3     =  3*sqrt(3)*Rcell*cos(theta);
    y3     =  3*sqrt(3)*Rcell*sin(theta);
    x      =  reshape([x1;x2;x3],numel([x1;x2;x3]),1);
    y      =  reshape([y1;y2;y3],numel([y1;y2;y3]),1);
    Cell.Position(20:Nbs,1:2) = [x(1:(Nbs-19)) y(1:(Nbs-19))];
end


%% cells layout plot
flag = 0;
if flag
    figure
%     figure(4298);%set(gcf,'DockControls','on','WindowStyle','docked','NumberTitle','off','Name','Multi-cell Scenario');
    clrstr    = [repmat([1 0 0],Nbs,1)];
    for Icell = 1 : Nbs
        x0 = Cell.Position(Icell,1);
        y0 = Cell.Position(Icell,2);
        plot(x0,y0,'^','MarkerSize',10,'color',clrstr(Icell,:),'MarkerFaceColor',clrstr(Icell,:));hold on;
        x1 = Rcell*cos(-pi/6:pi/3:2*pi) + x0;
        y1 = Rcell*sin(-pi/6:pi/3:2*pi) + y0;
        plot(x1,y1,'k');
        text(x0+35,y0+35,num2str(Icell),'FontSize',12);
    end
    axis equal;
end
