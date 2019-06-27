% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % MATLAB code to generate the model for IMAC case.
% % % To get results for other sections, slightly modification may apply.
% % % Code has been tested successfully on MATLAB 2016b platform.
% % % 
% % % For Example, [H, X, Y] = Generate_IMAC_function(20, 4, 10, 100, 0.2, 0, 1);
% % % Generate 10 samples of 20 BS each with 4 users, of radius 100
% % % Output the channel H (size 6400), input X (size 1600), and label Y (size 80, created by WMMSE) 
% % %
% % % References:
% % % [1] Haoran Sun, Xiangyi Chen, Qingjiang Shi, Mingyi Hong, Xiao Fu, and Nicholas D. Sidiropoulos.
% % % "Learning to optimize: Training deep neural networks for interference management."
% % % IEEE Transactions on Signal Processing 66, no. 20 (2018): 5438-5453.
% % % version 1.0 -- February 2017.
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

function [H, X, Y] = Generate_IMAC_function(num_BS, num_User, num_H, R, minR_ratio, seed, var_noise)
tic
rng(seed);
K = num_User * num_BS;
X=zeros(K*num_BS,num_H);
Y=zeros(K,num_H);
HHH=zeros(K, K, num_H);
temp_H=zeros(num_BS,K);
for i = 1:num_H
    HHH(:,:,i) = generate_IBC_channel(num_User,R,num_BS, minR_ratio);
    for l = 1 : num_BS
        temp_H(l,:)=HHH((l-1)*num_User+1,:,i);
    end
    Y(:,i) = WMMSE_sum_rate(ones(K,1), HHH(:,:,i), ones(K,1), var_noise);
    X(:,i) = reshape(temp_H, K*num_BS, 1);
end
toc
H=reshape(HHH, K*K, num_H);
end



function p_opt = WMMSE_sum_rate(p_int, H, Pmax, var_noise)
K = length(Pmax);
b = sqrt(p_int);
b_pre = zeros(K,1);
f = zeros(K, 1);
w = f;
vnew = 0;
for i=1:K
    f(i) = H(i,i)*b(i)/((H(i,:).^2)*(b.^2)+var_noise);
    w(i) = 1/(1-f(i)*b(i)*H(i,i));
    vnew = vnew + log2(w(i));
end

iter = 0;
while(1)
    iter = iter+1;
    vold = vnew;
    for i=1:K
        btmp = w(i)*f(i)*H(i,i)/sum(w.*(f.^2).*(H(:,i).^2));
        b_pre(i) = b(i);
        b(i) = min(btmp, sqrt(Pmax(i))) + max(btmp, 0) - btmp;
    end
    
    vnew = 0;
    for i=1:K
        f(i) = H(i,i)*b(i)/((H(i,:).^2)*(b.^2)+var_noise);
        w(i) = 1/(1-f(i)*b(i)*H(i,i));
        vnew = vnew + log2(w(i));
    end
    
    if vnew-vold <= 1e-5 || iter>500
        break;
    end
end
p_opt = b.^2;
end

function H_eq = generate_IBC_channel(Num_of_user_in_each_cell, cell_distance, Num_of_cell, minR_ratio)
% Generate Num_of_users*Num_of_users interference channel
% Num_of_cell cells, each with Num_of_user_in_each_cell user
T        = 1; % number of BS antennas
BaseNum  = 1; % number of BSs in each cell
UserNum  = Num_of_user_in_each_cell;
Distance = cell_distance;
CellNum  = Num_of_cell;

% Cell environment channel
Cell.Ncell      = CellNum;   % number of coordinated cells
Cell.Nintra     = UserNum;   % number of cell-intra users
Cell.NintraBase = BaseNum;   % number of cell-basestations
Cell.Rcell      = Distance*2/sqrt(3); % cell radius
Layout; % Layout of cells

% If consider cellular environment, generate the corresponding channel matrix
[MS, BS] = usergenerator(Cell, minR_ratio);
[HLarge] = channelsample(BS,MS,Cell);
H=(randn(T,BaseNum,CellNum,UserNum,CellNum)+sqrt(-1)*randn(T,BaseNum,CellNum,UserNum,CellNum))/sqrt(2);
for Base=1:BaseNum
    for CellBS=1:CellNum
        for User=1:UserNum
            for CellMS=1:CellNum
                H(:,Base,CellBS,User,CellMS)=H(:,Base,CellBS,User,CellMS)*sqrt(HLarge(User,CellMS,Base,CellBS));
                %between user in Cell MS and base station in Cell BS
            end
        end
    end
end

total_user_Num = CellNum*UserNum;
H_eq = zeros(total_user_Num, total_user_Num);
k = 0;
for CellMS=1:CellNum
    for User=1:UserNum
        k = k+1;
        k_INF = 0;
        for INFCellMS=1:CellNum
            for INFUser=1:UserNum
                k_INF = k_INF+1; %(INFCellMS-1)*CellNum+INFUser;
                H_eq(k, k_INF) = abs(H(:, Base, CellMS, INFUser, INFCellMS));
            end
        end
    end
end

%SINRRequirementdB=0; % The value of 1/\sigma_n^{2} in dB
%Sigma Square = 10^(-SINRRequirementdB/10);
end


function [Hlarge] = channelsample(BS,MS,Cell)
Ncell	= Cell.Ncell;      % # of cells
Nintra  = Cell.Nintra;     % # of MSs in each cell
Nbs     = Cell.NintraBase; % # of BSs in each cell

% Channel between BSs and Cell-intra MSs
Hlarge	  = zeros( Nintra, Ncell, Nbs, Ncell );

% large-scale fading
for CellBS = 1 : Ncell
    for CellMS = 1 : Ncell
        for Base = 1 : Nbs
            for User = 1 : Nintra
                d = norm(MS.Position{CellMS}(User,:)-BS.Position{CellBS}(Base,:));
                PL= 10^(randn*8/10)*(200/d)^(3);
                Hlarge(User,CellMS,Base,CellBS)=PL;
            end
        end
    end
end
end


function [MS, BS] = usergenerator(Cell, minR_ratio)
Ncell       = Cell.Ncell;         % number of cells
Nintra      = Cell.Nintra;        % number of cell-intra users
NintraBase  = Cell.NintraBase;    % number of cell-intra basestations
MS.Position = [];
BS.Position = [];
Nms         = Nintra;
NmsBase     = NintraBase;
Rcellmin    = minR_ratio*Cell.Rcell;

% User Deployment
MS.Position	= cell(Ncell,1);
if Nms>=1
    for n = 1 : Ncell % generate users for each cell
        theta = rand(Nms,1)*2*pi;
        [x,y] = distrnd(Cell.Rcell,Rcellmin,theta);
        MS.Position{n}	= [x+Cell.Position(n,1),y+Cell.Position(n,2)];
    end
end

% Basestation Deployment
BS.Position	= cell(Ncell,1);
if NmsBase>=1
    for n = 1 : Ncell % generate users for each cell
        theta = rand(NmsBase-1,1)*2*pi;
        [x,y] = distrnd(Cell.Rcell,Rcellmin,theta);
        BS.Position{n}	= [x+Cell.Position(n,1),y+Cell.Position(n,2)];
        BS.Position{n}  = [Cell.Position(n,:);BS.Position{n}];
    end
end
end

% User Distance Generator in Single Cell
function [x,y] = distrnd(Rcell,Rcellmin,theta)
MsNum   = numel(theta);
R       = Rcell - Rcellmin;            % effective cell radius

% generate the distance between MS and BS
d      = sum(rand(MsNum,2),2) * R;     % user distributed in the cell uniformly
d(d>R) = 2*R - d(d>R);
d      = d + Rcellmin;                 % real MS location

% generate the phase between MS and the normal of the BS
% theta = rand(MsNum,1)*2*pi;
x     = d.*cos(theta);
y     = d.*sin(theta);
end

