clc; clear;
addpath([pwd, '/funs']);

%% load dataset

dataname='COIL20';
load(strcat(dataname,'.mat'));

n_view = length(X);
c = length(unique(Y));  
n = size(X{1},1);

%% hyperparameter setting
opts.style = 3; 
iterMax = 50;
mu = 10e-1; 
max_mu = 10e9; 
eta = 1.6;
gamma = 0.2;

anchor_rate = 0.5;
rng(8);
k = 5;
order = 3;
beta = 10;

m = fix(n*anchor_rate);

%% Anchor Selection
B = cell(n_view,1); % initial bipartite graph storage cell
centers = cell(n_view,1); % anchor feature matrix storage cell
disp('----------Anchor Selection----------');
if opts.style == 1 % direct sample
    XX = [];
    for v = 1:length(X)
       XX = [XX X{v}];
    end
    [~,ind,~] = graphgen_anchor(XX,m);
    for v = 1:n_view
        centers{v} = X{v}(ind, :);
    end
elseif opts.style == 2 % rand sample
    vec = randperm(n);
    ind = vec(1:m);
    for v = 1:n_view
        centers{v} = X{v}(ind, :);
    end
elseif opts.style == 3 % KNP
    XX = [];
    for v = 1:n_view
        XX = [XX X{v}];
    end
    [~, ~, ~, ~, dis] = litekmeans(XX, m);
    [~,ind] = min(dis,[],1);
    ind = sort(ind,'ascend');
    for v = 1:n_view
        centers{v} = X{v}(ind, :);
    end
elseif opts.style == 4 % kmeans sample
    XX = [];
    for v = 1:n_view
       XX = [XX X{v}];
       len(v) = size(X{v},2);
    end
    [~, Cen, ~, ~, ~] = litekmeans(XX, m);
    t1 = 1;
    for v=1:n_view
       t2 = t1+len(v)-1;
       centers{v} = Cen(:,t1:t2);
       t1 = t2+1;
    end
end

%% First order Bipartite Graphs Inilization
disp('----------First order Bipartite Graphs Inilization----------');
for v = 1:n_view
    D = L2_distance_1(X{v}', centers{v}');
    [~, idx] = sort(D, 2); % sort each row
    B{v} = zeros(n,m);
    for ii = 1:n
        id = idx(ii,1:k+1);
        di = D(ii, id);
        B{v}(ii,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
    end
end

%% Generate high order 2P Graphs
disp('----------Generate high order 2P Graphs----------');
for v = 1:n_view
    TensorB_Cell{v,1} = B{v,1}./max(max(B{v,1},[],2));
    [U,sigma,Vt] = svd(B{v,1});
    for d = 2:order
        temp = U*sigma.^(2*d-1)*Vt';
        temp(temp<eps)=0;
        temp = temp./max(max(temp,[],2));
        %temp = temp./sum(temp,2);
        TensorB_Cell{v,d} = temp;
    end
end

%% Optimization
disp('----------Optimization----------');
nGraph = order*n_view;
w = ones(n_view,order)/(nGraph); % initialize w
% initialize P
P = zeros(n,m); 
for v = 1:n_view
    P= P + 1/n_view*TensorB_Cell{v,1};
end

% initialize A
TensorA_Cell = TensorB_Cell;
% initialize Q
TensorQ_Cell = cell(n_view,order);
for v = 1:n_view
    for d = 1:order
        TensorQ_Cell{v,d}= zeros(n,m);
    end
end

fun = [];
for iter = 1:iterMax
    fprintf('iter:%d-th!!!!!\n',iter);
    %% Update F
    Dn = diag(sum(P,2).^(-0.5));
    Dm = diag(sum(P).^(-0.5));   
    X = Dn*P*Dm;
    if any(isinf(X), 'all') || any(isnan(X), 'all')
        X(isinf(X)) = 0;
        X(isnan(X)) = 0;
    end
    [U,S,V] = svds(X,c+1);
    Fn = sqrt(2)*U(:,1:c)/2;
    Fm = sqrt(2)*V(:,1:c)/2;

    ev = diag(S);
    
    Fn_old = Fn;
    Fm_old = Fm;
    
    fn1 = sum(ev(1:c));
    fn2 = sum(ev(1:c+1));
    if fn1 < c-0.0000001
        gamma = 2*gamma;
    elseif fn2 > c+1-0.0000001
        gamma = gamma/2;   Fn = Fn_old; Fm = Fm_old;
    else
        break;
    end



    %% Update P
    % Compute G
    G = zeros(n,m); 
    for v = 1:n_view
        for d = 1:order
            G= G + 1/w(v,d)*TensorB_Cell{v,d};
        end
    end
    G = G./sum(sum(1./w));

    Fn_ = Dn*Fn;
    Fm_ = Dm*Fm;
    dist = L2_distance_1(Fn_',Fm_');

    P = zeros(n,m);
    for i = 1:n
        gi = G(i,:);
        ti = dist(i,:);
        ad = gi-0.5*gamma*ti;
        P(i,:) = EProjSimplex_new(ad);
    end
    if any(isnan(P))
        P = TensorB_Cell{1};
    end

    %% Update B
    for v = 1:n_view
        for d = 1:order
            H = 2*w(v,d)/(w(v,d)*mu+2)*(1/w(v,d)*P+mu/2*TensorA_Cell{v,d}-TensorQ_Cell{v,d}/2);
            Btemp = zeros(n,m);
            for i = 1:n
                Btemp(i,:) = EProjSimplex_new(H(i,:));
            end
            TensorB_Cell{v,d} = Btemp;
        end
    end

    %% Update A
    TensorQ_Cell = reshape(TensorQ_Cell,nGraph,1);
    Q_tensor = cat(3,TensorQ_Cell{:});
    TensorB_Cell = reshape(TensorB_Cell,nGraph,1);
    B_tensor = cat(3,TensorB_Cell{:});
    B_tensor = permute(B_tensor,[1 3 2]);
    Q_tensor = permute(Q_tensor,[1 3 2]);

    [A_tensor,tnn,trank] = solve_A_tensor(B_tensor+Q_tensor/mu,beta/mu);
    A_tensor = permute(A_tensor,[1 3 2]);
    for i = 1:nGraph
        TensorA_Cell{i} = A_tensor(:,:,i);
    end
    TensorA_Cell = reshape(TensorA_Cell,n_view,order);
    TensorQ_Cell = reshape(TensorQ_Cell,n_view,order);
    TensorB_Cell = reshape(TensorB_Cell,n_view,order);

    %% Update w
    w = zeros(n_view,order);
    for v = 1:n_view
        for d = 1:order
            h(v,d) = norm(P-TensorB_Cell{v,d}, 'fro');
        end
    end
    w = h./sum(sum(h));

    %% Update TensorQ
    for v = 1:n_view
        for d = 1:order
            TensorQ_Cell{v,d} = TensorQ_Cell{v,d}+mu*(TensorB_Cell{v,d}-TensorA_Cell{v,d});
        end
    end
    
   %% update mu
   mu = min(mu*eta,max_mu);

   fun_ele = 0;
   for v = 1:n_view
        for d = 1:order
            fun_ele = norm(P-TensorB_Cell{v,d}, 'fro') + fun_ele;
        end
   end
   fun = [fun fun_ele];
end

S = sparse(n+m,n+m);
S(1:n,n+1:end) = P;
S(n+1:end,1:n) = P';
G = graph(S);  % 创建图对象，其中 adjacencyMatrix 是你的邻接矩阵
y = conncomp(G);  % bins 返回一个数组，表示每个节点所属的连通组件
y1 = y(1:n)';
result = ClusteringMeasure1(Y, y1);
fprintf('acc = %f !!\n',result(1));