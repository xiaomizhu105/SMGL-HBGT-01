function B = bipartite_gen(X,m,k,style)
% Input:
%        X: data matrix  num*dim 
% 
% Output: 
%        B: the n*m bipartite graph
% Written by Rong Wang (wangrong07@tsinghua.org.cn), written in 2023/05/25

num = size(X,1);
%%
    disp('----------Anchor Selection----------');
    if style == 1 % direct sample
        [~,ind,~] = graphgen_anchor(X,m);
        centers = X(ind, :);
    elseif style == 2 % rand sample
        vec = randperm(num);
        ind = vec(1:m);
        centers = X(ind, :);
    elseif style == 3 % KNP
        [~, ~, ~, ~, dis] = litekmeans(X, m);
        [~,ind] = min(dis,[],1);
        ind = sort(ind,'ascend');
        centers = X(ind, :);
    elseif style == 4 % kmeans sample
        [~, centers, ~, ~, ~] = litekmeans(X, m);
    end

%%
    disp('----------Single Bipartite Graphs Inilization----------');

    D = L2_distance_1(X', centers');
    [~, idx] = sort(D, 2); % sort each row
    B = zeros(num,m);
    for ii = 1:num
        id = idx(ii,1:k+1);
        di = D(ii,id);
        B(ii,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
    end

end

