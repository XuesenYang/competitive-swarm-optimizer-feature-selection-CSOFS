%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Implementation of a competitive swarm optimizer (CSO) for feature selction
%  See the details of CSO in the following paper
%  Cite as:S. Gu, R. Cheng, Y. Jin, Feature Selection for High-Dimensional Classification using a Competitive Swarm Optimizer, Soft Computing. 22 (2018) 811-822.
%  The source code CSO is implemented by Yang Xuesen 
%  If you have any questions about the code, please contact: 
%  Yang Xuesen at 1348825332@qq.com
%  Institution: Shenzhen University
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath(genpath(pwd));
load ('sonar.mat')
%d: dimensionality
d = size(data,2)-1;
%maxfe: maximal number of fitness evaluations
maxfe = 100;
%runnum: the number of trial runs
runnum = 30;
%m:population size
m=50;
n=d;
lu = [-5 * ones(1, n); 5 * ones(1, n)];
fitness=zeros(m,1);
phi=0.1;
results=zeros(1,runnum);
% several runs
for run = 1 : runnum
    % initialization
    XRRmin = repmat(lu(1, :), m, 1);
    XRRmax = repmat(lu(2, :), m, 1);
    rand('seed', sum(100 * clock));
    p = XRRmin + (XRRmax - XRRmin) .* rand(m, d);
    bi_position=zeros(m,n);
    pop=sigmf(p,[1 0]);
    RandNum=rand(m,n);
    change_position=(pop>RandNum);
    bi_position(find(change_position))=1;
    for i=1:m
        feature = find(bi_position(i,:)==1);
        fitness(i,1) = func(data,feature);
    end
    v = zeros(m,d);
    bestever = 1e200;
    gen = 0;
    tic;
    % main loop
    while(gen < maxfe)
        
        % generate random pairs
        rlist = randperm(m);
        rpairs = [rlist(1:ceil(m/2)); rlist(floor(m/2) + 1:m)]';
        
        % calculate the center position
        center = ones(ceil(m/2),1)*mean(p);
        
        % do pairwise competitions
        mask = (fitness(rpairs(:,1)) > fitness(rpairs(:,2)));
        losers = mask.*rpairs(:,1) + ~mask.*rpairs(:,2); 
        winners = ~mask.*rpairs(:,1) + mask.*rpairs(:,2);
   
        
        %random matrix 
        randco1 = rand(ceil(m/2), d);
        randco2 = rand(ceil(m/2), d);
        randco3 = rand(ceil(m/2), d);
        
        % losers learn from winners
        v(losers,:) = randco1.*v(losers,:) ...,
                    + randco2.*(p(winners,:) - p(losers,:)) ...,
                    + phi*randco3.*(center - p(losers,:));
        p(losers,:) = p(losers,:) + v(losers,:);
         
        % boundary control
        for i = 1:ceil(m/2)
            p(losers(i),:) = max(p(losers(i),:), lu(1,:));
            p(losers(i),:) = min(p(losers(i),:), lu(2,:));
        end

        
        % fitness evaluation
        n_losers=length(losers);
        for id=1:n_losers
            pop(losers(id),:)=sigmf(p(losers(id),:),[1 0]);
            Randnu=rand(1,n);
            change_pos=(pop(losers(id),:)>Randnu);
            feature = find(change_pos==1);
            fitness(losers(id),1) = func(data,feature);
        end
        bestever = min(bestever, min(fitness));
        fprintf('Runs: %d\t Iter: %d\t Best fitness: %.4f\n',run,gen,bestever); 
        gen = gen + 1;
    end
    
    results(1, runnum) = bestever;
    fprintf('Run No.%d Done!\n', run); 
    disp(['CPU time: ',num2str(toc)]);
end

