%%% IC map assumed to be polynomial in X, no dependence on Y. Incremental lib for X
%%% Xeq assumed to be power-law in X and Y, fixed lib for X, incremental lib for Y
%%% Yeq assumed to be power-law in X and Y, fixed lib for Y, incremental for X

function [rhs_IC,W_IC,rhs_Y,W_Y,rhs_X,W_X,...
    lib_Y_IC,lib_X_IC,lib_Y_Yeq,lib_X_Yeq,lib_Y_Xeq,lib_X_Xeq,...
    WS_IC,WS_Yeq,WS_Xeq,...
    CovW_IC,CovW_Y,CovW_X,...
    Y_ns,errs_Yend,...
    loss_IC,loss_Y,loss_X]= ...
    wsindy_eco_fcn(toggle_zero_crossing,stop_tol,phifun_Y,tf_Y_params,WENDy_args,autowendy,tol,tol_min,...
        tol_dd_learn,pmax_IC,polys_Y_Yeq,polys_X_Xeq,pmax_X_Yeq,pmax_Y_Xeq,neg_Y,neg_X,boolT,boolTL,...
        Y_train,X_train,X_var,train_inds,train_time,t_epi,yearlength,nstates_X,nstates_Y,X_in,nX,nY,...
        custom_tags_X,custom_tags_Y,linregargs_fun_IC,linregargs_fun_Y,linregargs_fun_X)

    if isempty(X_var)
        X_var = X_train*0;
    end

    addpath(genpath('wsindy_obj_base'))
    %%% get wsindy_data 
    Uobj_Y = cellfun(@(x,t)wsindy_data(x,t),Y_train,train_time);
    Uobj_tot = arrayfun(@(i)...
        wsindy_data([[Uobj_Y(i).Uobs{:}] repmat(X_train(X_in(i),:),Uobj_Y(i).dims,1)],train_time{i}),(1:length(Uobj_Y))');
    for j=1:length(Uobj_tot)
        Uobj_tot(j).sigmas(nstates_Y+1:end) = num2cell(X_var(X_in(j),:));
    end

    IC = zeros(max(train_inds),nstates_Y+nstates_X);
    IC(train_inds,nstates_Y+1:end) = X_train;
    IC(train_inds(X_in),1:nstates_Y) = cell2mat(arrayfun(@(U)cellfun(@(x)x(1,:),U.Uobs),Uobj_Y,'uni',0));
    Uobj_IC = wsindy_data(IC,0:max(train_inds)-1);
    if autowendy>0
        S = arrayfun(@(U)cell2mat(U.estimate_sigma).^2,Uobj_Y,'uni',0);
        IC_cov = IC*0; 
        IC_cov(train_inds(X_in),1:nstates_Y) = cell2mat(S);
        IC_cov(train_inds(X_in),nstates_Y+1:end) = X_var(X_in,:).^2;
        Uobj_IC.R0 = spdiags(IC_cov(:),0,numel(IC_cov),numel(IC_cov));
    end
    E = eye(nstates_Y+nstates_X);
    
%%% get IC_map
    lib_Y_IC = library('tags',zeros(1,nstates_Y));
    lib_X_IC = library();
    tf_IC = testfcn(Uobj_IC,'meth','direct','param',0,'phifuns','delta','mtmin',0,'subinds',train_inds(X_in));
    lhs_IC = arrayfun(@(i)term('ftag',E(i,:)),(1:nstates_Y)','uni',0);
    [rhs_IC,W_IC,WS_IC,lib_X_IC,loss_IC,lambda_IC,W_its_IC,res_IC,res_0_IC,CovW_IC] = ...
        hybrid_MI(pmax_IC,lib_Y_IC,lib_X_IC,nstates_Y,nstates_X,Uobj_IC,tf_IC,lhs_IC,...
                    WENDy_args,linregargs_fun_IC,autowendy,tol,tol_min,nY,nX);
    rhs_IC = @(X) rhs_IC(zeros(nstates_Y,1),X(:));
    
%%% get parametric small scale model
    tf_Yeq = arrayfun(@(U)...
        arrayfun(@(i)testfcn(U,tf_Y_params{:},'phifuns',phifun_Y,'stateind',i),1:U.nstates,'uni',0),...
        Uobj_Y,'uni',0);
    lhs_Yeq = arrayfun(@(i)term('ftag',E(i,:),'linOp',1),(1:nstates_Y)','uni',0);
    lib_Y_Yeq = arrayfun(@(i)library('nstates',nstates_Y),1:nstates_Y);
    if isempty(boolTL)
        boolTL = repmat({[]},1,nstates_Y);
    end

    if ~isempty(custom_tags_Y)
        if length(custom_tags_Y)==1
            custom_tags_Y = repmat(custom_tags_Y,1,nstates_Y);
        end
        arrayfun(@(L,i)L.add_terms(custom_tags_Y{i}),lib_Y_Yeq,1:nstates_Y);
    end
    for i=1:nstates_Y
        lib_Y_Yeq(i).add_tags('polys',polys_Y_Yeq,'neg',neg_Y,'boolT',boolT,'boolTL',boolTL{i},'lhs',lhs_Yeq{i}); % library
    end
    lib_X_Yeq = library('tags',custom_tags_X);
    [rhs_Y,W_Y,WS_Yeq,lib_X_Yeq,loss_Y,lambda_Y,W_its_Y,res_Y,res_0_Y,CovW_Y] = ...
        hybrid_MI(pmax_X_Yeq,lib_Y_Yeq,lib_X_Yeq,nstates_Y,nstates_X,Uobj_tot,tf_Yeq,lhs_Yeq,...
                    WENDy_args,linregargs_fun_Y,autowendy,tol,tol_min,nY,nX);
    
%%% get Y(T)
    X_sub = find(diff(train_inds)==1);
    subinds = train_inds(X_sub);
    Yend = zeros(length(X_sub),nstates_Y);
    X_n = X_train(X_sub,:);
    errs_Yend = []; Y_new = {}; Y_ns = cell(length(X_sub),2);
    for n=1:length(X_sub)
        options_ode_sim = odeset('RelTol',tol_dd_learn,'AbsTol',tol_dd_learn*ones(1,nstates_Y),'Events',@(T,Y)myEvent(T,Y,stop_tol*max(max(cell2mat(Y_train))),toggle_zero_crossing));
        rhs_learned = @(y)rhs_Y(y,X_n(n,:).*nX);
        Y0 = rhs_IC(X_n(n,:).*nX);
        t_train = t_epi{train_inds(X_sub(n))}([1 end]);
        [t_n,Y_n,TE]=ode15s(@(t,x)rhs_learned(x),t_train,Y0,options_ode_sim);
        Y_ns(n,:) = {Y_n,t_n};
        if ismember(X_sub(n),X_in)
            try
                Y_new = [Y_new;{interp1(t_n,Y_n,train_time{X_in==X_sub(n)})./nY}];
                errs_Yend = [errs_Yend;std(Y_new{end}-Y_train{X_in==X_sub(n)})];
                % figure(1)
                % for i=1:nstates_Y
                %     subplot(nstates_Y,1,i)
                %     plot(t_n,Y_n(:,i)/nY(i),train_time{X_in==X_sub(n)},Y_train{X_in==X_sub(n)}(:,i)); 
                %     legend({'learn','data'})
                % end
                % drawnow
            catch
                Y_new = [Y_new;{NaN}];
                errs_Yend = [errs_Yend;max(abs(Y_n),'omitnan')];
            end
        % figure(1)
        % for i=1:nstates_Y
        %     subplot(nstates_Y,1,i)
        %     plot(t_n,Y_n(:,i)/nY(i),train_time{X_in==X_sub(n)},Y_train{X_in==X_sub(n)}(:,i),t_epi{train_inds(X_sub(n))},Ycell{train_inds(X_sub(n))}(:,i)/nY(i)); 
        %     legend({'learn','data','true'})
        % end
        % drawnow
        end
        Yend(n,:) = Y_n(end,:);
    end
    Yend = Yend./nY;

%%% get large scale model
    X_Yend = zeros(max(train_inds),nstates_X+nstates_Y);
    X_Yend(train_inds,1:nstates_X) = X_train;
    % Tends = cellfun(@(t)t(end),t_epi(subinds));
    % Yend = cell2mat(arrayfun(@(i)interp1(Y_ns{i,2},Y_ns{i,1},Tends(i))./nY,(1:length(subinds))','uni',0));
    X_Yend(subinds,nstates_X+1:end) = Yend;
    Uobj_X_Yend = wsindy_data(X_Yend,(0:max(train_inds)-1)*yearlength);
    if autowendy>0
        Xn_cov = X_Yend*0; 
        errs_Yend = fillmissing(interp1(train_inds(X_in),errs_Yend,subinds,'linear'),'linear');
        Xn_cov(subinds,1:nstates_X) = X_var(X_sub,:).^2;
        Xn_cov(subinds,nstates_X+1:end) = errs_Yend.^2;
        Uobj_X_Yend.R0 = spdiags(Xn_cov(:),0,numel(Xn_cov),numel(Xn_cov));
    end
    lib_X_Xeq = library('tags',get_tags(polys_X_Xeq,[],nstates_X));
    lib_Y_Xeq = library();
    tf_X = testfcn(Uobj_X_Yend,'meth','direct','param',1/2,'phifuns','delta','mtmin',0,'subinds',subinds);
    lhs = arrayfun(@(i)term('ftag',E(i,:),'linOp',1),(1:nstates_X)','uni',0);

    [rhs_X,W_X,WS_Xeq,lib_Y_Xeq,loss_X,lambda_X,w_its,res_X,res_0_X,CovW_X] = ...
        hybrid_MI(pmax_Y_Xeq,lib_X_Xeq,lib_Y_Xeq,nstates_X,nstates_Y,...
        Uobj_X_Yend,tf_X,lhs,WENDy_args,linregargs_fun_X,autowendy,tol,tol_min,nX,nY);


end
