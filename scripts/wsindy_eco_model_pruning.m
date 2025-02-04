%% load learned model or first run desired wsindy_eco_script

% load('../data/UQ_plots_uncorrected_red_model.mat')
% 
% sim_script;

%% display models
%%% each model is displayed in terms of f(u^s_1,...,u^s_2,u^p_1,...,u^p_2)
%%% for the IC map and Y maps, u^s = Y, u^p = X
%%% for the X map, u^s = X, u^p = Y
IC_mod = WS_IC.disp_mod('w',cell2mat(cellfun(@(w)w(:),W_IC,'Un',0)));
Y_mod = WS_Yeq.disp_mod('w',cell2mat(cellfun(@(w)w(:),W_Y,'Un',0)));
X_mod = WS_Xeq.disp_mod('w',cell2mat(cellfun(@(w)w(:),W_X,'Un',0)));

disp(['IC terms:'])
IC_mod{:}
disp(['Yeq terms:'])
Y_mod{:}
disp(['Xeq terms:'])
X_mod{:}

if exist('W_IC_true','var')
    coeff_compare;
end

%% view conf int
% close all
addpath(genpath('../vis'))
conf_tol = 0.25;

figure(21);clf
varget = 'IC';
view_conf_int;
IC_prune = find(abs(w_hat)<conf_tol*conf_int)

figure(22);clf
varget = 'Y';
view_conf_int;
Y_prune = find(abs(w_hat)<conf_tol*conf_int)

figure(23);clf
varget = 'X';
view_conf_int;
X_prune = find(abs(w_hat)<conf_tol*conf_int)

%% decide on terms to prune

W_IC_sparse = zero_ents(W_IC,IC_prune);
W_Y_sparse = zero_ents(W_Y,Y_prune);
W_X_sparse = zero_ents(W_X,X_prune);

tags_IC_X = lib_X_IC.tags;
tags_Y_Yeq = lib_Y_Yeq.tags;
tags_X_Yeq = lib_X_Yeq.tags;
tags_X_Xeq = lib_X_Xeq.tags;
tags_Y_Xeq = lib_Y_Xeq.tags;

%% extract model terms from wsindy_eco run

WENDy_args = {'maxits',10,'ittol',10^-4,'diag_reg',10^-6,'verbose',1};
[rhs_IC,W_IC,rhs_Y,W_Y,rhs_X,W_X,...
    lib_Y_IC,lib_X_IC,lib_Y_Yeq,lib_X_Yeq,lib_Y_Xeq,lib_X_Xeq,...
    WS_IC,WS_Yeq,WS_Xeq,...
    CovW_IC,CovW_Y,CovW_X,...
    errs_Yend]= ...
    wendy_eco_fcn(toggle_zero_crossing,stop_tol,phifun_Y,tf_Y_params,WENDy_args,autowendy,tol_dd_learn,...
        Y_train,X_train,X_var,train_inds,train_time,t_epi,yearlength,nstates_X,nstates_Y,X_in,nX,nY,...
        tags_IC_X,tags_Y_Yeq,tags_X_Yeq,tags_X_Xeq,tags_Y_Xeq,W_IC_sparse,W_Y_sparse,W_X_sparse);

%% sim full system

sim_script;