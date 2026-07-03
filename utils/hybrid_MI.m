% lib_X_Xeq = library('tags',get_tags(polys_X_Xeq,[],nstates_X));
% lib_Y_Xeq = library();
% tf_X = testfcn(Uobj_X_Yend,'meth','direct','param',1/2,'phifuns','delta','mtmin',0,'subinds',subinds);
% lhs = arrayfun(@(i)term('ftag',E(i,:),'linOp',1),(1:nstates_X)','uni',0);
% WENDy_args = {'maxits_wendy',maxits_wendy,'lambdas',lambdas,'ittol',wendy_ittol,'diag_reg',diag_reg,'verbose',wendy_verb};
% [rhs_X,W_X,WS_Xeq,lib_Xeq,loss_X,lambda_X,w_its,res_X,res_0_X,CovW_X] = ...
%     hybrid_MI(pmax_Y_Xeq,lib_X_Xeq,lib_Y_Xeq,nstates_X,nstates_Y,Uobj_X_Yend,tf_X,lhs,WENDy_args,linregargs_fun_X,autowendy,tol,tol_min,nX,nY);

function [rhs,W,WS,lib_param,MSTLS_loss,lambda_hat,W_its,res_WENDy,res_0,CovW] = ...
    hybrid_MI(pmax_param,lib_state,lib_param,n_state,n_param,Uobj,tf,lhs,...
                WENDy_args,linregargs_fun,autowendy,tol_cov,tol_libinc_min,scale_state,scale_param)

    addpath(genpath('wsindy_obj_base'));

    p=-1;check=1;
    c = norminv(autowendy);
    while and(check,p<pmax_param)
        p=p+1;
        polys_param = 0:p;
        tags_param = get_tags(polys_param,[],n_param);
        lib_param.add_terms(tags_param);
        lib = arrayfun(@(L)kron_lib(L,lib_param),lib_state);
        WS = wsindy_model(Uobj,lib,tf,'lhsterms', lhs);
        WS.cat_Gb('cat','blkdiag');
        linregargs = linregargs_fun(WS);
        WENDy_MSTLS_args = [WENDy_args,{'linregargs',linregargs}];
        [WS,MSTLS_loss,lambda_hat,W_its,res_WENDy,res_0,CovW] = WS_opt().MSTLS_WENDy(...
            WS,WENDy_MSTLS_args{:});
        if and(c>0,size(W_its,2)>1)
            if c~=0
                tol = max( mean(diag(WS.cov))+c*std(diag(WS.cov)),tol_libinc_min^2);
                check = rms(res_0(:,end))^2 > tol;
                fprintf('\nms(res_0)=%g, cov tol=%g',full(rms(res_0(:,end))^2),full(tol))
            else
                check = rms(res_0(:,end))^2 > max(tol_cov*mean(diag(WS.cov)),tol_libinc_min^2);
            end
            % mm = mean(diag(WS.cov))+var(diag(WS.cov))/3;
            % mm
            % check = rms(res_0(:,end))^2 > max( chi2inv(0.95,mm)*mean(diag(WS.cov)),tol_libinc_min^2);
        else
            check = norm(WS.res) > tol_libinc_min;
        end
        if and(p==pmax_param,check)
            disp(['tolerance not reached'])
        end
    end
    %%% ordering of rows=state coefficients, cols=param coefficients
    if length(lib_state)==1
        W = cellfun(@(w)reshape(w,length(lib_state.terms),[]),WS.reshape_w,'uni',0);
    else
        W = arrayfun(@(i,L)reshape(WS.reshape_w{i},length(L.terms),[]),(1:WS.numeq)',lib_state','uni',0);
    end
    [rhs,W,~,M] = shorttime_map(W,lib_state,lib_param,scale_state,scale_param);
    CovW = (M.*CovW).*(M');
end