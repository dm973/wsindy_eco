%% view
% dr = '~/Desktop/';%
dr = '~/Dropbox/Boulder/research/data/dukic collab/';
loadvars = {'results_cell','snr_Y','ntrain_inds','rngs','sim_cell'};
% peaks_ = false;subx = 1:3;
peaks_ = true;subx = 1:3;
% peaks_ = true;subx = 1;
for subt = 2
for ttf = [0.75]
for kk = [.05]
for sind = 11;%[1 3 4 6 7 9 11]

figure(sind);clf

% disp([sind kk])
% load([dr,'sweep_snrY_',num2str(kk),'_ttf_',num2str(ttf),'_subt_',num2str(subt),'.mat'],loadvars{:}) %ttf = 0.5; ntinds = 6:30
% load([dr,'sweep_snrY_',num2str(kk),'_ttf_',num2str(ttf),'_06-Apr-2024.mat'],loadvars{:})
% load([dr,'sweep_snrY_',num2str(kk),'_ttf_',num2str(ttf),'_07-Apr-2024.mat'],loadvars{:})
% load([dr,'sweep_snrY_',num2str(kk),'_ttf_',num2str(ttf),'_subt_',num2str(subt),'_10-Apr-2024.mat'],loadvars{:})

if ~peaks_
    % load([dr,'sweep_snrY_',num2str(kk),'_ttf_',num2str(ttf),'_subt_',num2str(subt),'.mat'],loadvars{:})
    load([dr,'sweep_snrY_',num2str(kk),'_ttf_',num2str(ttf),'_subt_',num2str(subt),'_mits_5.mat'],loadvars{:})
    x = ntrain_inds(subx);
else
    % load([dr,'sweep_snrY_',num2str(kk),'_ttf_',num2str(ttf),'_subt_',num2str(subt),'_mits_5_peaks.mat'],loadvars{:})
    load([dr,'sweep_snrY_',num2str(kk),'_ttf_',num2str(ttf),'_subt_',num2str(subt),'_mits_5_peaks.mat'])
    % load([dr,'sweep_snrY_',num2str(kk),'_ttf_',num2str(ttf),'_subt_',num2str(subt),'_mits_5_peaks_IC.mat'],loadvars{:})
    x = -ntrain_inds(subx);
end
err_tol = 0.5;
n_err_tols = cellfun(@(s) get_n_err_tol(s{2}{1},s{1}{1},err_tol) , sim_cell);
mean(n_err_tols')
median(n_err_tols')
ylabs = {'Coefficient error ($E_2^{IC}$)',[],'True Positivity Ratio (TPR$^{IC})$',...
    'Coefficient error ($E_2^{Y}$)',[],'True Positivity Ratio (TPR$^{Y})$',...
    'Coefficient error  ($E_2^{X}$)',[],'True Positivity Ratio (TPR$^{X})$','Walltime(sec)',...
    ['Generations predicted ($n_{',num2str(err_tol),'}(\hat{\bf w})$)'],['Generations predicted ($n_{',num2str(err_tol),'}(\hat{\bf w})$)']};

runs = length(rngs);
filter_fun = @(r)all([r(3)<=1 r(6)<=1 r(9)<=1]);

if isequal(subx,':')
    subx = 1:size(results_cell,1);
end

filter_inds = cellfun(@(r)filter_fun(r),results_cell(subx,:));
disp(['percent kept'])
arrayfun(@(i)length(find(filter_inds(i,:)))/size(filter_inds,2),1:size(filter_inds,1))'

if sind<11
    res_ind = cellfun(@(r)r(sind),results_cell(subx,:));
elseif sind ==11
    res_ind = n_err_tols;
elseif sind ==12
    res_ind = cellfun(@(r)mean(r(sind:end)),results_cell(subx,:));
end
% if sind==11
%     arrayfun(@(i) length(find(res_ind(i,:)>=79)),1:size(res_ind,1))
% end
res_ind = arrayfun(@(i)res_ind(i,filter_inds(i,:)),1:length(subx),'uni',0);

if ismember(sind,[1 4 7])
    % res_ind = cellfun(@(r)max(r,10^-6),res_ind,'uni',0);
    OL = cellfun(@(r)r(r>100),res_ind,'uni',0);
    disp(['percent remaining errs > 100'])
    cellfun(@(o,r)length(o)/length(r),OL,res_ind)
    res_ind = cellfun(@(r)r(r<=100),res_ind,'uni',0);
end

% g = arrayfun(@(i) zeros(length(res_ind{i}),1)+i-1,(1:length(res_ind))','uni',0);
% g = cell2mat(g);
% res_ind = cell2mat(res_ind)';
% h=boxplot(res_ind,g,'whisker',1.5);
% set(h,'LineWidth',2)
% set(h,'MarkerSize',10)

rr = cellfun(@(r)r',res_ind,'uni',0)';
if ~ismember(sind,[1 4 7])
    violin(rr','kernel','box',...
    'bw',range(cell2mat(rr))/size(results_cell,2)^(1/2),'support',[min(cell2mat(rr))-eps*range(cell2mat(rr)) max(cell2mat(rr))+eps*range(cell2mat(rr))],...
    'facecolor',[0 0.5 1],'plotlegend',0,'linewidth',1.7,'x',x);
else
    violin(cellfun(@(r)log10(r),rr','Un',0),'kernel','box',...
    'bw',range(log10(cell2mat(rr)))/size(results_cell,2)^(1/2),'support',[-7 2+eps],...
    'facecolor',[0 0.5 1],'plotlegend',0,'linewidth',1.7,'x',x);
end
set(gca,'Xtick',x,'Xlim',[min(x)-mean(diff(x))/2 max(x)+mean(diff(x))/2])
% drawnow

% replaceboxes

% ylim=[0 size(X,1)-1];
% ylabel('$n_{0.2}(\hat{\bf w})$','interpreter','latex')
ylabel(ylabs(sind),'interpreter','latex')
if peaks_
    xlabel('Peaks observed ($|\mathcal{I}|/4$)','interpreter','latex')
else
    xlabel('Generations observed ($|\mathcal{I}|$)','interpreter','latex')
end
% legend(h,['$\sigma_{NR}=',num2str(snr_Y),'$'],'interpreter','latex','location','best')
set(gca,'Xticklabels',x,'ticklabelinterpreter','latex','fontsize',18)
if ismember(sind,[1 4 7])
    % set(gca,'Yscale','log','Ytick',10.^(-6:2:2),'Ylim',[10^-6 10^2])
    set(gca,'Ytick',-6:2:2,'Yticklabels',num2str(10.^(-6:2:2)','%0.0e'),'Ylim',[-6 2])
elseif sind==11
    set(gca,'Ylim',[0 80])
end
grid on

% saveas(gcf,['~/Desktop/hybrid_snrY_',num2str(kk),'_ttf_',num2str(ttf),'_stat',num2str(sind),'.png'])
if peaks_
    saveas(gcf,['~/Desktop/hybrid_snrY_',num2str(kk),'_ttf_',num2str(ttf),'_stat',num2str(sind),'_peaks.png'])
else
    saveas(gcf,['~/Desktop/hybrid_snrY_',num2str(kk),'_ttf_',num2str(ttf),'_stat',num2str(sind),'.png'])
end
end
end
end
end

%%
ylabs = {'$E_2^{IC}$',[],'TPR$^{IC}$','$E_2^{Y}$',[],'TPR$^{Y}$','$E_2^{X}$',[],'TPR$^{X}$','Walltime(sec)','$n_{0.2}(\hat{\bf w})$'};
sind = 11;%[1 3 4 6 7 9]
subx = 2:9;
res_ind = cellfun(@(r)r(sind),results_cell(subx,:));
figure(1);clf
% histogram(res_ind(end,:),20)
violin(res_ind,'kernel','box','bw',500^(1/4),'facecolor',[0 1 1],'plotlegend',0,'linewidth',1.7);
ylabel(ylabs(sind),'interpreter','latex')
xlabel('Generations observed ($|\mathcal{I}|$)','interpreter','latex')
% legend(h,['$\sigma_{NR}=',num2str(snr_Y),'$'],'interpreter','latex','location','best')
% set(gca,'Xtick',x,'ticklabelinterpreter','latex','fontsize',20,'xlim',[min(x) max(x)])
set(gca,'Xticklabels',x,'ticklabelinterpreter','latex','fontsize',18)
if ismember(sind,[1 4 7])
    set(gca,'Yscale','log','Ytick',10.^(-6:2:2),'Ylim',[10^-6 10^2])
elseif sind==11
    set(gca,'Ylim',[0 80])
end
grid on

% distributionPlot(res_ind','colormap',copper)
ylim([0 80])
%%
sind = 11;
res_ind = cellfun(@(r)r(sind),results_cell(11,:));
figure(1);clf
histogram(res_ind,20)


%%

dr = '~/Dropbox/Boulder/research/data/dukic collab/';
kk = 0.01; ttf = 0.75; subt = 2;
loadvars = {'results_cell','snr_Y','ntrain_inds','rngs','sim_cell','coeffs_cell','libs_cell'};
load([dr,'sweep_snrY_',num2str(kk),'_ttf_',num2str(ttf),'_subt_',num2str(subt),'_mits_5_peaks.mat'],loadvars{:})

subx = 2; sind = 9;
res_ind = cellfun(@(r)r(sind),results_cell(subx,:));
inds = find(res_ind<1);
inds_1 = find(res_ind==1);

coeff_ref = coeffs_cell{subx,inds_1(1)}{end};

coeffs = cellfun(@(c)c{end},coeffs_cell(subx,inds),'uni',0);
libs = cellfun(@(c)c(end-1:end),libs_cell(subx,inds),'uni',0);
for coord = 1:2
    coeffs_coord = cellfun(@(c)c{coord},coeffs,'un',0);
    coeff_ref{coord}
    mean(cat(3,coeffs_coord{:}),3)
end
libs{1}{:}
% Q = quantile(res_ind,3,2);
% y = Q(:,2);
% y_l = Q(:,1);
% y_u = Q(:,3);
% y_f = [y_l;flipud(y_u)];
% x_f = [x';flipud(x')];
% fill(x_f,y_f,[0 1 1])
% hold on
% h=plot(x,y,'k-o',x,y_u,'r-',...
%     x,y_l,'r-','linewidth',3);
% hold off
% grid on