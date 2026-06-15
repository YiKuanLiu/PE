% run_all_features.m
% Extract 21 per-lobe features for ALL cases and save to lobe_features_matlab.mat.
% 對所有病例抽取 21 個逐肺葉特徵,存成 lobe_features_matlab.mat。
%
% Raw .mat live under <rawdir>/Positive_Anon (PE+) and <rawdir>/Negative_Anon (PE-).
% (Note: the original /mnt/hot label.csv was removed; here labels come from the folder.)
% 原始 .mat 放在 Positive_Anon(PE+)與 Negative_Anon(PE-);標籤由資料夾決定。

rawdir = '/home/yikuan/PE';            % <-- adjust if needed / 視情況調整
names = {'M_perf','V_vent','R_mism','HUin','HUex','HUchg','volin','volex', ...
         'volshrink','volratio','massin','massex','airin','airex', ...
         'stdin','stdex','skewin','skewex','hyper_max','hyper_p99','hyper_frac'};

pos = dir(fullfile(rawdir, 'Positive_Anon', '*.mat'));
neg = dir(fullfile(rawdir, 'Negative_Anon', '*.mat'));
files  = [pos; neg];
labels = [ones(numel(pos), 1); zeros(numel(neg), 1)];   % 1 = PE+, 0 = PE-
N = numel(files);

bags = nan(N, 5, 21);                  % N patients x 5 lobes x 21 features / 病人 x 肺葉 x 特徵
for i = 1:N
    f = fullfile(files(i).folder, files(i).name);
    try
        bags(i, :, :) = extract_lobe_features(f);
    catch ME
        fprintf('  [skip] %s : %s\n', files(i).name, ME.message);
    end
    if mod(i, 10) == 0, fprintf('  %d/%d\n', i, N); end
end

fnames = {files.name}';
save('lobe_features_matlab.mat', 'bags', 'labels', 'fnames', 'names');
fprintf('saved lobe_features_matlab.mat : %d patients x 5 lobes x 21 features\n', N);

% Quick sanity: hyper_frac (col 21) — PE+ should be higher (per-lobe mean over patients).
% 快速檢查:hyper_frac(第 21 欄)PE+ 應較高。
hf = squeeze(bags(:, :, 21));          % N x 5
fprintf('hyper_frac per-lobe (PE+ mean vs PE- mean, %%):\n');
for l = 1:5
    p = hf(labels == 1, l);  n = hf(labels == 0, l);
    fprintf('  L%d: %.3f%% vs %.3f%%\n', l, 100*mean(p(~isnan(p))), 100*mean(n(~isnan(n))));
end
