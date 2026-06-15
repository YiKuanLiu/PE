function feats = extract_lobe_features(matfile)
% EXTRACT_LOBE_FEATURES  Per-lobe V/Q + hyperdense features from inhale/exhale CT.
% 逐肺葉 V/Q + hyperdense 特徵(吸氣 T00 / 吐氣 T50 非顯影 CT)。
%
% This mirrors the Python pipeline (scripts/mil_patches.py); the resulting per-lobe
% feature vectors feed the MIL / RandomForest classifier (honest nested-10-fold AUC ~0.71).
% 對應 Python 的 scripts/mil_patches.py;產出的逐肺葉特徵餵給 MIL/RF(誠實 nested 10 折 AUC ~0.71)。
%
% INPUT  matfile : path to a .mat containing T00, T00_Lobe, T50, T50_Lobe, xymm, zmm
%                  T00/T50 are stored CT (uint16); HU = stored - 1024.
%                  T00_Lobe/T50_Lobe label the 5 lung lobes as 1..5 (0 = background).
% OUTPUT feats    : 5 x 21 matrix. Row l = lobe l (NaN row if that lobe is absent/tiny).
%                   Columns (same order as the Python feature names):
%        1 M_perf      |mass_in - mass_ex| / vol_ex      perfusion (blood-mass change) / 灌注
%        2 V_vent      |air_in  - air_ex | / vol_ex      ventilation (air change)      / 通氣
%        3 R_mism      log(V/M)                          V/Q mismatch                  / 失配
%        4 HUin        mean HU inhale
%        5 HUex        mean HU exhale
%        6 HUchg       HUin - HUex
%        7 volin       inhale lobe volume (mm^3)
%        8 volex       exhale lobe volume (mm^3)
%        9 volshrink   (volin - volex)/volex             deflation ratio / 收縮比
%       10 volratio    volex/volin
%       11 massin      sum(rho)*vox inhale               tissue+blood mass / 質量
%       12 massex      sum(rho)*vox exhale
%       13 airin       volin - massin                    air volume / 空氣量
%       14 airex       volex - massex
%       15 stdin       std HU inhale  (population, /N)    texture / 紋理
%       16 stdex       std HU exhale
%       17 skewin      skewness HU inhale (biased)
%       18 skewex      skewness HU exhale
%       19 hyper_max   min(max HU, 300)                  densest voxel (clot/vessel) / 最緻密
%       20 hyper_p99   99th-percentile HU                dense tail
%       21 hyper_frac  fraction of voxels HU in [50,150] HYPERDENSE LUMEN SIGN (key!) / 血栓密度比例
%
% Density model: rho = 1 + HU/1000  (g/cm^3; -1000 HU = air = 0, 0 HU = water = 1).
% 密度模型:rho = 1 + HU/1000(空氣 -1000HU→0、水 0HU→1)。

    S = load(matfile, 'T00', 'T00_Lobe', 'T50', 'T50_Lobe', 'xymm', 'zmm');
    xymm = double(S.xymm(1));  zmm = double(S.zmm(1));
    vox  = xymm * xymm * zmm;                  % voxel volume (mm^3) / 體素體積

    huIn = double(S.T00) - 1024;               % true HU / 還原 HU (= stored - 1024)
    huEx = double(S.T50) - 1024;
    rhoIn = 1 + huIn / 1000;                    % density / 密度
    rhoEx = 1 + huEx / 1000;
    lobeIn = S.T00_Lobe;  lobeEx = S.T50_Lobe;
    EPSV = 1e-4;

    feats = nan(5, 21);
    for l = 1:5
        mi = (lobeIn == l);  me = (lobeEx == l);   % inhale / exhale lobe masks / 吸吐肺葉遮罩
        if nnz(mi) < 50 || nnz(me) < 50            % missing/tiny lobe -> NaN row / 缺葉
            continue;
        end
        hi = huIn(mi);  he = huEx(me);             % HU values inside the lobe / 肺葉內 HU

        volIn  = nnz(mi) * vox;          volEx  = nnz(me) * vox;
        massIn = sum(rhoIn(mi)) * vox;   massEx = sum(rhoEx(me)) * vox;
        airIn  = volIn - massIn;         airEx  = volEx - massEx;

        M = abs(massIn - massEx) / volEx;          % perfusion / 灌注
        V = abs(airIn  - airEx ) / volEx;          % ventilation / 通氣
        R = log((V + EPSV) / (M + EPSV));          % V/Q mismatch / 失配

        feats(l, :) = [ ...
            M, V, R, ...
            mean(hi), mean(he), mean(hi) - mean(he), ...
            volIn, volEx, (volIn - volEx) / volEx, volEx / volIn, ...
            massIn, massEx, airIn, airEx, ...
            popstd(hi), popstd(he), bskew(hi), bskew(he), ...
            min(max(hi), 300), prctile_linear(hi, 99), mean(hi >= 50 & hi <= 150) ];
    end
end

% --- local helpers (match numpy/scipy defaults; no toolbox needed) ----------
% 本地子函式,與 numpy/scipy 預設一致,不需 toolbox。
function s = popstd(x)            % population std (numpy std, ddof=0) / 母體標準差
    m = mean(x);  s = sqrt(mean((x - m).^2));
end

function g = bskew(x)            % biased skewness (scipy.stats.skew default) / 有偏偏度
    m = mean(x);  m2 = mean((x - m).^2);  m3 = mean((x - m).^3);
    g = m3 / (m2^1.5);
end

function p = prctile_linear(x, q)   % linear-interp percentile (numpy.percentile default) / 線性插值百分位
    xs = sort(double(x(:)));  n = numel(xs);
    pos = (n - 1) * (q / 100);          % 0-based position / 0 起算位置
    lo = floor(pos);  fr = pos - lo;
    if lo + 2 <= n
        p = xs(lo + 1) + fr * (xs(lo + 2) - xs(lo + 1));
    else
        p = xs(n);
    end
end
