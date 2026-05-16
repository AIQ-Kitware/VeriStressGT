function run_nnv_verifier(onnx_path, vnnlib_path, timeout)

    if nargin < 3 || isempty(timeout)
        timeout = 0;
    end

    setup_nnv_path();

    fprintf('========================================\n');
    fprintf('NNV Verification\n');
    fprintf('ONNX: %s\n', onnx_path);
    fprintf('VNNLIB: %s\n', vnnlib_path);
    fprintf('Timeout: %.2f\n', timeout);
    fprintf('========================================\n');

    t_start = tic;

    try
        fprintf('Loading ONNX model...\n');
        matlab_net = load_onnx_network(onnx_path);

        fprintf('Parsing VNNLIB spec...\n');
        [lb, ub, output_spec] = parse_vnnlib(vnnlib_path);

        lb = lb(:);
        ub = ub(:);

        fprintf('Parsed %d input dimensions\n', length(lb));
        fprintf('lb range: [%.6g, %.6g]\n', min(lb), max(lb));
        fprintf('ub range: [%.6g, %.6g]\n', min(ub), max(ub));
        fprintf('True label: %d\n', output_spec.label);
        fprintf('Number of outputs: %d\n', output_spec.n_outputs);

        fprintf('Creating input Star set...\n');
        input_set = make_input_star(lb, ub);

        fprintf('Converting to NNV network...\n');
        nnv_net = matlab2nnv(matlab_net);

        fprintf('Running reachability...\n');

        reach_method = getenv('NNV_REACH_METHOD');
        if isempty(reach_method)
            reach_method = 'approx-star';
        end
        fprintf('Reach method: %s\n', reach_method);

        reachOptions = struct;
        reachOptions.reachMethod = reach_method;

        output_set = nnv_net.reach(input_set, reachOptions);

        fprintf('Checking VNNLIB violation margins...\n');
        result = check_output_spec_margin(output_set, output_spec);

        elapsed = toc(t_start);

        fprintf('========================================\n');
        fprintf('Result: %s\n', result);
        fprintf('Time: %.4f\n', elapsed);
        fprintf('========================================\n');

    catch ME
        elapsed = toc(t_start);
        fprintf('========================================\n');
        fprintf('Result: error\n');
        fprintf('Time: %.4f\n', elapsed);
        fprintf('Error: %s\n', ME.message);
        fprintf('Stack:\n');
        for i = 1:length(ME.stack)
            fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
        end
        fprintf('========================================\n');
    end
end


function setup_nnv_path()
% Configure MATLAB path for bundled NNV.

    this_file = mfilename('fullpath');
    wrapper_dir = fileparts(this_file);

    env_root = getenv('NNV_HOME');
    if ~isempty(env_root)
        NNV_ROOT = env_root;
    else
        NNV_ROOT = fullfile(wrapper_dir, 'nnv');
    end

    if ~isfolder(NNV_ROOT)
        error('NNV_ROOT does not exist: %s', NNV_ROOT);
    end

    % Remove stale saved paths from old checkout.
    stale_roots = {
        '/Desktop/VeriStressGT/nnv'
    };

    path_parts = strsplit(path, pathsep);
    for k = 1:numel(path_parts)
        p = path_parts{k};
        for r = 1:numel(stale_roots)
            if contains(p, stale_roots{r})
                try
                    rmpath(p);
                catch
                end
            end
        end
    end

    candidates = {
        fullfile(NNV_ROOT, 'code', 'nnv')
        fullfile(NNV_ROOT, 'nnv', 'code', 'nnv')
        NNV_ROOT
    };

    added = false;
    for k = 1:numel(candidates)
        candidate = candidates{k};
        if isfolder(candidate)
            addpath(genpath(candidate));
            fprintf('Added NNV path: %s\n', candidate);
            added = true;
            break;
        end
    end

    addpath(wrapper_dir);
    rehash toolboxcache;

    if ~added
        error('Could not find valid NNV MATLAB code directory under: %s', NNV_ROOT);
    end

    if isempty(which('Star'))
        error('NNV Star class not found on MATLAB path.');
    end
    if isempty(which('matlab2nnv'))
        error('NNV matlab2nnv function not found on MATLAB path.');
    end
end


function net = load_onnx_network(onnx_path)
% Import ONNX into MATLAB. Keep fallbacks because MATLAB ONNX import behavior
% differs across versions.

    try
        try
            layers = importONNXLayers( ...
                onnx_path, ...
                "ImportWeights", true, ...
                "InputDataFormats", "BC", ...
                "OutputDataFormats", "BC");
        catch
            layers = importONNXLayers(onnx_path, "ImportWeights", true);
        end

        net = assembleNetwork(layers);
        return;
    catch ME1
        fprintf('importONNXLayers failed: %s\n', ME1.message);
    end

    try
        net = importNetworkFromONNX( ...
            onnx_path, ...
            "InputDataFormats", "BC", ...
            "OutputDataFormats", "BC");
        return;
    catch ME2
        fprintf('importNetworkFromONNX failed: %s\n', ME2.message);
        error('Unable to import ONNX model.');
    end
end


function input_set = make_input_star(lb, ub)
% Make NNV Star set for box lb <= x <= ub.

    if any(lb > ub)
        error('Invalid input bounds: some lb > ub.');
    end

    try
        input_set = Star(lb, ub);
        return;
    catch
    end

    try
        input_box = Box(lb, ub);
        input_set = input_box.toStar();
        return;
    catch
    end

    n = length(lb);
    center = (lb + ub) / 2;
    basis = diag((ub - lb) / 2);

    % x = center + basis * alpha, alpha in [-1,1]^n
    C = [eye(n); -eye(n)];
    d = ones(2 * n, 1);

    input_set = Star(center, basis, C, d);
end


function [lb, ub, output_spec] = parse_vnnlib(vnnlib_path)
% Parse box input constraints and classification-violation output spec.

    fid = fopen(vnnlib_path, 'r');
    if fid == -1
        error('Could not open VNNLIB file: %s', vnnlib_path);
    end
    content = fread(fid, '*char')';
    fclose(fid);

    num_re = '[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?';

    % Input declarations.
    x_decl_matches = regexp(content, 'declare-const\s+X_(\d+)\s+Real', 'tokens');
    if isempty(x_decl_matches)
        error('No input declarations found in VNNLIB file.');
    end

    x_indices = zeros(numel(x_decl_matches), 1);
    for i = 1:numel(x_decl_matches)
        x_indices(i) = str2double(x_decl_matches{i}{1});
    end
    n_inputs = max(x_indices) + 1;

    fprintf('Found %d input declarations\n', n_inputs);

    lb = zeros(n_inputs, 1);
    ub = zeros(n_inputs, 1);
    lb_found = false(n_inputs, 1);
    ub_found = false(n_inputs, 1);

    lines = strsplit(content, newline);

    for i = 1:length(lines)
        line = strtrim(lines{i});

        % (assert (>= X_i value))
        lb_pat = ['\(assert\s*\(>=\s*X_(\d+)\s+(' num_re ')\s*\)\)'];
        lb_match = regexp(line, lb_pat, 'tokens');
        if ~isempty(lb_match)
            idx = str2double(lb_match{1}{1}) + 1;
            val = str2double(lb_match{1}{2});
            if idx >= 1 && idx <= n_inputs
                lb(idx) = val;
                lb_found(idx) = true;
            end
        end

        % (assert (<= X_i value))
        ub_pat = ['\(assert\s*\(<=\s*X_(\d+)\s+(' num_re ')\s*\)\)'];
        ub_match = regexp(line, ub_pat, 'tokens');
        if ~isempty(ub_match)
            idx = str2double(ub_match{1}{1}) + 1;
            val = str2double(ub_match{1}{2});
            if idx >= 1 && idx <= n_inputs
                ub(idx) = val;
                ub_found(idx) = true;
            end
        end
    end

    if ~all(lb_found)
        missing = find(~lb_found) - 1;
        error('Missing lower bounds for inputs: %s', num2str(missing(:)'));
    end
    if ~all(ub_found)
        missing = find(~ub_found) - 1;
        error('Missing upper bounds for inputs: %s', num2str(missing(:)'));
    end
    if any(lb > ub)
        error('Invalid VNNLIB bounds: some lb > ub.');
    end

    % Output declarations.
    y_decl_matches = regexp(content, 'declare-const\s+Y_(\d+)\s+Real', 'tokens');
    if isempty(y_decl_matches)
        error('No output declarations found in VNNLIB file.');
    end

    y_indices = zeros(numel(y_decl_matches), 1);
    for i = 1:numel(y_decl_matches)
        y_indices(i) = str2double(y_decl_matches{i}{1});
    end
    n_outputs = max(y_indices) + 1;

    % Parse violation comparisons: (>= Y_j Y_label)
    comp_matches = regexp(content, '\(>=\s*Y_(\d+)\s+Y_(\d+)\s*\)', 'tokens');

    output_spec = struct;
    output_spec.type = 'classification_violation';
    output_spec.n_outputs = n_outputs;

    if isempty(comp_matches)
        warning('No output comparisons found. Defaulting true label to 0.');
        output_spec.label = 0;
        output_spec.competitors = setdiff(0:n_outputs-1, output_spec.label);
    else
        lefts = zeros(numel(comp_matches), 1);
        rights = zeros(numel(comp_matches), 1);

        for i = 1:numel(comp_matches)
            lefts(i) = str2double(comp_matches{i}{1});
            rights(i) = str2double(comp_matches{i}{2});
        end

        % In standard robustness-violation VNNLIB:
        %   exists j != label with Y_j >= Y_label.
        % The true label is therefore the repeated right-hand side.
        unique_rights = unique(rights);
        counts = zeros(numel(unique_rights), 1);
        for i = 1:numel(unique_rights)
            counts(i) = sum(rights == unique_rights(i));
        end
        [~, max_idx] = max(counts);
        label = unique_rights(max_idx);

        competitors = unique(lefts(rights == label))';
        competitors = competitors(competitors ~= label);

        output_spec.label = label;
        output_spec.competitors = competitors;
    end

    fprintf('Output spec: %d classes, true label=%d\n', ...
            output_spec.n_outputs, output_spec.label);
    fprintf('Competitors: %s\n', num2str(output_spec.competitors));
end


function result = check_output_spec_margin(output_set, output_spec)
% Check the actual VNNLIB margins:
%   unsafe iff exists competitor j such that Y_j - Y_label >= 0.
%
% We prove safety if, for every reachable output set and every competitor,
%   max(Y_j - Y_label) < 0.
%
% This is stronger than checking independent scalar output ranges.

    sets = normalize_output_sets(output_set);

    label0 = output_spec.label;      % zero-indexed
    label = label0 + 1;              % MATLAB one-indexed
    competitors0 = output_spec.competitors(:)';

    tol = 1e-7;

    result = 'safe';

    for sidx = 1:numel(sets)
        S = sets{sidx};

        fprintf('--- Output set %d/%d ---\n', sidx, numel(sets));

        [out_lb, out_ub] = get_set_ranges(S);
        out_lb = out_lb(:);
        out_ub = out_ub(:);

        n_classes = length(out_lb);
        if output_spec.n_outputs > 0
            n_classes = min(n_classes, output_spec.n_outputs);
        end

        fprintf('Independent output ranges:\n');
        for j = 1:n_classes
            marker = '';
            if j == label
                marker = ' <-- true label';
            end
            fprintf('  Y_%d: [%.8g, %.8g]%s\n', j-1, out_lb(j), out_ub(j), marker);
        end

        for comp0 = competitors0
            comp = comp0 + 1;

            if comp < 1 || comp > n_classes
                warning('Skipping invalid competitor Y_%d.', comp0);
                continue;
            end

            c = zeros(1, n_classes);
            c(comp) = 1;
            c(label) = -1;

            [margin_ub, method] = upper_bound_linear_expr(S, c);

            fprintf('  max(Y_%d - Y_%d) <= %.10g   [%s]\n', ...
                    comp0, label0, margin_ub, method);

            if ~(margin_ub < -tol)
                fprintf('Cannot prove class %d margin negative: max(Y_%d - Y_%d) <= %.10g\n', ...
                        comp0, comp0, label0, margin_ub);
                result = 'unknown';
                return;
            end
        end
    end

    fprintf('Proved no violation: all competitor margins are strictly negative.\n');
end


function sets = normalize_output_sets(output_set)
% Convert NNV output to a cell array of sets.

    if iscell(output_set)
        sets = output_set;
        return;
    end

    if numel(output_set) > 1
        sets = cell(1, numel(output_set));
        for i = 1:numel(output_set)
            sets{i} = output_set(i);
        end
        return;
    end

    sets = {output_set};
end


function [lb, ub] = get_set_ranges(S)
% Get independent coordinate ranges for an output set.

    try
        [lb, ub] = S.getRanges();
        return;
    catch
    end

    try
        [lb, ub] = S.estimateRanges();
        return;
    catch
    end

    if isprop_or_field(S, 'lb') && isprop_or_field(S, 'ub')
        lb = get_prop_or_field(S, 'lb');
        ub = get_prop_or_field(S, 'ub');
        return;
    end

    error('Could not obtain ranges for output set.');
end


function [ub, method] = upper_bound_linear_expr(S, c)
% Upper bound max c*y over Star-like set S.
%
% Preferred:
%   1. Affine-map the Star to one-dimensional margin and call getRanges.
%   2. Directly solve LP over Star predicate variables.
%   3. Fallback to independent coordinate intervals.

    c = double(c(:)');

    % Method 1: affineMap margin Star, then range.
    try
        if ismethod(S, 'affineMap')
            try
                M = S.affineMap(c, 0);
            catch
                M = S.affineMap(c, []);
            end
            [~, m_ub] = get_set_ranges(M);
            ub = double(m_ub(1));
            method = 'affineMap+range';
            return;
        end
    catch
    end

    % Method 2: direct LP on Star representation.
    try
        [ub, method] = upper_bound_star_lp(S, c);
        return;
    catch ME
        fprintf('  Warning: direct Star LP failed: %s\n', ME.message);
    end

    % Method 3: conservative independent interval fallback.
    [lb, ub_vec] = get_set_ranges(S);
    lb = double(lb(:));
    ub_vec = double(ub_vec(:));

    pos = c >= 0;
    neg = c < 0;

    ub = sum(c(pos)' .* ub_vec(pos)) + sum(c(neg)' .* lb(neg));
    method = 'independent-ranges-fallback';
end


function [ub, method] = upper_bound_star_lp(S, c)
% Solve max c*y over an NNV Star using its predicate constraints.
%
% Star form:
%   y = V(:,1) + V(:,2:end) * alpha
%   C alpha <= d
%   pred_lb <= alpha <= pred_ub

    if ~isprop_or_field(S, 'V')
        error('Set has no V field/property.');
    end

    V = double(get_prop_or_field(S, 'V'));

    if size(c, 2) ~= size(V, 1)
        % If c is shorter than V dimension, pad. If longer, truncate.
        if size(c, 2) < size(V, 1)
            c = [c, zeros(1, size(V, 1) - size(c, 2))];
        else
            c = c(1:size(V, 1));
        end
    end

    nvar = size(V, 2) - 1;

    const = c * V(:, 1);
    coef = (c * V(:, 2:end))';

    if nvar == 0
        ub = const;
        method = 'star-constant';
        return;
    end

    A = [];
    b = [];

    if isprop_or_field(S, 'C') && isprop_or_field(S, 'd')
        A = double(get_prop_or_field(S, 'C'));
        b = double(get_prop_or_field(S, 'd'));
    end

    pred_lb = -inf(nvar, 1);
    pred_ub = inf(nvar, 1);

    if isprop_or_field(S, 'pred_lb')
        tmp = double(get_prop_or_field(S, 'pred_lb'));
        if numel(tmp) == nvar
            pred_lb = tmp(:);
        end
    elseif isprop_or_field(S, 'predicate_lb')
        tmp = double(get_prop_or_field(S, 'predicate_lb'));
        if numel(tmp) == nvar
            pred_lb = tmp(:);
        end
    end

    if isprop_or_field(S, 'pred_ub')
        tmp = double(get_prop_or_field(S, 'pred_ub'));
        if numel(tmp) == nvar
            pred_ub = tmp(:);
        end
    elseif isprop_or_field(S, 'predicate_ub')
        tmp = double(get_prop_or_field(S, 'predicate_ub'));
        if numel(tmp) == nvar
            pred_ub = tmp(:);
        end
    end

    % linprog solves min f'*alpha. We need max coef'*alpha.
    f = -coef;

    try
        options = optimoptions('linprog', 'Display', 'none');
        [~, fval, exitflag] = linprog(f, A, b, [], [], pred_lb, pred_ub, options);
    catch
        [~, fval, exitflag] = linprog(f, A, b, [], [], pred_lb, pred_ub);
    end

    if exitflag == 1 || exitflag == 3
        ub = const - fval;
        method = 'star-linprog';
        return;
    elseif exitflag == -2
        ub = -inf;
        method = 'empty-star-linprog';
        return;
    else
        error('linprog failed with exitflag %d.', exitflag);
    end
end


function tf = isprop_or_field(obj, name)
    tf = false;
    try
        if isobject(obj) && isprop(obj, name)
            tf = true;
            return;
        end
    catch
    end

    try
        if isstruct(obj) && isfield(obj, name)
            tf = true;
            return;
        end
    catch
    end
end


function val = get_prop_or_field(obj, name)
    if isobject(obj) && isprop(obj, name)
        val = obj.(name);
        return;
    end

    if isstruct(obj) && isfield(obj, name)
        val = obj.(name);
        return;
    end

    error('Missing property/field: %s', name);
end