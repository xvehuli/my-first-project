% 初始化环境
clear; clc; close all;
%% 模拟参数设置
sim_params = struct();
sim_params.num_neutrons = 5000;           % 模拟的中子数量
sim_params.max_thickness = 10;            % 最大屏蔽厚度(mfp)
sim_params.thickness_step = 0.5;          % 厚度步长(mfp)
sim_params.material_types = {'铅', '水', '混凝土'};  % 材料类型
% 材料属性: [Sigma_a, Sigma_s, anisotropy]
sim_params.material_properties = [0.8, 0.2, 0.1;
                                  0.1, 0.9, 0.8;
                                  0.4, 0.6, 0.5];
sim_params.plot_trajectories = true;      % 绘制中子轨迹
sim_params.plot_3D = false;               % 3D轨迹展示
sim_params.max_trajectories = 50;         % 最大显示轨迹数量
sim_params.colorblind_mode = true;        
sim_params.use_energy_dependent = false;  % 能量相关截面
sim_params.neutron_energy = 2.0;          % 中子能量(MeV)
sim_params.use_parallel = true;           % 并行计算
% 初始化并行计算
if sim_params.use_parallel && isempty(gcp('nocreate'))
    parpool('local');
end
%% 主模拟循环
for material_idx = 1:size(sim_params.material_properties, 1)
    % 获取当前材料属性
    material_name = sim_params.material_types{material_idx};
    Sigma_a = sim_params.material_properties(material_idx, 1);
    Sigma_s = sim_params.material_properties(material_idx, 2);
    anisotropy = sim_params.material_properties(material_idx, 3);
    % 更新截面值
    if sim_params.use_energy_dependent
        [Sigma_a, Sigma_s] = energy_dependent_xs(sim_params.neutron_energy);
    end
    Sigma_t = Sigma_a + Sigma_s;  % 总截面
    % 生成厚度范围
    thickness_range = 0:sim_params.thickness_step:sim_params.max_thickness;
    num_thickness = length(thickness_range);
    % 初始化概率数组
    penetration_prob = zeros(1, num_thickness);
    absorption_prob = zeros(1, num_thickness);
    escape_prob = zeros(1, num_thickness);
    % 创建图形窗口
    fig = figure('Position', [100, 100, 1400, 700], 'Color', 'white');
    set(fig, 'Name', sprintf('中子穿透模拟 - %s', material_name));
    % 对每个厚度进行模拟
    for t_idx = 1:num_thickness
        current_thickness = thickness_range(t_idx);
        final_state = zeros(1, sim_params.num_neutrons);
        history = cell(1, sim_params.num_neutrons);
        % 并行或串行模拟中子运动
        if sim_params.use_parallel
            parfor n = 1:sim_params.num_neutrons
                record_history = (n <= sim_params.max_trajectories);
                [final_state(n), ~, tmp_history] = simulate_neutron(...
                    current_thickness, Sigma_t, Sigma_a, Sigma_s, anisotropy, ...
                    record_history, sim_params.plot_3D);
                if record_history
                    history{n} = tmp_history;
                end
            end
        else
            for n = 1:sim_params.num_neutrons
                record_history = (n <= sim_params.max_trajectories);
                [final_state(n), ~, history{n}] = simulate_neutron(...
                    current_thickness, Sigma_t, Sigma_a, Sigma_s, anisotropy, ...
                    record_history, sim_params.plot_3D);
            end
        end
        % 计算概率
        penetration_count = sum(final_state == 2);
        absorption_count = sum(final_state == 4);
        escape_count = sum(final_state == 3);
        penetration_prob(t_idx) = penetration_count / sim_params.num_neutrons;
        absorption_prob(t_idx) = absorption_count / sim_params.num_neutrons;
        escape_prob(t_idx) = escape_count / sim_params.num_neutrons;
        % 打印当前进度
        fprintf('[%s] 厚度 %.1fmfp: 穿透=%.2f%%, 吸收=%.2f%%, 逸出=%.2f%%\n', ...
                material_name, current_thickness, ...
                penetration_prob(t_idx)*100, absorption_prob(t_idx)*100, ...
                escape_prob(t_idx)*100);
        % 更新可视化
        if sim_params.plot_trajectories
            update_visualization(history, t_idx, thickness_range, ...
                penetration_prob, absorption_prob, escape_prob, current_thickness, ...
                sim_params, material_idx, material_name, ...
                Sigma_t, Sigma_a, Sigma_s, anisotropy);
            drawnow;
        end
    end
    % 生成最终结果可视化并保存
    visualize_final_results(thickness_range, penetration_prob, absorption_prob, ...
        escape_prob, material_name, Sigma_t, Sigma_a, Sigma_s);
    save_results(material_name, thickness_range, ...
        penetration_prob, absorption_prob, escape_prob);
end% 清理并行池
if sim_params.use_parallel && ~isempty(gcp)
    delete(gcp);
end
%% 中子运动模拟函数
% 输入参数:
%   thickness: 屏蔽层厚度
%   Sigma_t: 总截面
%   Sigma_a: 吸收截面
%   Sigma_s: 散射截面
%   anisotropy: 各向异性因子
%   record_history: 记录轨迹
%   use_3D: 3D模拟
% 输出参数:
%   final_state: 最终状态(2=穿透, 3=逸出, 4=吸收)
%   final_pos: 最终位置
%   history: 轨迹历史记录
function [final_state, final_pos, history] = simulate_neutron(...
    thickness, Sigma_t, Sigma_a, Sigma_s, anisotropy, record_history, use_3D)
    % 初始化状态和位置
    final_state = 0;  % 初始状态
    final_pos = 0;    % 最终位置
    history = struct('x', [], 'y', [], 'z', [], 'status', []);  % 轨迹记录
    % 初始位置和方向
    x = 0; y = 0; z = 0;
    is_alive = true;
    direction = [1, 0, 0];  % 初始方向沿x轴
    % 记录初始状态
    if record_history
        history.x = x;
        if use_3D
            history.y = y;
            history.z = z;
        end
        history.status = 1;  % 1表示活跃状态
    end
    % 中子运动循环
    while is_alive
        % 计算自由程(根据总截面)
        step_length = -log(rand()) / Sigma_t;
        % 更新位置
        x = x + direction(1) * step_length;
        if use_3D
            y = y + direction(2) * step_length;
            z = z + direction(3) * step_length;
        end
        % 记录轨迹
        if record_history
            history.x(end+1) = x;
            if use_3D
                history.y(end+1) = y;
                history.z(end+1) = z;
            end
            history.status(end+1) = 1;
        end
        % 检查是否穿透屏蔽层
        if x >= thickness
            final_state = 2;  % 穿透
            final_pos = x;
            is_alive = false;
            if record_history
                history.status(end) = final_state;
            end
            break;
        end
        % 检查是否从入射面逸出
        if x < 0
            final_state = 3;  % 逸出
            final_pos = x;
            is_alive = false;
            if record_history
                history.status(end) = final_state;
            end
            break;
        end
        % 确定碰撞类型(吸收或散射)
        collision_prob = rand();
        if collision_prob < Sigma_a / Sigma_t
            % 吸收事件
            final_state = 4;  % 吸收
            final_pos = x;
            is_alive = false;
            if record_history
                history.status(end) = final_state;
            end
        else
            % 散射事件 - 计算新方向
            [theta, phi] = calculate_scattering_angles(anisotropy);
            % 计算新方向向量
            if use_3D
                new_dir = [sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)];
            else
                new_dir = [cos(theta), sin(theta), 0];
            end
            direction = new_dir / norm(new_dir);  % 归一化方向向量
        end
    end
end
%% 计算散射角度的辅助函数
% 输入参数:
%   anisotropy: 各向异性因子
% 输出参数:
%   theta: 极角
%   phi: 方位角
function [theta, phi] = calculate_scattering_angles(anisotropy)
    phi = 2 * pi * rand();  % 方位角均匀分布
    if anisotropy == 0
        % 各向同性散射
        cos_theta = 2 * rand() - 1;
    else
        % 各向异性散射 - 采用Henyey-Greenstein分布
        cos_theta = (1 + anisotropy^2 - ((1 - anisotropy^2) / ...
                    (1 - anisotropy + 2 * anisotropy * rand()))^2) / ...
                    (2 * anisotropy);
        cos_theta = max(min(cos_theta, 1), -1);  % 确保在有效范围内
    end
    theta = acos(cos_theta);  % 极角
end
%% 可视化更新函数
function update_visualization(histories, t_idx, thickness_range, ...
    penetration_prob, absorption_prob, escape_prob, current_thickness, ...
    sim_params, material_idx, material_name, ...
    Sigma_t, Sigma_a, Sigma_s, anisotropy)
    fig = gcf;
    clf(fig);
    % 设置颜色方案
    if sim_params.colorblind_mode
        colors = [
            0, 0.4470, 0.7410;    % 蓝色(活跃)
            0.4660, 0.6740, 0.1880;% 绿色(穿透)
            0.6350, 0.0780, 0.1840;% 红色(逸出)
            0.4940, 0.1840, 0.5560 % 紫色(吸收)
        ];
    else
        colors = [
            0, 0, 1;    % 蓝色(活跃)
            0, 0.8, 0;  % 绿色(穿透)
            1, 0, 0;    % 红色(逸出)
            0.5, 0, 0.5 % 紫色(吸收)
        ];
    end
    % 绘制轨迹子图
    if sim_params.plot_3D
        subplot(1, 2, 1, 'Projection', 'perspective');
    else
        subplot(1, 2, 1);
    end
    hold on;
    % 绘制屏蔽层
    if sim_params.plot_3D
        [X, Y] = meshgrid([0, current_thickness], [-0.5, 0.5]);
        Z = zeros(size(X));
        surf(X, Y, Z, 'FaceColor', [0.9 0.9 0.9], 'EdgeColor', 'k', 'FaceAlpha', 0.3);
    else
        rectangle('Position', [0, -0.5, current_thickness, 1], ...
                  'FaceColor', [0.9 0.9 0.9], 'EdgeColor', 'k', 'LineWidth', 1.5);
        line([0, 0], [-0.5, 0.5], 'Color', 'k', 'LineWidth', 2);
        line([current_thickness, current_thickness], [-0.5, 0.5], ...
             'Color', 'k', 'LineWidth', 2);
    end
    % 绘制中子轨迹
    max_plot = min(length(histories), sim_params.max_trajectories);
    for i = 1:max_plot
        if ~isempty(histories{i}) && ~isempty(histories{i}.x)
            x_vals = histories{i}.x;
            status = histories{i}.status;
            % 绘制轨迹线段
            for j = 1:(length(x_vals)-1)
                color_idx = min(status(j+1), 4);
                if sim_params.plot_3D
                    plot3(x_vals(j:j+1), histories{i}.y(j:j+1), histories{i}.z(j:j+1), ...
                          'Color', colors(color_idx, :), 'LineWidth', 1);
                else
                    plot(x_vals(j:j+1), zeros(1,2), ...
                         'Color', colors(color_idx, :), 'LineWidth', 1.5);
                end
            end
            % 绘制终点标记
            end_status = status(end);
            [marker, marker_size] = get_status_marker(end_status);
            
            if sim_params.plot_3D
                plot3(x_vals(end), histories{i}.y(end), histories{i}.z(end), ...
                      marker, 'Color', colors(end_status, :), ...
                      'MarkerSize', marker_size, 'LineWidth', 2);
            else
                plot(x_vals(end), 0, marker, ...
                     'Color', colors(end_status, :), ...
                     'MarkerSize', marker_size, 'LineWidth', 2);
            end
        end
    end
    % 设置轨迹图属性
    if sim_params.plot_3D
        xlim([-1, current_thickness + 1]);
        ylim([-1, 1]);
        zlim([-1, 1]);
        view(-30, 30);
        grid on;
        title(sprintf('%s - 3D中子轨迹 (厚度: %.1f mfp)', ...
              material_name, current_thickness), 'FontSize', 12);
        xlabel('X (mfp)');
        ylabel('Y (mfp)');
        zlabel('Z (mfp)');
    else
        xlim([-1, current_thickness + 1]);
        ylim([-1, 1]);
        title(sprintf('%s - 中子轨迹 (厚度: %.1f mfp)', ...
              material_name, current_thickness), 'FontSize', 12);
        xlabel('位置 (mfp)', 'FontSize', 10);
    end
    % 添加图例
    legend_labels = {'活跃中子', '穿透', '逸出', '吸收'};
    for i = 1:4
        if sim_params.plot_3D
            plot3(NaN, NaN, NaN, 'Color', colors(i, :), 'LineWidth', 2);
        else
            plot(NaN, NaN, 'Color', colors(i, :), 'LineWidth', 2);
        end
    end
    legend(legend_labels, 'Location', 'southoutside', 'Orientation', 'horizontal');
    % 绘制概率分布子图
    subplot(1, 2, 2);
    hold on;
    plot(thickness_range(1:t_idx), penetration_prob(1:t_idx), ...
         'b-o', 'LineWidth', 1.5, 'MarkerSize', 6, 'MarkerFaceColor', 'b');
    plot(thickness_range(1:t_idx), absorption_prob(1:t_idx), ...
         'm-s', 'LineWidth', 1.5, 'MarkerSize', 6, 'MarkerFaceColor', 'm');
    plot(thickness_range(1:t_idx), escape_prob(1:t_idx), ...
         'r-^', 'LineWidth', 1.5, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
    xlim([0, max(thickness_range)]);
    ylim([0, 1]);
    grid on;
    title('概率分布', 'FontSize', 12);
    xlabel('屏蔽层厚度 (mfp)', 'FontSize', 10);
    ylabel('概率', 'FontSize', 10);
    legend('穿透概率', '吸收概率', '逸出概率', 'Location', 'best');
    % 添加信息文本框
    info_str = sprintf('材料: %s\n厚度: %.1f mfp\n中子数: %d\n穿透率: %.2f%%\n吸收率: %.2f%%\n逸出率: %.2f%%', ...
                      material_name, current_thickness, ...
                      sim_params.num_neutrons, penetration_prob(t_idx)*100, ...
                      absorption_prob(t_idx)*100, escape_prob(t_idx)*100);
    annotation('textbox', [0.15, 0.75, 0.25, 0.15], 'String', info_str, ...
               'FitBoxToText', 'on', 'BackgroundColor', 'white', ...
               'FontSize', 10, 'EdgeColor', 'k', 'LineWidth', 1);
    % 添加参数文本框
    params_str = sprintf('Σt = %.2f mfp⁻¹\nΣa = %.2f mfp⁻¹\nΣs = %.2f mfp⁻¹\n各向异性: %.2f', ...
                         Sigma_t, Sigma_a, Sigma_s, anisotropy);
    annotation('textbox', [0.15, 0.55, 0.25, 0.1], 'String', params_str, ...
               'FitBoxToText', 'on', 'BackgroundColor', 'white', ...
               'FontSize', 10, 'EdgeColor', 'k', 'LineWidth', 1);
end
%% 获取状态标记的辅助函数
function [marker, marker_size] = get_status_marker(status)
    switch status
        case 2  % 穿透
            marker = '^';
            marker_size = 10;
        case 3  % 逸出
            marker = 'v';
            marker_size = 8;
        case 4  % 吸收
            marker = 'x';
            marker_size = 10;
        otherwise  % 活跃
            marker = 'o';
            marker_size = 8;
    end
end
%% 最终结果可视化函数
function visualize_final_results(thickness_range, penetration_prob, absorption_prob, ...
    escape_prob, material_name, Sigma_t, Sigma_a, Sigma_s)
    fig = figure('Position', [200, 200, 1200, 800], 'Color', 'white', ...
                 'Name', sprintf('%s - 最终结果', material_name));
    % 主概率曲线图
    subplot(2, 2, [1, 2]);
    hold on;
    plot(thickness_range, penetration_prob, 'b-o', 'LineWidth', 2, ...
         'MarkerSize', 8, 'MarkerFaceColor', 'b');
    plot(thickness_range, absorption_prob, 'm-s', 'LineWidth', 2, ...
         'MarkerSize', 8, 'MarkerFaceColor', 'm');
    plot(thickness_range, escape_prob, 'r-^', 'LineWidth', 2, ...
         'MarkerSize', 8, 'MarkerFaceColor', 'r');
    % 理论穿透概率(纯吸收模型)
    theory_penetration = exp(-thickness_range * Sigma_a);
    plot(thickness_range, theory_penetration, 'k--', 'LineWidth', 2);
    grid on;
    xlabel('屏蔽层厚度 (mfp)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('概率', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('%s: 中子穿透、吸收与逸出概率', material_name), ...
          'FontSize', 14, 'FontWeight', 'bold');
    legend('模拟穿透概率', '模拟吸收概率', '模拟逸出概率', '理论穿透概率(纯吸收)', ...
           'Location', 'best');
    set(gca, 'FontSize', 11);
    % 计算并标记半值厚度
    if any(penetration_prob <= 0.5)
        idx = find(penetration_prob <= 0.5, 1);
        if idx > 1
            % 线性插值获取更精确的半值厚度
            x1 = thickness_range(idx-1);
            x2 = thickness_range(idx);
            y1 = penetration_prob(idx-1);
            y2 = penetration_prob(idx);
            half_thickness = interp1([y1, y2], [x1, x2], 0.5);
        else
            half_thickness = thickness_range(idx);
        end
        % 绘制半值厚度参考线
        line([half_thickness, half_thickness], [0, 0.5], 'Color', 'k', 'LineStyle', '--');
        line([0, half_thickness], [0.5, 0.5], 'Color', 'k', 'LineStyle', '--');
        text(half_thickness, 0.55, sprintf('半厚度: %.2f mfp', half_thickness), ...
             'FontSize', 10, 'HorizontalAlignment', 'center');
    end
    % 平均概率饼图
    subplot(2, 2, 3);
    avg_penetration = mean(penetration_prob);
    avg_absorption = mean(absorption_prob);
    avg_escape = mean(escape_prob);
    pie_data = [avg_penetration, avg_absorption, avg_escape];
    pie_labels = {'穿透', '吸收', '逸出'};
    h = pie(pie_data, pie_labels);
    set(h(1), 'FaceColor', 'b');
    set(h(3), 'FaceColor', 'm');
    set(h(5), 'FaceColor', 'r');
    title('平均概率分布', 'FontSize', 12);
    % 累积概率图
    subplot(2, 2, 4);
    hold on;
    area(thickness_range, cumsum(penetration_prob), 'FaceColor', 'b', 'FaceAlpha', 0.3);
    area(thickness_range, cumsum(absorption_prob), 'FaceColor', 'm', 'FaceAlpha', 0.3);
    area(thickness_range, cumsum(escape_prob), 'FaceColor', 'r', 'FaceAlpha', 0.3);
    grid on;
    xlabel('屏蔽层厚度 (mfp)');
    ylabel('累积概率');
    title('累积概率分布');
    legend('穿透', '吸收', '逸出', 'Location', 'best');
end
%% 能量相关截面计算函数
function [Sigma_a, Sigma_s] = energy_dependent_xs(energy)
    % 基于能量计算中子截面
    % 输入: energy - 中子能量(MeV)
    % 输出: Sigma_a - 吸收截面, Sigma_s - 散射截面
    base_a = 0.3;  % 基础吸收截面
    base_s = 0.7;  % 基础散射截面
    % 能量相关因子(示例模型)
    energy_factor = 1 / sqrt(energy);  % 吸收截面随能量增加而减小
    Sigma_a = base_a * energy_factor;
    % 散射截面带有小的能量依赖性
    Sigma_s = base_s * (1 + 0.2 * sin(energy));
end
%% 结果保存函数
function save_results(material_name, thickness_range, penetration_prob, ...
    absorption_prob, escape_prob)
    % 组织结果数据
    results.material = material_name;
    results.thickness = thickness_range;
    results.penetration_prob = penetration_prob;
    results.absorption_prob = absorption_prob;
    results.escape_prob = escape_prob;
    results.timestamp = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
    % 保存为MAT文件
    mat_filename = sprintf('neutron_results_%s.mat', material_name);
    save(mat_filename, 'results');
    fprintf('结果已保存: %s\n', mat_filename);
    % 保存为CSV文件
    csv_filename = sprintf('neutron_results_%s.csv', material_name);
    T = table(thickness_range', penetration_prob', absorption_prob', escape_prob', ...
              'VariableNames', {'Thickness_mfp', 'Penetration_Probability', ...
              'Absorption_Probability', 'Escape_Probability'});
    writetable(T, csv_filename);
    fprintf('结果已保存: %s\n', csv_filename);
    % 保存图表
    fig_filename = sprintf('neutron_results_%s.png', material_name);
    saveas(gcf, fig_filename);
    fprintf('图表已保存: %s\n', fig_filename);
end