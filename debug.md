

# 0 目录约定
# 官方预训练模型：models/offical/      （8 个，只读，别覆盖）
# 自己训练的模型：models/exp-1118/     （训练后导出到这里）
# 训练中间产物：  logs/rsl_rl/<experiment>/<run>/   （.pt 权重、tensorboard、自动导出的 onnx）
# 注意：train 只把东西写进 logs/，绝不会自动写进 models/；要从 logs/ 导出到
#       models/exp-1118/ 必须手动跑第 3 节的 export 命令。

# 1 训练
uv run train Mjlab-Velocity-Flat-XgoDuck --env.scene.num-envs 4096 --agent.max-iterations 500 --agent.run-name exp-1118

# 2 在 viser 里看训练结果（读 logs/ 里最新的 .pt）
uv run play Mjlab-Velocity-Flat-XgoDuck --num-envs 1 --viewer viser

# 3 训练完把 checkpoint 导出到 models/exp-1118/，然后用 infer_policy 跑
#   把 <RUN> 换成 logs/rsl_rl/xgoduck_velocity/ 下带 _exp-1118 的那个目录名
uv run python scripts/export.py Mjlab-Velocity-Flat-XgoDuck --checkpoint-file logs/rsl_rl/xgoduck_velocity/<RUN>/model_500.pt --onnx-file models/exp-1118/xgoduck_walk.onnx
uv run python scripts/infer_policy.py --walking models/exp-1118/xgoduck_walk.onnx --new-cmd-obs

# 4 官方预训练模型测试（models/offical/，来自 xgoduck_runtime_arduino 的 ai-chat-audio-video-tof 分支）
# 这批是 61 维 obs 的新版 XgoDuck 策略，必须带 --new-cmd-obs，否则报 "Got: 51 Expected: 61"
# 机器人本体默认已是 xgoduck（--robot xgoduck），会自动选对场景和出生高度；只有跑老的
# MicroDuck checkpoint 才需要 --robot microduck。
#
# 键盘必须在启动脚本的那个终端里按（不是点 MuJoCo 窗口）。上下键每次 ±0.05 m/s，
# 起步值就是 --lin-vel-x；4.1 已经给了 0.3（= 上限），再按上键不会有变化，想感受
# 加速就用第 5 节那条从 0 起步的。

# 4.1 行走
uv run python scripts/infer_policy.py --walking models/offical/xgoduck_walk.onnx --new-cmd-obs --lin-vel-x 0.3

# 4.2 起身（站立）
uv run python scripts/infer_policy.py --standing models/offical/xgoduck_getup.onnx --new-cmd-obs

# 4.3 地面拾取（需要基线策略，按 G 触发）
uv run python scripts/infer_policy.py --standing models/offical/xgoduck_getup.onnx --ground-pick models/offical/xgoduck_pick.onnx --new-cmd-obs

# 4.4 坐下/起立（按 Y 切换）
uv run python scripts/infer_policy.py --sitstand models/offical/xgoduck_sitstand.onnx --new-cmd-obs

# 4.5 前滚翻（按 R 触发）
uv run python scripts/infer_policy.py --walking models/offical/xgoduck_walk.onnx --roulade models/offical/xgoduck_roulade.onnx --new-cmd-obs

# 4.6 右脚踢球（按 L 触发）
uv run python scripts/infer_policy.py --walking models/offical/xgoduck_walk.onnx --kick-right models/offical/xgoduck_ballkick_right.onnx --new-cmd-obs

# 4.7 左脚踢球（按 K 触发）
uv run python scripts/infer_policy.py --walking models/offical/xgoduck_walk.onnx --kick-left models/offical/xgoduck_ballkick_left.onnx --new-cmd-obs

# 4.8 轮滑（上下键加减速，左右键转向）
uv run python scripts/infer_policy.py --roller --walking models/offical/xgoduck_roller.onnx --new-cmd-obs --lin-vel-x 0.3

# 5 键盘手感自查：从 0 起步，连按上键看速度指令是否阶梯上升
uv run python scripts/infer_policy.py --walking models/offical/xgoduck_walk.onnx --new-cmd-obs
