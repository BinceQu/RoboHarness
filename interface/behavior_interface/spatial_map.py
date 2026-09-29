"""测试口 EgoMap：合规观测里程 + 路标 + 车头朝上小地图。

只使用官方 evaluator 允许的信息：
- 底盘工具返回的实际完成量（base_qvel 积分 / 指令完成量）
- 回放点选时：head depth_linear + cam_rel_pose 反解机体系地面点

禁止读取 WorldAPI.robot_pose、scene graph、房间名、分割或接触。
坐标系：本局第一次底盘更新为原点，当时车头为 +X，左为 +Y。
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from behavior_interface.coordinate_contract import relative_to_pixel
from behavior_interface.geometry_localization import (
    DEFAULT_WINDOWS as GEOMETRY_ROLLING_WINDOWS,
    GridSpec as GeometryGridSpec,
    GpuGeometryValidator,
    TemporalPoseHypothesisBank,
    extract_column_scan,
    transform_points as transform_geometry_points,
    unique_cells as unique_geometry_cells,
)
from behavior_interface.pose_graph import (
    PoseEdge,
    compose_pose,
    optimize_graph,
    relative_pose,
    total_error,
)


BACKEND = "behavior_interface.spatial_map.EgoMap"
BUILD = "egomap_gpu_multisensor_frontier_history_v24_candidate_20260901"
ENV_SPATIAL_MAP = "BEHAVIOR_SPATIAL_MAP"
ENV_OFFICIAL = "BEHAVIOR_OFFICIAL"

CHASSIS_TOOLS = frozenset(
    {
        "adjust_chassis",
        "spin_to_facing_point",
        "face_to_point",
        "move_chassis_to_floor_point",
        "move_base_to_point",
        "move_to_reach_point",
    }
)
PASSAGE_TOOLS = frozenset(
    {
        "move_chassis_to_floor_point",
        "move_base_to_point",
    }
)

# 这些工具成功时在地图上留一个事件点，方便认出「在哪儿干了什么」
EVENT_TOOLS = {
    "close_gripper": ("grasp", "抓取"),
    "open_gripper": ("place", "放下"),
}

_DEFAULT_MAP_SIZE = 640
_DEFAULT_RANGE_M = 8.0
_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
# 取景：画布用满多少、内容外包多松。以前 0.94 + 8% + 1m，四周空一大圈。
VIEW_FILL = 0.98
VIEW_PAD_RATIO = 1.03
VIEW_PAD_M = 0.28
VIEW_MIN_HALF_M = 2.0
VIEW_MAX_HALF_M = 26.0

# 占用栅格：只用 head depth + cam_rel_pose，属于合规观测
GRID_RES_M = 0.05
GRID_HALF_SPAN_M = 24.0
# 机体系离地高度分类阈值。地板判定要收紧：墙脚十几厘米的点如果被当成
# 地板，会顺手把墙自己的格子投成可通行，整堵墙就慢慢化掉了。
FLOOR_Z_MAX_M = 0.08
OBSTACLE_Z_MIN_M = 0.15
OBSTACLE_Z_MAX_M = 1.95
# 分层高度：底盘撞得到的算「挡路」，以上只算「头顶结构」。
# 桌面、吊柜、料理台底盘能从旁边过，压进同一层会把半个房间涂黑。
CHASSIS_BLOCK_Z_MAX_M = 0.80
# 近处是自身手臂/夹爪，远处位姿误差会被距离放大成重影
MIN_RANGE_M = 0.45
MAX_RANGE_M = 3.5
# 机器人自身包络（机体系）：双爪/手臂就悬在身前一臂距离，深度打到它们
# 会在地图上钉出两道跟着车走的假墙。这个框里高于地面的点一律不算障碍。
SELF_CLEAR_FORWARD_M = 0.95
SELF_CLEAR_HALF_WIDTH_M = 0.55
DEPTH_STRIDE = 4
# 深度图上物体边缘会插值出悬空飞点，在地图里拖成放射状假墙。
# 相邻像素深度跳变超过这个值就整片丢掉。
DEPTH_EDGE_JUMP_M = 0.12

# 占用用 log-odds：命中加分、穿透扣分，两边都钳住。
# 关键是 miss 能抵掉 hit——看错的格子后来被反复穿过就会自己翻回可通行。
# 旧的「票数比」实现里 obstacle 拿到 4 票后 free 打满上限也翻不了案，
# 噪点一旦落地就是永久黑块，走得越久图越黑。
LOGODDS_HIT = 0.90
LOGODDS_MISS = 0.30
LOGODDS_CLAMP = 6.0
LOGODDS_OCC_ENTER = 1.3
# 判成可通行要比判成墙更谨慎一点：位姿抖一下就会有几条射线穿墙而过，
# 门槛太低的话白色会从门缝里漏到墙外面去。
LOGODDS_FREE_ENTER = -0.9
# 短时层只表达“最近连续观测到什么”，不参与扫描匹配和位姿图。长期结构层
# 为了抗抖使用 6.0 饱和；短时层保留更短记忆，后来的多帧自由射线才能把
# 曾经关闭、现在已经打开的门翻成可通行。miss 比 hit 大，是因为端点命中一次
# 就足以取消“已清空”状态，而清空一堵旧墙仍需多帧一致的穿透证据。
RECENT_LOGODDS_HIT = 1.0
RECENT_LOGODDS_MISS = 0.70
RECENT_LOGODDS_CLAMP = 3.0
# 命中一律一格一票，不按点密度加权。按密度加权时近处墙面一帧就能顶过
# 成墙门槛，位姿抖出来的旁线跟着一起落地；改成固定票之后，一堵墙要连着
# 几帧都落在同一格才立得住，抖动描出的重影攒不够票。
# 去掉孤立噪点：3x3 邻域里至少还有这么多墙格
WALL_MIN_NEIGHBORS = 3
# 渲染时补墙缝：1 格 = 5cm。这个值必须小于重影间距，否则闭运算会把
# 并排的两条重影墙填成一整块实心（实测 2 格时 10~25cm 的重影全被填死）。
WALL_CLOSE_CELLS = 1
# 渲染时把墙细化成中轴线。位姿残差会把一堵墙描成几格厚的带子，
# 闭运算再把带子之间填实，图上就是一片黑块。平面图里墙本来就该是一条线，
# 厚度信息丢掉正合适——这是重建侧的保险，位姿还有残差时照样出细线。
WALL_THIN_MAX_ITER = 16
# 细化只能喂给墙。骨架化是中轴变换：给它一片实心块，吐出来的是一条穿过
# 块心的线；给它一个闭合的框，吐出来的是一个闭合的环，而且它保连通也保洞，
# 环一旦成形再迭代多少次也打不开。桌椅沙发压到同一层之后正好是这两种形状，
# 图上那些三角形和多边形就是这么来的。所以先把家具挑出去，只细化剩下的墙。
#
# 两个判据各管一种形状。厚：沙发座面、扶手、椅面都落在底盘那一层里，绕着走
# 一圈整个外壳都投下来，是接近一米深的一整片；墙哪怕被位姿残差描粗也就三五格。
FURNITURE_HALF_WIDTH_M = 0.30
# 围出小空腔：桌子的四条腿加一圈横撑，在图上连成一个一米见方的闭框。
# 房间也是闭的，但靠墙围出来的尺度比这大得多，不会被误判。
FURNITURE_POCKET_SPAN_M = 1.60
# 判洞之前先把缺口补上。家具的外壳是绕着它走一圈才拼出来的，边上难免缺几格，
# 差一格没接上就整个框都算不上「围死」，判据会白白失效。门洞、走廊比这宽
# 得多，封不住；万一真把房间封出个洞来，它也大到会被上面那条尺度判据排掉。
FURNITURE_SEAL_CELLS = 3
# 空腔紧贴着围它的那圈占据格，往外长几步就能把整圈连厚度一起收进来
FURNITURE_RING_STEPS = 8
# 墙面判据：把挡底盘那 65cm 拆成几个薄高度带，记下每格在哪几带见过障碍。
#
# 图上墙毛毛的、房子看着扭，主因不是位姿，是这一层压得太厚。0.15~0.80m
# 这一整带里有踢脚线、插座、暖气片、桌沿、靠墙的杂物，它们离墙面的距离
# 各不相同，全投到同一张栅格上就叠成一条毛边。标准 2D 激光没这问题——
# 它本身就是几毫米厚的一个平面；RGBD 做 2D SLAM 的常规做法也是取薄片
# 当虚拟激光，或者像 nav2 VoxelLayer 那样分层再逐列投票。
#
# 判据取「贴地起连续命中」：真墙从地面连着长上来，最低那几带必然都命中；
# 杂物只占中间某一两带。实测比「命中够多少带」好得多——在两个真实会话上
# 保留率高一倍、长墙段翻倍，弯曲 RMS 从 7cm 降到 2~3cm。
#
# 注意这只管「画不画成墙线」。挡不挡底盘仍看 occupied_mask，不能因为
# 某个东西只有半人高就当它不存在。
WALL_BAND_COUNT = 6
WALL_BAND_MIN_RUN = 2
# 弱证据定期淡出：位姿抖动会让同一堵墙在旁边描出好几道错线，
# 那些线只被看到一两次就再没回音，衰减掉正好；反复看实的真墙
# 早就顶到高置信区，不受影响，所以走远了老墙也不会凭空消失。
DECAY_EVERY_FRAMES = 20
DECAY_FACTOR = 0.94
DECAY_PROTECT_ABOVE = 2.5
# 射线穿透：机器人到观测点之间必定是空的，按半格步长全程清，
# 不能只挑几个比例点采样——那样一条 6m 射线只清得到 6% 的格子。
RAY_STEP_CELLS = 0.5
# 射线在离端点这么远的地方就收手，别把障碍自己清掉。必须是固定距离：
# 按比例收缩的话，近处射线只退几毫米，一帧就能把自己刚落的墙擦花。
RAY_BACKOFF_M = 0.12
# 射线抽稀：角度上已经足够密，全打太费 CPU
RAY_SOURCE_STRIDE = 2
# 扫描匹配：把本帧对齐到已建地图，抵消指令积分漂移
SCAN_MATCH_MIN_FRAMES = 3
SCAN_MATCH_MIN_POINTS = 120
SCAN_MATCH_MAX_POINTS = 1200
# 匹配只信近处点：远处点的角度误差被距离放大，会把匹配带跑偏
SCAN_MATCH_MAX_RANGE_M = 3.0
# (平移步长 m, 旋转步长 deg, 单边格数)：先粗后细
SCAN_MATCH_STAGES = ((0.10, 2.5, 2), (0.04, 1.0, 2))
MATCH_FREE_PENALTY = 0.6
# 偏离里程推算越远越不可信，按平方扣分
MATCH_REG_LIN = 6.0
MATCH_REG_YAW = 0.06
# 相对「不修正」必须有实质提升才接受
MATCH_MIN_GAIN = 0.02
# 视场里只剩一面墙时，沿墙滑动不改变任何观测，得分在这一维是平的，
# 匹配会一路滑到搜索窗边界（实测单步注入 0.26m，正好等于窗口上限）。
# 这种错误连回环都摊不掉：理想回环后残差仍有 0.97m，而锁住退化方向后
# 是 0.15m。信息矩阵 H = Σ 长度·法线·法线ᵀ 的小特征值方向就是没约束的
# 方向，把修正量投影掉再用。和 R5 拿线段方向定 yaw 是同一件事的两面。
MATCH_DEGENERATE_RATIO = 0.15
# 单次匹配最多把位姿挪这么远。搜索窗有 ±0.28m，退化时匹配会一路滑到窗口
# 边界——错一次注入的误差，远多于它平时一次能修回来的量。限幅之后对的修正
# 靠多次累积照样到位，错的则被摁住：实测绝对误差 0.80m → 0.20m，比干脆
# 不做平移匹配（0.35m）还好。
MATCH_MAX_STEP_M = 0.05
# 扫描匹配候选评分的设备。`auto` 在当前 interface 可见 CUDA 时使用批量
# 张量评分；不可用或发生运行时错误时只回退评分，不改变位姿门槛和地图逻辑。
SCAN_MATCH_BACKEND_ENV = "BEHAVIOR_SLAM_SCAN_MATCH_BACKEND"
SCAN_MATCH_BACKEND_AUTO = "auto"
SCAN_MATCH_BACKEND_CPU = "cpu"
SCAN_MATCH_BACKEND_CUDA = "cuda"
SCAN_MATCH_GPU_BATCH = 4096
# CPU / CUDA 的三角函数在最后一位可能不同。候选点数学上恰落在格线时，
# floor 会因此选到相邻格。评分统一加 1e-9 个栅格的偏置；它约等于
# 5e-11m，只消除浮点边界歧义，不改变地图积分的格子归属。
SCAN_MATCH_GRID_EPS = 1e-9
# 线段定向：把一帧点云里的直墙抽成线段，用线段方向定 yaw。
# 栅格匹配是逐点查表，head 视场只有 99°，常常整帧只看得见一面墙；
# 这时绕自己转一点等于让点云沿着那面墙滑，得分几乎不变，yaw 就修不动
# （离线实测注入 3° 误差：39% 的位姿完全没修，29% 修过了头）。
# 沿墙滑动确实不可观测的是「平移」，墙的方向角一直是能直接量的，
# 拿它定 yaw 在 4cm 深度噪声下仍有 86% 的样本残差 <0.5°。
SEG_ANGLE_BIN_DEG = 0.5
# 相邻方位上距离跳这么多就断开：两段墙之间、墙和家具之间都靠它分家
SEG_BREAK_JUMP_M = 0.25
# IEPF 分裂阈值：点离首尾连线超过这个距离，说明这段不是一条直线
SEG_FIT_TOL_M = 0.04
SEG_MIN_POINTS = 12
# 太短的段方向噪声大，量出来的角度不可信，不参与定向
SEG_MIN_LENGTH_M = 0.60
# 主方向从数据里学，不预设墙一定轴对齐：场景里柜子、沙发的边朝向本来
# 就任意，硬吸附到 0/90 会把斜家具当墙去纠 yaw，越纠越歪。
DIR_HIST_BINS = 180
DIR_HIST_DECAY = 0.98
DIR_HIST_BLUR_DEG = 2.0
# 主方向累计长度不够就还不算数，冷启动阶段只记方向不纠正
DIR_MIN_WEIGHT_M = 3.0
# 线段方向离所有主方向都超过这么多，当作家具而不是墙，不用它定向
DIR_MATCH_MAX_DEG = 6.0
# 圆弧和杂物会在直方图上堆出一排高度差不多的「峰」，每条线段都能就近找到
# 一个，拿它们纠 yaw 等于用噪声纠自己。要求峰显著高于平均、且数量不多，
# 方向确实成簇时才认：实测圆形房间 peak/mean 只有 1 点几并冒出十几个峰，
# 矩形和整间斜 23° 的房间都在十几倍、只有两个峰。
DIR_PEAK_PROMINENCE = 4.0
DIR_MAX_PEAKS = 6
# 单帧最多纠这么多，避免一次匹配错误就把整张图甩飞
YAW_FIX_MAX_DEG = 2.5
# 参与定向的线段总长下限：证据太少宁可不纠
YAW_FIX_MIN_SUPPORT_M = 1.2
# 轨迹抽稀：大概行踪即可
TRAIL_MIN_STEP_M = 0.6
# 图上只画最近这么多次底盘移动，更早的切掉
TRAIL_RECENT_MOVES = 10
# 轨迹审计采样间隔。采样只用于显示和评测，不能参与栅格更新。
TRAJECTORY_SAMPLE_STEP_M = 0.05
TRAJECTORY_SAMPLE_YAW_DEG = 2.0
# 建图只在证据充分且地图增长已经趋稳后单向冻结。阈值是通用的观测量，
# 不依赖 benchmark 帧号、房间名或标注点。
MAPPING_MIN_OBSERVATIONS = 180
MAPPING_MIN_TRAVEL_M = 5.0
MAPPING_MIN_VISUAL_KEYFRAMES = 12
MAPPING_MIN_LOOP_GAP = 80
# 一次通过几何门槛的长期重访已经足以启动“候选冻结”；真正切换还要经过
# 稳定增长、开放边界、轨迹覆盖和静止保持等独立门槛。把这里设成 1，避免
# 要求同一地点出现两次才冻结（那会让没有第二次重访的正常 episode 永不
# 进入定位态）。
MAPPING_MIN_REVISIT_EVIDENCE = 1
MAPPING_EVIDENCE_FRAME_GAP = 40
MAPPING_GROWTH_WINDOW = 30
MAPPING_MAX_GROWTH_RATIO = 0.05
MAPPING_MAX_GROWTH_CELLS = 400
# 长期重访已经说明机器人回到了旧结构。继续把同一区域写几十帧只会把
# 闭环残差描成双墙；是否成熟由增长、frontier 和轨迹覆盖独立判断。
MAPPING_CANDIDATE_HOLD_FRAMES = 0
# 冻结判断看整段已访问轨迹触达的自由连通域，而不是只看当前脚下那一格。
# 当前位姿在膨胀障碍边缘抖一格时，不能把之前走过的开放通道瞬间忘掉。
MAPPING_FRONTIER_CLEARANCE_M = 0.30
MAPPING_FRONTIER_SEED_MAX_DISTANCE_M = 0.45
MAPPING_MIN_FRONTIER_LENGTH_M = 0.50
# “没有可达前沿”也必须连续成立；单帧连通域抖动没有停止建图的权限。
MAPPING_FRONTIER_COMPLETE_HOLD_FRAMES = 24
# 稳定前沿只供探索调度判断“机器人是否卡住”，不能证明未知区域不存在，
# 因此绝不授权冻结。这个时序预算不绑定任务、场景或固定帧号。
MAPPING_FRONTIER_QUIESCENCE_FRAMES = 24
MAPPING_TRAJECTORY_COVERAGE_RADIUS_M = 0.15
MAPPING_MIN_TRAJECTORY_COVERAGE = 0.98
# 冻结不要求机器人停车。标准定位切换发生在传感器流中，要求静止会让
# 已经闭环的地图继续接收整段返程观测。
MAPPING_IDLE_HOLD_FRAMES = 0
MAPPING_IDLE_TRANSLATION_M = 0.025
MAPPING_IDLE_YAW_DEG = 1.5
# 冻结后只按这个频率探测长期视觉定位；每一帧仍可保留完整轨迹和图像。
LOCALIZATION_PROBE_INTERVAL = 8
# 候选绝对位姿相对当前里程的极端跳变保护，只负责在特征前端尽早丢掉
# 明显错配。所有实际校正还必须通过独立 GPU 深度几何门。
LOCALIZATION_MAX_POSE_DELTA_M = 1.5
LOCALIZATION_MAX_YAW_DELTA_DEG = 30.0
# 绝对定位状态不能由“走了多少帧”决定。这里按路程和转角累计一个保守的
# 运动模型方差；阈值只表达 5cm 栅格上还能否可靠区分通道与墙，不含任务、
# 房型或录制长度。相邻 RGB-D 只改善相对传播，不能冒充全局重定位来清零。
LOCALIZATION_POSITION_VARIANCE_PER_M = 0.0004
LOCALIZATION_POSITION_VARIANCE_PER_RAD = 0.0001
LOCALIZATION_YAW_VARIANCE_PER_M_DEG2 = 0.25
LOCALIZATION_YAW_VARIANCE_PER_RAD_DEG2 = 0.09
LOCALIZATION_LOST_POSITION_STD_M = 0.30
LOCALIZATION_LOST_YAW_STD_DEG = 8.0
# 冻结后关键帧仍按较密的时间间隔检查；这里不是建图门槛，不能拿它
# 代替长期证据，只负责让定位在刚停下或原地转身时及时接管。
LOCALIZATION_KEYFRAME_GAP = 8
# 几何搜索只保存短滚动窗口。视觉候选可跨过一个完整最大窗口继续提供
# 历史子图锚点，超过后必须重新由外观前端确认，不能无限期套用旧候选。
GEOMETRY_VISUAL_LEASE_MAX_FRAMES = max(GEOMETRY_ROLLING_WINDOWS) + 16
GEOMETRY_DEPTH_STRIDE = 8
GEOMETRY_COLUMN_M = 0.08
GEOMETRY_TARGET_NEIGHBOR_HOPS = 1
GEOMETRY_REPORT_KEEP = 64
# 保留多少张图的拍摄位姿，够 agent 回头点选前几步拍的图
IMAGE_POSE_KEEP = 64
# 换个名字标在附近时算改名而不是新增：房间名按房间尺度，点选的物体按物体尺度
PLACE_MERGE_M = 2.0
PICK_MERGE_M = 0.4
# submap：只覆盖机器人附近的一小张局部图。全局图攒着一路的漂移，拿它当
# 匹配基准等于让当前帧去对齐一张已经歪掉的图；submap 内部只有几十帧的
# 误差，基准干净得多。位姿图优化时也是以 submap 为单位整体挪，历史观测
# 才有可能跟着一起被纠正——直接写进全局栅格的话，落下去就再也改不动了。
# 半径要同时装下 submap 内的行走距离和一个量程，否则边上会被截掉。
SUBMAP_HALF_SPAN_M = 8.0
SUBMAP_MAX_FRAMES = 80
# 老的那张走到一半就开下一张，保证接班时新的已经攒了半张观测
SUBMAP_HANDOFF_FRAMES = SUBMAP_MAX_FRAMES // 2
SUBMAP_MAX_TRAVEL_M = 3.0
SUBMAP_MAX_TURN_DEG = 100.0
# 保留足够长的结构历史供全局回环和重建使用。淘汰 submap 会同时失去视觉
# 锚点和可重建观测，因此容量必须按模块内存预算而不是某条短路线设置；256
# 张覆盖数百米行程，仍是有界的数百 MiB。无限时长需要图边缘化，不能靠
# 悄悄丢掉老房间实现。
SUBMAP_MAX_COUNT = 256
# 位姿图。里程边是一路推算出来的，回环边是真匹配上了才连，后者更可信。
# 权重只有相对大小有意义（信息矩阵的对角）。
ODOM_EDGE_WEIGHT_XY = 1.0
ODOM_EDGE_WEIGHT_YAW = 1.0
LOOP_EDGE_WEIGHT_XY = 4.0
LOOP_EDGE_WEIGHT_YAW = 4.0
# 单帧长期视觉特征在重复门板和走廊里会产生高分别名，因此只能提出冻结候选。
# 只有连续候选给出一致的地图校正，并且当前视角有足够旋转/平移基线时，才把
# RGB-D 相对位姿变成鲁棒 submap 边。随后仍由统一位姿图摊回误差，禁止硬跳
# 当前位姿或按轨迹修改占据格。
MAPPING_REVISIT_MIN_INLIERS = 18
MAPPING_REVISIT_MIN_INLIER_RATIO = 0.60
MAPPING_REVISIT_MAX_RMSE_M = 0.06
MAPPING_REVISIT_MIN_ABOVE_FLOOR_INLIERS = 8
MAPPING_REVISIT_MIN_INDEPENDENT = 2
MAPPING_REVISIT_MIN_TRANSLATION_SPAN_M = 0.35
# 只有一条历史视点时，必须达到远高于普通定位的强重访门。它只能触发
# “停止写结构”，不能生成位姿图边或直接校正当前位姿。
MAPPING_REVISIT_SINGLE_MIN_INLIERS = 48
MAPPING_REVISIT_SINGLE_MIN_INLIER_RATIO = 0.85
MAPPING_REVISIT_SINGLE_MAX_RMSE_M = 0.035
MAPPING_REVISIT_SINGLE_MIN_ABOVE_FLOOR_INLIERS = 24
# 回环检测：只在原点够近的 submap 之间找，且要隔开足够多张——挨着的那
# 几张本来就由里程边连着，再连一条只是同义反复。
# 候选半径要盖得住「漂移 + submap 自身跨度」，否则真回环连候选都进不了
LOOP_SEARCH_RADIUS_M = 4.0
LOOP_MIN_SERIAL_GAP = 4
# 搜索窗要盖得住漂移，否则真回环落在窗外等于没找。而漂移正是越大越需要
# 回环，所以这个窗不能按「误差应该很小」来定——实测走两圈能漂 3m。
LOOP_SEARCH_SPAN_M = 3.0
LOOP_SEARCH_YAW_DEG = 24.0
# 参与回环匹配的点抽到这个数量级，全用太慢
LOOP_MATCH_POINTS = 220
# 粗图把 4x4 格并成一格（0.05m → 0.2m），正好配 0.25m 的粗步长
LOOP_COARSE_FACTOR = 4
# 三级搜索：粗级在降分辨率的膨胀图上扫全窗，后两级只在上一级的解附近细化。
# 每级的搜索半径都得盖住上一级的步长，否则细化会卡在粗解的格点上。
LOOP_STAGES = (
    # (半径 m, 步长 m, 半径 deg, 步长 deg, 用粗图, 点数)
    (LOOP_SEARCH_SPAN_M, 0.25, LOOP_SEARCH_YAW_DEG, 8.0, True, 90),
    (0.25, 0.08, 8.0, 2.5, False, LOOP_MATCH_POINTS),
    (0.08, 0.03, 2.5, 1.0, False, LOOP_MATCH_POINTS),
)
# 归一化命中率门槛：误回环比不回环更致命，宁可漏也不要错
LOOP_MIN_HIT_RATIO = 0.55
# 最优解得比次优的另一处高出这么多，否则说明是走廊那种到处都对得上的地方
LOOP_MIN_PEAK_MARGIN = 1.25

_MAPS: Dict[str, "EgoMap"] = {}
_MAPS_LOCK = threading.Lock()
# 最近一次真正产生内容的 session；server 侧 memory 没有 session 参数，靠它定位
_ACTIVE_SESSION = ""


def spatial_map_enabled() -> bool:
    """测试口开关。官方评测默认关，避免改 capture 合同。"""
    raw = str(os.environ.get(ENV_SPATIAL_MAP, "") or "").strip().lower()
    if raw in {"0", "false", "off", "no"}:
        return False
    if raw in {"1", "true", "on", "yes"}:
        return True
    official = str(os.environ.get(ENV_OFFICIAL, "") or "").strip().lower()
    if official in {"1", "true", "on", "yes"}:
        return False
    return False


def is_chassis_tool(tool: str) -> bool:
    return str(tool or "").strip() in CHASSIS_TOOLS


def wrap_deg(deg: float) -> float:
    """把角度包到 (-180, 180]。"""
    value = (float(deg) + 180.0) % 360.0 - 180.0
    if value <= -180.0:
        return 180.0
    return value


def body_twist_delta(
    forward_m: float,
    left_m: float,
    yaw_deg: float,
) -> Tuple[float, float, float]:
    """把一个 tick 的机体系速度积分成精确的 SE(2) 相对位姿。

    ``base_qvel`` 的平移和角速度在同一个 tick 内同时发生。把平移直接
    乘上 tick 开始时的朝向等价于“先走再转”，在连续转弯或长时间自转时
    会产生系统性的位置偏差。这里用常速度刚体运动的指数映射；返回的
    ``(forward, left, yaw)`` 仍是起始机体系下的有限相对位姿，方便和已有
    SE(2) 组合代码对接。
    """
    forward = float(forward_m)
    left = float(left_m)
    theta = math.radians(float(yaw_deg))
    if not all(math.isfinite(value) for value in (forward, left, theta)):
        raise ValueError("机体系运动必须是有限数")
    if abs(theta) < 1e-8:
        return forward, left, float(yaw_deg)
    # sin(theta)/theta 和 (1-cos(theta))/theta 在这里已经无量纲；输入的
    # 平移是一个 tick 内的积分量，所以不需要再除以角速度或 dt。
    sinc = math.sin(theta) / theta
    cosc = (1.0 - math.cos(theta)) / theta
    return (
        sinc * forward - cosc * left,
        cosc * forward + sinc * left,
        float(yaw_deg),
    )


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _nested_get(payload: Dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


@dataclass
class Landmark:
    name: str
    x: float
    y: float
    yaw_deg: float = 0.0
    image_id: str = ""
    tool: str = ""
    note: str = ""
    submap_serial: Optional[int] = None


@dataclass
class TrailPoint:
    x: float
    y: float
    yaw_deg: float
    tool: str = ""
    image_id: str = ""
    forward_m: float = 0.0
    translation_m: float = 0.0
    spin_deg: float = 0.0
    source: str = ""
    submap_serial: Optional[int] = None


@dataclass
class MapEvent:
    """地图上的事件点：在哪儿抓到 / 放下 / 卡住。"""

    x: float
    y: float
    kind: str
    label: str
    count: int = 1
    submap_serial: Optional[int] = None


@dataclass
class MotionDelta:
    forward_m: float = 0.0
    translation_m: float = 0.0
    spin_deg: float = 0.0
    source: str = "none"
    mark_passage: bool = False
    image_id: str = ""
    tool: str = ""


def _neighbor_count(mask: np.ndarray) -> np.ndarray:
    """3x3 邻域内 True 的个数（含自身）。"""
    padded = np.pad(mask.astype(np.int16), 1)
    total = np.zeros_like(mask, dtype=np.int16)
    h, w = mask.shape
    for dy in range(3):
        for dx in range(3):
            total += padded[dy : dy + h, dx : dx + w]
    return total


def _iepf_splits(xs: np.ndarray, ys: np.ndarray) -> List[Tuple[int, int]]:
    """Iterative End Point Fit：把一条折线拆成若干条够直的子段。

    反复找离首尾连线最远的点，超过容差就从那里断开。拐角会被切在角上，
    正好是我们想要的：每个子段对应一个平面。
    """
    n = xs.shape[0]
    out: List[Tuple[int, int]] = []
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j - i + 1 < SEG_MIN_POINTS:
            continue
        dx = float(xs[j] - xs[i])
        dy = float(ys[j] - ys[i])
        norm = math.hypot(dx, dy)
        if norm < 1e-9:
            continue
        dist = np.abs(
            (xs[i : j + 1] - xs[i]) * dy - (ys[i : j + 1] - ys[i]) * dx
        ) / norm
        k = int(np.argmax(dist))
        if float(dist[k]) > SEG_FIT_TOL_M and 0 < k < (j - i):
            stack.append((i, i + k))
            stack.append((i + k, j))
        else:
            out.append((i, j))
    return out


def _scan_segments(points: np.ndarray) -> List[Tuple[float, float]]:
    """一帧机体系点云 → 直线段列表 [(机体系方向角 deg, 长度 m)]。

    depth 投影出来的点是散的，先按方位角收成有序扫描（每个方位只留最近的
    那个点，远处被挡住的不算），再按距离跳变断开，最后 IEPF 分裂出直段。
    方向用 PCA 主轴而不是首尾连线，端点噪声不会把角度带偏。
    """
    if points is None or points.ndim != 2 or points.shape[0] < SEG_MIN_POINTS:
        return []
    x = np.asarray(points[:, 0], dtype=np.float64)
    y = np.asarray(points[:, 1], dtype=np.float64)
    radial = np.hypot(x, y)
    bins = np.floor(
        np.degrees(np.arctan2(y, x)) / SEG_ANGLE_BIN_DEG
    ).astype(np.int64)
    # 同一方位里按距离升序，取第一个就是最近的那个点
    order = np.lexsort((radial, bins))
    sorted_bins = bins[order]
    first = np.ones(sorted_bins.shape[0], dtype=bool)
    first[1:] = sorted_bins[1:] != sorted_bins[:-1]
    idx = order[first]
    xs, ys = x[idx], y[idx]
    bs, rs = bins[idx], radial[idx]
    if xs.shape[0] < SEG_MIN_POINTS:
        return []
    # 距离突变或方位上有空洞，都说明不是同一个连续表面
    cut = np.ones(xs.shape[0], dtype=bool)
    cut[1:] = (np.abs(np.diff(rs)) > SEG_BREAK_JUMP_M) | (np.diff(bs) > 2)
    edges = list(np.flatnonzero(cut)) + [xs.shape[0]]
    out: List[Tuple[float, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi - lo < SEG_MIN_POINTS:
            continue
        px, py = xs[lo:hi], ys[lo:hi]
        for i, j in _iepf_splits(px, py):
            sx, sy = px[i : j + 1], py[i : j + 1]
            length = math.hypot(
                float(sx[-1] - sx[0]), float(sy[-1] - sy[0])
            )
            if length < SEG_MIN_LENGTH_M:
                continue
            cx, cy = float(sx.mean()), float(sy.mean())
            cov = np.cov(np.stack([sx - cx, sy - cy]))
            if not np.all(np.isfinite(cov)):
                continue
            evals, evecs = np.linalg.eigh(cov)
            axis = evecs[:, 1]
            angle = math.degrees(math.atan2(float(axis[1]), float(axis[0])))
            out.append((angle % 180.0, length))
    return out



class OccupancyGrid:
    """地图系 2D 占用栅格，log-odds 累积，按高度分两层。

    ``low`` 是底盘撞得到的高度（约 0.15–0.8m），决定能不能走；
    ``high`` 是头顶结构（0.8–1.95m），只用来认「这是不是一堵真墙」和着色。
    真墙上下都有回波，桌面只在 ``high``，箱子只在 ``low``。
    """

    def __init__(
        self,
        resolution_m: float = GRID_RES_M,
        half_span_m: float = GRID_HALF_SPAN_M,
    ) -> None:
        self.resolution_m = float(resolution_m)
        self.half_span_m = float(half_span_m)
        self.n = int(round(2.0 * self.half_span_m / self.resolution_m))
        self.low = np.zeros((self.n, self.n), dtype=np.float32)
        self.high = np.zeros((self.n, self.n), dtype=np.float32)
        # ``recent_*`` 是同一批深度按时间顺序累积的短时观测层。它可以在结构
        # 冻结后继续更新，但永远不参与 scan matching / pose graph。
        self.recent_low = np.zeros((self.n, self.n), dtype=np.float32)
        self.recent_high = np.zeros((self.n, self.n), dtype=np.float32)
        # 每格在哪几个高度带见过障碍，一带一位。用来认「这是不是一个竖直墙面」
        self.bands = np.zeros((self.n, self.n), dtype=np.uint16)
        self.frames = 0
        self.recent_frames = 0
        self._rev = 0
        self._mask_cache: Optional[Tuple[int, np.ndarray]] = None
        self._wall_cache: Optional[
            Tuple[int, np.ndarray, np.ndarray, np.ndarray]
        ] = None
        self._ray_t: Optional[np.ndarray] = None
        # 扫描匹配只读这张低层 log-odds 图。按 revision 缓存 GPU 副本，栅格
        # 每次写入后自然失效；不把可变地图长期托管到 GPU，避免和渲染/导出
        # 的 NumPy 接口产生两份不一致的真相。
        self._cuda_low_cache: Any = None
        self._cuda_low_cache_revision: int = -1
        self._cuda_low_cache_device: str = ""
        self._scan_match_backend = "uninitialized"
        self._scan_match_device = "uninitialized"
        self._scan_match_fallback_reason = ""

    def _indices(self, xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ix = np.floor((xs + self.half_span_m) / self.resolution_m).astype(np.int64)
        iy = np.floor((ys + self.half_span_m) / self.resolution_m).astype(np.int64)
        return ix, iy

    def _unique_cells(self, xs: np.ndarray, ys: np.ndarray):
        """落进格子并去重。

        一帧里几千个点常常砸在同一个格子上，逐点累加等于让近处物体
        一帧就投几十票，瞬间冲到饱和后再也压不动。一帧一格只算一票。
        """
        if xs.size == 0:
            return None
        ix, iy = self._indices(xs, ys)
        ok = (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)
        if not np.any(ok):
            return None
        flat = (iy[ok] * self.n + ix[ok]).astype(np.int32, copy=False)
        uniq = np.unique(flat)
        return uniq // self.n, uniq % self.n

    def _bump(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        layer: np.ndarray,
        delta: float,
        *,
        recent_layer: Optional[np.ndarray] = None,
        recent_delta: Optional[float] = None,
        structural: bool = True,
    ) -> None:
        cells = self._unique_cells(xs, ys)
        if cells is None:
            return
        rows, cols = cells
        if structural:
            layer[rows, cols] = np.clip(
                layer[rows, cols] + np.float32(delta),
                -LOGODDS_CLAMP,
                LOGODDS_CLAMP,
            )
        if recent_layer is not None:
            current_delta = delta if recent_delta is None else recent_delta
            recent_layer[rows, cols] = np.clip(
                recent_layer[rows, cols] + np.float32(current_delta),
                -RECENT_LOGODDS_CLAMP,
                RECENT_LOGODDS_CLAMP,
            )
        self._rev += 1

    def add_floor(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        *,
        structural: bool = True,
    ) -> None:
        """看见地板：这一格在底盘高度上是空的。"""
        self._bump(
            xs,
            ys,
            self.low,
            -LOGODDS_MISS,
            recent_layer=self.recent_low,
            recent_delta=-RECENT_LOGODDS_MISS,
            structural=structural,
        )

    def add_obstacle(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        *,
        layer: str = "low",
        zs: Optional[np.ndarray] = None,
        structural: bool = True,
    ) -> None:
        """障碍命中：一帧一格一票，不看这一格里落了多少个点。

        按点密度加权时，近处墙面一帧就能把相邻好几格一起顶过成墙门槛，
        位姿抖出来的重影也跟着一次落地。固定票之后，墙要连着几帧都落在
        同一格才立得住，抖动描出的旁线攒不够票就会被 decay 拉掉。

        ``zs`` 是每个点的高度，用来记高度带、认竖直墙面。不给就当整根
        柱子都实——「墙悄悄画不出来」比「多画一条墙」难查得多。
        """
        target = self.high if layer == "high" else self.low
        recent = self.recent_high if layer == "high" else self.recent_low
        self._bump(
            xs,
            ys,
            target,
            LOGODDS_HIT,
            recent_layer=recent,
            recent_delta=RECENT_LOGODDS_HIT,
            structural=structural,
        )
        if structural and layer != "high":
            self.note_bands(xs, ys, zs)

    def note_bands(
        self, xs: np.ndarray, ys: np.ndarray, zs: Optional[np.ndarray] = None
    ) -> None:
        """记下每格在哪几个高度带见过障碍，一带一位。

        位是只加不减的。噪声顶多让某格多亮一位、判据松一点，不会凭空造出
        墙来——最终还要和 occupied_mask 求交，那一层有 log-odds 和衰减兜底。
        """
        if xs.size == 0:
            return
        if zs is None:
            # 高度未知：按「整根柱子都实」记，宁可多画墙也不少画
            cells = self._unique_cells(xs, ys)
            if cells is not None:
                rows, cols = cells
                self.bands[rows, cols] |= np.uint16((1 << WALL_BAND_COUNT) - 1)
                self._rev += 1
            return
        span = (CHASSIS_BLOCK_Z_MAX_M - OBSTACLE_Z_MIN_M) / WALL_BAND_COUNT
        idx = np.clip(
            ((np.asarray(zs) - OBSTACLE_Z_MIN_M) / span).astype(np.int64),
            0,
            WALL_BAND_COUNT - 1,
        )
        for k in range(WALL_BAND_COUNT):
            sel = idx == k
            if not np.any(sel):
                continue
            cells = self._unique_cells(xs[sel], ys[sel])
            if cells is None:
                continue
            rows, cols = cells
            self.bands[rows, cols] |= np.uint16(1 << k)
        self._rev += 1

    def add_pass_through(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        *,
        layer: str = "low",
        structural: bool = True,
    ) -> None:
        """射线穿过这一格，说明该高度上没东西。"""
        target = self.high if layer == "high" else self.low
        recent = self.recent_high if layer == "high" else self.recent_low
        self._bump(
            xs,
            ys,
            target,
            -LOGODDS_MISS,
            recent_layer=recent,
            recent_delta=-RECENT_LOGODDS_MISS,
            structural=structural,
        )

    def carve_rays(
        self,
        x0: float,
        y0: float,
        xs: np.ndarray,
        ys: np.ndarray,
        *,
        layer: str = "low",
        backoff_m: float = RAY_BACKOFF_M,
        structural: bool = True,
    ) -> None:
        """把机器人到每个观测点之间的整条视线清成空。

        以前只在射线上取 7 个固定比例点，6m 射线只清得到 6% 的格子，
        误判的黑块根本等不到被擦掉的机会。这里按半格步长走完全程。
        """
        if xs.size == 0:
            return
        # 成千上万个点常常砸在同一批格子上，先把端点收成唯一格心再发射线，
        # 覆盖范围一点不少，射线条数能少好几倍。
        cells = self._unique_cells(xs, ys)
        if cells is None:
            return
        rows, cols = cells
        xs = (cols.astype(np.float32) + 0.5) * self.resolution_m - self.half_span_m
        ys = (rows.astype(np.float32) + 0.5) * self.resolution_m - self.half_span_m
        if RAY_SOURCE_STRIDE > 1 and xs.size > 2 * RAY_SOURCE_STRIDE:
            xs = xs[::RAY_SOURCE_STRIDE]
            ys = ys[::RAY_SOURCE_STRIDE]
        dx = np.asarray(xs, dtype=np.float32) - np.float32(x0)
        dy = np.asarray(ys, dtype=np.float32) - np.float32(y0)
        dist = np.hypot(dx, dy)
        # 每条射线各自按固定米数收尾，端点附近的格子留给障碍自己
        keep = dist > (backoff_m + self.resolution_m)
        if not np.any(keep):
            return
        dx, dy, dist = dx[keep], dy[keep], dist[keep]
        stop = ((dist - np.float32(backoff_m)) / dist).astype(np.float32)
        longest = float(dist.max())
        steps = int(longest / (self.resolution_m * RAY_STEP_CELLS)) + 2
        if self._ray_t is None or self._ray_t.size != steps:
            self._ray_t = np.linspace(0.0, 1.0, steps, dtype=np.float32)
        t = self._ray_t[None, :] * stop[:, None]
        px = np.float32(x0) + dx[:, None] * t
        py = np.float32(y0) + dy[:, None] * t
        self.add_pass_through(
            px.ravel(),
            py.ravel(),
            layer=layer,
            structural=structural,
        )

    def decay_uncertain(self) -> None:
        """把还没站住脚的证据往零拉一点，站稳的留着。"""
        for layer in (self.low, self.high):
            weak = np.abs(layer) < DECAY_PROTECT_ABOVE
            if np.any(weak):
                layer[weak] *= np.float32(DECAY_FACTOR)
        self._rev += 1

    def occupied_mask(self, *, denoise: bool = True) -> np.ndarray:
        """挡底盘的格子。头顶结构不算——桌子底下照样能过。"""
        if denoise:
            cached = self._mask_cache
            if cached is not None and cached[0] == self._rev:
                return cached[1]
        mask = self.low >= LOGODDS_OCC_ENTER
        if denoise:
            mask = mask & (_neighbor_count(mask) >= WALL_MIN_NEIGHBORS)
            self._mask_cache = (self._rev, mask)
        return mask

    def observed_occupied_mask(self) -> np.ndarray:
        """按时间顺序融合后的原始占据证据，不做形态学或轨迹覆盖。

        长期结构仍是默认答案；只有短时层明确看到自由空间时才压掉旧结构，
        或明确看到新障碍时才补上占据。短时未知不会让旧墙凭空消失。
        """
        structural = self.low >= LOGODDS_OCC_ENTER
        recent_occupied = self.recent_low >= LOGODDS_OCC_ENTER
        recent_free = self.recent_low <= LOGODDS_FREE_ENTER
        return recent_occupied | (structural & ~recent_free)

    def observed_free_mask(self) -> np.ndarray:
        """按时间顺序融合后的原始自由空间证据。"""
        occupied = self.observed_occupied_mask()
        structural = self.low <= LOGODDS_FREE_ENTER
        recent = self.recent_low <= LOGODDS_FREE_ENTER
        return (structural | recent) & ~occupied

    def observed_wall_mask(self) -> np.ndarray:
        """由原始占据证据和高度带判定的墙面，不补洞、不细化。"""
        return self.observed_occupied_mask() & self.wall_face_mask()

    def observed_overhead_mask(self) -> np.ndarray:
        """头顶层的因果融合占据证据，不扩张。"""
        structural = self.high >= LOGODDS_OCC_ENTER
        recent_occupied = self.recent_high >= LOGODDS_OCC_ENTER
        recent_free = self.recent_high <= LOGODDS_FREE_ENTER
        return (
            recent_occupied | (structural & ~recent_free)
        ) & ~self.observed_occupied_mask()

    def wall_face_mask(self) -> np.ndarray:
        """从地面连着长上来的那些格子——竖直墙面长这样。

        位模式必须是低位一串连续的 1、高位全 0（0b0011、0b0111、0b1111…），
        也就是「最低带起，中间不断」。踢脚线、插座、桌沿只亮中间某一两位，
        位模式不连续或不贴地，就落选。
        """
        bits = self.bands
        # x & (x+1) == 0 只对 0b0…011…1 成立，一步同时管住「贴地」和「不断」
        contiguous = (bits & (bits + np.uint16(1))) == 0
        return contiguous & (bits >= np.uint16((1 << WALL_BAND_MIN_RUN) - 1))

    def _wall_shapes(self) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """墙体范围、墙的中轴、家具团块，一起算一次缓存到下次栅格变动。"""
        cached = self._wall_cache
        if cached is not None and cached[0] == self._rev:
            return cached
        occupied = self.observed_occupied_mask()
        occupied = occupied & (_neighbor_count(occupied) >= WALL_MIN_NEIGHBORS)
        body = _close(occupied, WALL_CLOSE_CELLS)
        # 只有竖直墙面配画成中轴线。剩下的照样挡底盘，但按团块画，
        # 免得骨架化把它们抽成假墙——那正是图上墙毛毛的来源。
        face = _close(occupied & self.wall_face_mask(), WALL_CLOSE_CELLS)
        wall, furniture = _split_furniture(
            body & face,
            fat_cells=int(round(FURNITURE_HALF_WIDTH_M / self.resolution_m)),
            pocket_cells=int(
                round(0.5 * FURNITURE_POCKET_SPAN_M / self.resolution_m)
            ),
        )
        entry = (self._rev, body, _thin(wall), furniture | (body & ~face))
        self._wall_cache = entry
        return entry

    def wall_body_mask(self) -> np.ndarray:
        """挡底盘的全部范围，墙和家具都算。用它挡住可通行判定。"""
        return self._wall_shapes()[1]

    def thinned_mask(self) -> np.ndarray:
        """墙体中轴：位姿残差描出的厚带子收成一条线，渲染按它画墙。"""
        return self._wall_shapes()[2]

    def furniture_mask(self) -> np.ndarray:
        """压扁之后成片的东西：沙发、椅子、桌子围出来的框。

        照样挡底盘，但它不是一条线，所以不细化、按团块原样画。
        """
        return self._wall_shapes()[3]

    def overhead_mask(self) -> np.ndarray:
        """只在头顶高度有东西：桌面、料理台、吊柜，底盘能从下面或旁边过。"""
        return self.observed_overhead_mask()

    def solid_wall_mask(self) -> np.ndarray:
        """上下两层都有回波，才当成真正的墙体。平面图的骨架就是它。"""
        high_structural = self.high >= LOGODDS_OCC_ENTER
        high_recent = self.recent_high >= LOGODDS_OCC_ENTER
        high_cleared = self.recent_high <= LOGODDS_FREE_ENTER
        high = high_recent | (high_structural & ~high_cleared)
        return self.observed_occupied_mask() & high

    def walkable_mask(self) -> np.ndarray:
        """可通行：只有深度射线明确观测为空的格子才算自由。"""
        return self.observed_free_mask()

    def free_mask(self) -> np.ndarray:
        return self.walkable_mask()

    def bounds_m(self) -> Optional[Tuple[float, float, float, float]]:
        xs, ys = self.known_xy()
        if xs is None:
            return None
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

    def known_xy(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """有内容的格子中心（地图系 XY）。去掉孤点，避免取景被噪点撑开。"""
        known = self.free_mask() | self.observed_occupied_mask()
        if not np.any(known):
            return None, None
        clustered = known & (_neighbor_count(known) >= 1)
        if np.any(clustered):
            known = clustered
        rows, cols = np.nonzero(known)
        xs = (cols.astype(np.float64) + 0.5) * self.resolution_m - self.half_span_m
        ys = (rows.astype(np.float64) + 0.5) * self.resolution_m - self.half_span_m
        return xs, ys

    def match_score(self, xs: np.ndarray, ys: np.ndarray) -> float:
        """扫描匹配得分：命中障碍加分，落进已知空地扣分。

        只加分会让匹配把点云吸进高密度区，导致地图整体收缩，
        所以必须对「本该是空地的位置出现障碍点」惩罚。
        """
        if xs.size == 0:
            return 0.0
        ix = np.floor(
            (xs + self.half_span_m) / self.resolution_m
            + SCAN_MATCH_GRID_EPS
        ).astype(np.int64)
        iy = np.floor(
            (ys + self.half_span_m) / self.resolution_m
            + SCAN_MATCH_GRID_EPS
        ).astype(np.int64)
        ok = (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)
        if not np.any(ok):
            return 0.0
        rows, cols = iy[ok], ix[ok]
        belief = self.low[rows, cols]
        hit = np.clip(belief, 0.0, None)
        miss = np.clip(-belief, 0.0, None)
        return float((hit - MATCH_FREE_PENALTY * miss).sum())

    @staticmethod
    def _requested_scan_match_backend() -> str:
        requested = str(
            os.environ.get(SCAN_MATCH_BACKEND_ENV, SCAN_MATCH_BACKEND_AUTO)
            or ""
        ).strip().lower()
        if requested not in {
            SCAN_MATCH_BACKEND_AUTO,
            SCAN_MATCH_BACKEND_CPU,
            SCAN_MATCH_BACKEND_CUDA,
        }:
            return SCAN_MATCH_BACKEND_AUTO
        return requested

    def _cuda_low(self):
        """返回当前 revision 的低层地图张量；失败由调用方记录并回退。"""
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch 看不到 CUDA")
        device_name = str(
            os.environ.get("BEHAVIOR_SLAM_CUDA_DEVICE", "cuda:0") or "cuda:0"
        )
        device = torch.device(device_name)
        if (
            self._cuda_low_cache is None
            or self._cuda_low_cache_revision != int(self._rev)
            or self._cuda_low_cache_device != str(device)
            or tuple(self._cuda_low_cache.shape) != tuple(self.low.shape)
        ):
            # 评分张量才上卡；CPU 仍保留唯一可导出的地图数组。
            torch.set_num_threads(1)
            self._cuda_low_cache = torch.from_numpy(
                np.ascontiguousarray(self.low, dtype=np.float32)
            ).to(device=device, non_blocking=True)
            self._cuda_low_cache_revision = int(self._rev)
            self._cuda_low_cache_device = str(device)
        return self._cuda_low_cache

    def scan_match_backend_info(self) -> Dict[str, str]:
        """报告扫描匹配实际设备，不触发一次昂贵的地图上传。"""
        requested = self._requested_scan_match_backend()
        if requested == SCAN_MATCH_BACKEND_CPU:
            return {
                "backend": "numpy_cpu",
                "device": "cpu",
                "requested": requested,
                "fallback_reason": "",
            }
        if self._scan_match_backend == "torch_cuda":
            return {
                "backend": self._scan_match_backend,
                "device": self._scan_match_device,
                "requested": requested,
                "fallback_reason": self._scan_match_fallback_reason,
            }
        # `auto` 的可用性在第一次评分时才确定；显式 cuda 也必须能安全回退，
        # 这样没有 GPU 的单测/旧 interface 不会因为新开关直接起不来。
        return {
            "backend": self._scan_match_backend
            if self._scan_match_backend != "uninitialized"
            else "numpy_cpu",
            "device": self._scan_match_device
            if self._scan_match_device != "uninitialized"
            else "cpu",
            "requested": requested,
            "fallback_reason": self._scan_match_fallback_reason,
        }

    def _match_score_candidates_cuda(
        self,
        bx: np.ndarray,
        by: np.ndarray,
        poses: np.ndarray,
    ) -> np.ndarray:
        """在 CUDA 上批量计算多个 (x, y, yaw_deg) 候选的栅格得分。"""
        import torch

        low = self._cuda_low()
        device = low.device
        body_x = torch.from_numpy(
            np.ascontiguousarray(np.asarray(bx, dtype=np.float64))
        ).to(device=device, dtype=torch.float64, non_blocking=True)
        body_y = torch.from_numpy(
            np.ascontiguousarray(np.asarray(by, dtype=np.float64))
        ).to(device=device, dtype=torch.float64, non_blocking=True)
        candidates = torch.from_numpy(
            np.ascontiguousarray(np.asarray(poses, dtype=np.float64))
        ).to(device=device, dtype=torch.float64, non_blocking=True)
        scores: list[np.ndarray] = []
        height, width = self.low.shape
        resolution = float(self.resolution_m)
        half_span = float(self.half_span_m)
        with torch.inference_mode():
            for first in range(0, int(candidates.shape[0]), SCAN_MATCH_GPU_BATCH):
                chunk = candidates[first:first + SCAN_MATCH_GPU_BATCH]
                angles = chunk[:, 2] * (math.pi / 180.0)
                cos_a = torch.cos(angles)[:, None]
                sin_a = torch.sin(angles)[:, None]
                world_x = chunk[:, 0, None] + cos_a * body_x[None, :] - sin_a * body_y[None, :]
                world_y = chunk[:, 1, None] + sin_a * body_x[None, :] + cos_a * body_y[None, :]
                columns = torch.floor(
                    (world_x + half_span) / resolution + SCAN_MATCH_GRID_EPS
                ).to(torch.long)
                rows = torch.floor(
                    (world_y + half_span) / resolution + SCAN_MATCH_GRID_EPS
                ).to(torch.long)
                inside = (
                    (columns >= 0)
                    & (columns < width)
                    & (rows >= 0)
                    & (rows < height)
                )
                clipped_columns = columns.clamp(0, width - 1)
                clipped_rows = rows.clamp(0, height - 1)
                belief = low[clipped_rows, clipped_columns].to(torch.float64)
                value = torch.clamp(belief, min=0.0) - (
                    MATCH_FREE_PENALTY * torch.clamp(-belief, min=0.0)
                )
                value = torch.where(inside, value, torch.zeros_like(value))
                scores.append(
                    value.sum(dim=1).detach().cpu().numpy().astype(
                        np.float64, copy=False
                    )
                )
        if low.device.type == "cuda":
            torch.cuda.synchronize(low.device)
        return np.concatenate(scores) if scores else np.zeros(0, dtype=np.float64)

    def match_score_candidates(
        self,
        bx: np.ndarray,
        by: np.ndarray,
        poses: np.ndarray,
    ) -> np.ndarray:
        """按候选顺序返回得分；CUDA 失败只降级本次调用。"""
        body_x = np.asarray(bx)
        body_y = np.asarray(by)
        candidates = np.asarray(poses, dtype=np.float64)
        if candidates.ndim != 2 or candidates.shape[1] != 3:
            raise ValueError("poses 必须是 Nx3")
        if body_x.size == 0 or candidates.shape[0] == 0:
            return np.zeros(candidates.shape[0], dtype=np.float64)
        requested = self._requested_scan_match_backend()
        # 诊断脚本会暂时替换 match_score 做插桩；此时必须保留其逐候选
        # 语义，避免“加速”悄悄绕过诊断或测试替身。
        patched_score = type(self).match_score is not OccupancyGrid.match_score
        if requested != SCAN_MATCH_BACKEND_CPU and not patched_score:
            try:
                result = self._match_score_candidates_cuda(body_x, body_y, candidates)
                self._scan_match_backend = "torch_cuda"
                self._scan_match_device = str(self._cuda_low_cache.device)
                self._scan_match_fallback_reason = ""
                return result
            except Exception as exc:
                self._scan_match_backend = "numpy_cpu"
                self._scan_match_device = "cpu"
                self._scan_match_fallback_reason = f"{type(exc).__name__}: {exc}"
        else:
            self._scan_match_backend = "numpy_cpu"
            self._scan_match_device = "cpu"
            self._scan_match_fallback_reason = "" if not patched_score else "patched_match_score"
        # 保持原实现的浮点/越界语义，便于逐候选回归和故障定位。
        out = np.empty(candidates.shape[0], dtype=np.float64)
        for index, (x, y, yaw) in enumerate(candidates):
            angle = math.radians(float(yaw))
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            xs = float(x) + cos_a * body_x - sin_a * body_y
            ys = float(y) + sin_a * body_x + cos_a * body_y
            out[index] = self.match_score(xs, ys)
        return out

    def cell_centers(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rows, cols = np.nonzero(mask)
        xs = (cols + 0.5) * self.resolution_m - self.half_span_m
        ys = (rows + 0.5) * self.resolution_m - self.half_span_m
        return xs, ys


class Submap:
    """一小段行程里攒出来的局部图，连同它在地图系里的位姿。

    栅格用的是 submap 自己的局部系（原点在 ``origin``，x 轴指向建图开始
    时的车头），所以整张 submap 可以事后被刚体挪动——位姿图优化就是靠这个
    把历史误差重新分摊回去。写进全局栅格的观测没有这个自由度。
    """

    def __init__(
        self,
        x: float,
        y: float,
        yaw_deg: float,
        *,
        half_span_m: float = SUBMAP_HALF_SPAN_M,
        serial: int = 0,
    ) -> None:
        # 位姿图的边用 serial 而不是列表下标：老 submap 被淘汰后下标会移位，
        # 边就全指错了地方
        self.serial = int(serial)
        self.origin_x = float(x)
        self.origin_y = float(y)
        self.origin_yaw_deg = float(yaw_deg)
        self.grid = OccupancyGrid(half_span_m=half_span_m)
        # 墙方向保存在 submap 局部系。位姿图旋转 submap 后，可据此重建
        # 地图系方向先验；若只留一份全局直方图，优化后它会继续把 yaw 拉回旧图。
        self.wall_dir_hist = np.zeros(DIR_HIST_BINS, dtype=np.float64)
        self.travel_m = 0.0
        self.turn_deg = 0.0
        # 定稿后不再接收新观测，只等着被位姿图挪动
        self.finished = False

    def to_local(
        self, x: float, y: float, yaw_deg: float
    ) -> Tuple[float, float, float]:
        """地图系位姿 → submap 局部系。"""
        angle = math.radians(-self.origin_yaw_deg)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        dx = float(x) - self.origin_x
        dy = float(y) - self.origin_y
        return (
            cos_a * dx - sin_a * dy,
            sin_a * dx + cos_a * dy,
            wrap_deg(float(yaw_deg) - self.origin_yaw_deg),
        )

    def to_map(
        self, x: float, y: float, yaw_deg: float
    ) -> Tuple[float, float, float]:
        """submap 局部系位姿 → 地图系。"""
        angle = math.radians(self.origin_yaw_deg)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        return (
            self.origin_x + cos_a * float(x) - sin_a * float(y),
            self.origin_y + sin_a * float(x) + cos_a * float(y),
            wrap_deg(float(yaw_deg) + self.origin_yaw_deg),
        )

    def local_delta_to_map(
        self, dx: float, dy: float
    ) -> Tuple[float, float]:
        """局部系里的一个位移增量转到地图系（只转不平移）。"""
        angle = math.radians(self.origin_yaw_deg)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        return (
            cos_a * float(dx) - sin_a * float(dy),
            sin_a * float(dx) + cos_a * float(dy),
        )

    def pose(self) -> Tuple[float, float, float]:
        return (self.origin_x, self.origin_y, self.origin_yaw_deg)

    def set_pose(self, x: float, y: float, yaw_deg: float) -> None:
        """位姿图优化后整张图刚体挪到新位置。栅格内容一格都不用动。"""
        self.origin_x = float(x)
        self.origin_y = float(y)
        self.origin_yaw_deg = wrap_deg(float(yaw_deg))

    def coarse_grid(self) -> OccupancyGrid:
        """降分辨率 + 取邻域最大值的墙图，专供回环粗搜。

        粗搜步长有几十厘米，而墙在细图上只占一两格，直接拿细图评分会从
        峰旁边整个跨过去——真回环于是永远搜不到。把格子放大并取块内最大
        值，峰就变宽了，粗搜只需保证「不错过」，位置精度交给后面几级。
        """
        cached = getattr(self, "_coarse", None)
        if cached is not None and cached[0] == self.grid.frames:
            return cached[1]
        source = self.grid
        factor = LOOP_COARSE_FACTOR
        size = source.n // factor
        coarse = OccupancyGrid(
            resolution_m=source.resolution_m * factor,
            half_span_m=source.half_span_m,
        )
        trimmed = source.low[:size * factor, :size * factor]
        coarse.low = trimmed.reshape(
            size, factor, size, factor
        ).max(axis=(1, 3)).astype(np.float32)
        coarse.frames = source.frames
        self._coarse = (source.frames, coarse)
        return coarse

    def wall_points(self, limit: int = LOOP_MATCH_POINTS) -> np.ndarray:
        """局部系里的墙点，抽稀到 limit 个，用来和别的 submap 对齐。"""
        grid = self.grid
        xs, ys = grid.cell_centers(grid.low > 0.0)
        if xs.size == 0:
            return np.zeros((0, 2), dtype=np.float64)
        if xs.size > limit:
            step = int(math.ceil(xs.size / limit))
            xs, ys = xs[::step], ys[::step]
        return np.column_stack([xs, ys])

    def wall_bounds_map(self) -> Optional[Tuple[float, float, float, float]]:
        """返回墙证据在地图系的 AABB；局部范围按帧数缓存。"""
        cached = getattr(self, "_wall_bounds_local", None)
        if cached is None or cached[0] != self.grid.frames:
            rows, cols = np.nonzero(self.grid.low > 0.0)
            if rows.size == 0:
                return None
            resolution = self.grid.resolution_m
            half = self.grid.half_span_m
            cached = (
                self.grid.frames,
                (
                    (float(cols.min()) + 0.5) * resolution - half,
                    (float(cols.max()) + 0.5) * resolution - half,
                    (float(rows.min()) + 0.5) * resolution - half,
                    (float(rows.max()) + 0.5) * resolution - half,
                ),
            )
            self._wall_bounds_local = cached
        xmin, xmax, ymin, ymax = cached[1]
        corners = [
            self.to_map(x, y, 0.0)[:2]
            for x, y in (
                (xmin, ymin),
                (xmin, ymax),
                (xmax, ymin),
                (xmax, ymax),
            )
        ]
        return (
            min(point[0] for point in corners),
            max(point[0] for point in corners),
            min(point[1] for point in corners),
            max(point[1] for point in corners),
        )

    def is_full(self) -> bool:
        return (
            self.grid.frames >= SUBMAP_MAX_FRAMES
            or self.travel_m >= SUBMAP_MAX_TRAVEL_M
            or abs(self.turn_deg) >= SUBMAP_MAX_TURN_DEG
        )

    def ready_for_handoff(self) -> bool:
        """走到一半就该开下一张，让它在接班前攒够观测。

        三个条件都得看：只按帧数判的话，行程和转角的上限就被架空了——
        机器人走得快时一张图会摊开好几米，既超出 submap 的半径，也让
        它内部攒的漂移不再「只有几十帧那么小」。
        """
        return (
            self.grid.frames >= SUBMAP_HANDOFF_FRAMES
            or self.travel_m >= 0.5 * SUBMAP_MAX_TRAVEL_M
            or abs(self.turn_deg) >= 0.5 * SUBMAP_MAX_TURN_DEG
        )

@dataclass
class EgoMap:
    session_id: str
    x: float = 0.0
    y: float = 0.0
    yaw_deg: float = 0.0
    initialized: bool = False
    trail: List[TrailPoint] = field(default_factory=list)
    # 完整 tick 轨迹，和用于地点标注的稀疏 action trail 分开保存。
    trajectory_samples: List[TrailPoint] = field(default_factory=list)
    landmarks: Dict[str, Landmark] = field(default_factory=dict)
    trail_len_m: float = 0.0
    # 所有相邻里程累计的绝对转角。视觉候选租约用它衡量候选生成后又传播了
    # 多少运动不确定性；只看首尾 yaw 会把转一圈后的 360 度错误抵消成零。
    odometry_abs_turn_deg: float = 0.0
    update_count: int = 0
    grid: OccupancyGrid = field(default_factory=OccupancyGrid)
    integrated_images: set = field(default_factory=set)
    events: List[MapEvent] = field(default_factory=list)
    places: List[MapEvent] = field(default_factory=list)
    # image_id -> 拍这张图时的 (x, y, yaw_deg)，供点选反解换算到地图系
    image_poses: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    # 图优化后图像位姿必须随拍摄当时的 submap 移动，不能按当前位置猜归属
    image_pose_submaps: Dict[str, int] = field(default_factory=dict)
    # 实时里程接管后，工具返回的完成量不再二次叠加，只用来切分轨迹段
    live_odometry: bool = False
    # 见过的墙方向（地图系，按 180° 折叠，1° 一格，值是累计线段长度）。
    # yaw 只能靠地图侧修，而这份直方图就是「这栋房子的墙朝哪几个方向」
    # 这一先验的载体。
    wall_dir_hist: np.ndarray = field(
        default_factory=lambda: np.zeros(DIR_HIST_BINS, dtype=np.float64)
    )
    yaw_fixes: int = 0
    yaw_fix_total_deg: float = 0.0
    # 局部图序列。全局栅格照旧用于渲染和查询，submap 额外留一份可以刚体
    # 挪动的副本：匹配拿它当基准（干净），位姿图优化后拿它重建全局图。
    submaps: List[Submap] = field(default_factory=list)
    submaps_dropped: int = 0
    # 位姿图：节点是 submap（按 serial 索引），边是相对位姿观测
    pose_edges: List[PoseEdge] = field(default_factory=list)
    next_submap_serial: int = 0
    loops_found: int = 0
    loops_rejected: int = 0
    graph_optimizations: int = 0
    # 扫描匹配运行账本。今晚的人工复现证明单看最终图不够：长时间原地
    # 旋转时，错误的平移修正可以一帧一帧累积，最后把机器人从起点附近推走。
    scan_match_attempts: int = 0
    scan_match_applied: int = 0
    scan_match_translation_locked: int = 0
    # human recording 离线回放可以先用整段真墙重观测做一次保守批优化。
    # 在线建图没有未来帧，这个字段保持空；离线则完整保留接受门槛和误差账本。
    global_pose_optimization: Dict[str, Any] = field(default_factory=dict)
    # 一次性的建图 -> 定位状态。冻结后结构栅格和 submap 都只读。
    mapping_state: str = "mapping"
    freeze_frame: Optional[int] = None
    freeze_reason: str = ""
    freeze_grid_revision: Optional[int] = None
    freeze_grid_digest: str = ""
    mapping_observations: int = 0
    mapping_travel_m: float = 0.0
    mapping_growth_history: List[int] = field(default_factory=list)
    mapping_visual_keyframes: int = 0
    mapping_freeze_candidate_frame: Optional[int] = None
    mapping_idle_observations: int = 0
    mapping_frontier_signature: str = ""
    mapping_frontier_quiescence_observations: int = 0
    mapping_frontier_complete_observations: int = 0
    mapping_coverage: Dict[str, Any] = field(default_factory=dict)
    # 只保留最近一小段冻结判定，便于证明“为什么此刻停/不停”，不会参与决策。
    mapping_freeze_checks: List[Dict[str, Any]] = field(default_factory=list)
    localization_attempts: int = 0
    localization_accepted: int = 0
    localization_rejected: int = 0
    localization_detail_frames: int = 0
    localization_last_reason: str = ""
    localization_status: str = "mapping"
    # 面向调用方的三态置信度：mapping / localized / ambiguous / lost。
    # status 保留更细的内部阶段，不能再拿它假装定位一直可靠。
    localization_state: str = "mapping"
    localization_state_reason: str = "mapping_active"
    localization_position_variance_m2: float = 0.0
    localization_yaw_variance_deg2: float = 0.0
    localization_ambiguous_observations: int = 0
    localization_last_validated_frame: Optional[int] = None
    localization_unvalidated_travel_m: float = 0.0
    localization_unvalidated_turn_deg: float = 0.0
    localization_visual_candidates: int = 0
    localization_geometry_attempts: int = 0
    # 外观只能选历史子图；只有 GPU 深度几何的唯一峰验证可触发冻结或校正。
    mapping_geometry_attempts: int = 0
    mapping_geometry_accepted: int = 0
    mapping_geometry_rejected: int = 0
    geometry_last_reason: str = ""
    geometry_reason_counts: Dict[str, int] = field(default_factory=dict)
    geometry_blocker_counts: Dict[str, int] = field(default_factory=dict)
    mapping_revisit_evidence: List[Dict[str, Any]] = field(default_factory=list)
    mapping_last_visual_probe_frame: Optional[int] = None
    mapping_last_keyframe_frame: Optional[int] = None
    _visual_localizer: Any = field(default=None, repr=False, compare=False)
    _geometry_validator: Any = field(default=None, repr=False, compare=False)
    _geometry_pose_bank: Any = field(default=None, repr=False, compare=False)
    _geometry_visual_lease: Optional[Dict[str, Any]] = field(
        default=None, repr=False, compare=False
    )
    _geometry_visual_leases: List[Dict[str, Any]] = field(
        default_factory=list, repr=False, compare=False
    )
    _geometry_reports: List[Dict[str, Any]] = field(
        default_factory=list, repr=False, compare=False
    )
    # 关键帧的绝对位姿会随结构位姿图优化失效；真正不变的是它在所属
    # submap 中的局部位姿。优化后的关键帧同步只读这份锚点。
    _visual_keyframe_anchors: Dict[
        str, Tuple[int, Tuple[float, float, float]]
    ] = field(default_factory=dict, repr=False, compare=False)
    _last_trajectory_sample: Optional[TrailPoint] = field(
        default=None, repr=False, compare=False
    )
    _mapping_last_activity_pose: Optional[Tuple[float, float, float]] = field(
        default=None, repr=False, compare=False
    )

    def advance_odometry(self, forward_m: float, left_m: float, yaw_deg: float) -> None:
        """应用有限的机体系相对位姿，连续更新位姿，不产生轨迹拐点。

        这个入口历史上接收的是相邻观测已经求出的有限 SE(2) 增量。速度
        形式的 ``base_qvel`` 请使用 ``advance_body_twist``，否则会把同一
        tick 的角速度再积分一次。
        """
        self.advance_relative_odometry(forward_m, left_m, yaw_deg)

    def advance_body_twist(
        self,
        forward_m: float,
        left_m: float,
        yaw_deg: float,
    ) -> None:
        """按常速度刚体模型积分一个 ``base_qvel`` tick。"""
        relative = body_twist_delta(forward_m, left_m, yaw_deg)
        self.advance_relative_odometry(*relative)

    def advance_relative_odometry(
        self,
        forward_m: float,
        left_m: float,
        yaw_deg: float,
    ) -> None:
        """应用已经在起始机体系表达的有限 SE(2) 相对位姿。

        录制帧之间的 ``local_command_odometry`` 差值已经是有限刚体变换，
        不能再次走 ``body_twist_delta``；该入口和 tick 级速度积分严格分开。
        """
        self.live_odometry = True
        self.ensure_start(tool="live_odometry")
        yaw = math.radians(self.yaw_deg)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        dx = float(forward_m) * cos_y - float(left_m) * sin_y
        dy = float(forward_m) * sin_y + float(left_m) * cos_y
        self.x += dx
        self.y += dy
        self.yaw_deg = wrap_deg(self.yaw_deg + float(yaw_deg))
        step_m = math.hypot(dx, dy)
        self.trail_len_m += step_m
        self.odometry_abs_turn_deg += abs(float(yaw_deg))
        if self.mapping_state == "mapping":
            self.mapping_travel_m += step_m
            # submap 该多大是按走了多远、转了多少来定的，只数帧数的话原地不动
            # 也会攒满一张，而快速穿堂而过又会把一大片塞进同一张图里。
            for submap in self.live_submaps():
                submap.travel_m += step_m
                submap.turn_deg += abs(float(yaw_deg))
        elif self.mapping_state == "localization":
            # 相邻 RGB-D/底盘会继续传播；在新的长期几何闭环到来前，同时
            # 累计可审计的行程以及下方与调用频率无关的保守运动模型方差。
            self.localization_unvalidated_travel_m += step_m
            self.localization_unvalidated_turn_deg += abs(float(yaw_deg))
            rotation_rad = abs(math.radians(float(yaw_deg)))
            self.localization_position_variance_m2 += (
                LOCALIZATION_POSITION_VARIANCE_PER_M * step_m
                + LOCALIZATION_POSITION_VARIANCE_PER_RAD * rotation_rad
            )
            self.localization_yaw_variance_deg2 += (
                LOCALIZATION_YAW_VARIANCE_PER_M_DEG2 * step_m
                + LOCALIZATION_YAW_VARIANCE_PER_RAD_DEG2 * rotation_rad
            )
            if self.localization_state != "ambiguous":
                self._refresh_localization_state("odometry_propagation")
        # 里程只负责传播位姿和记录轨迹，绝不把轨迹写回占据栅格。
        # 走过某处不是该处没有墙的观测证据；保留这种“刻空”会制造可通行
        # 通道，掩盖重影并改变地图拓扑。
        self._append_trajectory_sample(source="odometry")

    def correct_live_pose(self, x: float, y: float, yaw_deg: float) -> None:
        """用同一时段的视觉里程替换当前在线位姿，不重复累计路程。

        ``advance_odometry`` 已经让控制 tick 期间的地图查询保持连续；等下一帧
        RGB-D 到达后，只需把这一小段 qvel 终点校正到视觉观测给出的终点。
        这里不改 trail_len/submap 行程，也不跨校正量 carve，避免同一段运动被
        记两次或把位姿修正误画成机器人走过的通道。
        """
        pose = np.asarray([x, y, yaw_deg], dtype=np.float64)
        if not np.all(np.isfinite(pose)):
            raise ValueError("在线位姿校正必须是有限数")
        self.live_odometry = True
        self.x = float(pose[0])
        self.y = float(pose[1])
        self.yaw_deg = wrap_deg(float(pose[2]))
        self._append_trajectory_sample(source="visual_odometry", force=True)

    def _sync_latest_trajectory_pose(self) -> None:
        """把当前位姿校正同步到最近一个 tick 样本，不制造虚假的移动段。"""
        if self._last_trajectory_sample is not None:
            self._last_trajectory_sample.x = float(self.x)
            self._last_trajectory_sample.y = float(self.y)
            self._last_trajectory_sample.yaw_deg = float(self.yaw_deg)
        if self.trail:
            self.trail[-1].x = float(self.x)
            self.trail[-1].y = float(self.y)
            self.trail[-1].yaw_deg = float(self.yaw_deg)

    def _grid_digest(self) -> str:
        """对结构栅格做可复核摘要；摘要本身不参与地图生成。"""
        digest = hashlib.sha256()
        for layer in (self.grid.low, self.grid.high, self.grid.bands):
            digest.update(np.ascontiguousarray(layer).tobytes())
        digest.update(str(int(self.grid.frames)).encode("ascii"))
        return digest.hexdigest()

    def _refresh_localization_state(self, reason: str) -> str:
        """按累计不确定度更新定位三态，不读取场景或评测信息。"""
        if self.mapping_state != "localization":
            self.localization_state = "mapping"
            self.localization_state_reason = "mapping_active"
            return self.localization_state
        position_std = math.sqrt(max(
            0.0, float(self.localization_position_variance_m2)
        ))
        yaw_std = math.sqrt(max(
            0.0, float(self.localization_yaw_variance_deg2)
        ))
        lost = (
            position_std > LOCALIZATION_LOST_POSITION_STD_M
            or yaw_std > LOCALIZATION_LOST_YAW_STD_DEG
        )
        self.localization_state = "lost" if lost else "localized"
        self.localization_state_reason = str(
            "uncertainty_budget_exceeded" if lost else reason
        )
        return self.localization_state

    def _known_cell_count(self) -> int:
        """统计已观测证据格，供冻结稳定性门槛使用。"""
        # low/high 是两层证据，同一格同时有两层时只能算一个空间格。
        return int(np.count_nonzero((self.grid.low != 0.0) | (self.grid.high != 0.0)))

    def _update_mapping_activity(self) -> None:
        """记录最近观测期间是否仍在运动，供候选冻结的静止保持使用。"""
        if self.mapping_state != "mapping":
            return
        current = (float(self.x), float(self.y), float(self.yaw_deg))
        previous = self._mapping_last_activity_pose
        if previous is None:
            self.mapping_idle_observations = 0
        else:
            distance = math.hypot(current[0] - previous[0], current[1] - previous[1])
            yaw_delta = abs(wrap_deg(current[2] - previous[2]))
            if (
                distance <= MAPPING_IDLE_TRANSLATION_M
                and yaw_delta <= MAPPING_IDLE_YAW_DEG
            ):
                self.mapping_idle_observations += 1
            else:
                self.mapping_idle_observations = 0
        self._mapping_last_activity_pose = current

    def _mapping_coverage_metrics(self) -> Dict[str, Any]:
        """量化轨迹是否落在已观测自由区；只读栅格，不修改任何证据。"""
        grid = self.grid
        known = (grid.low != 0.0) | (grid.high != 0.0)
        known_count = int(np.count_nonzero(known))
        unknown = ~known
        padded = np.pad(unknown, 1, constant_values=True)
        frontier = known & (
            padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
        )
        frontier_ratio = float(np.count_nonzero(frontier) / max(1, known_count))

        # 机器人可达 frontier 与全图边界不是一回事。先按机身净空膨胀障碍，
        # 再找当前位姿和历史轨迹真正触达的自由连通域。只看当前位姿会在
        # 膨胀障碍边缘抖一格时丢掉整段历史通道，产生一帧“探索完成”假象。
        import cv2

        free = grid.observed_free_mask()
        occupied = grid.observed_occupied_mask()
        clearance_cells = max(0, int(math.ceil(
            MAPPING_FRONTIER_CLEARANCE_M / grid.resolution_m
        )))
        if clearance_cells:
            yy, xx = np.ogrid[
                -clearance_cells:clearance_cells + 1,
                -clearance_cells:clearance_cells + 1,
            ]
            kernel = (
                xx * xx + yy * yy <= clearance_cells * clearance_cells
            ).astype(np.uint8)
            inflated = cv2.dilate(occupied.astype(np.uint8), kernel) > 0
        else:
            inflated = occupied
        navigable = free & ~inflated
        component_count, component_labels = cv2.connectedComponents(
            navigable.astype(np.uint8), connectivity=4
        )
        pose_ix, pose_iy = grid._indices(
            np.asarray([self.x], dtype=np.float64),
            np.asarray([self.y], dtype=np.float64),
        )
        seed_label = 0
        seed_distance_m = math.inf
        if (
            0 <= int(pose_ix[0]) < grid.n
            and 0 <= int(pose_iy[0]) < grid.n
        ):
            seed_label = int(component_labels[int(pose_iy[0]), int(pose_ix[0])])
        if seed_label == 0:
            rows, columns = np.nonzero(navigable)
            if len(rows):
                distances = (
                    (rows - int(pose_iy[0])) ** 2
                    + (columns - int(pose_ix[0])) ** 2
                )
                nearest = int(np.argmin(distances))
                seed_label = int(component_labels[rows[nearest], columns[nearest]])
                seed_distance_m = float(
                    math.sqrt(float(distances[nearest])) * grid.resolution_m
                )
                if seed_distance_m > MAPPING_FRONTIER_SEED_MAX_DISTANCE_M:
                    seed_label = 0
        else:
            seed_distance_m = 0.0

        points = self.trajectory_samples or self.trail
        sampled_points = points
        max_samples = 2048
        if len(sampled_points) > max_samples:
            selection = np.linspace(
                0, len(sampled_points) - 1, max_samples
            ).astype(np.int64)
            sampled_points = [sampled_points[int(index)] for index in selection]
        visited_labels = {int(seed_label)} if seed_label else set()
        if sampled_points:
            trail_x = np.asarray(
                [float(point.x) for point in sampled_points], dtype=np.float64
            )
            trail_y = np.asarray(
                [float(point.y) for point in sampled_points], dtype=np.float64
            )
            trail_ix, trail_iy = grid._indices(trail_x, trail_y)
            trail_inside = (
                (trail_ix >= 0) & (trail_ix < grid.n)
                & (trail_iy >= 0) & (trail_iy < grid.n)
            )
            trail_seed = np.zeros_like(navigable, dtype=np.uint8)
            trail_seed[trail_iy[trail_inside], trail_ix[trail_inside]] = 1
            seed_radius = max(1, int(math.ceil(
                MAPPING_FRONTIER_SEED_MAX_DISTANCE_M / grid.resolution_m
            )))
            yy, xx = np.ogrid[
                -seed_radius:seed_radius + 1,
                -seed_radius:seed_radius + 1,
            ]
            seed_kernel = (
                xx * xx + yy * yy <= seed_radius * seed_radius
            ).astype(np.uint8)
            trail_near_navigable = (
                cv2.dilate(trail_seed, seed_kernel) > 0
            ) & navigable
            visited_labels.update(
                int(label)
                for label in np.unique(component_labels[trail_near_navigable])
                if int(label) > 0
            )
        reachable = (
            np.isin(
                component_labels,
                np.fromiter(sorted(visited_labels), dtype=np.int32),
            )
            if visited_labels
            else np.zeros_like(navigable)
        )
        unknown_neighbor = cv2.dilate(
            unknown.astype(np.uint8),
            np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8),
        ) > 0
        reachable_frontier = reachable & unknown_neighbor
        raw_cluster_count, cluster_labels = cv2.connectedComponents(
            reachable_frontier.astype(np.uint8), connectivity=8
        )
        minimum_frontier_cells = max(1, int(math.ceil(
            MAPPING_MIN_FRONTIER_LENGTH_M / grid.resolution_m
        )))
        cluster_cells = sorted((
            int(np.count_nonzero(cluster_labels == label))
            for label in range(1, raw_cluster_count)
        ), reverse=True)
        significant_clusters = [
            cells for cells in cluster_cells
            if cells >= minimum_frontier_cells
        ]
        frontier_complete = bool(
            seed_label and visited_labels and not significant_clusters
        )
        # 哈希的是已访问连通域里的显著前沿，而不是全图未知边界。
        # 相同签名表示传感器继续观测却没有打开新空间；它只参与停止判定，
        # 不会反向修改占据栅格或把未知区臆测成墙。
        significant_frontier = np.zeros_like(reachable_frontier, dtype=bool)
        significant_labels = {
            label
            for label in range(1, raw_cluster_count)
            if int(np.count_nonzero(cluster_labels == label))
            >= minimum_frontier_cells
        }
        if significant_labels:
            significant_frontier = np.isin(
                cluster_labels,
                np.fromiter(significant_labels, dtype=np.int32),
            )
        frontier_signature = hashlib.sha256(
            np.ascontiguousarray(
                np.packbits(significant_frontier, axis=None)
            ).tobytes()
        ).hexdigest()[:16]

        if not points:
            result = {
                "known_cells": known_count,
                "frontier_cells": int(np.count_nonzero(frontier)),
                "frontier_ratio": frontier_ratio,
                "reachable_frontier_cells": int(np.count_nonzero(reachable_frontier)),
                "reachable_frontier_clusters": len(significant_clusters),
                "reachable_frontier_lengths_m": [
                    float(cells * grid.resolution_m)
                    for cells in significant_clusters[:8]
                ],
                "frontier_complete": frontier_complete,
                "reachable_frontier_signature": frontier_signature,
                "pose_to_navigable_seed_m": (
                    None if not math.isfinite(seed_distance_m) else seed_distance_m
                ),
                "frontier_seed_components": len(visited_labels),
                "current_pose_seed_valid": bool(seed_label),
                "navigable_components": max(0, int(component_count) - 1),
                "trajectory_samples": 0,
                "trajectory_covered_ratio": 0.0,
                "trajectory_free_ratio": 0.0,
                "trajectory_collision_ratio": 0.0,
                "trajectory_outside_ratio": 1.0,
            }
            self.mapping_coverage = result
            return result

        # 轨迹可能按每个 qvel tick 保存；冻结判断只需有代表性的均匀样本。
        max_samples = 2048
        if len(points) > max_samples:
            selection = np.linspace(0, len(points) - 1, max_samples).astype(np.int64)
            points = [points[int(index)] for index in selection]
        xs = np.asarray([float(point.x) for point in points], dtype=np.float64)
        ys = np.asarray([float(point.y) for point in points], dtype=np.float64)
        ix, iy = grid._indices(xs, ys)
        inside = (ix >= 0) & (ix < grid.n) & (iy >= 0) & (iy < grid.n)
        total = int(len(points))
        inside_count = int(np.count_nonzero(inside))
        if inside_count == 0:
            result = {
                "known_cells": known_count,
                "frontier_cells": int(np.count_nonzero(frontier)),
                "frontier_ratio": frontier_ratio,
                "reachable_frontier_cells": int(np.count_nonzero(reachable_frontier)),
                "reachable_frontier_clusters": len(significant_clusters),
                "reachable_frontier_lengths_m": [
                    float(cells * grid.resolution_m)
                    for cells in significant_clusters[:8]
                ],
                "frontier_complete": frontier_complete,
                "reachable_frontier_signature": frontier_signature,
                "pose_to_navigable_seed_m": (
                    None if not math.isfinite(seed_distance_m) else seed_distance_m
                ),
                "frontier_seed_components": len(visited_labels),
                "current_pose_seed_valid": bool(seed_label),
                "navigable_components": max(0, int(component_count) - 1),
                "trajectory_samples": total,
                "trajectory_covered_ratio": 0.0,
                "trajectory_free_ratio": 0.0,
                "trajectory_collision_ratio": 0.0,
                "trajectory_outside_ratio": 1.0,
            }
            self.mapping_coverage = result
            return result

        ixv, iyv = ix[inside], iy[inside]
        radius = max(0, int(math.ceil(
            MAPPING_TRAJECTORY_COVERAGE_RADIUS_M / grid.resolution_m
        )))
        near_known = np.zeros(ixv.shape, dtype=bool)
        near_free = np.zeros(ixv.shape, dtype=bool)
        for dy in range(-radius, radius + 1):
            row = iyv + dy
            valid_row = (row >= 0) & (row < grid.n)
            if not np.any(valid_row):
                continue
            for dx in range(-radius, radius + 1):
                col = ixv + dx
                valid = valid_row & (col >= 0) & (col < grid.n)
                if not np.any(valid):
                    continue
                near_known[valid] |= known[row[valid], col[valid]]
                near_free[valid] |= free[row[valid], col[valid]]
        center_free = free[iyv, ixv]
        center_occupied = occupied[iyv, ixv]
        result = {
            "known_cells": known_count,
            "frontier_cells": int(np.count_nonzero(frontier)),
            "frontier_ratio": frontier_ratio,
            "reachable_frontier_cells": int(np.count_nonzero(reachable_frontier)),
            "reachable_frontier_clusters": len(significant_clusters),
            "reachable_frontier_lengths_m": [
                float(cells * grid.resolution_m)
                for cells in significant_clusters[:8]
            ],
            "frontier_complete": frontier_complete,
            "reachable_frontier_signature": frontier_signature,
            "pose_to_navigable_seed_m": (
                None if not math.isfinite(seed_distance_m) else seed_distance_m
            ),
            "frontier_seed_components": len(visited_labels),
            "current_pose_seed_valid": bool(seed_label),
            "navigable_components": max(0, int(component_count) - 1),
            "trajectory_samples": total,
            "trajectory_covered_ratio": float(np.count_nonzero(near_free) / max(1, inside_count)),
            "trajectory_free_ratio": float(np.count_nonzero(center_free) / max(1, inside_count)),
            "trajectory_collision_ratio": float(np.count_nonzero(center_occupied) / max(1, inside_count)),
            "trajectory_outside_ratio": float(1.0 - inside_count / max(1, total)),
            "trajectory_near_known_ratio": float(np.count_nonzero(near_known) / max(1, inside_count)),
        }
        self.mapping_coverage = result
        return result

    def _visual_tracker(self, feature_extractor=None):
        """延迟建立长期关键帧定位器，避免没有 RGB 的运行增加依赖开销。"""
        if self._visual_localizer is None:
            from behavior_interface.rgbd_odometry import FeatureKeyframeLocalizer

            self._visual_localizer = FeatureKeyframeLocalizer(
                feature_extractor=feature_extractor,
            )
        return self._visual_localizer

    def _submap_by_serial(self, serial: int) -> Optional[Submap]:
        for submap in self.submaps:
            if submap.serial == int(serial):
                return submap
        return None

    def _remember_visual_keyframe(self, image_id: str, tracker: Any) -> bool:
        """把长期关键帧锚定到当前时间 submap，而不是易漂的绝对坐标。"""
        owner = self._current_submap_serial()
        submap = self._submap_by_serial(owner)
        if submap is None:
            return False
        key = str(image_id)
        self._visual_keyframe_anchors[key] = (
            owner,
            relative_pose(submap.pose(), (self.x, self.y, self.yaw_deg)),
        )
        # 定位器有固定容量；同步清理已经被它淘汰的锚点，避免长 episode
        # 中字典只增不减。
        alive = set(getattr(tracker, "keyframe_ids", ()))
        if alive:
            self._visual_keyframe_anchors = {
                name: anchor
                for name, anchor in self._visual_keyframe_anchors.items()
                if name in alive
            }
        return True

    def _sync_visual_keyframe_poses(self) -> int:
        """按当前 submap 位姿重算定位器中的关键帧绝对位姿。"""
        tracker = self._visual_localizer
        update = getattr(tracker, "update_keyframe_poses", None)
        if not callable(update):
            return 0
        poses: Dict[str, Tuple[float, float, float]] = {}
        for image_id, (serial, local_pose) in self._visual_keyframe_anchors.items():
            submap = self._submap_by_serial(serial)
            if submap is not None:
                poses[image_id] = compose_pose(submap.pose(), local_pose)
        return int(update(poses))

    def _geometry_engine(self) -> GpuGeometryValidator:
        """延迟建立 CUDA 几何验证器；没有闭环候选时不占 GPU 显存。"""
        if self._geometry_validator is None:
            self._geometry_validator = GpuGeometryValidator(GeometryGridSpec(
                resolution_m=float(self.grid.resolution_m),
                half_span_m=float(self.grid.half_span_m),
            ))
        return self._geometry_validator

    def _temporal_pose_engine(self) -> TemporalPoseHypothesisBank:
        """跨探测保留竞争位姿假设；本体只做小规模 SE(2) 证据账本。"""
        if self._geometry_pose_bank is None:
            self._geometry_pose_bank = TemporalPoseHypothesisBank()
        return self._geometry_pose_bank

    def _clear_geometry_history(
        self,
        *,
        clear_pose_evidence: bool = True,
    ) -> None:
        """坐标系刚体变化后丢弃旧滚动点，禁止混用优化前后的坐标。"""
        engine = self._geometry_validator
        clear = getattr(engine, "clear", None)
        if callable(clear):
            clear()
        if clear_pose_evidence:
            pose_clear = getattr(self._geometry_pose_bank, "clear", None)
            if callable(pose_clear):
                pose_clear()

    def _authorize_temporal_pose(
        self,
        validation: Dict[str, Any],
        predicted_pose: Tuple[float, float, float],
        frame_index: int,
    ) -> bool:
        """把单次传感器共识记账；只有唯一跨时段假设可获得写权限。"""
        validation["single_probe_pose_correction_safe"] = bool(
            validation.get("pose_correction_safe")
        )
        validation["pose_correction_safe"] = False
        if (
            not validation.get("accepted")
            or not validation.get("revisit_evidence_safe")
        ):
            return False
        interval = validation.get("evidence_interval")
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            validation["temporal_pose_consensus"] = {
                "pose_correction_safe": False,
                "reason": "missing_evidence_interval",
            }
            return False
        correction = validation.get(
            "fused_correction", validation.get("geometry_correction")
        )
        if not isinstance(correction, (list, tuple)) or len(correction) != 3:
            validation["temporal_pose_consensus"] = {
                "pose_correction_safe": False,
                "reason": "missing_pose_correction",
            }
            return False
        target_group = validation.get("target_group") or []
        if not target_group and validation.get("target_serial") is not None:
            target_group = [int(validation["target_serial"])]
        corrected_pose = (
            float(predicted_pose[0]) + float(correction[0]),
            float(predicted_pose[1]) + float(correction[1]),
            wrap_deg(float(predicted_pose[2]) + float(correction[2])),
        )
        decision = self._temporal_pose_engine().observe(
            target_group=target_group,
            predicted_pose=predicted_pose,
            corrected_pose=corrected_pose,
            evidence_interval=(int(interval[0]), int(interval[1])),
            frame_index=int(frame_index),
        )
        validation["temporal_pose_consensus"] = decision
        if not decision.get("pose_correction_safe"):
            return False
        validation["pose_correction_safe"] = True
        validation["fused_correction"] = list(decision["correction"])
        validation["temporal_authorization"] = (
            "unique_visual_depth_hypothesis_across_disjoint_intervals"
        )
        return True

    def _remember_geometry_bundle(
        self,
        bundle: Dict[str, Any],
        frame_index: int,
    ) -> bool:
        """把当前深度压成小型滚动墙/自由证据，不读取任何语义。"""
        try:
            points = _depth_to_robot_points(bundle, stride=GEOMETRY_DEPTH_STRIDE)
            if points is None or points.size == 0:
                self.geometry_last_reason = "missing_depth_geometry"
                return False
            scan = extract_column_scan(
                points,
                column_m=GEOMETRY_COLUMN_M,
                min_range_m=MIN_RANGE_M,
                max_range_m=MAX_RANGE_M,
                obstacle_z_min_m=OBSTACLE_Z_MIN_M,
                obstacle_z_max_m=OBSTACLE_Z_MAX_M,
                chassis_block_z_max_m=CHASSIS_BLOCK_Z_MAX_M,
                wall_band_count=WALL_BAND_COUNT,
                wall_band_min_run=WALL_BAND_MIN_RUN,
                self_clear_forward_m=SELF_CLEAR_FORWARD_M,
                self_clear_half_width_m=SELF_CLEAR_HALF_WIDTH_M,
            )
            return bool(self._geometry_engine().remember(
                scan,
                (float(self.x), float(self.y), float(self.yaw_deg)),
                int(frame_index),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            self.geometry_last_reason = (
                f"geometry_capture_failed:{type(exc).__name__}"
            )
            return False

    @staticmethod
    def _geometry_grid_points(
        grid: OccupancyGrid,
        mask: np.ndarray,
    ) -> np.ndarray:
        x, y = grid.cell_centers(mask)
        return np.column_stack((x, y)).astype(np.float32)

    def _geometry_target_group(
        self,
        target_serial: int,
    ) -> Tuple[List[int], np.ndarray, np.ndarray]:
        """取视觉锚点及相邻时间子图；选择依据只有关键帧的 submap 归属。"""
        serials = sorted(
            int(submap.serial)
            for submap in self.submaps
            if abs(int(submap.serial) - int(target_serial))
            <= GEOMETRY_TARGET_NEIGHBOR_HOPS
        )
        wall_parts: List[np.ndarray] = []
        free_parts: List[np.ndarray] = []
        for serial in serials:
            submap = self._submap_by_serial(serial)
            if submap is None:
                continue
            wall_local = self._geometry_grid_points(
                submap.grid,
                submap.grid.occupied_mask() & submap.grid.wall_face_mask(),
            )
            free_local = self._geometry_grid_points(
                submap.grid,
                submap.grid.low <= LOGODDS_FREE_ENTER,
            )
            wall_parts.append(transform_geometry_points(wall_local, submap.pose()))
            free_parts.append(transform_geometry_points(free_local, submap.pose()))
        walls = (
            np.concatenate(wall_parts, axis=0)
            if wall_parts else np.empty((0, 2), dtype=np.float32)
        )
        frees = (
            np.concatenate(free_parts, axis=0)
            if free_parts else np.empty((0, 2), dtype=np.float32)
        )
        return (
            serials,
            unique_geometry_cells(walls, self.grid.resolution_m),
            unique_geometry_cells(frees, self.grid.resolution_m),
        )

    def _frozen_geometry_target(
        self,
    ) -> Tuple[List[int], np.ndarray, np.ndarray]:
        """返回完整冻结结构图，供定位态验证全局地点候选。

        定位查询的滚动窗口可能跨越多张 submap；仍只拿外观锚点左右各一张
        作目标会把真实墙截掉，并把截边误判成相关峰。冻结后结构栅格已经是
        只读的统一地图，验证可以安全使用全图；建图态回环仍走局部目标组。
        """
        occupied = self.grid.occupied_mask()
        wall = occupied & self.grid.wall_face_mask()
        free = (self.grid.low <= LOGODDS_FREE_ENTER) & ~occupied
        serials = sorted(int(submap.serial) for submap in self.submaps)
        return (
            serials,
            self._geometry_grid_points(self.grid, wall),
            self._geometry_grid_points(self.grid, free),
        )

    def _record_geometry_report(
        self,
        report: Dict[str, Any],
        *,
        phase: str,
        frame_index: int,
    ) -> None:
        report["phase"] = str(phase)
        report["frame_index"] = int(frame_index)
        report["task_scene_or_marks_consumed"] = False
        report["forbidden_inputs_consumed"] = []
        reason = str(report.get("reason") or "unknown")
        self.geometry_reason_counts[reason] = (
            int(self.geometry_reason_counts.get(reason, 0)) + 1
        )
        for row in report.get("rows") or ():
            for blocker in row.get("uniqueness_blockers") or ():
                key = str(blocker or "unknown")
                self.geometry_blocker_counts[key] = (
                    int(self.geometry_blocker_counts.get(key, 0)) + 1
                )
        self._geometry_reports.append(report)
        if len(self._geometry_reports) > GEOMETRY_REPORT_KEEP:
            del self._geometry_reports[:-GEOMETRY_REPORT_KEEP]

    def _validate_geometry_target(
        self,
        target_serial: int,
        visual_correction: Tuple[float, float, float],
        *,
        frame_index: int,
        phase: str,
        visual_metric: bool = True,
        visual_observability: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        frozen_global = (
            str(phase) == "localization"
            and self.mapping_state == "localization"
        )
        if frozen_global:
            serials, target_wall, target_free = self._frozen_geometry_target()
        else:
            serials, target_wall, target_free = self._geometry_target_group(
                int(target_serial)
            )
        if not serials:
            report: Dict[str, Any] = {
                "accepted": False,
                "reason": "missing_target_submap_group",
                "target_serial": int(target_serial),
                "target_group": [],
                "rows": [],
            }
        else:
            report = self._geometry_engine().validate(
                target_wall,
                target_free,
                pivot=(float(self.x), float(self.y)),
                visual_correction=visual_correction,
                visual_metric=bool(visual_metric),
                visual_observability=visual_observability,
            )
            report["target_serial"] = int(target_serial)
            report["target_group"] = serials
            report["target_scope"] = (
                "frozen_global_structure"
                if frozen_global else "anchored_submap_neighborhood"
            )
        self.geometry_last_reason = str(report.get("reason") or "")
        self._record_geometry_report(
            report, phase=phase, frame_index=frame_index
        )
        return report

    def _start_visual_geometry_lease(
        self,
        result: Any,
        predicted_pose: Tuple[float, float, float],
        frame_index: int,
    ) -> bool:
        """外观候选只租借历史子图锚点，不立即拥有修改位姿的权限。"""
        anchor = self._visual_keyframe_anchors.get(str(result.keyframe_id))
        if anchor is None:
            self.localization_last_reason = "missing_keyframe_submap_anchor"
            return False
        lease = {
            "frame_index": int(frame_index),
            "keyframe_id": str(result.keyframe_id),
            "target_serial": int(anchor[0]),
            "predicted_pose": tuple(float(value) for value in predicted_pose),
            "visual_pose": (
                float(result.x_m),
                float(result.y_m),
                float(result.yaw_deg),
            ),
            "inliers": int(getattr(result, "inliers", 0) or 0),
            "inlier_ratio": float(getattr(result, "inlier_ratio", 0.0) or 0.0),
            "rmse_m": float(getattr(result, "rmse_m", math.inf)),
            "evidence_type": str(
                getattr(result, "evidence_type", "direct_rgbd_geometry")
            ),
            "visual_metric": str(
                getattr(result, "evidence_type", "direct_rgbd_geometry")
            ) != "appearance_sequence",
            "appearance_score": float(
                getattr(result, "appearance_score", 0.0) or 0.0
            ),
            "sequence_observations": int(
                getattr(result, "sequence_observations", 0) or 0
            ),
            "candidate_count": int(
                getattr(result, "candidate_count", 0) or 0
            ),
            "independent_candidates": int(
                getattr(result, "independent_candidates", 0) or 0
            ),
            "geometry_major_span_m": float(
                getattr(result, "geometry_major_span_m", 0.0) or 0.0
            ),
            "geometry_minor_span_m": float(
                getattr(result, "geometry_minor_span_m", 0.0) or 0.0
            ),
            "pixel_coverage_x": float(
                getattr(result, "pixel_coverage_x", 0.0) or 0.0
            ),
            "pixel_coverage_y": float(
                getattr(result, "pixel_coverage_y", 0.0) or 0.0
            ),
            "translation_std_m": float(
                getattr(result, "translation_std_m", math.inf)
            ),
            "yaw_std_deg": float(
                getattr(result, "yaw_std_deg", math.inf)
            ),
            "normal_matrix_condition": float(
                getattr(result, "normal_matrix_condition", math.inf)
            ),
            "odometry_travel_m": float(self.trail_len_m),
            "odometry_abs_turn_deg": float(self.odometry_abs_turn_deg),
        }
        # 同一关键帧的新观测替换旧租约；不同地点必须同时保留，后面的并发
        # 几何唯一性门会把竞争位姿整体拒绝，不能让列表顺序决定结果。
        self._geometry_visual_leases = [
            old for old in self._geometry_visual_leases
            if str(old.get("keyframe_id")) != str(result.keyframe_id)
        ]
        self._geometry_visual_leases.append(lease)
        self._geometry_visual_leases = self._geometry_visual_leases[-8:]
        # 旧字段只作为状态兼容别名；生产验证始终消费完整候选列表。
        self._geometry_visual_lease = lease
        self.localization_visual_candidates += 1
        return True

    def _leased_visual_observability(
        self,
        lease: Dict[str, Any],
    ) -> Dict[str, float]:
        """把候选生成后的里程传播噪声计入视觉位姿协方差。"""
        metrics = {
            key: lease[key]
            for key in (
                "candidate_count",
                "independent_candidates",
                "geometry_major_span_m",
                "geometry_minor_span_m",
                "pixel_coverage_x",
                "pixel_coverage_y",
                "translation_std_m",
                "yaw_std_deg",
                "normal_matrix_condition",
            )
            if key in lease
        }
        travel_m = max(
            0.0,
            float(self.trail_len_m) - float(
                lease.get("odometry_travel_m", self.trail_len_m)
            ),
        )
        turn_deg = max(
            0.0,
            float(self.odometry_abs_turn_deg) - float(
                lease.get("odometry_abs_turn_deg", self.odometry_abs_turn_deg)
            ),
        )
        turn_rad = math.radians(turn_deg)
        translation_std = float(metrics.get("translation_std_m", math.inf))
        yaw_std = float(metrics.get("yaw_std_deg", math.inf))
        if math.isfinite(translation_std):
            metrics["translation_std_m"] = math.sqrt(
                translation_std * translation_std
                + LOCALIZATION_POSITION_VARIANCE_PER_M * travel_m
                + LOCALIZATION_POSITION_VARIANCE_PER_RAD * turn_rad
            )
        if math.isfinite(yaw_std):
            metrics["yaw_std_deg"] = math.sqrt(
                yaw_std * yaw_std
                + LOCALIZATION_YAW_VARIANCE_PER_M_DEG2 * travel_m
                + LOCALIZATION_YAW_VARIANCE_PER_RAD_DEG2 * turn_rad
            )
        metrics["lease_propagated_travel_m"] = travel_m
        metrics["lease_propagated_turn_deg"] = turn_deg
        return metrics

    def _leased_visual_corrections(
        self,
        frame_index: int,
    ) -> List[Tuple[Dict[str, Any], Tuple[float, float, float]]]:
        """沿相邻里程传播全部视觉候选，不提前丢掉竞争地点。"""
        current = (float(self.x), float(self.y), float(self.yaw_deg))
        active = []
        propagated = []
        for lease in self._geometry_visual_leases:
            if (
                int(frame_index) - int(lease["frame_index"])
                > GEOMETRY_VISUAL_LEASE_MAX_FRAMES
            ):
                continue
            active.append(lease)
            predicted_then = tuple(lease["predicted_pose"])
            visual_then = tuple(lease["visual_pose"])
            expected = compose_pose(
                visual_then,
                relative_pose(predicted_then, current),
            )
            correction = (
                float(expected[0] - current[0]),
                float(expected[1] - current[1]),
                float(wrap_deg(expected[2] - current[2])),
            )
            propagated.append((lease, correction))
        expired = len(active) != len(self._geometry_visual_leases)
        self._geometry_visual_leases = active
        self._geometry_visual_lease = active[-1] if active else None
        if expired and not active:
            self.localization_last_reason = "visual_geometry_lease_expired"
        return propagated

    def _leased_visual_correction(
        self,
        frame_index: int,
    ) -> Optional[Tuple[int, Tuple[float, float, float]]]:
        """兼容旧诊断：只返回完整候选列表中的第一项。"""
        leased = self._leased_visual_corrections(frame_index)
        if not leased:
            return None
        lease, correction = leased[0]
        return int(lease["target_serial"]), correction

    def _select_concurrent_geometry(
        self,
        validations: Iterable[Dict[str, Any]],
        predicted_pose: Tuple[float, float, float],
        *,
        frame_index: int,
        phase: str,
    ) -> Optional[Dict[str, Any]]:
        """要求同一时刻全部可行地点只支持一个位姿簇。"""
        accepted = [row for row in validations if row.get("accepted")]
        if not accepted:
            return None
        config = self._temporal_pose_engine().config

        def corrected(row: Dict[str, Any]) -> Tuple[float, float, float]:
            correction = row.get(
                "fused_correction", row.get("geometry_correction")
            )
            return (
                float(predicted_pose[0]) + float(correction[0]),
                float(predicted_pose[1]) + float(correction[1]),
                wrap_deg(float(predicted_pose[2]) + float(correction[2])),
            )

        def agree(
            first: Tuple[float, float, float],
            second: Tuple[float, float, float],
        ) -> bool:
            return (
                math.hypot(first[0] - second[0], first[1] - second[1])
                <= float(config.max_translation_m)
                and abs(wrap_deg(first[2] - second[2]))
                <= float(config.max_yaw_deg)
            )

        clusters: List[List[Dict[str, Any]]] = []
        for row in accepted:
            pose = corrected(row)
            owner = next((
                cluster for cluster in clusters
                if all(agree(pose, corrected(old)) for old in cluster)
            ), None)
            if owner is None:
                clusters.append([row])
            else:
                owner.append(row)
        summaries = [{
            "pose": list(corrected(cluster[0])),
            "target_serials": sorted({
                int(row["target_serial"]) for row in cluster
            }),
            "candidate_count": len(cluster),
        } for cluster in clusters]
        for row in accepted:
            row["concurrent_geometry_hypothesis_count"] = len(clusters)
            row["concurrent_geometry_hypotheses"] = summaries
        if len(clusters) != 1:
            summary = {
                "accepted": False,
                "reason": "ambiguous_concurrent_geometry_hypotheses",
                "revisit_evidence_safe": False,
                "pose_correction_safe": False,
                "hypothesis_count": len(clusters),
                "hypotheses": summaries,
                "rows": [],
            }
            self._record_geometry_report(
                summary, phase=phase, frame_index=frame_index
            )
            self.geometry_last_reason = str(summary["reason"])
            return None
        # 同一位姿簇中的候选等价；选视觉/深度分歧最小的一项继续进入跨时段
        # 授权。没有分歧字段的测试替身保持确定性地取第一项。
        winner = min(
            clusters[0],
            key=lambda row: (
                float(row.get("translation_disagreement_m", math.inf)),
                float(row.get("yaw_disagreement_deg", math.inf)),
                int(row.get("target_serial", 0)),
            ),
        )
        winner["concurrent_geometry_unique"] = True
        return winner

    def _commit_validated_loop(
        self,
        result: Any,
        validation: Dict[str, Any],
        frame_index: int,
    ) -> bool:
        """把视觉/深度共识写成一条鲁棒 submap 边，再由位姿图摊误差。"""
        if (
            not validation.get("accepted")
            or not (
                validation.get("pose_correction_safe")
                or validation.get("mapping_loop_pose_safe")
            )
            or self.mapping_state != "mapping"
        ):
            return False
        target_serial = int(validation["target_serial"])
        source_serial = int(self._current_submap_serial())
        validation["source_serial"] = source_serial
        if abs(source_serial - target_serial) < LOOP_MIN_SERIAL_GAP:
            validation.update(
                accepted=False,
                reason="temporally_adjacent_submap",
            )
            return False
        target = self._submap_by_serial(target_serial)
        source = self._submap_by_serial(source_serial)
        if target is None or source is None:
            validation.update(accepted=False, reason="missing_loop_submap")
            return False

        dx_m, dy_m, dyaw_deg = (
            float(value)
            for value in validation.get(
                "fused_correction", validation["geometry_correction"]
            )
        )
        pivot_x, pivot_y = float(self.x), float(self.y)
        angle = math.radians(dyaw_deg)
        offset_x = float(source.origin_x) - pivot_x
        offset_y = float(source.origin_y) - pivot_y
        corrected_source = (
            pivot_x + math.cos(angle) * offset_x
            - math.sin(angle) * offset_y + dx_m,
            pivot_y + math.sin(angle) * offset_x
            + math.cos(angle) * offset_y + dy_m,
            wrap_deg(float(source.origin_yaw_deg) + dyaw_deg),
        )
        submap_relative = relative_pose(target.pose(), corrected_source)
        edge = PoseEdge(
            target_serial,
            source_serial,
            *submap_relative,
            weight_xy=LOOP_EDGE_WEIGHT_XY,
            weight_yaw=LOOP_EDGE_WEIGHT_YAW,
            robust=True,
        )
        self.pose_edges.append(edge)
        self.loops_found += 1
        optimized = self.optimize_pose_graph()
        already_consistent = (
            math.hypot(dx_m, dy_m) <= 0.5 * GRID_RES_M
            and abs(dyaw_deg) <= 0.5
        )
        if not optimized and not already_consistent:
            self.pose_edges.pop()
            self.loops_found -= 1
            validation.update(accepted=False, reason="pose_graph_not_improved")
            return False
        if not optimized:
            self._clear_geometry_history()
        validation.update({
            "edge_measurement_source": "validated_visual_depth_se2_fusion",
            "submap_measurement": list(submap_relative),
            "graph_optimized": bool(optimized),
            "already_consistent": bool(already_consistent),
        })
        if not self._record_revisit_evidence(
            result,
            frame_index,
            geometry_validation=validation,
        ):
            return False
        self.maybe_freeze_mapping(frame_index)
        return True

    @classmethod
    def _mapping_loop_is_safe(
        cls,
        result: Any,
        validation: Dict[str, Any],
    ) -> bool:
        """强闭环可写鲁棒图边，但仍不授权冻结后的单次位姿跳变。"""
        if (
            not validation.get("accepted")
            or not validation.get("revisit_evidence_safe")
            or not validation.get("concurrent_geometry_unique")
            or int(validation.get("unique_window_count", 0)) < 2
            or int(validation.get("consensus_window_count", 0)) < 2
            or int(validation.get("independent_window_count", 0)) < 2
        ):
            return False
        evidence_type = str(
            getattr(result, "evidence_type", "direct_rgbd_geometry")
        )
        if evidence_type not in {
            "direct_rgbd_geometry",
            "place_rgbd_geometry",
        }:
            return False
        quality = cls._mapping_revisit_quality(result)
        if quality is None:
            return False
        if (
            int(quality.get("independent_candidates", 0))
            < MAPPING_REVISIT_MIN_INDEPENDENT
            or float(quality.get("independent_translation_span_m", 0.0))
            < MAPPING_REVISIT_MIN_TRANSLATION_SPAN_M
        ):
            return False
        if evidence_type == "place_rgbd_geometry" and int(
            getattr(result, "place_geometry_independent", 0) or 0
        ) < MAPPING_REVISIT_MIN_INDEPENDENT:
            return False
        correction = validation.get(
            "fused_correction", validation.get("geometry_correction")
        )
        if not isinstance(correction, (list, tuple)) or len(correction) != 3:
            return False
        return (
            math.hypot(float(correction[0]), float(correction[1]))
            <= LOCALIZATION_MAX_POSE_DELTA_M
            and abs(float(correction[2])) <= LOCALIZATION_MAX_YAW_DELTA_DEG
        )

    def _apply_validated_localization(
        self,
        validation: Dict[str, Any],
        frame_index: int,
    ) -> bool:
        """冻结后只修当前位姿；结构摘要前后必须逐字节一致。"""
        if not validation.get("accepted") or self.mapping_state != "localization":
            return False
        if not validation.get("pose_correction_safe"):
            validation["application_blocker"] = (
                "insufficient_independent_geometry_for_pose"
            )
            return False
        dx_m, dy_m, dyaw_deg = (
            float(value)
            for value in validation.get(
                "fused_correction", validation["geometry_correction"]
            )
        )
        digest_before = self._grid_digest()
        self.correct_live_pose(
            self.x + dx_m,
            self.y + dy_m,
            wrap_deg(self.yaw_deg + dyaw_deg),
        )
        if self._grid_digest() != digest_before:
            raise RuntimeError("定位校正意外修改了冻结结构栅格")
        self.localization_accepted += 1
        self.localization_last_validated_frame = int(frame_index)
        self.localization_unvalidated_travel_m = 0.0
        self.localization_unvalidated_turn_deg = 0.0
        self.localization_position_variance_m2 = 0.0
        self.localization_yaw_variance_deg2 = 0.0
        self.localization_state = "localized"
        self.localization_state_reason = "validated_visual_depth_consensus"
        self.localization_status = "geometry_validated"
        self.localization_last_reason = "visual_depth_consensus_applied"
        self._geometry_visual_lease = None
        self._geometry_visual_leases.clear()
        # 当前位姿发生了跳变，旧证据的里程传播链已断，必须从新坐标起算。
        self._clear_geometry_history()
        return True

    @staticmethod
    def _image_frame_number(image_id: str, fallback: int = -1) -> int:
        try:
            return int(str(image_id).rsplit("_", 1)[-1])
        except (TypeError, ValueError):
            return int(fallback)

    @staticmethod
    def _mapping_revisit_quality(result: Any) -> Optional[Dict[str, Any]]:
        """返回通过冻结/回环门的视觉证据摘要。"""
        if result is None:
            return None
        evidence_type = str(getattr(result, "evidence_type", ""))
        # 外观序列只产生候选 ID，绝不具备度量证据。只有 localize() 返回的
        # direct/place RGB-D geometry 结果才能进入冻结、回环和定位门。
        if evidence_type == "appearance_sequence":
            return None
        gap = int(getattr(result, "candidate_frame_gap", 0) or 0)
        independent = int(getattr(result, "independent_candidates", 0) or 0)
        translation_span = float(
            getattr(result, "independent_translation_span_m", 0.0) or 0.0
        )
        above_floor = int(getattr(result, "above_floor_inliers", 0) or 0)
        inliers = int(getattr(result, "inliers", 0) or 0)
        ratio = float(getattr(result, "inlier_ratio", 0.0) or 0.0)
        rmse = float(getattr(result, "rmse_m", math.inf))
        strong_single = (
            evidence_type != "place_rgbd_geometry"
            and independent >= 1
            and above_floor >= MAPPING_REVISIT_SINGLE_MIN_ABOVE_FLOOR_INLIERS
            and inliers >= MAPPING_REVISIT_SINGLE_MIN_INLIERS
            and ratio >= MAPPING_REVISIT_SINGLE_MIN_INLIER_RATIO
            and math.isfinite(rmse)
            and rmse <= MAPPING_REVISIT_SINGLE_MAX_RMSE_M
        )
        if (
            gap < MAPPING_MIN_LOOP_GAP
            or (
                independent < MAPPING_REVISIT_MIN_INDEPENDENT
                and not strong_single
            )
            or above_floor < MAPPING_REVISIT_MIN_ABOVE_FLOOR_INLIERS
            or inliers < MAPPING_REVISIT_MIN_INLIERS
            or ratio < MAPPING_REVISIT_MIN_INLIER_RATIO
            or not math.isfinite(rmse)
            or rmse > MAPPING_REVISIT_MAX_RMSE_M
        ):
            return None
        if (
            evidence_type == "place_rgbd_geometry"
            and (
                independent < MAPPING_REVISIT_MIN_INDEPENDENT
                or int(getattr(result, "place_geometry_independent", 0) or 0)
                < MAPPING_REVISIT_MIN_INDEPENDENT
            )
        ):
            return None
        return {
            "keyframe_id": str(getattr(result, "keyframe_id", "")),
            "keyframe_frame_index": int(
                getattr(result, "keyframe_frame_index", -1)
            ),
            "candidate_frame_gap": gap,
            "independent_candidates": independent,
            "independent_translation_span_m": translation_span,
            "visual_evidence_type": evidence_type,
            "evidence_mode": (
                "independent_history"
                if independent >= MAPPING_REVISIT_MIN_INDEPENDENT
                else "single_history_strong_geometry"
            ),
            "above_floor_inliers": above_floor,
            "inliers": inliers,
            "inlier_ratio": ratio,
            "rmse_m": rmse,
        }

    def _record_revisit_evidence(
        self,
        result: Any,
        frame_index: int,
        *,
        geometry_validation: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """登记视觉与深度共同确认的重访；登记本身不要求改写位姿图。"""
        if self.mapping_state != "mapping":
            return False
        quality = self._mapping_revisit_quality(result)
        if (
            quality is None
            or not geometry_validation
            or not geometry_validation.get("accepted")
            or not geometry_validation.get("revisit_evidence_safe")
        ):
            return False
        frame_no = int(frame_index)
        # 同一段连续视野只登记一次。后续开放区域仍可继续建图，走满独立
        # 时间间隔后可产生新的闭环，而不会把同一门板的连续帧重复投票。
        if self.mapping_revisit_evidence:
            previous = self.mapping_revisit_evidence[-1]
            if frame_no - int(previous.get("frame_index", -10**9)) < MAPPING_EVIDENCE_FRAME_GAP:
                return False
        validation_summary = {
            key: value
            for key, value in geometry_validation.items()
            if key != "rows"
        }
        visual_mode = str(quality.pop("evidence_mode", ""))
        self.mapping_revisit_evidence.append({
            "frame_index": frame_no,
            **quality,
            "evidence_mode": "validated_visual_depth_geometry",
            "visual_evidence_mode": visual_mode,
            "structural_map_mutated": bool(
                geometry_validation.get("graph_optimized")
            ),
            "geometry_validation": validation_summary,
        })
        self.mapping_freeze_candidate_frame = frame_no
        return True

    def maybe_freeze_mapping(self, frame_index: Optional[int] = None) -> bool:
        """在证据充分且增长趋稳时单向关闭结构写入。"""
        if self.mapping_state != "mapping":
            return self.mapping_state == "localization"
        if self.mapping_observations < MAPPING_MIN_OBSERVATIONS:
            return False
        if self.mapping_travel_m < MAPPING_MIN_TRAVEL_M:
            return False
        if self.mapping_visual_keyframes < MAPPING_MIN_VISUAL_KEYFRAMES:
            return False
        evidence = self.mapping_revisit_evidence
        if len(evidence) < MAPPING_MIN_REVISIT_EVIDENCE:
            return False
        if len(evidence) >= 2 and (
            int(evidence[-1].get("frame_index", -10**9))
            - int(evidence[-2].get("frame_index", -10**9))
            < MAPPING_EVIDENCE_FRAME_GAP
        ):
            return False
        frame_no = int(
            frame_index if frame_index is not None else self.mapping_observations
        )
        candidate_frame = self.mapping_freeze_candidate_frame
        if candidate_frame is None:
            candidate_frame = int(evidence[-1].get("frame_index", frame_no))
            self.mapping_freeze_candidate_frame = candidate_frame
        if frame_no - int(candidate_frame) < MAPPING_CANDIDATE_HOLD_FRAMES:
            return False
        history = self.mapping_growth_history
        growth = 0
        ratio = math.inf
        if len(history) >= MAPPING_GROWTH_WINDOW:
            first = int(history[-MAPPING_GROWTH_WINDOW])
            last = int(history[-1])
            growth = max(0, last - first)
            ratio = growth / max(1, last)
            stable = (
                ratio <= MAPPING_MAX_GROWTH_RATIO
                and growth <= MAPPING_MAX_GROWTH_CELLS
            )
        else:
            stable = False
        latest_evidence = evidence[-1]
        independent_revisit = (
            str(latest_evidence.get("evidence_mode", ""))
            == "validated_visual_depth_geometry"
            and bool(
                (latest_evidence.get("geometry_validation") or {}).get(
                    "accepted"
                )
            )
        )
        # 绝对新增格数来自 30 帧滑窗，会在机器人已经返回旧区域后继续携带
        # 先前扩张的尾巴。多个时间、位置均独立的历史视点已经确认这是重访时，
        # 允许低新增比例覆盖这个滞后的绝对数；frontier 和轨迹覆盖仍在下面
        # 独立把关。单历史强匹配不能走这条路。
        growth_override = (
            not stable
            and independent_revisit
            and math.isfinite(ratio)
            and ratio <= MAPPING_MAX_GROWTH_RATIO
        )
        idle_ready = not (
            MAPPING_IDLE_HOLD_FRAMES > 0
            and self.mapping_idle_observations < MAPPING_IDLE_HOLD_FRAMES
        )
        coverage = self._mapping_coverage_metrics()
        frontier_complete = bool(coverage.get("frontier_complete", False))
        frontier_signature = str(
            coverage.get("reachable_frontier_signature", "") or ""
        )
        frontier_quiescent = False
        if frontier_complete:
            self.mapping_frontier_complete_observations += 1
            # 下一次重新打开未知区域时从第一份签名重新计数。
            self.mapping_frontier_signature = ""
            self.mapping_frontier_quiescence_observations = 0
        elif frontier_signature:
            self.mapping_frontier_complete_observations = 0
            if frontier_signature == self.mapping_frontier_signature:
                self.mapping_frontier_quiescence_observations += 1
            else:
                self.mapping_frontier_signature = frontier_signature
                self.mapping_frontier_quiescence_observations = 1
            # 前沿稳定只说明机器人没有继续探索，不说明未知区域不存在。
            # 保留计数供调度器决定下一步探索目标，但绝不能据此关闭建图。
            frontier_quiescent = (
                self.mapping_frontier_quiescence_observations
                >= MAPPING_FRONTIER_QUIESCENCE_FRAMES
                and self.mapping_idle_observations
                >= MAPPING_FRONTIER_QUIESCENCE_FRAMES
                and (stable or growth_override)
            )
        else:
            self.mapping_frontier_complete_observations = 0
        blockers = []
        if not stable and not growth_override:
            blockers.append("map_growth_not_stable")
        if not idle_ready:
            blockers.append("idle_hold")
        if not frontier_complete:
            blockers.append("reachable_frontier")
        elif (
            self.mapping_frontier_complete_observations
            < MAPPING_FRONTIER_COMPLETE_HOLD_FRAMES
        ):
            blockers.append("frontier_completion_hold")
        if coverage["trajectory_covered_ratio"] < MAPPING_MIN_TRAJECTORY_COVERAGE:
            blockers.append("trajectory_coverage")
        self.mapping_freeze_checks.append({
            "frame_index": frame_no,
            "candidate_age": frame_no - int(candidate_frame),
            "blockers": blockers,
            "growth_cells": int(growth),
            "growth_ratio": (
                None if not math.isfinite(ratio) else float(ratio)
            ),
            "growth_override_by_independent_revisit": bool(growth_override),
            "frontier_ratio": float(coverage["frontier_ratio"]),
            "reachable_frontier_clusters": int(
                coverage.get("reachable_frontier_clusters", 0)
            ),
            "reachable_frontier_lengths_m": list(
                coverage.get("reachable_frontier_lengths_m", ())
            ),
            "reachable_frontier_signature": frontier_signature,
            "pose_to_navigable_seed_m": coverage.get(
                "pose_to_navigable_seed_m"
            ),
            "frontier_seed_components": int(
                coverage.get("frontier_seed_components", 0)
            ),
            "current_pose_seed_valid": bool(
                coverage.get("current_pose_seed_valid", False)
            ),
            "frontier_complete_observations": int(
                self.mapping_frontier_complete_observations
            ),
            "frontier_quiescence_observations": int(
                self.mapping_frontier_quiescence_observations
            ),
            "frontier_quiescent": bool(frontier_quiescent),
            # quiescent 只是诊断量；任何可达未知前沿都继续阻止自动冻结。
            "unexplored_frontier": bool(not frontier_complete),
            "trajectory_covered_ratio": float(
                coverage["trajectory_covered_ratio"]
            ),
            "idle_observations": int(self.mapping_idle_observations),
        })
        if len(self.mapping_freeze_checks) > 64:
            del self.mapping_freeze_checks[:-64]
        if blockers:
            return False
        reason = (
            "evidence_stable"
            f":observations={self.mapping_observations}"
            f":travel_m={self.mapping_travel_m:.2f}"
            f":revisits={len(evidence)}"
            f":candidate_age={frame_no - int(candidate_frame)}"
            f":idle={self.mapping_idle_observations}"
            f":coverage={coverage['trajectory_covered_ratio']:.3f}"
            f":frontier={coverage['frontier_ratio']:.3f}"
            f":frontier_hold={self.mapping_frontier_complete_observations}"
            f":frontier_quiescent={int(frontier_quiescent)}"
        )
        return self.freeze_mapping(frame_index=frame_index, reason=reason)

    def freeze_mapping(
        self,
        *,
        frame_index: Optional[int] = None,
        reason: str = "",
    ) -> bool:
        """一次性切换到定位态；切换后禁止任何栅格/submap 写入。"""
        if self.mapping_state == "localization":
            return False
        if self.mapping_state != "mapping":
            raise RuntimeError(f"未知建图状态: {self.mapping_state}")
        if self.grid.frames <= 0:
            return False
        # 冻结是结构写入的单向边界，不能在边界上顺手引入一个此前未验证的
        # 全局形变。尚未成熟的尾 submap 只定稿，不做新的 scan-only 回环；
        # 已经成熟的 submap 仍会在 advance_submaps() 中按原流程检测回环。
        for submap in list(self.live_submaps()):
            submap.finished = True
        self.mapping_state = "localization"
        self.freeze_frame = None if frame_index is None else int(frame_index)
        self.freeze_reason = str(reason or "explicit")
        self.freeze_grid_revision = int(getattr(self.grid, "_rev", 0))
        self.freeze_grid_digest = self._grid_digest()
        # 只读不是删除 submap；保留它们供诊断，但不再滚动或优化。
        for submap in self.live_submaps():
            submap.finished = True
        self.localization_status = "propagating_adjacent_odometry"
        self.localization_state = "localized"
        self.localization_state_reason = "mapping_frozen_at_current_pose"
        self.localization_position_variance_m2 = 0.0
        self.localization_yaw_variance_deg2 = 0.0
        self._geometry_visual_lease = None
        self._geometry_visual_leases.clear()
        tracker = self._visual_localizer
        seal = getattr(tracker, "seal_place_database", None)
        if callable(seal):
            seal()
        # 冻结没有改变地图坐标系，映射期的第一段重访仍可和冻结后的独立
        # 观测段组成定位证据；只清掉含旧结构写入时刻的滚动点云。
        self._clear_geometry_history(clear_pose_evidence=False)
        return True

    def commit_visual_keyframe(
        self,
        bundle: Optional[Dict[str, Any]],
        image_id: str,
        frame_index: int,
        *,
        force_keyframe: bool = False,
        image_features: Optional[Any] = None,
        feature_extractor: Optional[Any] = None,
    ) -> bool:
        """在结构帧成功融合后，按最终位姿提交长期视觉关键帧。"""
        if self.mapping_state != "mapping" or not bundle:
            return False
        # 闭环刚发生时跳过同一返程视野，避免把高度相关的重复帧塞进长期库。
        # 如果开放边界阻止了冻结，间隔期后恢复关键帧提交，才能继续覆盖任意
        # 大场景；永久停更会让长 episode 后半段失去定位锚点。
        if self.mapping_revisit_evidence:
            last_evidence_frame = int(
                self.mapping_revisit_evidence[-1].get("frame_index", -10**9)
            )
            if int(frame_index) - last_evidence_frame < MAPPING_EVIDENCE_FRAME_GAP:
                return False
        rgb = bundle.get("rgb")
        depth = bundle.get("depth")
        camera = bundle.get("camera")
        if rgb is None or depth is None or not isinstance(camera, dict):
            return False
        tracker = self._visual_tracker(feature_extractor=feature_extractor)
        shared_features = image_features
        source_backend = str(getattr(shared_features, "backend", ""))
        if (
            shared_features is not None
            and source_backend
            and source_backend != str(tracker.feature_backend)
        ):
            shared_features = None
        # 地点检索需要稠密时间序列，度量匹配需要稀疏、低相关的 RGB-D 锚点。
        # 两者共享同一次 CUDA 提取缓存，但分别维护集合，避免检索密度改变建图。
        place_added = tracker.add_place_keyframe(
            str(image_id),
            rgb,
            depth,
            camera,
            (self.x, self.y, self.yaw_deg),
            frame_index=int(frame_index),
            force=force_keyframe,
        )
        metric_added = tracker.add_keyframe(
            str(image_id),
            rgb,
            depth,
            camera,
            (self.x, self.y, self.yaw_deg),
            frame_index=int(frame_index),
            force=force_keyframe,
            image_features=shared_features,
            appearance_features=image_features,
        )
        if metric_added:
            self.mapping_visual_keyframes = tracker.keyframe_count
            self.mapping_last_keyframe_frame = int(frame_index)
        if place_added or metric_added:
            self._remember_visual_keyframe(str(image_id), tracker)
        return bool(place_added or metric_added)

    def observe_visual_frame(
        self,
        bundle: Optional[Dict[str, Any]],
        image_id: str,
        frame_index: int,
        *,
        force_keyframe: bool = False,
        image_features: Optional[Any] = None,
        feature_extractor: Optional[Any] = None,
        commit_mapping_keyframe: bool = True,
    ) -> Optional[Any]:
        """探测长期外观候选，并用独立深度几何决定闭环或定位校正。

        在线建图传 ``commit_mapping_keyframe=False``，等结构融合成功后再调用
        :meth:`commit_visual_keyframe`。默认值保留旧调用者的一步式行为。
        """
        if not bundle:
            return None
        rgb = bundle.get("rgb")
        depth = bundle.get("depth")
        camera = bundle.get("camera")
        if rgb is None or depth is None or not isinstance(camera, dict):
            self.localization_last_reason = "missing_rgbd_bundle"
            return None
        frame_no = int(frame_index)
        # 每帧都积累短滚动几何；昂贵的 CUDA 搜索只在稀疏长期候选出现后运行。
        self._remember_geometry_bundle(bundle, frame_no)
        tracker = self._visual_tracker(feature_extractor=feature_extractor)
        observe_appearance = getattr(tracker, "observe_appearance", None)
        appearance_candidates: Tuple[Any, ...] = ()
        if callable(observe_appearance):
            appearance_candidates = tuple(observe_appearance(
                str(image_id),
                rgb,
                depth,
                camera,
                frame_index=frame_no,
                min_frame_gap=(
                    MAPPING_MIN_LOOP_GAP
                    if self.mapping_state == "mapping"
                    else LOCALIZATION_KEYFRAME_GAP
                ),
            ))
        shared_features = image_features
        source_backend = str(getattr(shared_features, "backend", ""))
        if (
            shared_features is not None
            and source_backend
            and source_backend != str(tracker.feature_backend)
        ):
            shared_features = None
        if self.mapping_state == "mapping":
            self._update_mapping_activity()
            should_probe = (
                tracker.keyframe_count >= MAPPING_MIN_VISUAL_KEYFRAMES
                and (
                    self.mapping_last_visual_probe_frame is None
                    or frame_no - self.mapping_last_visual_probe_frame
                    >= LOCALIZATION_PROBE_INTERVAL
                )
            )
            if not should_probe:
                self.maybe_freeze_mapping(frame_no)
                if commit_mapping_keyframe and self.mapping_state == "mapping":
                    self.commit_visual_keyframe(
                        bundle,
                        image_id,
                        frame_no,
                        force_keyframe=force_keyframe,
                        image_features=image_features,
                        feature_extractor=feature_extractor,
                    )
                return None
            self.mapping_last_visual_probe_frame = frame_no
            result = tracker.localize(
                str(image_id),
                rgb,
                depth,
                camera,
                predicted_pose=(self.x, self.y, self.yaw_deg),
                frame_index=frame_no,
                min_frame_gap=MAPPING_MIN_LOOP_GAP,
                max_pose_jump_m=LOCALIZATION_MAX_POSE_DELTA_M,
                max_pose_yaw_jump_deg=LOCALIZATION_MAX_YAW_DELTA_DEG,
                image_features=shared_features,
                place_candidates=appearance_candidates,
            )
            recent_evidence = bool(
                self.mapping_revisit_evidence
                and frame_no - int(
                    self.mapping_revisit_evidence[-1].get(
                        "frame_index", -10**9
                    )
                ) < MAPPING_EVIDENCE_FRAME_GAP
            )
            proposals = []
            proposal_ids = set()
            if self._mapping_revisit_quality(result) is not None:
                proposals.append(result)
                proposal_ids.add(str(result.keyframe_id))
            if proposals and not recent_evidence:
                predicted = (float(self.x), float(self.y), float(self.yaw_deg))
                source_serial = int(self._current_submap_serial())
                validations = []
                proposal_for_validation: Dict[int, Any] = {}
                for proposal in proposals:
                    anchor = self._visual_keyframe_anchors.get(
                        str(proposal.keyframe_id)
                    )
                    self.mapping_geometry_attempts += 1
                    if anchor is None:
                        validation = {
                            "accepted": False,
                            "reason": "missing_keyframe_submap_anchor",
                            "rows": [],
                        }
                        self._record_geometry_report(
                            validation,
                            phase="mapping",
                            frame_index=frame_no,
                        )
                    else:
                        target_serial = int(anchor[0])
                        if abs(source_serial - target_serial) < LOOP_MIN_SERIAL_GAP:
                            validation = {
                                "accepted": False,
                                "reason": "temporally_adjacent_submap",
                                "target_serial": target_serial,
                                "source_serial": source_serial,
                                "rows": [],
                            }
                            self._record_geometry_report(
                                validation,
                                phase="mapping",
                                frame_index=frame_no,
                            )
                        else:
                            visual_correction = (
                                float(proposal.x_m - predicted[0]),
                                float(proposal.y_m - predicted[1]),
                                float(wrap_deg(
                                    proposal.yaw_deg - predicted[2]
                                )),
                            )
                            validation = self._validate_geometry_target(
                                target_serial,
                                visual_correction,
                                frame_index=frame_no,
                                phase="mapping",
                                visual_observability={
                                    "candidate_count": int(getattr(
                                        proposal, "candidate_count", 0
                                    ) or 0),
                                    "independent_candidates": int(getattr(
                                        proposal, "independent_candidates", 0
                                    ) or 0),
                                    "geometry_major_span_m": float(getattr(
                                        proposal, "geometry_major_span_m", 0.0
                                    ) or 0.0),
                                    "geometry_minor_span_m": float(getattr(
                                        proposal, "geometry_minor_span_m", 0.0
                                    ) or 0.0),
                                    "pixel_coverage_x": float(getattr(
                                        proposal, "pixel_coverage_x", 0.0
                                    ) or 0.0),
                                    "pixel_coverage_y": float(getattr(
                                        proposal, "pixel_coverage_y", 0.0
                                    ) or 0.0),
                                    "translation_std_m": float(getattr(
                                        proposal, "translation_std_m", math.inf
                                    )),
                                    "yaw_std_deg": float(getattr(
                                        proposal, "yaw_std_deg", math.inf
                                    )),
                                    "normal_matrix_condition": float(getattr(
                                        proposal,
                                        "normal_matrix_condition",
                                        math.inf,
                                    )),
                                },
                            )
                            validation["source_serial"] = source_serial
                    validation["visual_candidate"] = {
                        "keyframe_id": str(proposal.keyframe_id),
                        "evidence_type": str(getattr(
                            proposal, "evidence_type", "direct_rgbd_geometry"
                        )),
                        "appearance_score": float(getattr(
                            proposal, "appearance_score", 0.0
                        ) or 0.0),
                    }
                    validations.append(validation)
                    proposal_for_validation[id(validation)] = proposal
                validation = self._select_concurrent_geometry(
                    validations,
                    predicted,
                    frame_index=frame_no,
                    phase="mapping",
                )
                handled = False
                if validation is not None:
                    selected_proposal = proposal_for_validation[id(validation)]
                    self._authorize_temporal_pose(
                        validation, predicted, frame_no
                    )
                    validation["mapping_loop_pose_safe"] = (
                        self._mapping_loop_is_safe(
                            selected_proposal, validation
                        )
                    )
                    if (
                        validation.get("pose_correction_safe")
                        or validation.get("mapping_loop_pose_safe")
                    ):
                        handled = self._commit_validated_loop(
                            selected_proposal, validation, frame_no
                        )
                    else:
                        handled = self._record_revisit_evidence(
                            selected_proposal,
                            frame_no,
                            geometry_validation=validation,
                        )
                        if handled:
                            self.maybe_freeze_mapping(frame_no)
                if handled:
                    self.mapping_geometry_accepted += 1
                else:
                    self.mapping_geometry_rejected += 1
                    if validation is not None:
                        self.geometry_last_reason = str(
                            validation.get("reason") or "geometry_rejected"
                        )
            self.maybe_freeze_mapping(frame_no)
            if commit_mapping_keyframe and self.mapping_state == "mapping":
                self.commit_visual_keyframe(
                    bundle,
                    image_id,
                    frame_no,
                    force_keyframe=force_keyframe,
                    image_features=image_features,
                    feature_extractor=feature_extractor,
                )
            return result

        if self.mapping_state != "localization":
            self.localization_last_reason = "invalid_mapping_state"
            return None
        self.localization_detail_frames += 1
        if (
            self.mapping_last_visual_probe_frame is not None
            and frame_no - self.mapping_last_visual_probe_frame
            < LOCALIZATION_PROBE_INTERVAL
        ):
            return None
        self.mapping_last_visual_probe_frame = frame_no
        attempts_before = tracker.attempts
        result = tracker.localize(
            str(image_id),
            rgb,
            depth,
            camera,
            predicted_pose=(self.x, self.y, self.yaw_deg),
            frame_index=frame_no,
            min_frame_gap=LOCALIZATION_KEYFRAME_GAP,
            # 单张历史关键帧只能提出候选；真正应用还必须通过下面三次、
            # 跨平移视点的一致性保持门，不能凭一次相似外观跳位姿。
            min_independent=1,
            allow_single_strong=True,
            single_min_inliers=24,
            single_min_ratio=0.65,
            single_min_above_floor=8,
            max_pose_jump_m=LOCALIZATION_MAX_POSE_DELTA_M,
            max_pose_yaw_jump_deg=LOCALIZATION_MAX_YAW_DELTA_DEG,
            image_features=shared_features,
            place_candidates=appearance_candidates,
        )
        self.localization_attempts += max(0, tracker.attempts - attempts_before)
        self.localization_last_reason = str(tracker.last_reason or "")
        predicted = (float(self.x), float(self.y), float(self.yaw_deg))
        metric_keyframe_id = (
            "" if result is None else str(result.keyframe_id)
        )
        # 序列外观候选只提供历史子图和粗搜索中心；它本身没有米制度量权。
        # 与度量结果指向同一关键帧时，以后者覆盖，避免把更强证据降级。
        for candidate in appearance_candidates:
            if str(candidate.keyframe_id) == metric_keyframe_id:
                continue
            self._start_visual_geometry_lease(candidate, predicted, frame_no)
        if result is not None:
            self._start_visual_geometry_lease(result, predicted, frame_no)
        leased = self._leased_visual_corrections(frame_no)
        if not leased:
            self.localization_status = "propagating_adjacent_odometry"
            if self.localization_state == "ambiguous":
                self._refresh_localization_state("ambiguity_expired")
            return result
        validations = []
        seen_hypotheses = set()
        for lease, visual_correction in leased:
            target_serial = int(lease["target_serial"])
            key = (
                target_serial,
                round(float(visual_correction[0]), 2),
                round(float(visual_correction[1]), 2),
                round(float(visual_correction[2]), 1),
            )
            if key in seen_hypotheses:
                continue
            seen_hypotheses.add(key)
            self.localization_geometry_attempts += 1
            validation = self._validate_geometry_target(
                target_serial,
                visual_correction,
                frame_index=frame_no,
                phase="localization",
                visual_metric=bool(lease.get("visual_metric", True)),
                visual_observability=self._leased_visual_observability(lease),
            )
            validation["visual_candidate"] = {
                "keyframe_id": str(lease["keyframe_id"]),
                "evidence_type": str(lease["evidence_type"]),
                "appearance_score": float(lease["appearance_score"]),
                "sequence_observations": int(lease["sequence_observations"]),
                "visual_metric": bool(lease.get("visual_metric", True)),
            }
            validations.append(validation)
        selected = self._select_concurrent_geometry(
            validations,
            predicted,
            frame_index=frame_no,
            phase="localization",
        )
        if selected is not None:
            self._authorize_temporal_pose(selected, predicted, frame_no)
        if (
            selected is None
            or not self._apply_validated_localization(selected, frame_no)
        ):
            self.localization_rejected += 1
            self.localization_status = "uncertain_geometry"
            report = selected or (validations[-1] if validations else {})
            if self.geometry_last_reason == (
                "ambiguous_concurrent_geometry_hypotheses"
            ):
                self.localization_state = "ambiguous"
                self.localization_state_reason = self.geometry_last_reason
                self.localization_ambiguous_observations += 1
            elif self.localization_state != "ambiguous":
                self._refresh_localization_state("geometry_not_yet_authorized")
            self.localization_last_reason = (
                "geometry_rejected:"
                f"{report.get('application_blocker') or self.geometry_last_reason or report.get('reason', 'unknown')}"
            )
        return result

    def mark_place(
        self,
        label: str,
        *,
        at: Optional[Tuple[float, float]] = None,
        merge_m: Optional[float] = None,
        submap_serial: Optional[int] = None,
    ) -> Optional[MapEvent]:
        """起个中文地名标在地图上。at 缺省是脚下，也可以给地图系 (x, y)。

        重名一律当成同一个地方挪位置。不同名字只有在**同类**标记挨得很近时
        才算改名：房间名之间可以互相顶掉，但点选出来的物体和脚下的地名不能
        互相吞——它们本来就常常只隔一两米。
        """
        label = str(label or "").strip()[:12]
        if not label:
            return None
        picked = at is not None
        mark_x, mark_y = (float(at[0]), float(at[1])) if picked else (self.x, self.y)
        owner = (
            int(submap_serial)
            if submap_serial is not None
            else self._current_submap_serial()
        )
        kind = "picked" if picked else "named"
        radius = float(
            merge_m if merge_m is not None
            else (PICK_MERGE_M if picked else PLACE_MERGE_M)
        )
        mark_active_session(self.session_id)
        for existing in self.places:
            if existing.label != label:
                if existing.kind != kind:
                    continue
                if math.hypot(existing.x - mark_x, existing.y - mark_y) > radius:
                    continue
            existing.x, existing.y = mark_x, mark_y
            existing.label = label
            existing.kind = kind
            existing.count += 1
            existing.submap_serial = owner
            return existing
        place = MapEvent(
            x=mark_x,
            y=mark_y,
            kind=kind,
            label=label,
            submap_serial=owner,
        )
        self.places.append(place)
        return place

    def note_image_pose(self, image_id: str) -> None:
        """记下这张图是在哪个位姿拍的，之后点选反解要按当时的位姿换算。"""
        key = str(image_id or "").strip()
        if not key:
            return
        self.image_poses[key] = (self.x, self.y, self.yaw_deg)
        self.image_pose_submaps[key] = self._current_submap_serial()
        if len(self.image_poses) > IMAGE_POSE_KEEP:
            for stale in list(self.image_poses)[:-IMAGE_POSE_KEEP]:
                self.image_poses.pop(stale, None)
                self.image_pose_submaps.pop(stale, None)

    def robot_xy_to_map(
        self,
        rx: float,
        ry: float,
        *,
        image_id: str = "",
    ) -> Tuple[float, float]:
        """机体系 XY → 地图系 XY，按拍这张图时的位姿换算。

        点选反解出来的坐标是拍照那一刻的机体系；如果拍完又走了几步，
        套用当前位姿会把物体标错地方。
        """
        pose = self.image_poses.get(str(image_id or "").strip())
        x, y, yaw_deg = pose if pose else (self.x, self.y, self.yaw_deg)
        yaw = math.radians(yaw_deg)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        return x + cos_y * rx - sin_y * ry, y + sin_y * rx + cos_y * ry

    def mark_event(self, kind: str, label: str, *, merge_m: float = 1.0) -> None:
        """在当前里程位姿记一个事件点（抓到/放下/卡住）。"""
        for existing in self.events:
            if existing.kind != kind:
                continue
            if math.hypot(existing.x - self.x, existing.y - self.y) <= merge_m:
                existing.count += 1
                return
        self.events.append(
            MapEvent(
                x=self.x,
                y=self.y,
                kind=str(kind),
                label=str(label),
                submap_serial=self._current_submap_serial(),
            )
        )

    def _append_trajectory_sample(
        self,
        *,
        source: str = "",
        force: bool = False,
        tool: str = "",
        image_id: str = "",
        submap_serial: Optional[int] = None,
    ) -> None:
        """记录完整运动轨迹；这条路径永远不参与地图更新。"""
        point = TrailPoint(
            x=float(self.x),
            y=float(self.y),
            yaw_deg=float(self.yaw_deg),
            tool=str(tool or ""),
            image_id=str(image_id or ""),
            source=str(source or ""),
            submap_serial=(
                self._current_submap_serial()
                if submap_serial is None
                else int(submap_serial)
            ),
        )
        previous = self._last_trajectory_sample
        if previous is not None and not force:
            distance = math.hypot(point.x - previous.x, point.y - previous.y)
            yaw_delta = abs(wrap_deg(point.yaw_deg - previous.yaw_deg))
            if distance < TRAJECTORY_SAMPLE_STEP_M and yaw_delta < TRAJECTORY_SAMPLE_YAW_DEG:
                return
        self.trajectory_samples.append(point)
        self._last_trajectory_sample = point

    def full_trail(self) -> List[TrailPoint]:
        """返回从 episode 起点开始的完整轨迹，不截断早期路径。"""
        if self.trajectory_samples:
            points = list(self.trajectory_samples)
            if math.hypot(points[-1].x - self.x, points[-1].y - self.y) > 1e-3:
                points.append(TrailPoint(
                    x=self.x,
                    y=self.y,
                    yaw_deg=self.yaw_deg,
                    source="current_pose",
                    submap_serial=self._current_submap_serial(),
                ))
            return points
        # 兼容只调用 apply_motion 的旧离线消费者。
        points = list(self.trail)
        if points and math.hypot(points[-1].x - self.x, points[-1].y - self.y) > 1e-3:
            points.append(TrailPoint(
                x=self.x,
                y=self.y,
                yaw_deg=self.yaw_deg,
                source="current_pose",
                submap_serial=self._current_submap_serial(),
            ))
        return points

    def recent_trail(self, moves: int = TRAIL_RECENT_MOVES) -> List[TrailPoint]:
        """最近 N 次底盘移动的折线；更早的行踪切掉，避免图上糊成一团。"""
        keep = max(1, int(moves)) + 1
        points = list(self.trail[-keep:])
        # 实时里程下当前位姿已经走过最后一个拐点，补上让线连到机器人
        if points and math.hypot(points[-1].x - self.x, points[-1].y - self.y) > 1e-3:
            points.append(TrailPoint(
                x=self.x,
                y=self.y,
                yaw_deg=self.yaw_deg,
                source="live",
                submap_serial=self._current_submap_serial(),
            ))
        return points

    def sparse_trail(self, min_step_m: float = TRAIL_MIN_STEP_M) -> List[TrailPoint]:
        """抽稀成大概行踪，避免原地转圈把地图糊满。"""
        if not self.trail:
            return []
        kept = [self.trail[0]]
        for point in self.trail[1:]:
            last = kept[-1]
            if math.hypot(point.x - last.x, point.y - last.y) >= float(min_step_m):
                kept.append(point)
        if kept[-1] is not self.trail[-1]:
            kept.append(self.trail[-1])
        return kept

    def integrate_capture(
        self,
        bundle: Optional[Dict[str, Any]],
        *,
        image_id: str = "",
        stride: int = DEPTH_STRIDE,
        scan_match: bool = True,
        match_translation: bool = True,
    ) -> bool:
        """把一帧 head depth 投影进占用栅格。只用 depth + cam_rel_pose。"""
        if not bundle:
            return False
        if self.mapping_state != "mapping":
            # 冻结后的 RGB-D 只供视觉定位/细节缓存，不能改变结构栅格。
            return False
        key = str(image_id or "")
        if key and key in self.integrated_images:
            return False
        points = _depth_to_robot_points(bundle, stride=stride)
        if points is None or points.size == 0:
            return False
        if key:
            self.integrated_images.add(key)
        return self.integrate_points(
            points,
            scan_match=scan_match,
            match_translation=match_translation,
        )

    def integrate_detail_capture(
        self,
        bundle: Optional[Dict[str, Any]],
        *,
        stride: int = DEPTH_STRIDE,
    ) -> bool:
        """冻结后只更新短时观测层，不动结构、submap 或位姿图。"""
        if self.mapping_state != "localization" or not bundle:
            return False
        points = _depth_to_robot_points(bundle, stride=stride)
        if points is None or points.size == 0:
            return False
        radial = np.hypot(points[:, 0], points[:, 1])
        usable = (radial >= MIN_RANGE_M) & (radial <= MAX_RANGE_M)
        usable &= ~(
            (points[:, 0] <= SELF_CLEAR_FORWARD_M)
            & (np.abs(points[:, 1]) <= SELF_CLEAR_HALF_WIDTH_M)
            & (points[:, 2] > FLOOR_Z_MAX_M)
        )
        points = points[usable]
        if points.size == 0:
            return False
        self._write_frame(
            self.grid,
            self.x,
            self.y,
            self.yaw_deg,
            points,
            structural=False,
        )
        return True

    def integrate_points(
        self,
        points: np.ndarray,
        *,
        scan_match: bool = True,
        match_translation: bool = True,
    ) -> bool:
        """把一帧机体系点云并入占用栅格。"""
        if self.mapping_state != "mapping":
            return False
        if points is None or points.size == 0:
            return False
        radial = np.hypot(points[:, 0], points[:, 1])
        usable = (radial >= MIN_RANGE_M) & (radial <= MAX_RANGE_M)
        # 身前一臂范围内、离地有高度的点就是自己的手臂和爪子，扔掉；
        # 贴地的点仍然当地板用，免得脚下留一块空洞。扫描匹配也不该看见它们，
        # 因为它们在机体系里不动，会把匹配硬拽向「没移动」。
        usable &= ~(
            (points[:, 0] <= SELF_CLEAR_FORWARD_M)
            & (np.abs(points[:, 1]) <= SELF_CLEAR_HALF_WIDTH_M)
            & (points[:, 2] > FLOOR_Z_MAX_M)
        )
        points = points[usable]
        if points.size == 0:
            return False
        # 先用墙的方向定 yaw，再让栅格匹配只管平移。两者是不同的可观测性：
        # 视场里只有一面墙时，沿墙滑动确实定不出平移，但墙的方向角照样能
        # 量准。混在一起搜的话，平坦的方向会把好不容易定准的角度拖走。
        segments = _scan_segments(self._match_points(points))
        base, live = self.advance_submaps()
        if scan_match:
            self.scan_match_attempts += 1
            self.align_yaw_to_walls(segments)
            # 原地转身时 proprio 已经明确给出平移近似为零。此时不同视角下
            # 的单墙/门框会制造平移伪峰；每次只挪 5cm 仍会在长 spin 里累成
            # 数米随机游走。保留墙方向纠 yaw，但不允许凭深度凭空造平移。
            if match_translation:
                if self._scan_match(points, base, segments, lock_yaw=True):
                    self.scan_match_applied += 1
            else:
                self.scan_match_translation_locked += 1
        self._note_wall_dirs(segments)
        self._note_submap_wall_dirs(live, segments)
        self._write_frame(self.grid, self.x, self.y, self.yaw_deg, points)
        # 同一帧再按各自局部系写进 submap：这份副本能被整体挪动，
        # 是位姿图事后修正历史观测的唯一途径。
        for submap in live:
            self._write_frame(
                submap.grid,
                *submap.to_local(self.x, self.y, self.yaw_deg),
                points,
            )
        if self.grid.frames % DECAY_EVERY_FRAMES == 0:
            self.grid.decay_uncertain()
        self.mapping_observations += 1
        self.mapping_growth_history.append(self._known_cell_count())
        if len(self.mapping_growth_history) > 4 * MAPPING_GROWTH_WINDOW:
            del self.mapping_growth_history[:-4 * MAPPING_GROWTH_WINDOW]
        return True

    @staticmethod
    def _write_frame(
        grid: OccupancyGrid,
        px: float,
        py: float,
        yaw_deg: float,
        points: np.ndarray,
        *,
        structural: bool = True,
    ) -> None:
        """把一帧机体系点云按给定位姿落进结构层或只落短时层。"""
        yaw = math.radians(yaw_deg)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        map_x = px + cos_y * points[:, 0] - sin_y * points[:, 1]
        map_y = py + sin_y * points[:, 0] + cos_y * points[:, 1]
        z = points[:, 2]
        floor = np.abs(z) <= FLOOR_Z_MAX_M
        low = (z >= OBSTACLE_Z_MIN_M) & (z <= CHASSIS_BLOCK_Z_MAX_M)
        high = (z > CHASSIS_BLOCK_Z_MAX_M) & (z <= OBSTACLE_Z_MAX_M)

        # 先清视线再落障碍：同一帧里，射线扫过的格子该是空的，
        # 而端点该是实的。顺序反了会把刚落的墙自己擦掉。
        # 贴地射线可靠，可以拿来清底盘高度；头顶射线可能是从桌面上方
        # 掠过去的，只能清头顶层，否则会把桌子腿连带擦没。
        if np.any(floor):
            # 贴地射线可以一路清到脚下那一格，只留一格余量防墙脚
            grid.carve_rays(
                px,
                py,
                map_x[floor],
                map_y[floor],
                layer="low",
                backoff_m=grid.resolution_m,
                structural=structural,
            )
        if np.any(low):
            grid.carve_rays(
                px,
                py,
                map_x[low],
                map_y[low],
                layer="low",
                structural=structural,
            )
        if np.any(high):
            grid.carve_rays(
                px,
                py,
                map_x[high],
                map_y[high],
                layer="high",
                structural=structural,
            )
        grid.add_floor(map_x[floor], map_y[floor], structural=structural)
        grid.add_obstacle(
            map_x[low],
            map_y[low],
            layer="low",
            zs=z[low],
            structural=structural,
        )
        grid.add_obstacle(
            map_x[high],
            map_y[high],
            layer="high",
            structural=structural,
        )
        grid.recent_frames += 1
        if structural:
            grid.frames += 1

    def live_submaps(self) -> List[Submap]:
        """还在接收观测的 submap，按建立先后排列。"""
        return [s for s in self.submaps if not s.finished]

    def _current_submap_serial(self) -> int:
        """返回当前观测所属的时间 submap；首帧建图前预留 serial 0。"""
        live = self.live_submaps()
        if live:
            return live[-1].serial
        if self.submaps:
            return self.submaps[-1].serial
        return self.next_submap_serial

    def advance_submaps(self) -> Tuple[Submap, List[Submap]]:
        """滚动 submap 队列，返回 (匹配基准, 本帧要写入的所有 submap)。

        始终让两张 submap 同时收观测，进度错开半张。新开的那张几乎是空的，
        拿它当匹配基准等于没有基准，所以基准永远用先建的那张——它已经攒够
        观测，又只累积了几十帧的漂移，比全局图干净得多。
        """
        if not self.submaps:
            self._open_submap()
        live = self.live_submaps()
        if len(live) > 1 and live[0].is_full():
            settled = live[0]
            settled.finished = True
            live = live[1:]
            # 刚定稿的这张不会再变了，正是拿它去找回环的时候
            if self.detect_loop(settled) is not None:
                self.optimize_pose_graph()
        if len(live) == 1 and live[0].ready_for_handoff():
            live.append(self._open_submap())
        return live[0], live

    def _open_submap(self) -> Submap:
        previous = self.submaps[-1] if self.submaps else None
        fresh = Submap(
            self.x, self.y, self.yaw_deg, serial=self.next_submap_serial
        )
        self.next_submap_serial += 1
        self.submaps.append(fresh)
        if previous is not None:
            # 里程边：两张图的原点之差就是这段路的推算结果
            self.pose_edges.append(PoseEdge(
                previous.serial,
                fresh.serial,
                *relative_pose(previous.pose(), fresh.pose()),
                weight_xy=ODOM_EDGE_WEIGHT_XY,
                weight_yaw=ODOM_EDGE_WEIGHT_YAW,
            ))
        while len(self.submaps) > SUBMAP_MAX_COUNT:
            # 丢最老的：它离机器人已经很远，既不会被拿来匹配，回环也够不着。
            # 记个数，别让「能重建」这件事悄悄失效。
            dropped = self.submaps.pop(0)
            self.submaps_dropped += 1
            # 边指向已经不存在的节点就没意义了，一起清掉
            self.pose_edges = [
                e for e in self.pose_edges
                if e.i != dropped.serial and e.j != dropped.serial
            ]
            self._visual_keyframe_anchors = {
                image_id: anchor
                for image_id, anchor in self._visual_keyframe_anchors.items()
                if anchor[0] != dropped.serial
            }
        return fresh

    def detect_loop(self, source: Submap) -> Optional[PoseEdge]:
        """看 source 是不是又回到了某张老 submap 上，是就连一条回环边。

        没有回环，位姿和地图只会一起漂——匹配再准也只保证局部自洽。
        误回环比不回环更糟（会把整张图撕坏），所以宁可漏也不要错。
        """
        points = source.wall_points()
        if points.shape[0] < 40:
            return None
        best: Optional[PoseEdge] = None
        for target in self.submaps:
            if target is source:
                continue
            gap = abs(source.serial - target.serial)
            if gap < LOOP_MIN_SERIAL_GAP:
                # 挨着的几张本来就由里程边连着，再连一条只是同义反复
                continue
            if not self._loop_footprints_overlap(source, target):
                continue
            edge = self._match_submaps(source, target, points)
            if edge is None:
                self.loops_rejected += 1
            elif best is None or edge.weight_xy > best.weight_xy:
                best = edge
        if best is not None:
            self.pose_edges.append(best)
            self.loops_found += 1
        return best

    @staticmethod
    def _loop_footprints_overlap(source: Submap, target: Submap) -> bool:
        """观测足迹能否在 LOOP 搜索窗内重叠，用于廉价候选预筛。"""
        source_bounds = source.wall_bounds_map()
        target_bounds = target.wall_bounds_map()
        if source_bounds is None or target_bounds is None:
            return math.hypot(
                source.origin_x - target.origin_x,
                source.origin_y - target.origin_y,
            ) <= LOOP_SEARCH_RADIUS_M
        sx0, sx1, sy0, sy1 = source_bounds
        tx0, tx1, ty0, ty1 = target_bounds
        gap_x = max(0.0, sx0 - tx1, tx0 - sx1)
        gap_y = max(0.0, sy0 - ty1, ty0 - sy1)
        return math.hypot(gap_x, gap_y) <= LOOP_SEARCH_SPAN_M

    def _match_submaps(
        self, source: Submap, target: Submap, points: np.ndarray
    ) -> Optional[PoseEdge]:
        """把 source 的墙点摆到 target 的栅格上找最佳位姿。

        逐级细化：粗级在降分辨率的膨胀图上扫遍整个窗口，只负责找对山头；
        后面两级在细图上收敛到格点精度。
        """
        pose = relative_pose(target.pose(), source.pose())
        margin = math.inf
        for span_m, step_m, span_deg, step_deg, coarse, limit in LOOP_STAGES:
            grid = target.coarse_grid() if coarse else target.grid
            sampled = points
            if sampled.shape[0] > limit:
                sampled = sampled[::int(math.ceil(sampled.shape[0] / limit))]
            found = self._search_pose(
                grid, sampled, pose,
                span_m=span_m, step_m=step_m,
                span_deg=span_deg, step_deg=step_deg,
            )
            if found is None:
                return None
            pose, score, runner_up = found
            if coarse:
                # 走廊那种到处都对得上的地方，最优和次优几乎一样高。
                # 只在粗级判：细级的搜索窗太小，次优必然贴着最优。
                margin = score / runner_up if runner_up > 0.0 else math.inf
        if margin < LOOP_MIN_PEAK_MARGIN:
            return None
        # 命中率 = 得分 / 「每个点都压在一格典型墙上」。参考强度得取 target
        # 自己墙格的实际值：log-odds 要攒很多帧才饱和，拿理论上限当分母会
        # 把命中率系统性地压低好几倍，真回环全被挡在门外。
        belief = target.grid.low[target.grid.low > 0.0]
        if belief.size == 0:
            return None
        ceiling = sampled.shape[0] * float(np.median(belief))
        if ceiling <= 0.0 or score / ceiling < LOOP_MIN_HIT_RATIO:
            return None
        return PoseEdge(
            target.serial, source.serial, pose[0], pose[1], pose[2],
            weight_xy=LOOP_EDGE_WEIGHT_XY,
            weight_yaw=LOOP_EDGE_WEIGHT_YAW,
            robust=True,
        )

    @staticmethod
    def _search_pose(
        grid: OccupancyGrid,
        points: np.ndarray,
        center: Tuple[float, float, float],
        *,
        span_m: float,
        step_m: float,
        span_deg: float,
        step_deg: float,
    ) -> Optional[Tuple[Tuple[float, float, float], float, float]]:
        """在栅格上网格搜索，返回 (最佳位姿, 得分, 远处次优分)。

        次优分只取离最佳解足够远的那些候选——峰顶旁边一格分数当然也高，
        拿它当次优就永远判不出「这地方到处都对得上」。
        """
        cx, cy, cyaw = center
        bx, by = points[:, 0], points[:, 1]
        lin = np.arange(-span_m, span_m + 1e-9, step_m)
        ang = np.arange(-span_deg, span_deg + 1e-9, step_deg)
        # 先按旧的 yaw -> dx -> dy 顺序生成候选，再一次性交给栅格评分器；
        # 这样 CUDA 路径只改变执行设备，不改变 tie-break 或搜索窗口。
        spots = []
        for dyaw in ang:
            for dx in lin:
                for dy in lin:
                    spots.append((cx + dx, cy + dy, cyaw + dyaw))
        candidates = np.asarray(spots, dtype=np.float64)
        scores = grid.match_score_candidates(bx, by, candidates)
        if scores.size == 0:
            return None
        best_index = int(np.argmax(scores))
        best_score = float(scores[best_index])
        best_pose = tuple(float(value) for value in candidates[best_index])
        if best_score <= 0.0:
            return None
        distances = np.hypot(
            candidates[:, 0] - best_pose[0],
            candidates[:, 1] - best_pose[1],
        )
        far = max(
            scores[distances > 3.0 * step_m].tolist() or [0.0],
            default=0.0,
        )
        return best_pose, best_score, far

    def optimize_pose_graph(self) -> bool:
        """跑一遍位姿图，把回环处的落差摊回整条轨迹，然后重画全局图。"""
        if self.mapping_state != "mapping":
            return False
        if len(self.submaps) < 2 or not self.pose_edges:
            return False
        order = {s.serial: k for k, s in enumerate(self.submaps)}
        edges = [
            PoseEdge(order[e.i], order[e.j], e.dx, e.dy, e.dyaw_deg,
                     weight_xy=e.weight_xy, weight_yaw=e.weight_yaw,
                     robust=e.robust)
            for e in self.pose_edges
            if e.i in order and e.j in order
        ]
        if not any(e.robust for e in edges):
            # 只有里程边时，位姿图的解就是里程本身，跑了也白跑
            return False
        poses = [s.pose() for s in self.submaps]
        before = total_error(poses, edges)
        solved = optimize_graph(poses, edges)
        if total_error(solved, edges) >= before:
            return False
        shifts = [
            relative_pose(old, tuple(new))
            for old, new in zip(poses, solved)
        ]
        for submap, pose in zip(self.submaps, solved):
            submap.set_pose(*pose)
        self._apply_graph_shift(poses, solved, shifts)
        self._sync_visual_keyframe_poses()
        self.graph_optimizations += 1
        self.rebuild_global_from_submaps()
        # 滚动深度点已经按优化前的地图系保存；继续使用会把两个坐标系拼在
        # 同一次搜索里，制造看似很尖的假峰。
        self._clear_geometry_history()
        return True

    def _apply_graph_shift(
        self,
        before: List[Tuple[float, float, float]],
        after: np.ndarray,
        shifts: List[Tuple[float, float, float]],
    ) -> None:
        """submap 挪了，挂在它上面的轨迹、标记和图像位姿一起移动。"""
        if not before:
            return
        origins = np.array([[p[0], p[1]] for p in before], dtype=np.float64)
        order = {submap.serial: k for k, submap in enumerate(self.submaps)}

        def moved(
            x: float,
            y: float,
            yaw_deg: float,
            submap_serial: Optional[int] = None,
        ) -> Tuple[float, float, float]:
            k = order.get(submap_serial) if submap_serial is not None else None
            if k is None:
                # 兼容旧会话：没有时间归属时才退回空间最近邻。
                k = int(np.argmin(np.hypot(origins[:, 0] - x, origins[:, 1] - y)))
            local = relative_pose(before[k], (x, y, yaw_deg))
            angle = math.radians(after[k][2])
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            return (
                after[k][0] + cos_a * local[0] - sin_a * local[1],
                after[k][1] + sin_a * local[0] + cos_a * local[1],
                wrap_deg(after[k][2] + local[2]),
            )

        for point in self.trail:
            point.x, point.y, point.yaw_deg = moved(
                point.x, point.y, point.yaw_deg, point.submap_serial
            )
        for point in self.trajectory_samples:
            point.x, point.y, point.yaw_deg = moved(
                point.x, point.y, point.yaw_deg, point.submap_serial
            )
        for landmark in self.landmarks.values():
            landmark.x, landmark.y, landmark.yaw_deg = moved(
                landmark.x,
                landmark.y,
                landmark.yaw_deg,
                landmark.submap_serial,
            )
        for marker in list(self.places) + list(self.events):
            marker.x, marker.y, _ = moved(
                marker.x, marker.y, 0.0, marker.submap_serial
            )
        for image_id, pose in list(self.image_poses.items()):
            self.image_poses[image_id] = moved(
                *pose, self.image_pose_submaps.get(image_id)
            )
        self.x, self.y, self.yaw_deg = moved(
            self.x,
            self.y,
            self.yaw_deg,
            self._current_submap_serial(),
        )
    def rebuild_global_from_submaps(self) -> bool:
        """按各 submap 当前的位姿重新拼出全局栅格。

        位姿图优化改的是 submap 的位姿，全局图得跟着重画才算数。
        轨迹只用于显示和审计，不会被重新投影或覆盖传感器证据。
        """
        if self.mapping_state != "mapping" or not self.submaps:
            return False
        rebuilt = OccupancyGrid(
            resolution_m=self.grid.resolution_m,
            half_span_m=self.grid.half_span_m,
        )
        for submap in self.submaps:
            source = submap.grid
            for name in ("low", "high", "recent_low", "recent_high"):
                layer = getattr(source, name)
                rows, cols = np.nonzero(layer)
                if rows.size == 0:
                    continue
                lx = (cols + 0.5) * source.resolution_m - source.half_span_m
                ly = (rows + 0.5) * source.resolution_m - source.half_span_m
                angle = math.radians(submap.origin_yaw_deg)
                cos_a, sin_a = math.cos(angle), math.sin(angle)
                mx = submap.origin_x + cos_a * lx - sin_a * ly
                my = submap.origin_y + sin_a * lx + cos_a * ly
                ix, iy = rebuilt._indices(mx, my)
                ok = (ix >= 0) & (ix < rebuilt.n) & (iy >= 0) & (iy < rebuilt.n)
                if not np.any(ok):
                    continue
                target = getattr(rebuilt, name)
                np.add.at(
                    target,
                    (iy[ok], ix[ok]),
                    layer[rows[ok], cols[ok]],
                )
            # 高度带也得跟着搬，否则位姿图一优化，全局图的墙面判据就空了
            rows, cols = np.nonzero(source.bands)
            if rows.size:
                lx = (cols + 0.5) * source.resolution_m - source.half_span_m
                ly = (rows + 0.5) * source.resolution_m - source.half_span_m
                angle = math.radians(submap.origin_yaw_deg)
                cos_a, sin_a = math.cos(angle), math.sin(angle)
                mx = submap.origin_x + cos_a * lx - sin_a * ly
                my = submap.origin_y + sin_a * lx + cos_a * ly
                ix, iy = rebuilt._indices(mx, my)
                ok = (ix >= 0) & (ix < rebuilt.n) & (iy >= 0) & (iy < rebuilt.n)
                if np.any(ok):
                    np.bitwise_or.at(
                        rebuilt.bands,
                        (iy[ok], ix[ok]),
                        source.bands[rows[ok], cols[ok]],
                    )
            rebuilt.frames += source.frames
            rebuilt.recent_frames += source.recent_frames
        np.clip(rebuilt.low, -LOGODDS_CLAMP, LOGODDS_CLAMP, out=rebuilt.low)
        np.clip(rebuilt.high, -LOGODDS_CLAMP, LOGODDS_CLAMP, out=rebuilt.high)
        np.clip(
            rebuilt.recent_low,
            -RECENT_LOGODDS_CLAMP,
            RECENT_LOGODDS_CLAMP,
            out=rebuilt.recent_low,
        )
        np.clip(
            rebuilt.recent_high,
            -RECENT_LOGODDS_CLAMP,
            RECENT_LOGODDS_CLAMP,
            out=rebuilt.recent_high,
        )
        # 轨迹是展示/审计数据，不能在重建时覆盖观测到的自由/占据证据。
        self.grid = rebuilt
        self._rebuild_wall_dir_hist_from_submaps()
        return True

    def _match_points(self, points: np.ndarray) -> np.ndarray:
        """挑出能用来对齐的墙点：底盘高度、近处。

        头顶的桌面家具会被搬走，远处点的角度误差又被距离放大，
        两者都会把对齐带跑偏。定向和匹配都只信这批点。
        """
        if points is None or points.ndim != 2 or points.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float64)
        z = points[:, 2]
        radial = np.hypot(points[:, 0], points[:, 1])
        return points[
            (z >= OBSTACLE_Z_MIN_M)
            & (z <= CHASSIS_BLOCK_Z_MAX_M)
            & (radial <= SCAN_MATCH_MAX_RANGE_M)
        ]

    @staticmethod
    def _accumulate_wall_dirs(
        histogram: np.ndarray,
        segments: List[Tuple[float, float]],
        frame_yaw_deg: float,
    ) -> None:
        """把一帧墙方向按长度累计到指定坐标系的环形直方图。"""
        if not segments:
            return
        histogram *= DIR_HIST_DECAY
        scale = DIR_HIST_BINS / 180.0
        for angle, length in segments:
            # 四舍五入而不是截断：截断会让每一格都系统性地偏半格，
            # 这半格最后原样变成 yaw 的稳态残差。
            index = int(
                round((angle + frame_yaw_deg) % 180.0 * scale)
            ) % DIR_HIST_BINS
            histogram[index] += float(length)

    def _note_wall_dirs(self, segments: List[Tuple[float, float]]) -> None:
        """把本帧线段的地图系方向按长度记进主方向直方图。

        必须在 yaw 修正之后调用，否则等于把当前的朝向误差写进先验里，
        以后就再也纠不回来了。
        """
        self._accumulate_wall_dirs(self.wall_dir_hist, segments, self.yaw_deg)

    def _note_submap_wall_dirs(
        self,
        submaps: Iterable[Submap],
        segments: List[Tuple[float, float]],
    ) -> None:
        """同时保存每张 submap 的局部墙方向，供图优化后无损换坐标。"""
        for submap in submaps:
            histogram = getattr(submap, "wall_dir_hist", None)
            if histogram is None:
                histogram = np.zeros(DIR_HIST_BINS, dtype=np.float64)
                submap.wall_dir_hist = histogram
            self._accumulate_wall_dirs(
                histogram,
                segments,
                wrap_deg(self.yaw_deg - submap.origin_yaw_deg),
            )

    def _rebuild_wall_dir_hist_from_submaps(self) -> None:
        """按优化后的 submap 朝向重建地图系墙方向先验。"""
        rebuilt = np.zeros(DIR_HIST_BINS, dtype=np.float64)
        scale = DIR_HIST_BINS / 180.0
        for submap in self.submaps:
            local = getattr(submap, "wall_dir_hist", None)
            if local is None:
                continue
            indices = np.flatnonzero(local > 0.0)
            if indices.size == 0:
                continue
            world_angles = (
                indices.astype(np.float64) / scale + submap.origin_yaw_deg
            ) % 180.0
            world_indices = np.rint(world_angles * scale).astype(np.int64)
            world_indices %= DIR_HIST_BINS
            np.add.at(rebuilt, world_indices, local[indices])
        # 没有局部证据时必须清空旧先验，宁可暂时不纠 yaw，也不能使用
        # 已经和结构图不在同一坐标系里的缓存。
        self.wall_dir_hist = rebuilt

    def _smoothed_dir_hist(self) -> np.ndarray:
        """环形高斯模糊：墙不会正好落在整度上，不糊一下峰会被切碎。"""
        sigma = DIR_HIST_BLUR_DEG * DIR_HIST_BINS / 180.0
        span = max(1, int(math.ceil(3.0 * sigma)))
        offsets = np.arange(-span, span + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
        kernel /= kernel.sum()
        padded = np.concatenate(
            [self.wall_dir_hist[-span:], self.wall_dir_hist,
             self.wall_dir_hist[:span]]
        )
        return np.convolve(padded, kernel, mode="valid")

    def wall_directions(self) -> np.ndarray:
        """这栋房子的墙都朝哪几个方向（地图系，度）。

        从观测里学，不预设轴对齐：斜着的隔断、L 形料理台都是真实结构，
        硬吸附到 0/90 只会把 yaw 越纠越偏。
        """
        # 证据量看模糊前的累计长度：模糊只是为了把峰定在正确的角度上，
        # 它会把权重摊到邻角去，拿摊薄后的峰值和长度阈值比是量纲不对。
        if float(self.wall_dir_hist.sum()) < DIR_MIN_WEIGHT_M:
            return np.zeros(0, dtype=np.float64)
        hist = self._smoothed_dir_hist()
        peak = float(hist.max()) if hist.size else 0.0
        average = float(hist.mean()) if hist.size else 0.0
        # 方向没成簇就说明这儿是圆弧或一堆杂物，此时任何「主方向」都是噪声
        if peak <= 0.0 or peak < DIR_PEAK_PROMINENCE * average:
            return np.zeros(0, dtype=np.float64)
        left = np.roll(hist, 1)
        right = np.roll(hist, -1)
        is_peak = (hist >= left) & (hist > right) & (hist >= 0.25 * peak)
        index = np.flatnonzero(is_peak)
        if index.size == 0 or index.size > DIR_MAX_PEAKS:
            return np.zeros(0, dtype=np.float64)
        # 抛物线顶点插值：直方图是 1° 一格，光取整格的话这 1° 的量化误差
        # 会原封不动变成 yaw 的稳态残差。
        low = hist[(index - 1) % DIR_HIST_BINS]
        mid = hist[index]
        high = hist[(index + 1) % DIR_HIST_BINS]
        curvature = low - 2.0 * mid + high
        shift = np.where(
            np.abs(curvature) > 1e-12, 0.5 * (low - high) / curvature, 0.0
        )
        offset = index + np.clip(shift, -0.5, 0.5)
        return (offset * (180.0 / DIR_HIST_BINS)) % 180.0

    def align_yaw_to_walls(
        self, segments: List[Tuple[float, float]]
    ) -> Optional[float]:
        """用线段方向和已知墙方向的偏差纠正 yaw，返回实际纠了多少。

        每条线段各投一票（票重是它的长度），取加权中位数——中位数比均值
        抗离群，一两条被家具带偏的线段拽不动结果。
        """
        peaks = self.wall_directions()
        if peaks.size == 0 or not segments:
            return None
        errors: List[float] = []
        weights: List[float] = []
        for angle, length in segments:
            world = (angle + self.yaw_deg) % 180.0
            delta = (peaks - world + 90.0) % 180.0 - 90.0
            k = int(np.argmin(np.abs(delta)))
            if abs(float(delta[k])) > DIR_MATCH_MAX_DEG:
                continue
            errors.append(float(delta[k]))
            weights.append(float(length))
        if not errors or sum(weights) < YAW_FIX_MIN_SUPPORT_M:
            return None
        order = np.argsort(errors)
        sorted_err = np.asarray(errors, dtype=np.float64)[order]
        cumulative = np.cumsum(np.asarray(weights, dtype=np.float64)[order])
        median = float(
            sorted_err[int(np.searchsorted(cumulative, 0.5 * cumulative[-1]))]
        )
        fix = max(-YAW_FIX_MAX_DEG, min(YAW_FIX_MAX_DEG, median))
        if abs(fix) < 1e-6:
            return None
        self.yaw_deg = wrap_deg(self.yaw_deg + fix)
        self.yaw_fixes += 1
        self.yaw_fix_total_deg += fix
        self._sync_latest_trajectory_pose()
        return fix

    @staticmethod
    def _observable_axes(
        segments: List[Tuple[float, float]]
    ) -> Optional[np.ndarray]:
        """机体系里平移可观测的方向，按列给出。None 表示证据不足。

        墙只约束它自己的法线方向；沿墙那一维要靠别的朝向的墙来补。
        """
        info = np.zeros((2, 2), dtype=np.float64)
        for angle_deg, length_m in segments:
            angle = math.radians(angle_deg)
            normal = np.array([-math.sin(angle), math.cos(angle)])
            info += float(length_m) * np.outer(normal, normal)
        if info.trace() <= 1e-9:
            return None
        values, vectors = np.linalg.eigh(info)
        return vectors[:, values >= MATCH_DEGENERATE_RATIO * values[-1]]

    def _scan_match(
        self,
        points: np.ndarray,
        submap: Submap,
        segments: List[Tuple[float, float]],
        *,
        lock_yaw: bool = False,
    ) -> bool:
        """把本帧障碍点对齐到 submap，修正里程漂移。只用 depth。

        基准用 submap 而不是全局图：全局图攒的是一路的漂移，拿它对齐等于
        让当前帧去迁就一张已经歪掉的图，误差只会越描越深。submap 里只有
        几十帧的相对误差，对齐它得到的才是干净的局部约束。

        ``lock_yaw`` 下只搜平移：朝向已经由线段方向定过，再让逐点得分去动
        它只会把定准的角度重新拖平（得分沿墙方向本来就是平的）。
        """
        if submap.grid.frames < SCAN_MATCH_MIN_FRAMES:
            return False
        wall = self._match_points(points)
        if wall.shape[0] < SCAN_MATCH_MIN_POINTS:
            return False
        if wall.shape[0] > SCAN_MATCH_MAX_POINTS:
            step = int(math.ceil(wall.shape[0] / SCAN_MATCH_MAX_POINTS))
            wall = wall[::step]
        bx = wall[:, 0]
        by = wall[:, 1]
        # 整个搜索在 submap 局部系里做，最后再把增量转回地图系
        pose = submap.to_local(self.x, self.y, self.yaw_deg)
        baseline = self._offset_score(submap, pose, bx, by, 0.0, 0.0, 0.0)
        if baseline <= 0.0:
            return False
        best_dx, best_dy, best_dyaw = 0.0, 0.0, 0.0
        best_score = baseline
        for lin_step, yaw_step, count in SCAN_MATCH_STAGES:
            found = self._best_offset(
                submap,
                pose,
                bx,
                by,
                center=(best_dx, best_dy, best_dyaw),
                lin_step=lin_step,
                yaw_step=0.0 if lock_yaw else yaw_step,
                count=count,
            )
            if found is None:
                break
            dx, dy, dyaw, score = found
            if score > best_score:
                best_dx, best_dy, best_dyaw, best_score = dx, dy, dyaw, score
        if best_score < baseline * (1.0 + MATCH_MIN_GAIN):
            return False
        axes = self._observable_axes(segments)
        if axes is None:
            # 点云得分本身不能说明平移在哪些方向可观测。线段提取没有给出
            # 任何法向约束时，网格搜索的最优平移只是噪声峰，不能写回位姿。
            return False
        if axes.shape[1] < 2:
            # 退化：只保留可观测子空间里的那一份修正，其余是噪声搜出来的
            angle = math.radians(pose[2])
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            basis = np.array([[cos_a, -sin_a], [sin_a, cos_a]]) @ axes
            best_dx, best_dy = basis @ (
                basis.T @ np.array([best_dx, best_dy])
            )
        step = math.hypot(best_dx, best_dy)
        if step > MATCH_MAX_STEP_M:
            scale = MATCH_MAX_STEP_M / step
            best_dx, best_dy = best_dx * scale, best_dy * scale
        if (
            abs(best_dx) < 1e-6
            and abs(best_dy) < 1e-6
            and abs(best_dyaw) < 1e-6
        ):
            return False
        map_dx, map_dy = submap.local_delta_to_map(best_dx, best_dy)
        self.x += map_dx
        self.y += map_dy
        self.yaw_deg = wrap_deg(self.yaw_deg + best_dyaw)
        self._sync_latest_trajectory_pose()
        return True

    @staticmethod
    def _offset_score(
        submap: Submap,
        pose: Tuple[float, float, float],
        bx: np.ndarray,
        by: np.ndarray,
        dx: float,
        dy: float,
        dyaw: float,
    ) -> float:
        px, py, pyaw = pose
        raw = submap.grid.match_score_candidates(
            bx,
            by,
            np.asarray([[px + dx, py + dy, pyaw + dyaw]], dtype=np.float64),
        )[0]
        penalty = MATCH_REG_LIN * (dx * dx + dy * dy) + MATCH_REG_YAW * dyaw * dyaw
        return raw - penalty

    @staticmethod
    def _best_offset(
        submap: Submap,
        pose: Tuple[float, float, float],
        bx: np.ndarray,
        by: np.ndarray,
        *,
        center: Tuple[float, float, float],
        lin_step: float,
        yaw_step: float,
        count: int,
    ) -> Optional[Tuple[float, float, float, float]]:
        px, py, pyaw = pose
        cdx, cdy, cdyaw = center
        steps = [i for i in range(-count, count + 1)]
        # 步长为 0 表示这一维被锁住了，只留中心一个候选，省掉整层循环
        yaw_steps = steps if yaw_step > 0.0 else [0]
        candidates = []
        for iy in yaw_steps:
            dyaw = cdyaw + iy * yaw_step
            for ix in steps:
                dx = cdx + ix * lin_step
                for jy in steps:
                    dy = cdy + jy * lin_step
                    candidates.append((
                        px + dx,
                        py + dy,
                        pyaw + dyaw,
                        dx,
                        dy,
                        dyaw,
                    ))
        if not candidates:
            return None
        candidate_array = np.asarray(candidates, dtype=np.float64)
        raw = submap.grid.match_score_candidates(
            bx,
            by,
            candidate_array[:, :3],
        )
        penalty = MATCH_REG_LIN * (
            candidate_array[:, 3] ** 2 + candidate_array[:, 4] ** 2
        ) + MATCH_REG_YAW * candidate_array[:, 5] ** 2
        scores = raw - penalty
        best_index = int(np.argmax(scores))
        best_row = candidate_array[best_index]
        return (
            float(best_row[3]),
            float(best_row[4]),
            float(best_row[5]),
            float(scores[best_index]),
        )

    def ensure_start(self, *, image_id: str = "", tool: str = "") -> None:
        if self.initialized:
            return
        self.initialized = True
        mark_active_session(self.session_id)
        start_serial = (
            self.submaps[0].serial if self.submaps else self.next_submap_serial
        )
        self.landmarks["start"] = Landmark(
            name="start",
            x=0.0,
            y=0.0,
            yaw_deg=0.0,
            image_id=str(image_id or ""),
            tool=str(tool or ""),
            note="本局第一次底盘观测",
            submap_serial=start_serial,
        )
        self.trail.append(
            TrailPoint(
                x=0.0,
                y=0.0,
                yaw_deg=0.0,
                tool=tool,
                image_id=image_id,
                source="origin",
                submap_serial=start_serial,
            )
        )
        self._append_trajectory_sample(
            source="origin",
            force=True,
            tool=tool,
            image_id=image_id,
            submap_serial=start_serial,
        )

    def apply_motion(self, motion: MotionDelta) -> TrailPoint:
        image_id = str(motion.image_id or "")
        tool = str(motion.tool or "")
        self.ensure_start(image_id=image_id, tool=tool)
        if self.live_odometry:
            # 实时里程已经连续积分过这段运动，这里只在当前位姿切一个轨迹拐点
            self.update_count += 1
            point = TrailPoint(
                x=self.x,
                y=self.y,
                yaw_deg=self.yaw_deg,
                tool=tool,
                image_id=image_id,
                forward_m=float(motion.forward_m),
                translation_m=float(motion.translation_m),
                spin_deg=float(motion.spin_deg),
                source="live_odometry",
                submap_serial=self._current_submap_serial(),
            )
            self.trail.append(point)
            self._append_trajectory_sample(
                source="live_odometry",
                force=True,
                tool=tool,
                image_id=image_id,
            )
            return point
        yaw = math.radians(self.yaw_deg)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        dx = float(motion.forward_m) * cos_y - float(motion.translation_m) * sin_y
        dy = float(motion.forward_m) * sin_y + float(motion.translation_m) * cos_y
        self.x += dx
        self.y += dy
        self.yaw_deg = wrap_deg(self.yaw_deg + float(motion.spin_deg))
        step = math.hypot(dx, dy)
        self.trail_len_m += step
        self.update_count += 1
        point = TrailPoint(
            x=self.x,
            y=self.y,
            yaw_deg=self.yaw_deg,
            tool=tool,
            image_id=image_id,
            forward_m=float(motion.forward_m),
            translation_m=float(motion.translation_m),
            spin_deg=float(motion.spin_deg),
            source=str(motion.source or ""),
            submap_serial=self._current_submap_serial(),
        )
        self.trail.append(point)
        self._append_trajectory_sample(
            source=str(motion.source or "motion"),
            force=True,
            tool=tool,
            image_id=image_id,
        )
        if motion.mark_passage:
            last = Landmark(
                name="last_passage",
                x=self.x,
                y=self.y,
                yaw_deg=self.yaw_deg,
                image_id=image_id,
                tool=tool,
                note="点地/过门工具成功后的底盘位置",
                submap_serial=self._current_submap_serial(),
            )
            self.landmarks["last_passage"] = last
            if "first_passage" not in self.landmarks:
                self.landmarks["first_passage"] = Landmark(
                    name="first_passage",
                    x=last.x,
                    y=last.y,
                    yaw_deg=last.yaw_deg,
                    image_id=last.image_id,
                    tool=last.tool,
                    note=last.note,
                    submap_serial=last.submap_serial,
                )
        return point

    def place_reports(self) -> List[Dict[str, Any]]:
        """自己标记的地点相对当前底盘的方位，供工具返回与 memory 展示。

        spin_deg_to_face 直接就是 adjust_chassis 的 spin 参数（左正右负）。
        """
        reports: List[Dict[str, Any]] = []
        for place in self.places:
            range_m, bearing = self.bearing_to(place.x, place.y)
            reports.append(
                {
                    "name": place.label,
                    "range_m": round(range_m, 2),
                    "spin_deg_to_face": round(bearing, 1),
                    "direction": _bearing_text(bearing),
                }
            )
        return reports

    def bearing_to(self, x: float, y: float) -> Tuple[float, float]:
        """机体系：正前 0°，左正右负。返回 (range_m, bearing_deg)。"""
        dx = float(x) - self.x
        dy = float(y) - self.y
        range_m = math.hypot(dx, dy)
        world_bearing = math.degrees(math.atan2(dy, dx)) if range_m > 1e-9 else self.yaw_deg
        return range_m, wrap_deg(world_bearing - self.yaw_deg)

    def scan_match_backend_info(self) -> Dict[str, Any]:
        """汇总全局图和所有 submap 的扫描匹配实际设备。"""
        grids = [self.grid, *(submap.grid for submap in self.submaps)]
        infos = [grid.scan_match_backend_info() for grid in grids]
        cuda_infos = [info for info in infos if info.get("backend") == "torch_cuda"]
        fallbacks = [
            str(info.get("fallback_reason", ""))
            for info in infos
            if info.get("fallback_reason")
        ]
        first = infos[0] if infos else {
            "backend": "numpy_cpu",
            "device": "cpu",
            "requested": SCAN_MATCH_BACKEND_AUTO,
            "fallback_reason": "",
        }
        if cuda_infos:
            backend = "torch_cuda"
            device = str(cuda_infos[0].get("device", "cuda:0"))
        else:
            backend = str(first.get("backend", "numpy_cpu"))
            device = str(first.get("device", "cpu"))
        return {
            "backend": backend,
            "device": device,
            "requested": str(first.get("requested", SCAN_MATCH_BACKEND_AUTO)),
            "cuda_grid_count": len(cuda_infos),
            "grid_count": len(infos),
            "fallback_reason": "; ".join(dict.fromkeys(fallbacks)),
        }

    def query(self) -> Dict[str, Any]:
        if not self.initialized:
            # 还没挪过窝，但实时建图可能已经把眼前这片房间画出来了
            if self.grid.frames > 0:
                self.ensure_start()
            else:
                return {
                    "ok": True,
                    "backend": BACKEND,
                    "build": BUILD,
                    "session_id": self.session_id,
                    "empty": True,
                    "hint": "地图还是空的。先做一次底盘移动。",
                }
        start = self.landmarks.get("start")
        first = self.landmarks.get("first_passage")
        last = self.landmarks.get("last_passage")

        def _pack(landmark: Optional[Landmark]) -> Optional[Dict[str, Any]]:
            if landmark is None:
                return None
            range_m, bearing = self.bearing_to(landmark.x, landmark.y)
            return {
                "name": landmark.name,
                "range_m": round(range_m, 3),
                "bearing_deg": round(bearing, 1),
                "image_id": landmark.image_id,
                "note": landmark.note,
            }

        start_pack = _pack(start)
        first_pack = _pack(first)
        last_pack = _pack(last)
        hint = _format_hint(start_pack, first_pack, last_pack)
        visual_tracker = self._visual_localizer
        geometry_engine = self._geometry_validator
        geometry_backend = getattr(geometry_engine, "backend_info", None)
        last_geometry = self._geometry_reports[-1] if self._geometry_reports else {}
        last_geometry_summary = {
            key: value for key, value in last_geometry.items() if key != "rows"
        }
        if last_geometry.get("rows"):
            last_geometry_summary["windows"] = [
                {
                    "window_frames": row.get("window_frames"),
                    "valid": row.get("valid"),
                    "unique_geometry": row.get("unique_geometry"),
                    "uniqueness_blockers": row.get("uniqueness_blockers", []),
                    "best_prominence_robust": row.get(
                        "best_prominence_robust"
                    ),
                    "top_one_percent_cluster_ratio": row.get(
                        "top_one_percent_cluster_ratio"
                    ),
                    "coarse_best": row.get("coarse_best"),
                    "peaks": list(row.get("peaks") or ())[:3],
                    "probe_hypothesis": row.get("probe_hypothesis"),
                    "target_support": row.get("target_support"),
                }
                for row in last_geometry["rows"]
            ]
        return {
            "ok": True,
            "backend": BACKEND,
            "build": BUILD,
            "session_id": self.session_id,
            "empty": False,
            "pose": {
                "x": round(self.x, 3),
                "y": round(self.y, 3),
                "yaw_deg": round(self.yaw_deg, 1),
                "frame": "episode_start_odometry",
            },
            "trail_len_m": round(self.trail_len_m, 3),
            "update_count": self.update_count,
            "start": start_pack,
            "first_passage": first_pack,
            "last_passage": last_pack,
            "places": self.place_reports(),
            "events": [
                {
                    "kind": event.kind,
                    "label": event.label,
                    "range_m": round(self.bearing_to(event.x, event.y)[0], 2),
                    "bearing_deg": round(self.bearing_to(event.x, event.y)[1], 1),
                }
                for event in self.events
            ],
            "mapped_frames": self.grid.frames,
            "mapping": {
                "state": self.mapping_state,
                "freeze_frame": self.freeze_frame,
                "freeze_reason": self.freeze_reason,
                "freeze_grid_revision": self.freeze_grid_revision,
                "freeze_grid_digest": self.freeze_grid_digest,
                "observations": self.mapping_observations,
                "travel_m": round(self.mapping_travel_m, 3),
                "visual_keyframes": self.mapping_visual_keyframes,
                "freeze_candidate_frame": self.mapping_freeze_candidate_frame,
                "idle_observations": self.mapping_idle_observations,
                "frontier_signature": self.mapping_frontier_signature,
                "frontier_quiescence_observations": (
                    self.mapping_frontier_quiescence_observations
                ),
                "coverage": dict(self.mapping_coverage),
                "revisit_evidence": list(self.mapping_revisit_evidence),
                "freeze_checks": list(self.mapping_freeze_checks),
                "geometry_attempts": self.mapping_geometry_attempts,
                "geometry_accepted": self.mapping_geometry_accepted,
                "geometry_rejected": self.mapping_geometry_rejected,
            },
            "localization": {
                "attempts": self.localization_attempts,
                "accepted": self.localization_accepted,
                "rejected": self.localization_rejected,
                "detail_frames": self.localization_detail_frames,
                "last_reason": self.localization_last_reason,
                "status": self.localization_status,
                "state": self.localization_state,
                "state_reason": self.localization_state_reason,
                "position_std_m": round(math.sqrt(max(
                    0.0, self.localization_position_variance_m2
                )), 4),
                "yaw_std_deg": round(math.sqrt(max(
                    0.0, self.localization_yaw_variance_deg2
                )), 3),
                "ambiguous_observations": (
                    self.localization_ambiguous_observations
                ),
                "last_validated_frame": self.localization_last_validated_frame,
                "unvalidated_travel_m": round(
                    self.localization_unvalidated_travel_m, 3
                ),
                "unvalidated_turn_deg": round(
                    self.localization_unvalidated_turn_deg, 1
                ),
                "visual_candidates": self.localization_visual_candidates,
                "geometry_attempts": self.localization_geometry_attempts,
                "geometry_last_reason": self.geometry_last_reason,
                "geometry_reason_counts": dict(self.geometry_reason_counts),
                "geometry_blocker_counts": dict(self.geometry_blocker_counts),
                "geometry_backend": (
                    {
                        "backend": "uninitialized",
                        "device": "uninitialized",
                        "cpu_fallback": False,
                    }
                    if not callable(geometry_backend)
                    else geometry_backend()
                ),
                "last_geometry_validation": last_geometry_summary,
                "feature_backend": (
                    "uninitialized"
                    if visual_tracker is None
                    else str(getattr(visual_tracker, "feature_backend", "unknown"))
                ),
                "feature_device": (
                    "uninitialized"
                    if visual_tracker is None
                    else str(getattr(visual_tracker, "feature_device", "unknown"))
                ),
                "feature_fallback_reason": (
                    ""
                    if visual_tracker is None
                    else str(getattr(visual_tracker, "feature_fallback_reason", ""))
                ),
                "matching_backend": (
                    "uninitialized"
                    if visual_tracker is None
                    else str(getattr(visual_tracker, "matching_backend", "unknown"))
                ),
                "matching_device": (
                    "uninitialized"
                    if visual_tracker is None
                    else str(getattr(visual_tracker, "matching_device", "unknown"))
                ),
                "matching_fallback_reason": (
                    ""
                    if visual_tracker is None
                    else str(getattr(
                        visual_tracker, "matching_fallback_reason", ""
                    ))
                ),
                "sequence_place": {
                    "metric_keyframes": (
                        0 if visual_tracker is None else int(getattr(
                            visual_tracker, "keyframe_count", 0
                        ))
                    ),
                    "place_keyframes": (
                        0 if visual_tracker is None else int(getattr(
                            visual_tracker, "place_keyframe_count", 0
                        ))
                    ),
                    "backend": (
                        "uninitialized"
                        if visual_tracker is None
                        else str(getattr(
                            visual_tracker, "sequence_backend", "unknown"
                        ))
                    ),
                    "device": (
                        "uninitialized"
                        if visual_tracker is None
                        else str(getattr(
                            visual_tracker, "sequence_device", "unknown"
                        ))
                    ),
                    "attempts": (
                        0 if visual_tracker is None else int(getattr(
                            visual_tracker, "sequence_attempts", 0
                        ))
                    ),
                    "candidates": (
                        0 if visual_tracker is None else int(getattr(
                            visual_tracker, "sequence_candidates", 0
                        ))
                    ),
                    "last_reason": (
                        "uninitialized"
                        if visual_tracker is None
                        else str(getattr(
                            visual_tracker, "sequence_last_reason", "unknown"
                        ))
                    ),
                    "last_candidates": (
                        []
                        if visual_tracker is None
                        else [
                            {
                                "keyframe_id": str(item.keyframe_id),
                                "keyframe_frame_index": int(
                                    item.keyframe_frame_index
                                ),
                                "candidate_frame_gap": int(
                                    item.candidate_frame_gap
                                ),
                                "appearance_score": float(
                                    item.appearance_score
                                ),
                                "robust_z": float(item.robust_z),
                                "sequence_observations": int(
                                    item.sequence_observations
                                ),
                            }
                            for item in getattr(
                                visual_tracker, "last_place_candidates", ()
                            )
                        ]
                    ),
                    "geometry_attempts": (
                        0 if visual_tracker is None else int(getattr(
                            visual_tracker, "place_geometry_attempts", 0
                        ))
                    ),
                    "geometry_accepted": (
                        0 if visual_tracker is None else int(getattr(
                            visual_tracker, "place_geometry_accepted", 0
                        ))
                    ),
                    "geometry_rejected": (
                        0 if visual_tracker is None else int(getattr(
                            visual_tracker, "place_geometry_rejected", 0
                        ))
                    ),
                    "geometry_last_reason": (
                        "uninitialized"
                        if visual_tracker is None
                        else str(getattr(
                            visual_tracker, "place_geometry_last_reason", ""
                        ))
                    ),
                    "geometry_reason_counts": (
                        {}
                        if visual_tracker is None
                        else dict(getattr(
                            visual_tracker,
                            "place_geometry_reason_counts",
                            {},
                        ))
                    ),
                    "last_geometry_rows": (
                        []
                        if visual_tracker is None
                        else list(getattr(
                            visual_tracker, "last_place_geometry_rows", ()
                        ))
                    ),
                    "forbidden_inputs_consumed": [],
                },
            },
            # 朝向是靠墙的方向定的，把依据一起报出来：主方向为空说明这一带
            # 几何退化（圆弧、杂物），此时不纠 yaw 是预期行为而不是故障。
            "wall_alignment": {
                "directions_deg": [
                    round(float(d), 1) for d in self.wall_directions()
                ],
                "yaw_fixes": self.yaw_fixes,
                "yaw_fix_total_deg": round(self.yaw_fix_total_deg, 2),
            },
            # 回环为 0 说明还没走回过老地方，位置漂移就只能一路累积下去
            "pose_graph": {
                "submaps": len(self.submaps),
                "loops": self.loops_found,
                "optimizations": self.graph_optimizations,
            },
            "scan_matching": {
                "attempts": self.scan_match_attempts,
                "applied": self.scan_match_applied,
                "translation_locked": self.scan_match_translation_locked,
                **self.scan_match_backend_info(),
            },
            "global_pose_optimization": self.global_pose_optimization,
            "hint": hint,
        }


def _format_hint(
    start: Optional[Dict[str, Any]],
    first: Optional[Dict[str, Any]],
    last: Optional[Dict[str, Any]],
) -> str:
    parts = []
    if first:
        parts.append(
            f"来时通道在{_bearing_text(first['bearing_deg'])}，约 {first['range_m']:.1f}m"
        )
    if last and (not first or last.get("image_id") != first.get("image_id")):
        parts.append(
            f"最近一次点地在{_bearing_text(last['bearing_deg'])}，约 {last['range_m']:.1f}m"
        )
    if start:
        parts.append(
            f"起点在{_bearing_text(start['bearing_deg'])}，约 {start['range_m']:.1f}m"
        )
    if not parts:
        return "已记录当前位置，还没有通道路标。"
    return "；".join(parts) + "。沿轨迹回退可到。"


def _bearing_text(bearing_deg: float) -> str:
    deg = wrap_deg(bearing_deg)
    abs_deg = abs(deg)
    if abs_deg <= 15.0:
        return "正前方"
    if abs_deg >= 165.0:
        return "正后方"
    side = "左" if deg > 0.0 else "右"
    if abs_deg < 75.0:
        return f"{side}前方 {abs_deg:.0f}°"
    if abs_deg <= 105.0:
        return f"正{side}侧"
    return f"{side}后方 {abs_deg:.0f}°"


def get_map(session_id: str) -> EgoMap:
    key = str(session_id or "").strip() or "default"
    with _MAPS_LOCK:
        if key not in _MAPS:
            _MAPS[key] = EgoMap(session_id=key)
        return _MAPS[key]


def mark_active_session(session_id: str) -> None:
    global _ACTIVE_SESSION
    key = str(session_id or "").strip()
    if key:
        _ACTIVE_SESSION = key


def _has_content(ego: Optional["EgoMap"]) -> bool:
    """有内容 = 走过路或者建过图；只被轮询创建出来的空壳不算。"""
    return ego is not None and (ego.initialized or ego.grid.frames > 0)


def active_session_id() -> str:
    """当前这张实时地图的 session：优先最近活跃的，否则挑唯一一张非空图。"""
    if _ACTIVE_SESSION:
        return _ACTIVE_SESSION
    with _MAPS_LOCK:
        live = [key for key, ego in _MAPS.items() if _has_content(ego)]
    return live[0] if len(live) == 1 else ""


def adopt_session(session_id: str) -> None:
    """把已经建好的那张地图交给新来的 session，而不是另起一张空图。

    一个 episode 里会冒出好几个 session id：实时建图在任何工具调用之前就
    开始了（那时只能先记在 default 上），UI 轮询带的是自己的 id，agent 用
    的又是另一个。之前只认 default，于是换一个 id 就新建空图、实时建图跟着
    切过去，起点附近已经建好的墙面和空地凭空消失。
    """
    key = str(session_id or "").strip()
    if not key:
        return
    with _MAPS_LOCK:
        if _has_content(_MAPS.get(key)):
            return
        src_key = _ACTIVE_SESSION if _has_content(_MAPS.get(_ACTIVE_SESSION)) else ""
        if not src_key:
            candidates = [k for k, ego in _MAPS.items() if _has_content(ego)]
            if len(candidates) != 1:
                return
            src_key = candidates[0]
        if src_key == key:
            return
        src = _MAPS[src_key]
        src.session_id = key
        _MAPS[key] = src
        _MAPS.pop(src_key, None)
    mark_active_session(key)


def reset_map(session_id: str) -> None:
    key = str(session_id or "").strip() or "default"
    with _MAPS_LOCK:
        _MAPS.pop(key, None)
    try:
        from behavior_interface_eval_test.navigation_route_overlay import clear_route

        clear_route(key)
    except (ImportError, RuntimeError, ValueError):
        pass


def reset_all_maps() -> None:
    """episode 重置：丢掉所有地图，下一帧从零重新建。"""
    global _ACTIVE_SESSION
    with _MAPS_LOCK:
        _MAPS.clear()
    _ACTIVE_SESSION = ""
    _SNAPSHOT_CACHE.clear()
    _RGBA_CACHE.clear()
    try:
        from behavior_interface_eval_test.navigation_route_overlay import (
            clear_all_routes,
        )

        clear_all_routes()
    except (ImportError, RuntimeError):
        pass


def extract_chassis_motion(
    tool: str,
    args: Optional[Dict[str, Any]] = None,
    result: Optional[Dict[str, Any]] = None,
    *,
    depth_lookup: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
) -> MotionDelta:
    """从工具参数/结果提取机体系完成量，不读全局位姿。"""
    args = dict(args or {})
    result = dict(result or {})
    name = str(tool or result.get("tool") or args.get("tool") or "").strip()
    image_id = str(
        result.get("image_id")
        or result.get("output_image_id")
        or args.get("image_id")
        or ""
    )
    motion = MotionDelta(tool=name, image_id=image_id)
    if name not in CHASSIS_TOOLS:
        return motion

    # official_v2 把 base_qvel 积分出的实际完成量放在嵌套 actual 下，优先用它
    actual_f = _first_number(
        result,
        ("actual", "forward_m"),
        "forward_actual_m",
        ("ground_guard", "safe_forward_m"),
    )
    actual_t = _first_number(
        result,
        ("actual", "translation_m"),
        "translation_actual_m",
        ("ground_guard", "safe_translation_m"),
    )
    actual_s = _first_number(
        result,
        ("actual", "spin_deg"),
        "spin_actual_deg",
        "spin_deg",
    )
    if actual_f is not None or actual_t is not None or actual_s is not None:
        motion.forward_m = _finite(actual_f)
        motion.translation_m = _finite(actual_t)
        motion.spin_deg = _finite(actual_s)
        motion.source = "result_actual"
        motion.mark_passage = name in PASSAGE_TOOLS
        return motion

    if name in {"adjust_chassis"}:
        motion.forward_m = _finite(args.get("forward"))
        motion.translation_m = _finite(
            args.get("translation", args.get("leftward"))
        )
        motion.spin_deg = _finite(args.get("spin"))
        motion.source = "command_args"
        return motion

    if name in {"spin_to_facing_point", "face_to_point"}:
        spin = _first_number(result, "spin_deg", ("exec_args", "spin"))
        if spin is None:
            spin = estimate_spin_from_uv(args, result, depth_lookup=depth_lookup)
        motion.spin_deg = _finite(spin)
        motion.source = "spin_from_uv" if spin is not None else "none"
        return motion

    if name in PASSAGE_TOOLS:
        forward = _first_number(result, "forward_m")
        translation = _first_number(result, "translation_m")
        if forward is None or translation is None:
            est = estimate_floor_point_delta(args, result, depth_lookup=depth_lookup)
            if est is not None:
                forward, translation = est
                motion.source = "depth_unproject"
            else:
                forward, translation = 0.0, 0.0
                motion.source = "none"
        else:
            motion.source = "result_commanded_delta"
        motion.forward_m = _finite(forward)
        motion.translation_m = _finite(translation)
        motion.mark_passage = True
        return motion

    if name == "move_to_reach_point":
        est = estimate_reach_point_delta(args, result, depth_lookup=depth_lookup)
        if est is not None:
            motion.forward_m, motion.translation_m = est
            motion.source = "depth_unproject_reach"
        return motion

    return motion


def _first_number(payload: Dict[str, Any], *keys: Any) -> Optional[float]:
    for key in keys:
        if isinstance(key, tuple):
            value = _nested_get(payload, *key)
        else:
            value = payload.get(key)
        if value is None or value == "":
            continue
        number = _finite(value, default=float("nan"))
        if math.isfinite(number):
            return number
    return None


def estimate_spin_from_uv(
    args: Dict[str, Any],
    result: Dict[str, Any],
    *,
    depth_lookup: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
) -> Optional[float]:
    """与 face_to_point 同一套针孔公式，保证回放转角和当时一致。"""
    if args.get("u") is None:
        return None
    width = int(_finite(result.get("image_width"), 0.0))
    fl = _finite(result.get("focal_length"), 0.0)
    ha = _finite(result.get("horizontal_aperture"), 0.0)
    image_id = str(args.get("image_id") or "")
    if (width <= 0 or fl <= 0.0 or ha <= 0.0) and depth_lookup and image_id:
        bundle = depth_lookup(image_id) or {}
        camera = dict(bundle.get("camera") or {})
        width = int(_finite(camera.get("image_width"), width))
        fl = _finite(camera.get("focal_length"), fl)
        ha = _finite(camera.get("horizontal_aperture"), ha)
    if width <= 0:
        width = 720
    if fl <= 0.0:
        fl = 17.0
    if ha <= 0.0:
        ha = 40.0
    from behavior_interface.skills.face_to_point import compute_face_to_point_spin_deg

    spin = compute_face_to_point_spin_deg(
        u=_finite(args.get("u")),
        image_width=width,
        focal_length=fl,
        horizontal_aperture=ha,
    )
    limit = abs(_finite(args.get("max_abs_spin"), 60.0))
    if limit > 0.0:
        spin = max(-limit, min(limit, spin))
    return float(spin)


def estimate_floor_point_delta(
    args: Dict[str, Any],
    result: Dict[str, Any],
    *,
    depth_lookup: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
) -> Optional[Tuple[float, float]]:
    xy = unproject_uv_robot_xy(
        str(args.get("image_id") or ""),
        args.get("u"),
        args.get("v"),
        depth_lookup=depth_lookup,
        result=result,
    )
    if xy is None:
        return None
    from behavior_interface.skills.viz_base_path_overlay import (
        BASE_FRONT_OFFSET_M,
        PATH_Y_CENTER_M,
    )

    return (
        float(xy[0] - BASE_FRONT_OFFSET_M),
        float(xy[1] - PATH_Y_CENTER_M),
    )


def estimate_reach_point_delta(
    args: Dict[str, Any],
    result: Dict[str, Any],
    *,
    depth_lookup: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
) -> Optional[Tuple[float, float]]:
    xy = unproject_uv_robot_xy(
        str(args.get("image_id") or ""),
        args.get("u"),
        args.get("v"),
        depth_lookup=depth_lookup,
        result=result,
    )
    if xy is None:
        return None
    reach = _finite(args.get("reach"), 0.6)
    dist = math.hypot(xy[0], xy[1])
    if dist <= 1e-6:
        return (0.0, 0.0)
    remain = max(0.0, dist - max(0.0, reach))
    scale = remain / dist
    return (float(xy[0] * scale), float(xy[1] * scale))


def unproject_uv_robot_xy(
    image_id: str,
    u: Any,
    v: Any,
    *,
    depth_lookup: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
    result: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[float, float]]:
    """把 0..1000 点反解到机体系 XY。只用 depth + cam_rel_pose。"""
    if depth_lookup is None or not image_id or u is None or v is None:
        return None
    bundle = depth_lookup(str(image_id)) or {}
    depth = bundle.get("depth")
    camera = dict(bundle.get("camera") or {})
    if result:
        camera.setdefault("image_width", result.get("image_width"))
        camera.setdefault("image_height", result.get("image_height"))
    if depth is None:
        return None
    arr = np.asarray(depth, dtype=np.float32).squeeze()
    if arr.ndim != 2:
        return None
    height, width = arr.shape
    try:
        col = relative_to_pixel(u, width)
        row = relative_to_pixel(v, height)
    except Exception:
        return None
    row = min(max(row, 0), height - 1)
    col = min(max(col, 0), width - 1)
    sampled = float(arr[row, col])
    if not math.isfinite(sampled) or sampled <= 1e-4:
        return None
    fl = _finite(camera.get("focal_length"), 17.0)
    ha = _finite(camera.get("horizontal_aperture"), 40.0)
    fx = fl * float(width) / ha if ha > 1e-9 else _finite(camera.get("fx"), 306.0)
    fy = _finite(camera.get("fy"), fx)
    cx = _finite(camera.get("cx"), width * 0.5)
    cy = _finite(camera.get("cy"), height * 0.5)
    cam_point = np.array(
        [
            (col - cx) / fx * sampled,
            -(row - cy) / fy * sampled,
            -sampled,
        ],
        dtype=np.float64,
    )
    rel = dict(camera.get("robot_relative_pose") or {})
    if not rel:
        return None
    from behavior_interface.skills.reach_point_pitch_recovery import quat_to_mat_xyzw

    cam_pos = np.asarray(rel.get("pos"), dtype=np.float64).reshape(3)
    cam_rot = quat_to_mat_xyzw(rel.get("quat"))
    robot_point = cam_pos + cam_rot @ cam_point
    return float(robot_point[0]), float(robot_point[1])


def _drop_depth_edges(depth: np.ndarray) -> np.ndarray:
    """丢掉物体边缘的插值飞点。

    深度图在前后景交界处会插出一串中间值，反投影后变成悬在空中的点，
    在地图上拖成放射状假墙。相邻像素跳变过大的位置整片置零，
    后面按最小距离过滤时自然被扔掉。
    """
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return arr
    dx = np.abs(np.diff(arr, axis=1, prepend=arr[:, :1]))
    dy = np.abs(np.diff(arr, axis=0, prepend=arr[:1, :]))
    edge = (dx > DEPTH_EDGE_JUMP_M) | (dy > DEPTH_EDGE_JUMP_M)
    if not np.any(edge):
        return arr
    # 跳变检出的是交界的一侧，把邻域一起抹掉才干净
    edge |= np.roll(edge, 1, axis=1) | np.roll(edge, 1, axis=0)
    out = arr.copy()
    out[edge] = 0.0
    return out


def _depth_to_robot_points(
    bundle: Dict[str, Any],
    *,
    stride: int = DEPTH_STRIDE,
) -> Optional[np.ndarray]:
    depth = bundle.get("depth")
    camera = dict(bundle.get("camera") or {})
    rel = dict(camera.get("robot_relative_pose") or {})
    if depth is None or not rel:
        return None
    depth = _drop_depth_edges(depth)
    from behavior_interface.skills.base_forward_observation_guard import (
        depth_to_robot_points,
    )

    try:
        return depth_to_robot_points(
            depth,
            camera_relative_pose=rel,
            focal_length=_finite(camera.get("focal_length"), 17.0),
            horizontal_aperture=_finite(camera.get("horizontal_aperture"), 40.0),
            stride=int(stride),
        )
    except Exception:
        return None


def load_capture_bundle(session_id: str, image_id: str) -> Optional[Dict[str, Any]]:
    """从 agent_runs 读 depth + camera meta，供回放反解。"""
    try:
        from behavior_interface import agent_runs
    except Exception:
        return None
    try:
        meta = agent_runs.load_image_meta(session_id, image_id)
    except Exception:
        meta = {}
    depth = None
    try:
        depth_path = agent_runs.image_path(session_id, image_id, ".depth.npy")
        if os.path.isfile(depth_path):
            depth = np.load(depth_path)
    except Exception:
        depth = None
    if not meta and depth is None:
        return None
    return {"camera": dict((meta or {}).get("camera") or {}), "depth": depth, "meta": meta}


_COLOR_UNKNOWN = (214, 216, 220)
_COLOR_FREE = (255, 255, 255)
_COLOR_WALL = (12, 12, 14)
# 只在头顶高度挡住的东西（桌面、料理台、吊柜）：底盘过得去，画淡灰
_COLOR_OVERHEAD = (176, 182, 192)
_COLOR_TRAIL = (20, 90, 220)
# Remaining navigate_to route.  It is a transient vector layer, never occupancy
# evidence, and therefore cannot influence mapping or scan matching.
_COLOR_NAVIGATION_ROUTE = (238, 126, 132)
_COLOR_PLACE = (12, 48, 150)
_COLOR_ROBOT = (20, 90, 220)
_COLOR_VIEW_CONE = (120, 190, 255)
_COLOR_START = (20, 90, 220)

# 未探索浅灰要半透明，叠到 head 上才能透出后面的画面
UNKNOWN_ALPHA = 72
FREE_ALPHA = 236
OVERHEAD_ALPHA = 210
# 视场光锥：延伸多远、最浓处多浓、侧边羽化几度
VIEW_CONE_REACH_M = 3.5
VIEW_CONE_ALPHA = 0.58
VIEW_CONE_FEATHER_DEG = 3.0
HEAD_MINIMAP_FRACTION = 0.32
HEAD_MINIMAP_MARGIN_FRAC = 0.016


def render_minimap(
    ego: EgoMap,
    *,
    size: int = _DEFAULT_MAP_SIZE,
    range_m: Optional[float] = None,
    heading_up: bool = True,
) -> np.ndarray:
    """场景俯视图：占用栅格画墙与地板，叠加最近行踪和自己标的地点。

    图上只有几何信息：墙 / 空地 / 一条蓝色行进线 / 机器人视场光锥 /
    绿色起点 / 自己标的地点（带名字）。不画标题、图例、比例尺、提示。
    默认车头朝上（FPS 小地图）；heading_up=False 时改为起点朝向朝上的固定图。
    """
    return flatten_rgba(
        render_minimap_rgba(ego, size=size, range_m=range_m, heading_up=heading_up)
    )


def render_minimap_rgba(
    ego: EgoMap,
    *,
    size: int = _DEFAULT_MAP_SIZE,
    range_m: Optional[float] = None,
    heading_up: bool = True,
) -> np.ndarray:
    """带透明通道的小地图：未探索浅灰是半透明，给叠到 head 上用。"""
    size = max(320, int(size))
    route = _route_overlay_snapshot(ego.session_id)
    route_points = () if route is None else route.points_xy_m
    center_x, center_y, span = _view_window(
        ego,
        range_m=range_m,
        heading_up=heading_up,
        extra_points_xy_m=route_points,
    )
    scale = (size * VIEW_FILL) / (2.0 * span)
    cx_px = size * 0.5
    cy_px = size * 0.5

    def to_px(x: float, y: float) -> Tuple[float, float]:
        up, left = _view_axes(ego, x, y, center_x, center_y, heading_up)
        return cx_px - left * scale, cy_px - up * scale

    base = _render_grid_layer(
        ego,
        width=size,
        height=size,
        center_x=center_x,
        center_y=center_y,
        scale=scale,
        heading_up=heading_up,
    )
    robot_px, robot_py = to_px(ego.x, ego.y)
    _blend_view_cone(base, ego, robot_px, robot_py, scale, heading_up=heading_up)
    image = Image.fromarray(base)
    draw = ImageDraw.Draw(image, "RGBA")
    _draw_navigation_route(draw, route, to_px, canvas=size)
    _draw_trail(draw, ego, to_px)
    _draw_start(draw, ego, to_px)
    _draw_places(draw, ego, to_px, canvas=size)
    _draw_robot(draw, ego, to_px, heading_up=heading_up)
    return np.asarray(image)


render_minimap_rgba._navigation_route_overlay_native = True


def _view_window(
    ego: EgoMap,
    *,
    range_m: Optional[float],
    heading_up: bool,
    extra_points_xy_m=(),
) -> Tuple[float, float, float]:
    """返回视窗中心（地图系）与半宽（米）。"""
    chunks_x = [np.asarray([ego.x], dtype=np.float64)]
    chunks_y = [np.asarray([ego.y], dtype=np.float64)]
    trajectory = ego.full_trail()
    if trajectory:
        chunks_x.append(np.asarray([p.x for p in trajectory], dtype=np.float64))
        chunks_y.append(np.asarray([p.y for p in trajectory], dtype=np.float64))
    if ego.landmarks:
        chunks_x.append(np.asarray([lm.x for lm in ego.landmarks.values()], dtype=np.float64))
        chunks_y.append(np.asarray([lm.y for lm in ego.landmarks.values()], dtype=np.float64))
    if ego.places:
        chunks_x.append(np.asarray([p.x for p in ego.places], dtype=np.float64))
        chunks_y.append(np.asarray([p.y for p in ego.places], dtype=np.float64))
    if extra_points_xy_m:
        chunks_x.append(
            np.asarray([p[0] for p in extra_points_xy_m], dtype=np.float64)
        )
        chunks_y.append(
            np.asarray([p[1] for p in extra_points_xy_m], dtype=np.float64)
        )
    # 用格子本身取框，不要只用世界包围盒四角：车头朝上旋转后，
    # 四角框会比真实内容大一圈，斜着的房间四周就会空出来。
    cell_x, cell_y = ego.grid.known_xy()
    if cell_x is not None:
        chunks_x.append(cell_x)
        chunks_y.append(cell_y)
    # 在最终显示的坐标系里取景，这样旋转后也能填满画布
    yaw = math.radians(ego.yaw_deg) if heading_up else 0.0
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    arr_x = np.concatenate(chunks_x)
    arr_y = np.concatenate(chunks_y)
    ups = cos_y * arr_x + sin_y * arr_y
    lefts = -sin_y * arr_x + cos_y * arr_y
    mid_up = 0.5 * (float(ups.min()) + float(ups.max()))
    mid_left = 0.5 * (float(lefts.min()) + float(lefts.max()))
    center_x = cos_y * mid_up - sin_y * mid_left
    center_y = sin_y * mid_up + cos_y * mid_left
    if range_m is not None:
        return center_x, center_y, max(1.5, float(range_m))
    half = 0.5 * max(float(ups.max() - ups.min()), float(lefts.max() - lefts.min()))
    return center_x, center_y, max(
        VIEW_MIN_HALF_M, min(VIEW_MAX_HALF_M, half * VIEW_PAD_RATIO + VIEW_PAD_M)
    )


def _view_axes(
    ego: EgoMap,
    x: float,
    y: float,
    center_x: float,
    center_y: float,
    heading_up: bool,
) -> Tuple[float, float]:
    """返回 (向上量, 向左量)，单位米。"""
    dx = float(x) - center_x
    dy = float(y) - center_y
    if not heading_up:
        return dx, dy
    yaw = math.radians(ego.yaw_deg)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    return cos_y * dx + sin_y * dy, -sin_y * dx + cos_y * dy


def _render_grid_layer(
    ego: EgoMap,
    *,
    width: int,
    height: int,
    center_x: float,
    center_y: float,
    scale: float,
    heading_up: bool,
) -> np.ndarray:
    """逆映射采样占用栅格，直接生成地图底图。"""
    layer = np.empty((height, width, 4), dtype=np.uint8)
    layer[..., 0] = _COLOR_UNKNOWN[0]
    layer[..., 1] = _COLOR_UNKNOWN[1]
    layer[..., 2] = _COLOR_UNKNOWN[2]
    layer[..., 3] = UNKNOWN_ALPHA
    grid = ego.grid
    if grid.frames <= 0:
        return layer
    us = np.arange(width, dtype=np.float64)
    vs = np.arange(height, dtype=np.float64)
    grid_u, grid_v = np.meshgrid(us, vs)
    left = (width * 0.5 - grid_u) / scale
    up = (height * 0.5 - grid_v) / scale
    if heading_up:
        yaw = math.radians(ego.yaw_deg)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        world_x = center_x + cos_y * up - sin_y * left
        world_y = center_y + sin_y * up + cos_y * left
    else:
        world_x = center_x + up
        world_y = center_y + left
    ix = np.floor((world_x + grid.half_span_m) / grid.resolution_m).astype(np.int64)
    iy = np.floor((world_y + grid.half_span_m) / grid.resolution_m).astype(np.int64)
    inside = (ix >= 0) & (ix < grid.n) & (iy >= 0) & (iy < grid.n)
    ix = np.clip(ix, 0, grid.n - 1)
    iy = np.clip(iy, 0, grid.n - 1)
    # 生产渲染只显示传感器已经观测到的格子。这里不能调用闭运算、细化、
    # 膨胀或轨迹刻空；那些操作会把缺失证据变成一堵“看起来完整”的墙。
    occupied = grid.observed_occupied_mask()
    if np.any(grid.bands):
        line = grid.observed_wall_mask()
    else:
        # 兼容没有高度带的旧 capture：所有低层占据都按原始障碍显示。
        line = occupied.copy()
    furniture = occupied & ~line
    walkable = grid.observed_free_mask()
    overhead = grid.observed_overhead_mask() | furniture
    free = walkable[iy, ix] & inside
    over = overhead[iy, ix] & inside & ~free
    wall = line[iy, ix] & inside
    layer[free, :3] = _COLOR_FREE
    layer[free, 3] = FREE_ALPHA
    # 头顶结构画成淡灰家具块：看得见「那儿有张桌子」，又不会当成死路
    layer[over, :3] = _COLOR_OVERHEAD
    layer[over, 3] = OVERHEAD_ALPHA
    layer[wall, :3] = _COLOR_WALL
    layer[wall, 3] = 255
    return layer


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """二值膨胀：上下左右各扩 radius 格。"""
    out = mask
    height, width = mask.shape
    for _ in range(max(0, int(radius))):
        padded = np.pad(out, 1)
        out = (
            out
            | padded[0:height, 1 : width + 1]
            | padded[2 : height + 2, 1 : width + 1]
            | padded[1 : height + 1, 0:width]
            | padded[1 : height + 1, 2 : width + 2]
        )
    return out


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    """二值腐蚀。图外一律当成实的，免得把贴边的墙啃掉。"""
    out = mask
    height, width = mask.shape
    for _ in range(max(0, int(radius))):
        padded = np.pad(out, 1, constant_values=True)
        out = (
            out
            & padded[0:height, 1 : width + 1]
            & padded[2 : height + 2, 1 : width + 1]
            & padded[1 : height + 1, 0:width]
            & padded[1 : height + 1, 2 : width + 2]
        )
    return out


def _close(mask: np.ndarray, radius: int) -> np.ndarray:
    """闭运算：接上墙面里几格宽的小豁口，但不把墙整体撑粗。

    一堵墙很少能被每一帧都看全，落到栅格上就是断断续续的点线。
    半径按厘米级取，门洞、走廊这些真开口比它宽得多，不会被堵死。
    """
    radius = max(0, int(radius))
    if radius <= 0:
        return mask
    return _erode(_dilate(mask, radius), radius)


def _grow_within(seed: np.ndarray, host: np.ndarray, max_steps: int) -> np.ndarray:
    """种子在 host 里一格一格长，长到不动了或者步数用完为止。

    每步都跟 host 求交，所以水漫不过障碍——这点跟先膨胀 n 格再求交不一样，
    后者中间几步没约束，会直接穿墙过去。
    """
    out = seed & host
    for _ in range(max(0, int(max_steps))):
        grown = _dilate(out, 1) & host
        if np.array_equal(grown, out):
            break
        out = grown
    return out


def _box_pass(mask: np.ndarray, radius: int, *, grow: bool) -> np.ndarray:
    """方形结构元的膨胀/腐蚀：先沿行做 radius 次，再沿列做 radius 次。

    _dilate / _erode 用的是菱形结构元，拿它做开运算会把直角削成斜角。
    这里用方形：轴对齐的家具开完能原样恢复，不用再额外长几步去补角。
    """
    out = mask
    for axis in (0, 1):
        for _ in range(max(0, int(radius))):
            padded = np.pad(out, 1)
            if axis == 0:
                near, far = padded[0:-2, 1:-1], padded[2:, 1:-1]
            else:
                near, far = padded[1:-1, 0:-2], padded[1:-1, 2:]
            out = (out | near | far) if grow else (out & near & far)
    return out


def _fat_parts(mask: np.ndarray, radius: int) -> np.ndarray:
    """厚过 radius 的那些地方。

    开运算：腐蚀掉薄的部分，剩下的核再膨胀回来，留下的正好是「放得进
    一个 radius 见方的块」的那些位置。

    别改成形态学重建（核在整块里一路长回原形）。沙发贴着墙摆时两者在
    栅格上是连通的，重建会顺着墙爬出去，把沙发两侧一大段墙也吸进家具，
    那段墙线就再也画不出来了。要取整块请用 _fat_blobs，且只对本来就
    互不连通的东西用。
    """
    if radius <= 0 or not mask.any():
        return np.zeros_like(mask)
    core = _box_pass(mask, radius, grow=False)
    if not core.any():
        return np.zeros_like(mask)
    return _box_pass(core, radius, grow=True) & mask


def _fat_blobs(mask: np.ndarray, radius: int) -> np.ndarray:
    """含有厚核的那些整块，连边角一起取出来。

    只有在「块与块本来就不连通」时才能用，比如一堆各自封闭的空腔。
    对占据格不能用，理由见 _fat_parts。
    """
    if radius <= 0 or not mask.any():
        return np.zeros_like(mask)
    core = _box_pass(mask, radius, grow=False)
    if not core.any():
        return np.zeros_like(mask)
    # 腐蚀削的是厚度不是长度，核长回整块只要 radius 量级的步数
    return _grow_within(core, mask, max_steps=4 * radius + 16)


def _reach_from_border(free: np.ndarray, max_rounds: int = 64) -> np.ndarray:
    """从四边往里灌水，返回水能流到的空地。

    每轮沿四个方向各做一次扫描：一格能不能进水，看它这一行里最近的水
    是不是比最近的障碍还近。这样一轮就能跨过一整条直走廊，绕几个弯就
    收敛，不用一格一格往前推。
    """
    water = np.zeros_like(free)
    water[0, :] |= free[0, :]
    water[-1, :] |= free[-1, :]
    water[:, 0] |= free[:, 0]
    water[:, -1] |= free[:, -1]
    blocked = ~free
    for _ in range(max(1, int(max_rounds))):
        before = water
        for axis in (0, 1):
            span = free.shape[axis]
            pos = np.arange(span).reshape((-1, 1) if axis == 0 else (1, -1))
            for flip in (False, True):
                b = np.flip(blocked, axis) if flip else blocked
                w = np.flip(water, axis) if flip else water
                last_wall = np.maximum.accumulate(np.where(b, pos, -1), axis=axis)
                last_water = np.maximum.accumulate(np.where(w, pos, -1), axis=axis)
                wet = last_water > last_wall
                water = water | ((np.flip(wet, axis) if flip else wet) & free)
        if np.array_equal(water, before):
            break
    return water


def _enclosed_pockets(
    mask: np.ndarray, radius: int, seal: int = FURNITURE_SEAL_CELLS
) -> np.ndarray:
    """被占据格围死、而且围出来不大的空腔。

    桌子的四条腿加一圈横撑会在图上连成一个闭框，框里那块空地就是这里说的
    空腔。房间也是围死的，但它大得多——塞得进半径 radius 的圆就不算家具。
    """
    out = np.zeros_like(mask)
    if radius <= 0 or not mask.any():
        return out
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1
    # 只在有内容的范围里灌，外面留一圈空边当灌水口
    sealed = _close(np.pad(mask[top:bottom, left:right], seal + 1), seal)
    free = ~sealed
    holes = free & ~_reach_from_border(free)
    if not holes.any():
        return out
    # 房间那种大空腔整个排除掉，只留家具围出来的小兜。空腔之间隔着墙、
    # 本来就互不连通，这里可以放心用取整块的那个版本。
    holes &= ~_fat_blobs(holes, radius)
    edge = seal + 1
    out[top:bottom, left:right] = holes[edge:-edge, edge:-edge]
    return out


def _split_furniture(
    body: np.ndarray, *, fat_cells: int, pocket_cells: int
) -> Tuple[np.ndarray, np.ndarray]:
    """把占据格分成「墙」和「家具」两摊，只有前者该拿去细化。

    返回 (墙, 家具)。两者都挡底盘，区别只在画法：墙收成中轴线，
    家具保持团块——它压根就不是一条线，硬画成线就是图上那些假环。
    """
    furniture = _fat_parts(body, fat_cells)
    pockets = _enclosed_pockets(body, pocket_cells)
    if pockets.any():
        # 空腔的边界就是围它那一圈。补缺口时空腔缩了一点，膨胀要把这段让回来，
        # 再往外长几步，把圈本身的厚度也收进来。
        ring = _dilate(pockets, FURNITURE_SEAL_CELLS + 1) & body
        furniture |= _grow_within(ring, body, max_steps=FURNITURE_RING_STEPS)
    return body & ~furniture, furniture


def _thin(mask: np.ndarray, max_iter: int = WALL_THIN_MAX_ITER) -> np.ndarray:
    """Zhang-Suen 细化：把厚墙收成一格宽的中轴，且不切断连通性。

    位姿残差会让同一堵墙在相邻几格上各描一遍，闭运算再把中间填实，
    最后图上是一片黑块而不是一条墙。平面图里墙本来就是一条线，这里
    把厚度收掉只留中轴——位姿层还有残差时，画出来也仍然是线。
    """
    if not np.any(mask):
        return mask
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1
    # 只在有内容的范围里迭代，外面再留一圈空边，免得贴边的墙被当成边界
    img = np.pad(mask[top:bottom, left:right], 1)
    for _ in range(max(0, int(max_iter))):
        changed = False
        for step in (0, 1):
            padded = np.pad(img, 1)
            # 8 邻域，从正上方起顺时针
            around = (
                padded[0:-2, 1:-1],
                padded[0:-2, 2:],
                padded[1:-1, 2:],
                padded[2:, 2:],
                padded[2:, 1:-1],
                padded[2:, 0:-2],
                padded[1:-1, 0:-2],
                padded[0:-2, 0:-2],
            )
            north, east, south, west = around[0], around[2], around[4], around[6]
            neighbors = np.zeros(img.shape, dtype=np.int8)
            for one in around:
                neighbors += one.astype(np.int8)
            # 绕一圈只跨过一次「空→实」，说明这一格不是连接两段墙的桥
            ring = around + (around[0],)
            crossings = np.zeros(img.shape, dtype=np.int8)
            for i in range(8):
                crossings += (~ring[i] & ring[i + 1]).astype(np.int8)
            if step == 0:
                open_side = ~(north & east & south) & ~(east & south & west)
            else:
                open_side = ~(north & east & west) & ~(north & south & west)
            peel = (
                img
                & (neighbors >= 2)
                & (neighbors <= 6)
                & (crossings == 1)
                & open_side
            )
            if np.any(peel):
                img &= ~peel
                changed = True
        if not changed:
            break
    out = np.zeros_like(mask)
    out[top:bottom, left:right] = img[1:-1, 1:-1]
    return out


def _draw_trail(draw: ImageDraw.ImageDraw, ego: EgoMap, to_px) -> None:
    """一条蓝线，画完整 episode 轨迹；不把轨迹写回地图。"""
    points = ego.full_trail()
    if len(points) < 2:
        return
    pts = [to_px(p.x, p.y) for p in points]
    draw.line(pts, fill=_COLOR_TRAIL + (230,), width=3, joint="curve")


def _route_overlay_snapshot(session_id: str):
    """Read the optional policy-owned route without making mapping depend on it."""

    try:
        from behavior_interface_eval_test.navigation_route_overlay import (
            get_route_for_display,
        )

        return get_route_for_display(session_id)
    except (ImportError, RuntimeError, ValueError):
        return None


def _route_overlay_version(session_id: str) -> int:
    snapshot = _route_overlay_snapshot(session_id)
    return 0 if snapshot is None else int(snapshot.revision)


def _draw_navigation_route(
    draw: ImageDraw.ImageDraw,
    snapshot,
    to_px,
    *,
    canvas: int,
) -> None:
    """Draw only the untravelled part of the active planned polyline."""

    if snapshot is None or len(snapshot.points_xy_m) < 2:
        return
    points = [to_px(x_m, y_m) for x_m, y_m in snapshot.points_xy_m]
    draw.line(
        points,
        fill=_COLOR_NAVIGATION_ROUTE + (224,),
        width=max(3, int(round(int(canvas) / 150.0))),
        joint="curve",
    )


def _place_font_size(canvas: int) -> int:
    """按画布比例取字号：UI 会把图缩到小格子里，字必须一开始就够大。"""
    return max(22, int(round(int(canvas) * 0.055)))


def _draw_places(
    draw: ImageDraw.ImageDraw,
    ego: EgoMap,
    to_px,
    *,
    canvas: int = _DEFAULT_MAP_SIZE,
) -> None:
    """自己标的地点：唯一保留文字的元素。"""
    font = _load_font(_place_font_size(canvas))
    r = max(7, int(round(canvas * 0.012)))
    for place in ego.places:
        px, py = to_px(place.x, place.y)
        draw.polygon(
            [(px, py - r), (px + r, py), (px, py + r), (px - r, py)],
            fill=_COLOR_PLACE + (255,),
            outline=(15, 18, 24, 255),
        )
        tx, ty = _clamp_label_xy(draw, font, place.label, px + r + 4, py - r - 2, canvas)
        _text_with_halo(draw, (tx, ty), place.label, _COLOR_PLACE + (255,), font)


def _clamp_label_xy(
    draw: ImageDraw.ImageDraw,
    font,
    text: str,
    x: float,
    y: float,
    canvas: int,
) -> Tuple[float, float]:
    """把标签留在画布里，收紧取景后字不会被裁掉半截。"""
    try:
        left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    except Exception:
        return x, y
    pad = 4
    dx = 0.0
    dy = 0.0
    if right > canvas - pad:
        dx = (canvas - pad) - right
    if left + dx < pad:
        dx = pad - left
    if top < pad:
        dy = pad - top
    if bottom + dy > canvas - pad:
        dy = (canvas - pad) - bottom
    return x + dx, y + dy


def _draw_start(draw: ImageDraw.ImageDraw, ego: EgoMap, to_px) -> None:
    start = ego.landmarks.get("start")
    if start is None:
        return
    px, py = to_px(start.x, start.y)
    r = 5
    draw.ellipse(
        (px - r, py - r, px + r, py + r),
        fill=_COLOR_START + (255,),
        outline=(15, 18, 24, 255),
    )


def head_fov_deg() -> float:
    """head 相机水平视场角，由出厂内参算出（约 99°）。"""
    try:
        from behavior_interface.head_capture import (
            HEAD_FOCAL_LENGTH,
            HEAD_HORIZONTAL_APERTURE,
        )

        fl = float(HEAD_FOCAL_LENGTH)
        ha = float(HEAD_HORIZONTAL_APERTURE)
    except Exception:
        fl, ha = 17.0, 40.0
    if fl <= 1e-6:
        return 90.0
    return 2.0 * math.degrees(math.atan(ha * 0.5 / fl))


def _blend_view_cone(
    base: np.ndarray,
    ego: EgoMap,
    cx_px: float,
    cy_px: float,
    scale: float,
    *,
    heading_up: bool,
) -> None:
    """把视场画成从机器人向外发散、越远越淡的无边框光锥。

    顶角取 head 相机水平 FOV，直接在底图上按距离做 alpha 混合：PIL 画
    半透明多边形只能得到一块均匀的色片，做不出径向渐变。
    """
    height, width = base.shape[:2]
    reach_px = float(np.clip(VIEW_CONE_REACH_M * scale, 80.0, 180.0))
    # 车头朝上时画布已随 yaw 旋转过，光锥就该指正上方（0°），不能再转
    theta = math.radians(0.0 if heading_up else ego.yaw_deg)
    half_fov = math.radians(head_fov_deg() * 0.5)

    u0 = int(max(0, math.floor(cx_px - reach_px)))
    u1 = int(min(width, math.ceil(cx_px + reach_px) + 1))
    v0 = int(max(0, math.floor(cy_px - reach_px)))
    v1 = int(min(height, math.ceil(cy_px + reach_px) + 1))
    if u1 <= u0 or v1 <= v0:
        return

    us, vs = np.meshgrid(
        np.arange(u0, u1, dtype=np.float32),
        np.arange(v0, v1, dtype=np.float32),
    )
    # to_px 的逆：px = cx - left*scale, py = cy - up*scale
    left = cx_px - us
    up = cy_px - vs
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    forward = up * cos_t + left * sin_t
    lateral = -up * sin_t + left * cos_t
    dist = np.hypot(forward, lateral)
    with np.errstate(invalid="ignore"):
        angle = np.abs(np.arctan2(lateral, forward))

    alpha = VIEW_CONE_ALPHA * np.clip(1.0 - dist / reach_px, 0.0, 1.0)
    # 侧边羽化几度，避免出现一条硬邦邦的直边
    feather = math.radians(VIEW_CONE_FEATHER_DEG)
    alpha *= np.clip((half_fov - angle) / feather, 0.0, 1.0)
    alpha = alpha.astype(np.float32)[:, :, None]

    patch = base[v0:v1, u0:u1, :3].astype(np.float32)
    tint = np.asarray(_COLOR_VIEW_CONE, dtype=np.float32)
    blended = np.clip(patch + (tint - patch) * alpha, 0.0, 255.0)
    base[v0:v1, u0:u1, :3] = blended.astype(np.uint8)
    if base.shape[2] == 4:
        src_a = base[v0:v1, u0:u1, 3].astype(np.float32)
        cone = alpha[:, :, 0]
        base[v0:v1, u0:u1, 3] = np.clip(
            src_a + (255.0 - src_a) * cone * 0.88, 0.0, 255.0
        ).astype(np.uint8)


def _draw_robot(draw: ImageDraw.ImageDraw, ego: EgoMap, to_px, *, heading_up: bool) -> None:
    """机器人本体：光锥的顶点。锥体本身在底图上画（见 _blend_view_cone）。"""
    px, py = to_px(ego.x, ego.y)
    r = 4.0
    draw.ellipse(
        (px - r, py - r, px + r, py + r),
        fill=_COLOR_ROBOT + (255,),
        outline=(16, 22, 34, 255),
    )


def _text_with_halo(draw: ImageDraw.ImageDraw, xy, text: str, fill, font) -> None:
    x, y = xy
    for ox, oy in (
        (-2, 0), (2, 0), (0, -2), (0, 2),
        (-1, -1), (1, -1), (-1, 1), (1, 1),
        (-1, 0), (1, 0), (0, -1), (0, 1),
    ):
        draw.text((x + ox, y + oy), text, fill=(230, 232, 236, 240), font=font)
    draw.text((x, y), text, fill=fill, font=font)


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def flatten_rgba(
    rgba: np.ndarray,
    background: Tuple[int, int, int] = _COLOR_UNKNOWN,
) -> np.ndarray:
    """把半透明小地图铺到浅灰底上，给单独展示用。"""
    src = np.asarray(rgba)
    if src.ndim != 3 or src.shape[2] != 4:
        return np.asarray(src[..., :3], dtype=np.uint8)
    alpha = src[:, :, 3:4].astype(np.float32) / 255.0
    bg = np.asarray(background, dtype=np.float32)
    return np.clip(
        src[:, :, :3].astype(np.float32) * alpha + bg * (1.0 - alpha),
        0.0,
        255.0,
    ).astype(np.uint8)


def overlay_minimap_on_head(
    head_rgb: np.ndarray,
    minimap_rgba: np.ndarray,
    *,
    fraction: float = HEAD_MINIMAP_FRACTION,
) -> np.ndarray:
    """把半透明小地图贴到 head 画面右上角。"""
    head = np.asarray(head_rgb)
    if head.ndim != 3 or head.shape[2] < 3:
        raise ValueError("head_rgb 必须是 HxWx3")
    out = np.ascontiguousarray(head[:, :, :3].copy())
    mini = np.asarray(minimap_rgba)
    if mini.ndim != 3 or mini.shape[2] != 4:
        raise ValueError("minimap 必须是 HxWx4")
    height, width = out.shape[:2]
    side = max(96, int(round(min(height, width) * float(fraction))))
    margin = max(6, int(round(min(height, width) * HEAD_MINIMAP_MARGIN_FRAC)))
    try:
        resample = Image.Resampling.BILINEAR
    except AttributeError:
        resample = Image.BILINEAR
    mini_img = Image.fromarray(mini).resize((side, side), resample)
    src = np.asarray(mini_img)
    x0 = width - margin - side
    y0 = margin
    if x0 < 0 or y0 + side > height:
        return out
    patch = out[y0 : y0 + side, x0 : x0 + side].astype(np.float32)
    alpha = src[:, :, 3:4].astype(np.float32) / 255.0
    out[y0 : y0 + side, x0 : x0 + side] = np.clip(
        src[:, :, :3].astype(np.float32) * alpha + patch * (1.0 - alpha),
        0.0,
        255.0,
    ).astype(np.uint8)
    return out


def minimap_rgb_to_overlay_rgba(rgb: np.ndarray) -> np.ndarray:
    """把已画出的 RGB 小地图收成叠图层：未探索变半透明浅灰。

    兼容旧的深色底图：暗背景当未探索，浅色墙收成黑，中间灰收成白。
    """
    src = np.asarray(rgb)[:, :, :3]
    red = src[:, :, 0].astype(np.int16)
    green = src[:, :, 1].astype(np.int16)
    blue = src[:, :, 2].astype(np.int16)
    lum = (red + green + blue) / 3.0
    blueish = (blue > red + 25) & (blue > green)
    rgba = np.empty(src.shape[:2] + (4,), dtype=np.uint8)
    rgba[:, :, :3] = src
    rgba[:, :, 3] = 255

    unk = (
        (np.abs(red - _COLOR_UNKNOWN[0]) < 18)
        & (np.abs(green - _COLOR_UNKNOWN[1]) < 18)
        & (np.abs(blue - _COLOR_UNKNOWN[2]) < 18)
    ) | ((lum < 48) & ~blueish)
    rgba[unk, :3] = _COLOR_UNKNOWN
    rgba[unk, 3] = UNKNOWN_ALPHA

    old_free = (lum >= 48) & (lum < 115) & ~blueish & (np.abs(red - green) < 22)
    rgba[old_free, :3] = _COLOR_FREE
    rgba[old_free, 3] = FREE_ALPHA

    old_wall = (lum > 175) & ~blueish & (np.abs(red - green) < 28)
    rgba[old_wall, :3] = _COLOR_WALL
    rgba[old_wall, 3] = 255

    new_free = (red > 248) & (green > 248) & (blue > 248)
    rgba[new_free, :3] = _COLOR_FREE
    rgba[new_free, 3] = FREE_ALPHA
    return rgba


_RGBA_CACHE: Dict[str, Tuple[str, np.ndarray]] = {}


def render_minimap_rgba_cached(
    session_id: str = "",
    *,
    size: int = 480,
    heading_up: bool = True,
) -> Optional[np.ndarray]:
    """给 HUD 用的半透明小地图，内容没变就复用。"""
    key = str(session_id or "").strip() or active_session_id() or "default"
    ego = get_map(key)
    if not ego.initialized and ego.grid.frames <= 0:
        return None
    version = "{}|{}|{}".format(map_version(ego), int(heading_up), int(size))
    cache_key = f"{key}|{int(heading_up)}|{int(size)}"
    cached = _RGBA_CACHE.get(cache_key)
    if cached is not None and cached[0] == version:
        return cached[1]
    rgba = render_minimap_rgba(ego, size=int(size), heading_up=bool(heading_up))
    _RGBA_CACHE[cache_key] = (version, rgba)
    return rgba


def stamp_minimap_on_image(head_rgb: np.ndarray, session_id: str = "") -> np.ndarray:
    """在 head RGB 右上角盖当前小地图。地图还是空的就原样返回。"""
    if not spatial_map_enabled():
        return np.asarray(head_rgb)
    rgba = render_minimap_rgba_cached(session_id)
    if rgba is None:
        return np.asarray(head_rgb)
    return overlay_minimap_on_head(head_rgb, rgba)


def stamp_minimap_on_bgr(head_bgr: np.ndarray, session_id: str = "") -> np.ndarray:
    """live HUD 走 OpenCV，输入是 BGR。"""
    rgb = np.asarray(head_bgr)[:, :, ::-1]
    return stamp_minimap_on_image(rgb, session_id)[:, :, ::-1]


def compose_head_minimap(
    head_rgb: np.ndarray,
    minimap_rgb: np.ndarray,
    *,
    query: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """头视图 + 小地图并排，供回放看效果。"""
    head = np.asarray(head_rgb)
    mini = np.asarray(minimap_rgb)
    if head.ndim != 3:
        raise ValueError("head_rgb 必须是 HxWx3")
    target_h = int(head.shape[0])
    scale = target_h / float(mini.shape[0])
    new_w = max(1, int(round(mini.shape[1] * scale)))
    try:
        resample = Image.Resampling.NEAREST
    except AttributeError:
        resample = Image.NEAREST
    mini_img = Image.fromarray(mini).resize((new_w, target_h), resample)
    mini = np.asarray(mini_img)
    gap = np.full((target_h, 8, 3), 8, dtype=np.uint8)
    return np.concatenate([head, gap, mini], axis=1)


def _note_result_image_pose(session_id: str, payload: Dict[str, Any]) -> None:
    """把工具结果里带回的 image_id 和当前位姿绑上，供之后点选反解。"""
    image_id = str(payload.get("image_id") or "").strip()
    if image_id:
        get_map(session_id).note_image_pose(image_id)


def attach_to_result(
    session_id: str,
    tool: str,
    args: Optional[Dict[str, Any]],
    result: Optional[Dict[str, Any]],
    *,
    force: bool = False,
    depth_lookup: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
    save: bool = True,
) -> Dict[str, Any]:
    """底盘工具收尾：更新地图并附 hint / 小地图。"""
    payload = dict(result or {})
    if not force and not spatial_map_enabled():
        return payload
    adopt_session(session_id)
    name = str(tool or payload.get("tool") or "").strip()
    if name in EVENT_TOOLS and payload.get("ok", True) is not False:
        kind, label = EVENT_TOOLS[name]
        get_map(session_id).mark_event(kind, label)
    if not is_chassis_tool(name):
        # 非底盘工具（主要是各种 capture）位姿没变，直接记下这张图的拍摄位姿
        _note_result_image_pose(session_id, payload)
        return payload
    motion = extract_chassis_motion(
        name, args, payload, depth_lookup=depth_lookup
    )
    if (
        abs(motion.forward_m) < 1e-6
        and abs(motion.translation_m) < 1e-6
        and abs(motion.spin_deg) < 1e-6
        and motion.source == "none"
    ):
        query = get_map(session_id).query()
        payload["spatial_map"] = query
        return payload
    ego = get_map(session_id)
    ego.apply_motion(motion)
    # 底盘工具带回的图是动作走完之后拍的，位姿要在 apply_motion 之后才对
    _note_result_image_pose(session_id, payload)
    _integrate_session_capture(ego, session_id, motion.image_id, depth_lookup)
    query = ego.query()
    payload["spatial_map"] = query
    rgb = render_minimap(ego)
    image_id = str(motion.image_id or payload.get("image_id") or "map")
    if save and session_id:
        try:
            from behavior_interface import agent_runs

            path = agent_runs.image_path(session_id, image_id, ".minimap.png")
            Image.fromarray(rgb).save(path)
            payload["minimap_path"] = path
            payload["rgb_minimap"] = agent_runs.file_to_data_url(path)
        except Exception:
            payload["rgb_minimap"] = _rgb_to_data_url(rgb)
    else:
        payload["rgb_minimap"] = _rgb_to_data_url(rgb)
    return payload


def _integrate_session_capture(
    ego: EgoMap,
    session_id: str,
    image_id: str,
    depth_lookup: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
) -> bool:
    """底盘动作后的 exit capture：把这一帧 depth 并进占用栅格。"""
    image_id = str(image_id or "")
    if not image_id:
        return False
    bundle = None
    if depth_lookup is not None:
        bundle = depth_lookup(image_id)
    if bundle is None and session_id:
        bundle = load_capture_bundle(session_id, image_id)
    return ego.integrate_capture(bundle, image_id=image_id)


def build_query_result(
    session_id: str,
    *,
    save: bool = True,
    label: str = "",
    heading_up: bool = True,
) -> Dict[str, Any]:
    adopt_session(session_id)
    ego = get_map(session_id)
    marked = ego.mark_place(label) if label else None
    query = ego.query()
    if marked is not None:
        query["marked_place"] = marked.label
    rgb = render_minimap(ego, heading_up=bool(heading_up))
    result = {
        "ok": True,
        "tool": "query_map",
        "build": BUILD,
        "spatial_map": query,
        "hint": query.get("hint"),
        "rgb_main": _rgb_to_data_url(rgb),
        "image_id": "minimap_current",
    }
    if save and session_id:
        try:
            from behavior_interface import agent_runs

            path = agent_runs.image_path(session_id, "minimap_current", ".png")
            Image.fromarray(rgb).save(path)
            result["rgb_main_path"] = path
            result["rgb_main"] = agent_runs.file_to_data_url(path) or result["rgb_main"]
        except Exception:
            pass
    return result


MARK_TOOL = "mark_on_map"


def build_mark_on_map_result(
    session_id: str,
    name: str,
    *,
    image_id: str = "",
    u: Any = None,
    v: Any = None,
    depth_lookup: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """在地图上标一个具名地点：点选 head 图上的物体，或者标记脚下。

    给了 image_id + (u, v) 就把那个像素按该帧 depth 反解成三维点，取它的
    地面投影；否则标记当前底盘位置。
    """
    label = str(name or "").strip()
    if not label:
        return {"ok": False, "tool": MARK_TOOL, "error": "需要 name"}
    adopt_session(session_id)
    ego = get_map(session_id)
    ego.ensure_start(tool=MARK_TOOL)

    picked = str(image_id or "").strip()
    at: Optional[Tuple[float, float]] = None
    source = "robot_position"
    if picked or u is not None or v is not None:
        if not (picked and u is not None and v is not None):
            return {
                "ok": False,
                "tool": MARK_TOOL,
                "error": "点选要同时给 image_id、u、v；只标脚下就三个都别给",
            }
        if depth_lookup is None:
            def depth_lookup(key: str) -> Optional[Dict[str, Any]]:
                return load_capture_bundle(session_id, key)

        robot_xy = unproject_uv_robot_xy(picked, u, v, depth_lookup=depth_lookup)
        if robot_xy is None:
            return {
                "ok": False,
                "tool": MARK_TOOL,
                "error": (
                    f"{picked} 的 ({u},{v}) 反解不出三维点："
                    "该图没有可用 depth，或点到了无穷远/无效像素。"
                    "换 capture_head_camera 新拍一张，点在物体实体上再试。"
                ),
            }
        at = ego.robot_xy_to_map(robot_xy[0], robot_xy[1], image_id=picked)
        source = "image_pick"

    marked = ego.mark_place(
        label,
        at=at,
        submap_serial=(ego.image_pose_submaps.get(picked) if picked else None),
    )
    if marked is None:
        return {"ok": False, "tool": MARK_TOOL, "error": "name 无效"}
    range_m, bearing = ego.bearing_to(marked.x, marked.y)
    where = (
        f"点选 {picked}({u},{v})" if source == "image_pick" else "机器人脚下"
    )
    return {
        "ok": True,
        "tool": MARK_TOOL,
        "build": BUILD,
        "marked": marked.label,
        "marked_from": source,
        "marked_range_m": round(range_m, 2),
        "marked_spin_deg_to_face": round(wrap_deg(bearing), 1),
        "places": ego.place_reports(),
        "hint": (
            f"已把{where}标记为「{marked.label}」，距当前底盘约 {range_m:.1f}m，"
            f"spin {wrap_deg(bearing):+.0f}° 可正对。地图上会一直显示。"
        ),
    }


# 旧名字：早期只能标脚下，保留给已经写死这个入口的调用方
build_mark_position_result = build_mark_on_map_result


def places_memory_lines(session_id: str = "") -> List[str]:
    """给 memory 面板用的一行一个地点：方位 + 距离 + 该转多少度。"""
    key = str(session_id or "").strip() or active_session_id()
    if not key:
        return []
    ego = get_map(key)
    lines: List[str] = []
    for item in ego.place_reports():
        spin = item["spin_deg_to_face"]
        lines.append(
            f"{item['name']}：{item['direction']}，约 {item['range_m']:.1f}m"
            f"（spin {spin:+.0f}° 正对）"
        )
    return lines


_SNAPSHOT_CACHE: Dict[str, Tuple[str, bytes]] = {}


def map_version(ego: EgoMap) -> str:
    """地图内容指纹：只有这些变了才需要重画。

    位姿必须算进来。车头朝上时整张画布跟着 yaw 转，视窗中心也跟着机器人走，
    位姿一变画面就完全不同。而实时里程只更新位姿，既不碰 update_count 也不
    碰 grid.frames——漏掉位姿的话，原地转身后拿到的还是上一个朝向画的图，
    看上去就是「明明正对着桌子，图上桌子却在右边」。
    """
    return "{}:{}:{}:{}:{}:{}:{}:{}:{}:{:.2f}:{:.2f}:{:.1f}:{}:{}".format(
        ego.update_count,
        ego.grid.frames,
        len(ego.places),
        len(ego.events),
        len(ego.landmarks),
        ego.mapping_state,
        ego.freeze_frame if ego.freeze_frame is not None else -1,
        ego.freeze_grid_revision if ego.freeze_grid_revision is not None else -1,
        ego.grid._rev,
        ego.x,
        ego.y,
        ego.yaw_deg,
        ego.freeze_grid_digest[:12],
        _route_overlay_version(ego.session_id),
    )


map_version._navigation_route_overlay_native = True


def map_snapshot_png(
    session_id: str,
    *,
    heading_up: bool = True,
    size: int = _DEFAULT_MAP_SIZE,
) -> Tuple[bytes, str]:
    """渲染当前地图为 PNG。内容没变就复用上次结果，避免轮询反复重画。"""
    import io

    # 只读展示，不能 adopt：UI 轮询带的是它自己的 session id，接管过去会让
    # agent 之后用真实 session 查到一张空图。
    ego = get_map(session_id)
    version = "{}|{}|{}".format(map_version(ego), int(heading_up), int(size))
    cache_key = f"{session_id}|{int(heading_up)}|{int(size)}"
    cached = _SNAPSHOT_CACHE.get(cache_key)
    if cached is not None and cached[0] == version:
        return cached[1], version
    rgb = render_minimap(ego, size=size, heading_up=heading_up)
    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgb)).save(buf, format="PNG")
    png = buf.getvalue()
    _SNAPSHOT_CACHE[cache_key] = (version, png)
    return png, version


def _rgb_to_data_url(rgb: np.ndarray) -> str:
    import base64
    import io

    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgb)).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def replay_tool_cards(
    cards: Iterable[Dict[str, Any]],
    *,
    session_id: str,
    depth_lookup: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
    only_ok: bool = True,
) -> EgoMap:
    """按 monitor 卡片回放底盘工具，构建地图。"""
    reset_map(session_id)
    ego = get_map(session_id)
    for card in cards:
        if not isinstance(card, dict):
            continue
        if only_ok and card.get("ok") is False:
            continue
        tool = str(card.get("tool") or "")
        if tool in EVENT_TOOLS:
            kind, label = EVENT_TOOLS[tool]
            ego.mark_event(kind, label)
            continue
        if not is_chassis_tool(tool):
            continue
        args = dict(card.get("args") or {})
        nested = card.get("result")
        result = dict(nested) if isinstance(nested, dict) else {}
        result.setdefault("ok", card.get("ok", True))
        result.setdefault("tool", tool)
        result["image_id"] = str(
            card.get("output_image_id")
            or card.get("image_id")
            or result.get("image_id")
            or ""
        )
        motion = extract_chassis_motion(
            tool, args, result, depth_lookup=depth_lookup
        )
        motion.image_id = str(result.get("image_id") or "")
        motion.tool = tool
        ego.apply_motion(motion)
        _integrate_session_capture(
            ego, session_id, motion.image_id, depth_lookup
        )
    return ego


def load_human_turns(turn_dir: str) -> List[Dict[str, Any]]:
    """读取 human_recording 的 turns/turn_XXXX.json。"""
    turns: List[Dict[str, Any]] = []
    for path in sorted(Path(turn_dir).glob("turn_*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(item, dict):
            turns.append(item)
    return turns


def human_turns_to_cards(
    turns: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """把 human_recording turn 转成 monitor card 的兼容字段。"""
    cards: List[Dict[str, Any]] = []
    for index, turn in enumerate(turns, 1):
        if not isinstance(turn, dict):
            continue
        action = dict(turn.get("action") or {})
        output = dict(turn.get("output") or {})
        response = dict(output.get("response") or {})
        displayed = dict((turn.get("input") or {}).get("displayed_media") or {})
        image_id = str(response.get("image_id") or "")
        cards.append({
            "turn_index": index,
            "tool": str(action.get("tool_name") or ""),
            "args": dict(action.get("args") or {}),
            "ok": response.get("ok"),
            "input_image_id": str(displayed.get("image_id") or ""),
            "image_id": image_id,
            "output_image_id": image_id,
            # replay_tool_cards 会把完整 response 交给 extract_chassis_motion，
            # 这样 actual.* 不会在字段转换时丢掉。
            "result": response,
        })
    return cards


def capture_local_odometry_pose(
    bundle: Optional[Dict[str, Any]],
) -> Optional[Tuple[float, float, float]]:
    """取归档帧里的 local_command_odometry 位姿；它是本体感知，不是 GT。"""
    meta = dict((bundle or {}).get("meta") or {})
    robot = dict(meta.get("robot") or {})
    pose = dict(robot.get("base_pose") or {})
    pos = pose.get("pos") or []
    if len(pos) < 2 or pose.get("yaw_deg") is None:
        return None
    values = (
        _finite(pos[0], default=float("nan")),
        _finite(pos[1], default=float("nan")),
        _finite(pose.get("yaw_deg"), default=float("nan")),
    )
    if not all(math.isfinite(value) for value in values):
        return None
    return values


def recorded_capture_motion(
    previous: Optional[Dict[str, Any]],
    current: Optional[Dict[str, Any]],
) -> Optional[MotionDelta]:
    """用两帧 local_command_odometry 求前一机体系下的真实完成增量。"""
    before = capture_local_odometry_pose(previous)
    after = capture_local_odometry_pose(current)
    if before is None or after is None:
        return None

    before_meta = dict((previous or {}).get("meta") or {})
    after_meta = dict((current or {}).get("meta") or {})
    before_robot = dict(before_meta.get("robot") or {})
    after_robot = dict(after_meta.get("robot") or {})
    before_episode = str(before_robot.get("episode_id") or "")
    after_episode = str(after_robot.get("episode_id") or "")
    if before_episode and after_episode and before_episode != after_episode:
        return None
    before_epoch = _first_number(before_robot, "motion_epoch")
    after_epoch = _first_number(after_robot, "motion_epoch")
    if (
        before_epoch is not None
        and after_epoch is not None
        and after_epoch < before_epoch
    ):
        return None

    x0, y0, yaw0 = before
    x1, y1, yaw1 = after
    dx, dy = x1 - x0, y1 - y0
    angle = math.radians(yaw0)
    cos_y, sin_y = math.cos(angle), math.sin(angle)
    return MotionDelta(
        forward_m=cos_y * dx + sin_y * dy,
        translation_m=-sin_y * dx + cos_y * dy,
        spin_deg=wrap_deg(yaw1 - yaw0),
        source="recorded_local_command_odometry",
    )


def replay_human_turns(
    turns: Iterable[Dict[str, Any]],
    *,
    session_id: str,
    depth_lookup: Callable[[str], Optional[Dict[str, Any]]],
    capture_image_ids: Optional[Iterable[str]] = None,
    through_turn: Optional[int] = None,
    scan_match: bool = True,
    global_optimize: bool = False,
) -> EgoMap:
    """离线重放 human_recording；cancelled turn 也按录制里程处理。

    turn 的 exit response 偶尔不带 image_id，但动作已经发生。相反，runs
    里还有工具内部留下、未被 turn 直接引用的 depth。这里以连续录制帧为
    主线，相邻帧 base_pose 的差就是该段真实完成量；因此既不会漏帧，也
    不会把 cancelled 的命令目标误当成实际完成量。
    """
    selected = [turn for turn in turns if isinstance(turn, dict)]
    if through_turn is not None:
        selected = selected[:max(0, int(through_turn))]
    cards = human_turns_to_cards(selected)
    output_ids = [
        str(card.get("output_image_id") or "")
        for card in cards
        if card.get("output_image_id")
    ]
    if capture_image_ids is None:
        image_ids = list(dict.fromkeys(output_ids))
    else:
        image_ids = sorted({str(image_id) for image_id in capture_image_ids if image_id})
        if through_turn is not None:
            # turn 1--2 可能仍引用上一 episode 的 img_0136。它不在本归档里，
            # 不能拿字符串大小当截止点，否则会把当前包的 27 帧全放进“turn 1”。
            # 只信当前归档真实存在的输出帧；没有一个命中就应当重放 0 帧。
            positions = {image_id: index for index, image_id in enumerate(image_ids)}
            known_cutoffs = [
                positions[image_id]
                for image_id in output_ids
                if image_id in positions
            ]
            image_ids = (
                image_ids[:max(known_cutoffs) + 1]
                if known_cutoffs
                else []
            )

    tool_by_image = {
        str(card.get("output_image_id")): str(card.get("tool") or "")
        for card in cards
        if card.get("output_image_id")
    }

    # 先把整段完整载入；这样归档缺帧会在写地图之前报错，不能留下半张
    # 看似可用的结果。这里明确不调用任何“未来帧”批优化：它会把离线
    # 标注信息泄漏回历史位姿，和在线建图契约不一致。
    captures: List[Tuple[str, Dict[str, Any]]] = []
    for image_id in image_ids:
        bundle = depth_lookup(image_id)
        if not bundle or bundle.get("depth") is None:
            # 没给 capture 清单时只能从 turn 猜帧；human session 开头可能引用
            # 上一 episode 的图片，明确允许跳过。给了清单则缺帧就是归档损坏，
            # 必须报错，不能把完整性问题悄悄吞掉。
            if capture_image_ids is None:
                continue
            raise ValueError(f"{image_id} 缺少可回放 depth")
        if capture_local_odometry_pose(bundle) is None:
            raise ValueError(f"{image_id} 缺少 local_command_odometry base_pose")
        captures.append((image_id, bundle))

    optimization_stats: Dict[str, Any] = {
        "accepted": False,
        "applied": False,
        "reason": (
            "too_few_frames"
            if len(captures) < 8
            else "future_frame_optimizer_removed"
        ),
        "requested": bool(global_optimize),
    }

    reset_map(session_id)
    ego = get_map(session_id)
    ego.global_pose_optimization = optimization_stats
    previous: Optional[Dict[str, Any]] = None
    for index, (image_id, bundle) in enumerate(captures):
        if previous is None:
            motion = MotionDelta(source="recorded_local_command_odometry")
        else:
            motion = recorded_capture_motion(previous, bundle)
            if motion is None:
                raise ValueError(f"{image_id} 与前一帧不属于连续里程序列")
        motion.image_id = image_id
        motion.tool = tool_by_image.get(image_id, "recorded_capture")
        # 走 live_odometry 分支，submap 的 travel/turn 阈值才能和在线路径一致。
        ego.advance_relative_odometry(
            motion.forward_m, motion.translation_m, motion.spin_deg
        )
        ego.apply_motion(motion)
        ego.note_image_pose(image_id)
        ego.integrate_capture(
            bundle,
            image_id=image_id,
            scan_match=scan_match,
        )
        previous = bundle
    return ego


def load_cards_jsonl(path: str) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                cards.append(item)
    return cards
