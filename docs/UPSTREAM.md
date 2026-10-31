# 上游来源

本仓库基于 Huy-Quang-Dao 的 [InstantTrack](https://github.com/Huy-Quang-Dao/instantTrack)，固定提交为 `a33691a08bca8802d9d2985f62e2567479735a8e`。

InstantTrack 自身基于 Chong Zhang 的 [supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) mjlab 实现，并引用 SuperMimic 的可微世界模型训练方法。保留的许可位于根目录 `LICENSE`、`THIRD_PARTY_NOTICES.md`。

新增工作包括根节点位置反馈、保持共同初始化的额外输入连接、三种子控制实验，以及便于独立使用的配置、结果复算和检查点工具。`torso` 是已记录的历史消融，推荐配置为 `root`。

`reproducibility/validated_position_feedback.patch` 是完成三种子实验时相对于上游的源码补丁。发布整理中的额外修改包括默认根节点入口、可配置数据目录、旧检查点兼容和 ONNX 位置元数据；新整理未重新训练三个种子。

本仓库使用独立、精简的 Git 历史，以上游固定提交和补丁记录来源。未复制上游的示范动图、历史实验数据包或其性能宣传表。
