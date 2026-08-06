# 第 3 章完整源码导读

本文件配合项目源码使用。源码保持“一项操作一段代码”的教学风格，
因此很多表达式刻意拆成多行，方便逐行阅读和调试。

## 建议运行顺序

1. 把 `bank-full.csv` 放到 `data/raw/`。
2. 运行 `python -m src.train_baseline`。
3. 运行 `python -m src.train_xgboost`。
4. 运行 `python -m src.threshold_tuning`。
5. 运行 `python -m src.explain_model`。
6. 运行 `python -m src.predict`。

## 文件之间的依赖关系

```text
config.py
   ↓
data_loader.py
   ↓
preprocessing.py
   ↓
evaluate.py
   ↓
train_baseline.py / train_xgboost.py
   ↓
threshold_tuning.py
   ↓
explain_model.py
   ↓
predict.py
```

## 逐文件说明

### `src/__init__.py`

- 把 `src` 声明为 Python package。
- 因此可以用 `python -m src.train_xgboost` 运行模块。
- Docstring 说明 package 的用途。

### `src/config.py`

- `Path(__file__).resolve()`：取得当前文件的绝对路径。
- 第一个 `.parent`：进入 `src`。
- 第二个 `.parent`：进入项目根目录。
- `Final[...]`：表达“这是项目常量，不应在运行中重新赋值”。
- 所有数据、模型和报告路径集中在一个文件，避免其他模块重复硬编码。
- `TARGET_COLUMN`：原始目标字段 `y`。
- `TARGET_MAPPING`：把 `no/yes` 映射成 `0/1`。
- `LEAKAGE_COLUMNS`：明确删除 `duration`。
- `TEST_SIZE`、`RANDOM_STATE`：保证所有模型使用同样拆分。
- `create_project_directories()`：训练前统一创建目录。

### `src/data_loader.py`

- `load_bank_data()` 接收可选 `Path`，默认读取配置文件中的路径。
- `file_path.exists()` 在读取前主动检查文件。
- `pd.read_csv(..., sep=";")` 使用该数据集正确的分隔符。
- 空 DataFrame、缺少 Target、重复列名都会主动报错。
- 函数只负责读取和基础结构验证，不负责建模。

### `src/preprocessing.py`

#### `create_features_and_target()`

- 检查 Target 和 Leakage 字段。
- `.map(TARGET_MAPPING)` 创建整数 Target。
- `target.isnull()` 用于发现无法映射的新值。
- `columns_to_drop` 同时包含 Target 与 Leakage。
- 返回 `features, target`，不在此函数内拆分数据。

#### `split_training_and_test_data()`

- 统一调用 `train_test_split`。
- `test_size=0.20`。
- `random_state=42`。
- `stratify=target` 保持类别比例。
- 所有模型都调用它，保证比较公平。

#### `identify_feature_types()`

- `select_dtypes(include="number")` 找数值字段。
- `object/category/bool` 作为分类字段。
- 主动检查未识别 dtype，避免字段被静默丢弃。

#### `build_preprocessor()`

- Numeric Pipeline：Median Imputer → StandardScaler。
- Categorical Pipeline：Most-Frequent Imputer → OneHotEncoder。
- `handle_unknown="ignore"` 允许预测时出现训练中未见类别。
- `sparse_output=True` 节省 One-Hot 数据内存。
- `ColumnTransformer` 把两套处理应用到相应列。
- `verbose_feature_names_out=True` 产生 `numeric__age` 一类清晰名称。

### `src/evaluate.py`

#### `_to_numpy()`

- 把 Series、List 或其他 array-like 统一转换成 NumPy Array。
- 下划线前缀表示内部辅助函数。

#### `validate_binary_evaluation_inputs()`

- 检查三个输入长度一致。
- 检查真实值和预测值只包含 0/1。
- 检查概率没有 NaN 且位于 0–1。

#### `calculate_classification_metrics()`

- 计算 Accuracy、Precision、Recall、F1。
- ROC-AUC 和 Average Precision 使用 Probability。
- `confusion_matrix(..., labels=[0, 1])` 固定矩阵顺序。
- `.ravel()` 按 TN、FP、FN、TP 展平。
- 所有 NumPy 数值转为标准 `float/int`，方便 JSON 保存。

#### 图表函数

- 每个函数都自己创建 Figure。
- 保存后调用 `plt.close()`，避免批量运行时积累内存。
- ROC 图包含随机分类器基准线。
- Precision-Recall 图包含 Positive-Class Prevalence 基准线。

### `src/train_baseline.py`

- 保留 Logistic Regression 作为比较基准。
- 使用同一个 `build_preprocessor()`。
- `class_weight="balanced"` 处理类别不平衡。
- 保存完整 Pipeline，而不是只保存 classifier。
- 生成 Baseline Metrics、Predictions 和 Confusion Matrix。
- 返回训练对象和数组，方便 Notebook 或其他模块复用。

### `src/train_xgboost.py`

#### `calculate_scale_pos_weight()`

- 统计训练集 Negative 和 Positive 数量。
- 返回 `negative_count / positive_count`。
- 只根据 `y_train` 计算，不根据 Test Set 计算。
- 任一类别为空时主动报错。

#### `build_xgboost_pipeline()`

- 识别字段类型并建立同一套 Preprocessor。
- 创建 `XGBClassifier`。
- `n_estimators`：树的数量。
- `learning_rate`：每棵树的更新步长。
- `max_depth`：树的最大深度。
- `min_child_weight`：叶节点继续分裂的保守程度。
- `subsample`：每棵树抽取的记录比例。
- `colsample_bytree`：每棵树抽取的特征比例。
- `objective="binary:logistic"`：二元概率分类。
- `eval_metric="logloss"`：训练评估指标。
- `scale_pos_weight`：提高少数 Positive Class 的训练权重。
- `tree_method="hist"`：使用高效 Histogram Tree Method。

#### `run_cross_validation()`

- 使用 `StratifiedKFold(n_splits=5)`。
- Cross Validation 只在 `X_train/y_train` 内执行。
- Estimator 是完整 Pipeline，因此每个 Fold 都重新 Fit Preprocessing。
- 同时返回 Train 和 Validation 指标。
- Train 与 Validation 差距可用于判断 Overfitting。

#### `compare_with_baseline()`

- Baseline Model 存在时加载并在同一 Test Set 上预测。
- 不存在时仍可完成 XGBoost 训练，只生成一行比较结果。

#### `train_xgboost_model()`

- 完成读取、拆分、CV、Fit、Predict、Evaluate、Save、Plot。
- 最终 Test Set 只用于最后评估。
- 保存 `xgboost_pipeline.pkl` 和所有第 3 章报告。

### `src/threshold_tuning.py`

#### `predictions_from_probability()`

- 使用 `probability >= threshold` 产生 Boolean Array。
- `.astype(int)` 转成 0/1。
- 这段是从 Probability 到 Business Decision 的核心代码。

#### `evaluate_thresholds()`

- 对每个 Threshold 重复生成预测。
- ROC-AUC 与 AP 不随 Threshold 改变，因为它们仍使用原始 Probability。
- Precision、Recall、F1、FP、FN 会随 Threshold 改变。

#### `select_threshold_by_recall()`

- 先过滤 `recall >= minimum_recall`。
- 再选 Precision 最高的候选。
- Precision 相同时依次比较 F1 和 Threshold。
- 这是教学用的明确业务规则，不代表真实 AML Policy。

### `src/explain_model.py`

#### Feature Importance

- 从 `pipeline.named_steps` 取得已训练的 Preprocessor 和 Classifier。
- `get_feature_names_out()` 获得 One-Hot 后名称。
- `feature_importances_` 获得模型的重要性数值。
- 两者长度必须完全一致。
- 输出 CSV 和 Top-20 图。

#### SHAP

- 先用已训练 Preprocessor `.transform(X_test)`。
- 绝不能对 Test Set 使用 `fit_transform()`。
- Sparse Matrix 仅在抽样解释前转 Dense。
- `TreeExplainer` 专门解释 Tree Ensemble。
- Summary Plot 表示全局贡献分布。
- Waterfall Plot 只解释一条记录。
- SHAP 解释模型行为，不证明现实中的因果关系。

### `src/predict.py`

- 加载完整 Pipeline 和 Threshold 配置。
- `feature_names_in_` 是模型期待的原始字段。
- 主动检查 Missing 和 Extra Fields。
- 按训练时字段顺序重新排列输入。
- `predict_proba(... )[:, 1]` 取得 Positive Probability。
- Threshold 转换成 0/1 Decision。
- 支持 Single 和 Batch Prediction。
- Risk Band 边界仅是演示占位值，不是正式 AML Policy。
- `MODEL_VERSION` 为第 4 章 API Response 做准备。

## 必须熟练掌握的代码

```python
pipeline.fit(
    X_train,
    y_train,
)
```

```python
positive_probability = (
    pipeline
    .predict_proba(
        X_test
    )[:, 1]
)
```

```python
prediction = (
    positive_probability
    >= threshold
).astype(int)
```

```python
scale_pos_weight = (
    negative_count
    / positive_count
)
```

```python
pipeline.named_steps[
    "preprocessor"
]
```

```python
pipeline.named_steps[
    "classifier"
]
```

```python
joblib.dump(
    pipeline,
    model_path,
)
```

```python
pipeline = joblib.load(
    model_path
)
```

## 只需理解、暂不要求默写

- 完整 Matplotlib 绘图实现。
- SHAP 图形保存细节。
- Cross Validation 汇总字典循环。
- 输入验证函数的全部边界条件。
- Type Hint 中较复杂的嵌套类型。
- `RandomizedSearchCV`，本版本暂未默认运行，以避免教材项目执行成本过高。

## 重要业务说明

当前项目虽然以 AML Risk Scoring 为学习目标，但使用的是 Bank Marketing
Dataset。当前 Target 的真实含义是“是否订阅定期存款”。Probability、
Threshold、Risk Band 和 SHAP 流程可迁移到未来 AML 数据，但真实 AML
项目必须重新设计 Label、Observation Window、Performance Window、
Leakage Rules、Governance 和 Business Threshold。
