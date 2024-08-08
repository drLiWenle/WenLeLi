
# install.packages("tidymodels")
library(tidymodels)
library(bonsai)
setwd("D:/R work")
source("tidyfuncs4cls2_v18.R")

# 多核并行
library(doParallel)
registerDoParallel(
  makePSOCKcluster(
    max(1, (parallel::detectCores(logical = F))-1)
  )
)
# 读取数据
# file.choose()
Heart <- readr::read_csv(file.choose())
colnames(Heart) 
# 修正变量类型
# 将分类变量转换为factor
for(i in c(1:8)){ 
  Heart[[i]] <- factor(Heart[[i]])
}

# 删除无关变量在此处进行
Heart$Id <- NULL
# 删除含有缺失值的样本在此处进行，填充缺失值在后面
Heart <- na.omit(Heart)
# Heart <- Heart %>%
#   drop_na(Thal)

# 变量类型修正后数据概况
skimr::skim(Heart)    

# 设定阳性类别和阴性类别
yourpositivelevel <- "Yes"
yournegativelevel <- "No"
# 转换因变量的因子水平，将阳性类别设定为第二个水平
levels(Heart$Operation_time)
table(Heart$Operation_time)
Heart$Operation_time <- factor(
  Heart$Operation_time,
  levels = c(yournegativelevel, yourpositivelevel)
)
levels(Heart$Operation_time)
table(Heart$Operation_time)


# 数据拆分
set.seed(42)
datasplit <- initial_split(Heart, prop = 0.7, strata = Operation_time)
traindata <- training(datasplit)
testdata <- testing(datasplit)


#不拆分，外部验证
traindata <- Heart
aa <- readr::read_csv(file.choose())
colnames(aa) 
# 修正变量类型
# 将分类变量转换为factor
for(i in c(1:8)){ 
  aa[[i]] <- factor(aa[[i]])
}
testdata <- aa


# 重抽样设定-5折交叉验证
set.seed(42)
folds <- vfold_cv(traindata, v = 10, strata = Operation_time)
folds



# 数据预处理配方
datarecipe_dt <- recipe(formula = Operation_time ~ ., traindata)
datarecipe_dt



# 设定模型
model_dt <- decision_tree(
  mode = "classification",
  engine = "rpart",
  tree_depth = tune(),
  min_n = tune(),
  cost_complexity = tune()
) %>%
  set_args(model=T)
model_dt

# workflow
wk_dt <- 
  workflow() %>%
  add_recipe(datarecipe_dt) %>%
  add_model(model_dt)
wk_dt

##############################################################
############################  超参数寻优2选1-网格搜索

# 超参数寻优网格
set.seed(42)
hpgrid_dt <- parameters(
  tree_depth(range = c(3, 7)),
  min_n(range = c(5, 10)),
  cost_complexity(range = c(-6, -3))
) %>%
  # grid_regular(levels = c(3, 2, 4)) # 常规网格
  grid_random(size = 20) # 随机网格
# grid_latin_hypercube(size = 10) # 拉丁方网格
# grid_max_entropy(size = 10) # 最大熵网格
hpgrid_dt
log10(hpgrid_dt$cost_complexity)
# 网格也可以自己手动生成expand.grid()
# hpgrid_dt <- expand.grid(
#   tree_depth = c(2:5),
#   min_n = c(5, 11),
#   cost_complexity = 10^(-5:-1)
# )

# 交叉验证网格搜索过程
set.seed(42)
tune_dt <- wk_dt %>%
  tune_grid(
    resamples = folds,
    grid = hpgrid_dt,
    metrics = metricset_cls2,
    control = control_grid(save_pred = T, 
                           verbose = T,
                           event_level = "second",
                           parallel_over = "everything",
                           save_workflow = T)
  )

#########################  超参数寻优2选1-贝叶斯优化

# 贝叶斯优化超参数
set.seed(42)
tune_dt <- wk_dt %>%
  tune_bayes(
    resamples = folds,
    initial = 10,
    iter = 50,
    metrics = metricset_cls2,
    control = control_bayes(save_pred = T, 
                            verbose = T,
                            no_improve = 10,
                            uncertain = 5,
                            event_level = "second",
                            parallel_over = "everything",
                            save_workflow = T)
  )

########################  超参数寻优结束


# 交叉验证结果
eval_tune_dt <- tune_dt %>%
  collect_metrics()
eval_tune_dt

# 图示
# autoplot(tune_dt)
eval_tune_dt %>% 
  filter(.metric == "roc_auc") %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'cost_complexity', values = ~cost_complexity),
      list(label = 'tree_depth', values = ~tree_depth),
      list(label = 'min_n', values = ~min_n)
    )
  ) %>%
  plotly::layout(title = "DT HPO Guided by AUCROC",
                 font = list(family = "serif"))

# 经过交叉验证得到的最优超参数
hpbest_dt <- tune_dt %>%
  select_by_one_std_err(metric = "roc_auc", desc(cost_complexity))
hpbest_dt

# 采用最优超参数组合训练最终模型
set.seed(42)
final_dt <- wk_dt %>%
  finalize_workflow(hpbest_dt) %>%
  fit(traindata)
final_dt

##################################################################

# 训练集预测评估
predtrain_dt <- eval4cls2(
  model = final_dt, 
  dataset = traindata, 
  yname = "Operation_time", 
  modelname = "DT", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_dt$prediction
predtrain_dt$predprobplot
predtrain_dt$rocplot
predtrain_dt$prplot
predtrain_dt$caliplot
predtrain_dt$cmplot
predtrain_dt$metrics
predtrain_dt$diycutoff
predtrain_dt$ksplot

# pROC包auc值及其置信区间
pROC::auc(predtrain_dt$proc)
pROC::ci.auc(predtrain_dt$proc)

# 预测评估测试集预测评估
predtest_dt <- eval4cls2(
  model = final_dt, 
  dataset = testdata, 
  yname = "Operation_time", 
  modelname = "DT", 
  datasetname = "testdata",
  cutoff = predtrain_dt$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_dt$prediction
predtest_dt$predprobplot
predtest_dt$rocplot
predtest_dt$prplot
predtest_dt$caliplot
predtest_dt$cmplot
predtest_dt$metrics
predtest_dt$diycutoff
predtest_dt$ksplot

# pROC包auc值及其置信区间
pROC::auc(predtest_dt$proc)
pROC::ci.auc(predtest_dt$proc)

# ROC比较检验
pROC::roc.test(predtrain_dt$proc, predtest_dt$proc)


# 合并训练集和测试集上ROC曲线
predtrain_dt$rocresult %>%
  bind_rows(predtest_dt$rocresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上PR曲线
predtrain_dt$prresult %>%
  bind_rows(predtest_dt$prresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上校准曲线
predtrain_dt$caliresult %>%
  bind_rows(predtest_dt$caliresult) %>%
  mutate(data = forcats::as_factor(data)) %>%
  ggplot(aes(x = predprobgroup,
             y = Fraction, 
             color = data)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3, pch = 15) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  labs(x = "Bin Midpoint", y = "Event Rate", color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上性能指标
predtrain_dt$metrics %>%
  bind_rows(predtest_dt$metrics) %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


# 最优超参数交叉验证的结果
evalcv_dt <- bestcv4cls2(
  wkflow = wk_dt,
  tuneresult = tune_dt,
  hpbest = hpbest_dt,
  yname = "Operation_time",
  modelname = "DT",
  v = 10,
  positivelevel = yourpositivelevel
)
evalcv_dt$cvroc
evalcv_dt$cvpr
evalcv_dt$evalcv










# 数据预处理配方
datarecipe_rf <- recipe(formula = Operation_time ~ ., traindata)
datarecipe_rf


# 设定模型
model_rf <- rand_forest(
  mode = "classification",
  engine = "randomForest", # ranger
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_args(importance = T)
model_rf

# workflow
wk_rf <- 
  workflow() %>%
  add_recipe(datarecipe_rf) %>%
  add_model(model_rf)
wk_rf

##############################################################
############################  超参数寻优2选1-网格搜索

# 超参数寻优网格
set.seed(42)
hpgrid_rf <- parameters(
  mtry(range = c(2, 10)), 
  trees(range = c(200, 500)),
  min_n(range = c(20, 50))
) %>%
  # grid_regular(levels = c(3, 2, 4)) # 常规网格
  grid_random(size = 20) # 随机网格
# grid_latin_hypercube(size = 10) # 拉丁方网格
# grid_max_entropy(size = 10) # 最大熵网格
hpgrid_rf
# 网格也可以自己手动生成expand.grid()

# 交叉验证网格搜索过程
set.seed(42)
tune_rf <- wk_rf %>%
  tune_grid(
    resamples = folds,
    grid = hpgrid_rf,
    metrics = metricset_cls2,
    control = control_grid(save_pred = T, 
                           verbose = T,
                           event_level = "second",
                           parallel_over = "everything",
                           save_workflow = T)
  )

#########################  超参数寻优2选1-贝叶斯优化

# 更新超参数范围
param_rf <- model_rf %>%
  extract_parameter_set_dials() %>%
  update(mtry = mtry(c(2, 10)),
         trees = trees(c(100, 1000)),
         min_n = min_n(c(7, 55)))

# 贝叶斯优化超参数
set.seed(42)
tune_rf <- wk_rf %>%
  tune_bayes(
    resamples = folds,
    initial = 10,
    iter = 50,
    param_info = param_rf,
    metrics = metricset_cls2,
    control = control_bayes(save_pred = T, 
                            verbose = T,
                            no_improve = 10,
                            uncertain = 5,
                            event_level = "second",
                            parallel_over = "everything",
                            save_workflow = T)
  )

########################  超参数寻优结束


# 交叉验证结果
eval_tune_rf <- tune_rf %>%
  collect_metrics()
eval_tune_rf

# 图示
# autoplot(tune_rf)
eval_tune_rf %>% 
  filter(.metric == "roc_auc") %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'mtry', values = ~mtry),
      list(label = 'trees', values = ~trees),
      list(label = 'min_n', values = ~min_n)
    )
  ) %>%
  plotly::layout(title = "RF HPO Guided by AUCROC",
                 font = list(family = "serif"))

# 经过交叉验证得到的最优超参数
hpbest_rf <- tune_rf %>%
  select_by_one_std_err(metric = "roc_auc", desc(min_n))
hpbest_rf

# 采用最优超参数组合训练最终模型
set.seed(42)
final_rf <- wk_rf %>%
  finalize_workflow(hpbest_rf) %>%
  fit(traindata)
final_rf

##################################################################

# 训练集预测评估
predtrain_rf <- eval4cls2(
  model = final_rf, 
  dataset = traindata, 
  yname = "Operation_time", 
  modelname = "RF", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_rf$prediction
predtrain_rf$predprobplot
predtrain_rf$rocplot
predtrain_rf$prplot
predtrain_rf$caliplot
predtrain_rf$cmplot
predtrain_rf$metrics
predtrain_rf$diycutoff
predtrain_rf$ksplot

# pROC包auc值及其置信区间
pROC::auc(predtrain_rf$proc)
pROC::ci.auc(predtrain_rf$proc)

# 预测评估测试集预测评估
predtest_rf <- eval4cls2(
  model = final_rf, 
  dataset = testdata, 
  yname = "Operation_time", 
  modelname = "RF", 
  datasetname = "testdata",
  cutoff = predtrain_rf$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_rf$prediction
predtest_rf$predprobplot
predtest_rf$rocplot
predtest_rf$prplot
predtest_rf$caliplot
predtest_rf$cmplot
predtest_rf$metrics
predtest_rf$diycutoff
predtest_rf$ksplot

# pROC包auc值及其置信区间
pROC::auc(predtest_rf$proc)
pROC::ci.auc(predtest_rf$proc)

# ROC比较检验
pROC::roc.test(predtrain_rf$proc, predtest_rf$proc)


# 合并训练集和测试集上ROC曲线
predtrain_rf$rocresult %>%
  bind_rows(predtest_rf$rocresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上PR曲线
predtrain_rf$prresult %>%
  bind_rows(predtest_rf$prresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上校准曲线
predtrain_rf$caliresult %>%
  bind_rows(predtest_rf$caliresult) %>%
  mutate(data = forcats::as_factor(data)) %>%
  ggplot(aes(x = predprobgroup,
             y = Fraction, 
             color = data)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3, pch = 15) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  labs(x = "Bin Midpoint", y = "Event Rate", color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上性能指标
predtrain_rf$metrics %>%
  bind_rows(predtest_rf$metrics) %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


# 最优超参数交叉验证的结果
evalcv_rf <- bestcv4cls2(
  wkflow = wk_rf,
  tuneresult = tune_rf,
  hpbest = hpbest_rf,
  yname = "Operation_time",
  modelname = "RF",
  v = 10,
  positivelevel = yourpositivelevel
)
evalcv_rf$cvroc
evalcv_rf$cvpr
evalcv_rf$evalcv







# 数据预处理配方
datarecipe_xgboost <- recipe(formula = Operation_time ~ ., traindata) %>%
  step_dummy(all_nominal_predictors(), naming = new_dummy_names) %>%
  step_nzv(all_predictors())
datarecipe_xgboost


# 设定模型
model_xgboost <- boost_tree(
  mode = "classification",
  engine = "xgboost",
  mtry = tune(),
  trees = 1000,
  min_n = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  stop_iter = 25
) %>%
  set_args(validation = 0.2,
           event_level = "second")
model_xgboost

# workflow
wk_xgboost <- 
  workflow() %>%
  add_recipe(datarecipe_xgboost) %>%
  add_model(model_xgboost)
wk_xgboost

##############################################################
############################  超参数寻优2选1-网格搜索

# 超参数寻优网格
set.seed(42)
hpgrid_xgboost <- parameters(
  mtry(range = c(2, 8)),
  min_n(range = c(5, 20)),
  tree_depth(range = c(1, 3)),
  learn_rate(range = c(-3, -1)),
  loss_reduction(range = c(-3, 0)),
  sample_prop(range = c(0.8, 1))
) %>%
  # grid_regular(levels = c(3, 2, 2, 3, 2, 2)) # 常规网格
  grid_random(size = 20) # 随机网格
# grid_latin_hypercube(size = 10) # 拉丁方网格
# grid_max_entropy(size = 10) # 最大熵网格
hpgrid_xgboost
# 网格也可以自己手动生成expand.grid()

# 交叉验证网格搜索过程
set.seed(42)
tune_xgboost <- wk_xgboost %>%
  tune_grid(
    resamples = folds,
    grid = hpgrid_xgboost,
    metrics = metricset_cls2,
    control = control_grid(save_pred = T, 
                           verbose = T,
                           event_level = "second",
                           parallel_over = "everything",
                           save_workflow = T)
  )

#########################  超参数寻优2选1-贝叶斯优化

# 更新超参数范围
param_xgboost <- model_xgboost %>%
  extract_parameter_set_dials() %>%
  update(mtry = mtry(c(2, 10)))

# 贝叶斯优化超参数
set.seed(42)
tune_xgboost <- wk_xgboost %>%
  tune_bayes(
    resamples = folds,
    initial = 10,
    iter = 50,
    param_info = param_xgboost,
    metrics = metricset_cls2,
    control = control_bayes(save_pred = T, 
                            verbose = T,
                            no_improve = 10,
                            uncertain = 5,
                            event_level = "second",
                            parallel_over = "everything",
                            save_workflow = T)
  )

########################  超参数寻优结束


# 交叉验证结果
eval_tune_xgboost <- tune_xgboost %>%
  collect_metrics()
eval_tune_xgboost

# 图示
# autoplot(tune_xgboost)
eval_tune_xgboost %>% 
  filter(.metric == "roc_auc") %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'mtry', values = ~mtry),
      list(label = 'min_n', values = ~min_n),
      list(label = 'tree_depth', values = ~tree_depth),
      list(label = 'learn_rate', values = ~learn_rate),
      list(label = 'loss_reduction', values = ~loss_reduction),
      list(label = 'sample_size', values = ~sample_size)
    )
  ) %>%
  plotly::layout(title = "xgboost HPO Guided by AUCROC",
                 font = list(family = "serif"))

# 经过交叉验证得到的最优超参数
hpbest_xgboost <- tune_xgboost %>%
  select_by_one_std_err(metric = "roc_auc", desc(min_n))
hpbest_xgboost

# 采用最优超参数组合训练最终模型
set.seed(42)
final_xgboost <- wk_xgboost %>%
  finalize_workflow(hpbest_xgboost) %>%
  fit(traindata)
final_xgboost

##################################################################

# 训练集预测评估
predtrain_xgboost <- eval4cls2(
  model = final_xgboost, 
  dataset = traindata, 
  yname = "Operation_time", 
  modelname = "Xgboost", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_xgboost$prediction
predtrain_xgboost$predprobplot
predtrain_xgboost$rocplot
predtrain_xgboost$prplot
predtrain_xgboost$caliplot
predtrain_xgboost$cmplot
predtrain_xgboost$metrics
predtrain_xgboost$diycutoff
predtrain_xgboost$ksplot

# pROC包auc值及其置信区间
pROC::auc(predtrain_xgboost$proc)
pROC::ci.auc(predtrain_xgboost$proc)

# 预测评估测试集预测评估
predtest_xgboost <- eval4cls2(
  model = final_xgboost, 
  dataset = testdata, 
  yname = "Operation_time", 
  modelname = "Xgboost", 
  datasetname = "testdata",
  cutoff = predtrain_xgboost$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_xgboost$prediction
predtest_xgboost$predprobplot
predtest_xgboost$rocplot
predtest_xgboost$prplot
predtest_xgboost$caliplot
predtest_xgboost$cmplot
predtest_xgboost$metrics
predtest_xgboost$diycutoff
predtest_xgboost$ksplot

# pROC包auc值及其置信区间
pROC::auc(predtest_xgboost$proc)
pROC::ci.auc(predtest_xgboost$proc)

# ROC比较检验
pROC::roc.test(predtrain_xgboost$proc, predtest_xgboost$proc)


# 合并训练集和测试集上ROC曲线
predtrain_xgboost$rocresult %>%
  bind_rows(predtest_xgboost$rocresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上PR曲线
predtrain_xgboost$prresult %>%
  bind_rows(predtest_xgboost$prresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上校准曲线
predtrain_xgboost$caliresult %>%
  bind_rows(predtest_xgboost$caliresult) %>%
  mutate(data = forcats::as_factor(data)) %>%
  ggplot(aes(x = predprobgroup,
             y = Fraction, 
             color = data)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3, pch = 15) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  labs(x = "Bin Midpoint", y = "Event Rate", color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上性能指标
predtrain_xgboost$metrics %>%
  bind_rows(predtest_xgboost$metrics) %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


# 最优超参数交叉验证的结果
evalcv_xgboost <- bestcv4cls2(
  wkflow = wk_xgboost,
  tuneresult = tune_xgboost,
  hpbest = hpbest_xgboost,
  yname = "Operation_time",
  modelname = "Xgboost",
  v = 10,
  positivelevel = yourpositivelevel
)
evalcv_xgboost$cvroc
evalcv_xgboost$cvpr
evalcv_xgboost$evalcv





# 数据预处理配方
datarecipe_lightgbm <- recipe(formula = Operation_time ~ ., traindata) %>%
  step_dummy(all_nominal_predictors(), naming = new_dummy_names) %>%
  step_nzv(all_predictors())
datarecipe_lightgbm


# 设定模型
model_lightgbm <- boost_tree(
  mode = "classification",
  engine = "lightgbm",
  tree_depth = tune(),
  trees = tune(),
  learn_rate = tune(),
  mtry = tune(),
  min_n = tune(),
  loss_reduction = tune()
)
model_lightgbm

# workflow
wk_lightgbm <- 
  workflow() %>%
  add_recipe(datarecipe_lightgbm) %>%
  add_model(model_lightgbm)
wk_lightgbm

##############################################################
############################  超参数寻优2选1-网格搜索

# 超参数寻优网格
set.seed(42)
hpgrid_lightgbm <- parameters(
  tree_depth(range = c(1, 3)),
  trees(range = c(100, 500)),
  learn_rate(range = c(-3, -1)),
  mtry(range = c(2, 8)),
  min_n(range = c(5, 10)),
  loss_reduction(range = c(-3, 0))
) %>%
  # grid_regular(levels = c(3, 2, 2, 3, 2, 2)) # 常规网格
  grid_random(size = 20) # 随机网格
# grid_latin_hypercube(size = 10) # 拉丁方网格
# grid_max_entropy(size = 10) # 最大熵网格
hpgrid_lightgbm
# 网格也可以自己手动生成expand.grid()

# 交叉验证网格搜索过程
set.seed(42)
tune_lightgbm <- wk_lightgbm %>%
  tune_grid(
    resamples = folds,
    grid = hpgrid_lightgbm,
    metrics = metricset_cls2,
    control = control_grid(save_pred = T, 
                           verbose = T,
                           event_level = "second",
                           parallel_over = "everything",
                           save_workflow = T)
  )

#########################  超参数寻优2选1-贝叶斯优化

# 更新超参数范围
param_lightgbm <- model_lightgbm %>%
  extract_parameter_set_dials() %>%
  update(mtry = mtry(c(2, 10)),
         min_n = min_n(c(15, 55)))

# 贝叶斯优化超参数
set.seed(42)
tune_lightgbm <- wk_lightgbm %>%
  tune_bayes(
    resamples = folds,
    initial = 10,
    iter = 50,
    param_info = param_lightgbm,
    metrics = metricset_cls2,
    control = control_bayes(save_pred = T, 
                            verbose = T,
                            no_improve = 10,
                            uncertain = 5,
                            event_level = "second",
                            parallel_over = "everything",
                            save_workflow = T)
  )

########################  超参数寻优结束

# 交叉验证结果
eval_tune_lightgbm <- tune_lightgbm %>%
  collect_metrics()
eval_tune_lightgbm

# 图示
# autoplot(tune_lightgbm)
eval_tune_lightgbm %>% 
  filter(.metric == "roc_auc") %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'mtry', values = ~mtry),
      list(label = 'trees', values = ~trees),
      list(label = 'min_n', values = ~min_n),
      list(label = 'tree_depth', values = ~tree_depth),
      list(label = 'learn_rate', values = ~learn_rate),
      list(label = 'loss_reduction', values = ~loss_reduction)
    )
  ) %>%
  plotly::layout(title = "lightgbm HPO Guided by AUCROC",
                 font = list(family = "serif"))

# 经过交叉验证得到的最优超参数
hpbest_lightgbm <- tune_lightgbm %>%
  select_by_one_std_err(metric = "roc_auc", desc(min_n))
hpbest_lightgbm

# 采用最优超参数组合训练最终模型
set.seed(42)
final_lightgbm <- wk_lightgbm %>%
  finalize_workflow(hpbest_lightgbm) %>%
  fit(traindata)
final_lightgbm

##################################################################

# 训练集预测评估
predtrain_lightgbm <- eval4cls2(
  model = final_lightgbm, 
  dataset = traindata, 
  yname = "Operation_time", 
  modelname = "Lightgbm", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_lightgbm$prediction
predtrain_lightgbm$predprobplot
predtrain_lightgbm$rocplot
predtrain_lightgbm$prplot
predtrain_lightgbm$caliplot
predtrain_lightgbm$cmplot
predtrain_lightgbm$metrics
predtrain_lightgbm$diycutoff
predtrain_lightgbm$ksplot

# pROC包auc值及其置信区间
pROC::auc(predtrain_lightgbm$proc)
pROC::ci.auc(predtrain_lightgbm$proc)

# 预测评估测试集预测评估
predtest_lightgbm <- eval4cls2(
  model = final_lightgbm, 
  dataset = testdata, 
  yname = "Operation_time", 
  modelname = "Lightgbm", 
  datasetname = "testdata",
  cutoff = predtrain_lightgbm$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_lightgbm$prediction
predtest_lightgbm$predprobplot
predtest_lightgbm$rocplot
predtest_lightgbm$prplot
predtest_lightgbm$caliplot
predtest_lightgbm$cmplot
predtest_lightgbm$metrics
predtest_lightgbm$diycutoff
predtest_lightgbm$ksplot

# pROC包auc值及其置信区间
pROC::auc(predtest_lightgbm$proc)
pROC::ci.auc(predtest_lightgbm$proc)

# ROC比较检验
pROC::roc.test(predtrain_lightgbm$proc, predtest_lightgbm$proc)


# 合并训练集和测试集上ROC曲线
predtrain_lightgbm$rocresult %>%
  bind_rows(predtest_lightgbm$rocresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上PR曲线
predtrain_lightgbm$prresult %>%
  bind_rows(predtest_lightgbm$prresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上校准曲线
predtrain_lightgbm$caliresult %>%
  bind_rows(predtest_lightgbm$caliresult) %>%
  mutate(data = forcats::as_factor(data)) %>%
  ggplot(aes(x = predprobgroup,
             y = Fraction, 
             color = data)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3, pch = 15) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  labs(x = "Bin Midpoint", y = "Event Rate", color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上性能指标
predtrain_lightgbm$metrics %>%
  bind_rows(predtest_lightgbm$metrics) %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


# 最优超参数交叉验证的结果
evalcv_lightgbm <- bestcv4cls2(
  wkflow = wk_lightgbm,
  tuneresult = tune_lightgbm,
  hpbest = hpbest_lightgbm,
  yname = "Operation_time",
  modelname = "Lightgbm",
  v = 10,
  positivelevel = yourpositivelevel
)
evalcv_lightgbm$cvroc
evalcv_lightgbm$cvpr
evalcv_lightgbm$evalcv






# 数据预处理配方
datarecipe_svm <- recipe(formula = Operation_time ~ ., traindata) %>%
  step_dummy(all_nominal_predictors(), naming = new_dummy_names) %>%
  step_nzv(all_predictors()) %>% 
  step_normalize(all_predictors())
datarecipe_svm


# 设定模型
model_svm <- svm_rbf(
  mode = "classification",
  engine = "kernlab",
  cost = tune(),
  rbf_sigma = tune()
) %>%
  set_args(class.weights = c("No" = 1, "Yes" = 2)) 
# 此处NoYes是因变量的取值水平，换数据之后要相应更改
# 后面的数字12表示分类错误时的成本权重，无需设定时都等于1即可
model_svm

# workflow
wk_svm <- 
  workflow() %>%
  add_recipe(datarecipe_svm) %>%
  add_model(model_svm)
wk_svm

##############################################################
############################  超参数寻优2选1-网格搜索

# 超参数寻优网格
set.seed(42)
hpgrid_svm <- parameters(
  cost(range = c(-5, 5)), 
  rbf_sigma(range = c(-4, -1))
) %>%
  # grid_regular(levels = c(2,3)) # 常规网格
  grid_random(size = 20) # 随机网格
# grid_latin_hypercube(size = 10) # 拉丁方网格
# grid_max_entropy(size = 10) # 最大熵网格
hpgrid_svm
# 网格也可以自己手动生成expand.grid()

# 交叉验证网格搜索过程
set.seed(42)
tune_svm <- wk_svm %>%
  tune_grid(
    resamples = folds,
    grid = hpgrid_svm,
    metrics = metricset_cls2,
    control = control_grid(save_pred = T, 
                           verbose = T,
                           event_level = "second",
                           parallel_over = "everything",
                           save_workflow = T)
  )

#########################  超参数寻优2选1-贝叶斯优化

# 贝叶斯优化超参数
set.seed(42)
tune_svm <- wk_svm %>%
  tune_bayes(
    resamples = folds,
    initial = 10,
    iter = 50,
    metrics = metricset_cls2,
    control = control_bayes(save_pred = T, 
                            verbose = T,
                            no_improve = 10,
                            uncertain = 5,
                            event_level = "second",
                            parallel_over = "everything",
                            save_workflow = T)
  )

########################  超参数寻优结束


# 交叉验证结果
eval_tune_svm <- tune_svm %>%
  collect_metrics()
eval_tune_svm

# 图示
# autoplot(tune_svm)
eval_tune_svm %>% 
  filter(.metric == "roc_auc") %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'cost', values = ~cost),
      list(label = 'rbf_sigma', values = ~rbf_sigma,
           font = list(family = "serif"))
    )
  ) %>%
  plotly::layout(title = "SVM HPO Guided by AUCROC",
                 font = list(family = "serif"))

# 经过交叉验证得到的最优超参数
hpbest_svm <- tune_svm %>%
  select_best(metric = "roc_auc")
hpbest_svm

# 采用最优超参数组合训练最终模型
set.seed(42)
final_svm <- wk_svm %>%
  finalize_workflow(hpbest_svm) %>%
  fit(traindata)
final_svm

##################################################################

# 训练集预测评估
predtrain_svm <- eval4cls2(
  model = final_svm, 
  dataset = traindata, 
  yname = "Operation_time", 
  modelname = "SVM", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_svm$prediction
predtrain_svm$predprobplot
predtrain_svm$rocplot
predtrain_svm$prplot
predtrain_svm$caliplot
predtrain_svm$cmplot
predtrain_svm$metrics
predtrain_svm$diycutoff
predtrain_svm$ksplot

# pROC包auc值及其置信区间
pROC::auc(predtrain_svm$proc)
pROC::ci.auc(predtrain_svm$proc)

# 预测评估测试集预测评估
predtest_svm <- eval4cls2(
  model = final_svm, 
  dataset = testdata, 
  yname = "Operation_time", 
  modelname = "SVM", 
  datasetname = "testdata",
  cutoff = predtrain_svm$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_svm$prediction
predtest_svm$predprobplot
predtest_svm$rocplot
predtest_svm$prplot
predtest_svm$caliplot
predtest_svm$cmplot
predtest_svm$metrics
predtest_svm$diycutoff
predtest_svm$ksplot

# pROC包auc值及其置信区间
pROC::auc(predtest_svm$proc)
pROC::ci.auc(predtest_svm$proc)

# ROC比较检验
pROC::roc.test(predtrain_svm$proc, predtest_svm$proc)


# 合并训练集和测试集上ROC曲线
predtrain_svm$rocresult %>%
  bind_rows(predtest_svm$rocresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上PR曲线
predtrain_svm$prresult %>%
  bind_rows(predtest_svm$prresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上校准曲线
predtrain_svm$caliresult %>%
  bind_rows(predtest_svm$caliresult) %>%
  mutate(data = forcats::as_factor(data)) %>%
  ggplot(aes(x = predprobgroup,
             y = Fraction, 
             color = data)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3, pch = 15) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  labs(x = "Bin Midpoint", y = "Event Rate", color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上性能指标
predtrain_svm$metrics %>%
  bind_rows(predtest_svm$metrics) %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


# 最优超参数交叉验证的结果
evalcv_svm <- bestcv4cls2(
  wkflow = wk_svm,
  tuneresult = tune_svm,
  hpbest = hpbest_svm,
  yname = "Operation_time",
  modelname = "SVM",
  v = 10,
  positivelevel = yourpositivelevel
)
evalcv_svm$cvroc
evalcv_svm$cvpr
evalcv_svm$evalcv








# 数据预处理配方
datarecipe_mlp <- recipe(formula = Operation_time ~ ., traindata) %>%
  step_dummy(all_nominal_predictors(), naming = new_dummy_names) %>%
  step_nzv(all_predictors()) %>% 
  step_range(all_predictors())
datarecipe_mlp


# 设定模型
model_mlp <- mlp(
  mode = "classification",
  engine = "nnet",
  hidden_units = tune(),
  penalty = tune(),
  epochs = tune()
) %>%
  set_args(MaxNWts = 10000)
model_mlp

# workflow
wk_mlp <- 
  workflow() %>%
  add_recipe(datarecipe_mlp) %>%
  add_model(model_mlp)
wk_mlp

##############################################################
############################  超参数寻优2选1-网格搜索

# 超参数寻优网格
set.seed(42)
hpgrid_mlp <- parameters(
  hidden_units(range = c(15, 24)),
  penalty(range = c(-3, 0)),
  epochs(range = c(50, 150))
) %>%
  grid_regular(levels = 3) # 常规网格
# grid_random(size = 5) # 随机网格
# grid_latin_hypercube(size = 10) # 拉丁方网格
# grid_max_entropy(size = 10) # 最大熵网格
hpgrid_mlp
# 网格也可以自己手动生成expand.grid()

# 交叉验证网格搜索过程
set.seed(42)
tune_mlp <- wk_mlp %>%
  tune_grid(
    resamples = folds,
    grid = hpgrid_mlp,
    metrics = metricset_cls2,
    control = control_grid(save_pred = T, 
                           verbose = T,
                           event_level = "second",
                           parallel_over = "everything",
                           save_workflow = T)
  )

#########################  超参数寻优2选1-贝叶斯优化

# 贝叶斯优化超参数
set.seed(42)
tune_mlp <- wk_mlp %>%
  tune_bayes(
    resamples = folds,
    initial = 10,
    iter = 50,
    metrics = metricset_cls2,
    control = control_bayes(save_pred = T, 
                            verbose = T,
                            no_improve = 10,
                            uncertain = 5,
                            event_level = "second",
                            parallel_over = "everything",
                            save_workflow = T)
  )

########################  超参数寻优结束


# 交叉验证结果
eval_tune_mlp <- tune_mlp %>%
  collect_metrics()
eval_tune_mlp

# 图示
# autoplot(tune_mlp)
eval_tune_mlp %>% 
  filter(.metric == "roc_auc") %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'hidden_units', values = ~hidden_units),
      list(label = 'penalty', values = ~penalty),
      list(label = 'epochs', values = ~epochs)
    )
  ) %>%
  plotly::layout(title = "MLP HPO Guided by AUCROC",
                 font = list(family = "serif"))

# 经过交叉验证得到的最优超参数
hpbest_mlp <- tune_mlp %>%
  select_by_one_std_err(metric = "roc_auc", desc(penalty))
hpbest_mlp

# 采用最优超参数组合训练最终模型
set.seed(42)
final_mlp <- wk_mlp %>%
  finalize_workflow(hpbest_mlp) %>%
  fit(traindata)
final_mlp

##################################################################

# 训练集预测评估
predtrain_mlp <- eval4cls2(
  model = final_mlp, 
  dataset = traindata, 
  yname = "Operation_time", 
  modelname = "MLP", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_mlp$prediction
predtrain_mlp$predprobplot
predtrain_mlp$rocplot
predtrain_mlp$prplot
predtrain_mlp$caliplot
predtrain_mlp$cmplot
predtrain_mlp$metrics
predtrain_mlp$diycutoff
predtrain_mlp$ksplot

# pROC包auc值及其置信区间
pROC::auc(predtrain_mlp$proc)
pROC::ci.auc(predtrain_mlp$proc)

# 预测评估测试集预测评估
predtest_mlp <- eval4cls2(
  model = final_mlp, 
  dataset = testdata, 
  yname = "Operation_time", 
  modelname = "MLP", 
  datasetname = "testdata",
  cutoff = predtrain_mlp$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_mlp$prediction
predtest_mlp$predprobplot
predtest_mlp$rocplot
predtest_mlp$prplot
predtest_mlp$caliplot
predtest_mlp$cmplot
predtest_mlp$metrics
predtest_mlp$diycutoff
predtest_mlp$ksplot

# pROC包auc值及其置信区间
pROC::auc(predtest_mlp$proc)
pROC::ci.auc(predtest_mlp$proc)

# ROC比较检验
pROC::roc.test(predtrain_mlp$proc, predtest_mlp$proc)


# 合并训练集和测试集上ROC曲线
predtrain_mlp$rocresult %>%
  bind_rows(predtest_mlp$rocresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上PR曲线
predtrain_mlp$prresult %>%
  bind_rows(predtest_mlp$prresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上校准曲线
predtrain_mlp$caliresult %>%
  bind_rows(predtest_mlp$caliresult) %>%
  mutate(data = forcats::as_factor(data)) %>%
  ggplot(aes(x = predprobgroup,
             y = Fraction, 
             color = data)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3, pch = 15) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  labs(x = "Bin Midpoint", y = "Event Rate", color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上性能指标
predtrain_mlp$metrics %>%
  bind_rows(predtest_mlp$metrics) %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


# 最优超参数交叉验证的结果
evalcv_mlp <- bestcv4cls2(
  wkflow = wk_mlp,
  tuneresult = tune_mlp,
  hpbest = hpbest_mlp,
  yname = "Operation_time",
  modelname = "MLP",
  v = 10,
  positivelevel = yourpositivelevel
)
evalcv_mlp$cvroc
evalcv_mlp$cvpr
evalcv_mlp$evalcv







# 数据预处理配方
datarecipe_knn <- recipe(formula = Operation_time ~ ., traindata) %>%
  step_dummy(all_nominal_predictors(), naming = new_dummy_names) %>%
  step_nzv(all_predictors()) %>% 
  step_normalize(all_predictors())
datarecipe_knn


# 设定模型
model_knn <- nearest_neighbor(
  mode = "classification",
  engine = "kknn",
  
  neighbors = tune(),
  weight_func = tune(),
  dist_power = 2
)
model_knn

# workflow
wk_knn <- 
  workflow() %>%
  add_recipe(datarecipe_knn) %>%
  add_model(model_knn)
wk_knn

##############################################################
############################  超参数寻优2选1-网格搜索

# 超参数寻优网格
set.seed(42)
hpgrid_knn <- parameters(
  neighbors(range = c(3, 11)),
  weight_func()
) %>%
  # grid_regular(levels = c(5)) # 常规网格
  grid_random(size = 20) # 随机网格
# grid_latin_hypercube(size = 10) # 拉丁方网格
# grid_max_entropy(size = 10) # 最大熵网格
hpgrid_knn
# 网格也可以自己手动生成expand.grid()
# 交叉验证网格搜索过程
set.seed(42)
tune_knn <- wk_knn %>%
  tune_grid(
    resamples = folds,
    grid = hpgrid_knn,
    metrics = metricset_cls2,
    control = control_grid(save_pred = T, 
                           verbose = T,
                           event_level = "second",
                           parallel_over = "everything",
                           save_workflow = T)
  )

#########################  超参数寻优2选1-贝叶斯优化

# 更新超参数范围
param_knn <- model_knn %>%
  extract_parameter_set_dials() %>%
  update(neighbors = neighbors(c(5, 35)),
         weight_func = weight_func(c("rectangular",  "triangular")))

# 贝叶斯优化超参数
set.seed(42)
tune_knn <- wk_knn %>%
  tune_bayes(
    resamples = folds,
    initial = 20,
    iter = 50,
    param_info = param_knn,
    metrics = metricset_cls2,
    control = control_bayes(save_pred = T, 
                            verbose = T,
                            no_improve = 10,
                            uncertain = 5,
                            event_level = "second",
                            parallel_over = "everything",
                            save_workflow = T)
  )

########################  超参数寻优结束


# 交叉验证结果
eval_tune_knn <- tune_knn %>%
  collect_metrics()
eval_tune_knn

# 图示
# autoplot(tune_knn)
eval_tune_knn %>% 
  filter(.metric == "roc_auc") %>%
  mutate(weight_func2 = as.numeric(as.factor(weight_func))) %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'neighbors', values = ~neighbors),
      list(label = 'weight_func', values = ~weight_func2,
           range = c(1,length(unique(eval_tune_knn$weight_func))), 
           tickvals = 1:length(unique(eval_tune_knn$weight_func)),
           ticktext = sort(unique(eval_tune_knn$weight_func)))
    )
  ) %>%
  plotly::layout(title = "KNN HPO Guided by AUCROC",
                 font = list(family = "serif"))

# 经过交叉验证得到的最优超参数
hpbest_knn <- tune_knn %>%
  select_by_one_std_err(metric = "roc_auc", desc(neighbors))
hpbest_knn

# 采用最优超参数组合训练最终模型
set.seed(42)
final_knn <- wk_knn %>%
  finalize_workflow(hpbest_knn) %>%
  fit(traindata)
final_knn

##################################################################

# 训练集预测评估
predtrain_knn <- eval4cls2(
  model = final_knn, 
  dataset = traindata, 
  yname = "Operation_time", 
  modelname = "KNN", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_knn$prediction
predtrain_knn$predprobplot
predtrain_knn$rocplot
predtrain_knn$prplot
predtrain_knn$caliplot
predtrain_knn$cmplot
predtrain_knn$metrics
predtrain_knn$diycutoff
predtrain_knn$ksplot

# pROC包auc值及其置信区间
pROC::auc(predtrain_knn$proc)
pROC::ci.auc(predtrain_knn$proc)

# 预测评估测试集预测评估
predtest_knn <- eval4cls2(
  model = final_knn, 
  dataset = testdata, 
  yname = "Operation_time", 
  modelname = "KNN", 
  datasetname = "testdata",
  cutoff = predtrain_knn$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_knn$prediction
predtest_knn$predprobplot
predtest_knn$rocplot
predtest_knn$prplot
predtest_knn$caliplot
predtest_knn$cmplot
predtest_knn$metrics
predtest_knn$diycutoff
predtest_knn$ksplot

# pROC包auc值及其置信区间
pROC::auc(predtest_knn$proc)
pROC::ci.auc(predtest_knn$proc)

# ROC比较检验
pROC::roc.test(predtrain_knn$proc, predtest_knn$proc)


# 合并训练集和测试集上ROC曲线
predtrain_knn$rocresult %>%
  bind_rows(predtest_knn$rocresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上PR曲线
predtrain_knn$prresult %>%
  bind_rows(predtest_knn$prresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上校准曲线
predtrain_knn$caliresult %>%
  bind_rows(predtest_knn$caliresult) %>%
  mutate(data = forcats::as_factor(data)) %>%
  ggplot(aes(x = predprobgroup,
             y = Fraction, 
             color = data)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3, pch = 15) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  labs(x = "Bin Midpoint", y = "Event Rate", color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上性能指标
predtrain_knn$metrics %>%
  bind_rows(predtest_knn$metrics) %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


# 最优超参数交叉验证的结果
evalcv_knn <- bestcv4cls2(
  wkflow = wk_knn,
  tuneresult = tune_knn,
  hpbest = hpbest_knn,
  yname = "Operation_time",
  modelname = "KNN",
  v = 10,
  positivelevel = yourpositivelevel
)
evalcv_knn$cvroc
evalcv_knn$cvpr
evalcv_knn$evalcv






# 数据预处理配方
datarecipe_logistic <- recipe(Operation_time ~ ., traindata)
datarecipe_logistic


# 设定模型
model_logistic <- logistic_reg(
  mode = "classification",
  engine = "glm"
)
model_logistic

# workflow
wk_logistic <- 
  workflow() %>%
  add_recipe(datarecipe_logistic) %>%
  add_model(model_logistic)
wk_logistic

# 训练模型
set.seed(42)
final_logistic <- wk_logistic %>%
  fit(traindata)
final_logistic

##################################################################

# 训练集预测评估
predtrain_logistic <- eval4cls2(
  model = final_logistic, 
  dataset = traindata, 
  yname = "Operation_time", 
  modelname = "Logistic", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_logistic$prediction
predtrain_logistic$predprobplot
predtrain_logistic$rocplot
predtrain_logistic$prplot
predtrain_logistic$caliplot
predtrain_logistic$cmplot
predtrain_logistic$metrics
predtrain_logistic$diycutoff
predtrain_logistic$ksplot

# pROC包auc值及其置信区间
pROC::auc(predtrain_logistic$proc)
pROC::ci.auc(predtrain_logistic$proc)

# 预测评估测试集预测评估
predtest_logistic <- eval4cls2(
  model = final_logistic, 
  dataset = testdata, 
  yname = "Operation_time", 
  modelname = "Logistic", 
  datasetname = "testdata",
  cutoff = predtrain_logistic$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_logistic$prediction
predtest_logistic$predprobplot
predtest_logistic$rocplot
predtest_logistic$prplot
predtest_logistic$caliplot
predtest_logistic$cmplot
predtest_logistic$metrics
predtest_logistic$diycutoff
predtest_logistic$ksplot

# pROC包auc值及其置信区间
pROC::auc(predtest_logistic$proc)
pROC::ci.auc(predtest_logistic$proc)

# ROC比较检验
pROC::roc.test(predtrain_logistic$proc, predtest_logistic$proc)


# 合并训练集和测试集上ROC曲线
predtrain_logistic$rocresult %>%
  bind_rows(predtest_logistic$rocresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上PR曲线
predtrain_logistic$prresult %>%
  bind_rows(predtest_logistic$prresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上校准曲线
predtrain_logistic$caliresult %>%
  bind_rows(predtest_logistic$caliresult) %>%
  mutate(data = forcats::as_factor(data)) %>%
  ggplot(aes(x = predprobgroup,
             y = Fraction, 
             color = data)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3, pch = 15) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  labs(x = "Bin Midpoint", y = "Event Rate", color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上性能指标
predtrain_logistic$metrics %>%
  bind_rows(predtest_logistic$metrics) %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)

##################################################################

# 交叉验证
set.seed(42)
cv_logistic <- 
  wk_logistic %>%
  fit_resamples(
    folds,
    metrics = metricset_cls2,
    control = control_resamples(save_pred = T,
                                verbose = T,
                                event_level = "second",
                                parallel_over = "everything",
                                save_workflow = T)
  )
cv_logistic

# 交叉验证指标结果
evalcv_logistic <- list()
# 评估指标设定
metrictemp <- metric_set(yardstick::roc_auc, yardstick::pr_auc)
evalcv_logistic$evalcv <- 
  collect_predictions(cv_logistic) %>%
  group_by(id) %>%
  metrictemp(Operation_time, .pred_Yes, event_level = "second") %>%
  group_by(.metric) %>%
  mutate(model = "logistic",
         mean = mean(.estimate),
         sd = sd(.estimate)/sqrt(length(folds$splits)))
evalcv_logistic$evalcv

# 交叉验证预测结果图示
# ROC
evalcv_logistic$cvroc <- 
  collect_predictions(cv_logistic) %>%
  group_by(id) %>%
  roc_curve(Operation_time, .pred_Yes, event_level = "second") %>%
  ungroup() %>%
  left_join(evalcv_logistic$evalcv %>% filter(.metric == "roc_auc"), 
            by = "id") %>%
  mutate(idAUC = paste(id, " ROCAUC:", round(.estimate, 4)),
         idAUC = forcats::as_factor(idAUC)) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = idAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0, 0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.2),
                     labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
  scale_y_continuous(expand = c(0, 0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.2),
                     labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = c(1,0),
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))
evalcv_logistic$cvroc

# PR
evalcv_logistic$cvpr <- 
  collect_predictions(cv_logistic) %>%
  group_by(id) %>%
  pr_curve(Operation_time, .pred_Yes, event_level = "second") %>%
  ungroup() %>%
  left_join(evalcv_logistic$evalcv %>% filter(.metric == "pr_auc"), 
            by = "id") %>%
  mutate(idAUC = paste(id, " PRAUC:", round(.estimate, 4)),
         idAUC = forcats::as_factor(idAUC)) %>%
  ggplot(aes(x = recall, y = precision, color = idAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", intercept = 1, slope = -1) +
  scale_x_continuous(expand = c(0, 0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.2),
                     labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
  scale_y_continuous(expand = c(0, 0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.2),
                     labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = c(0,0),
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))
evalcv_logistic$cvpr






# 数据预处理配方
datarecipe_lasso <- recipe(formula = Operation_time ~ ., traindata) %>%
  step_dummy(all_nominal_predictors(), naming = new_dummy_names) %>%
  step_nzv(all_predictors()) %>%
  step_normalize(all_predictors())
datarecipe_lasso


# 设定模型
model_lasso <- logistic_reg(
  mode = "classification",
  engine = "glmnet",
  mixture = 1,   # LASSO
  penalty = tune()
)
model_lasso

# workflow
wk_lasso <- 
  workflow() %>%
  add_recipe(datarecipe_lasso) %>%
  add_model(model_lasso)
wk_lasso

##############################################################
############################   超参数寻优2选1-网格搜索

# 超参数寻优网格
set.seed(42)
hpgrid_lasso <- parameters(
  penalty(range = c(-5, 0))
) %>%
  grid_regular(levels = c(20)) # 常规网格
# grid_random(size = 5) # 随机网格
# grid_latin_hypercube(size = 10) # 拉丁方网格
# grid_max_entropy(size = 10) # 最大熵网格
hpgrid_lasso
# 网格也可以自己手动生成expand.grid()

# 交叉验证网格搜索过程
set.seed(42)
tune_lasso <- wk_lasso %>%
  tune_grid(
    resamples = folds,
    grid = hpgrid_lasso,
    metrics = metricset_cls2,
    control = control_grid(save_pred = T, 
                           verbose = T,
                           event_level = "second",
                           parallel_over = "everything",
                           save_workflow = T)
  )

#########################  超参数寻优2选1-贝叶斯优化

# 贝叶斯优化超参数
set.seed(42)
tune_lasso <- wk_lasso %>%
  tune_bayes(
    resamples = folds,
    initial = 10,
    iter = 50,
    metrics = metricset_cls2,
    control = control_bayes(save_pred = T, 
                            verbose = T,
                            no_improve = 10,
                            uncertain = 5,
                            event_level = "second",
                            parallel_over = "everything",
                            save_workflow = T)
  )

########################  超参数寻优结束


# 交叉验证结果
eval_tune_lasso <- tune_lasso %>%
  collect_metrics()
eval_tune_lasso

# 图示
autoplot(tune_lasso)

# 经过交叉验证得到的最优超参数
hpbest_lasso <- tune_lasso %>%
  select_by_one_std_err(metric = "roc_auc", desc(penalty))
hpbest_lasso

# 采用最优超参数组合训练最终模型
set.seed(42)
final_lasso <- wk_lasso %>%
  finalize_workflow(hpbest_lasso) %>%
  fit(traindata)
final_lasso

##################################################################

# 训练集预测评估
predtrain_lasso <- eval4cls2(
  model = final_lasso, 
  dataset = traindata, 
  yname = "Operation_time", 
  modelname = "LASSO", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_lasso$prediction
predtrain_lasso$predprobplot
predtrain_lasso$rocplot
predtrain_lasso$prplot
predtrain_lasso$caliplot
predtrain_lasso$cmplot
predtrain_lasso$metrics
predtrain_lasso$diycutoff
predtrain_lasso$ksplot

# pROC包auc值及其置信区间
pROC::auc(predtrain_lasso$proc)
pROC::ci.auc(predtrain_lasso$proc)

# 预测评估测试集预测评估
predtest_lasso <- eval4cls2(
  model = final_lasso, 
  dataset = testdata, 
  yname = "Operation_time", 
  modelname = "LASSO", 
  datasetname = "testdata",
  cutoff = predtrain_lasso$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_lasso$prediction
predtest_lasso$predprobplot
predtest_lasso$rocplot
predtest_lasso$prplot
predtest_lasso$caliplot
predtest_lasso$cmplot
predtest_lasso$metrics
predtest_lasso$diycutoff
predtest_lasso$ksplot

# pROC包auc值及其置信区间
pROC::auc(predtest_lasso$proc)
pROC::ci.auc(predtest_lasso$proc)

# ROC比较检验
pROC::roc.test(predtrain_lasso$proc, predtest_lasso$proc)


# 合并训练集和测试集上ROC曲线
predtrain_lasso$rocresult %>%
  bind_rows(predtest_lasso$rocresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上PR曲线
predtrain_lasso$prresult %>%
  bind_rows(predtest_lasso$prresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上校准曲线
predtrain_lasso$caliresult %>%
  bind_rows(predtest_lasso$caliresult) %>%
  mutate(data = forcats::as_factor(data)) %>%
  ggplot(aes(x = predprobgroup,
             y = Fraction, 
             color = data)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3, pch = 15) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  labs(x = "Bin Midpoint", y = "Event Rate", color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上性能指标
predtrain_lasso$metrics %>%
  bind_rows(predtest_lasso$metrics) %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


# 最优超参数交叉验证的结果
evalcv_lasso <- bestcv4cls2(
  wkflow = wk_lasso,
  tuneresult = tune_lasso,
  hpbest = hpbest_lasso,
  yname = "Operation_time",
  modelname = "LASSO",
  v = 10,
  positivelevel = yourpositivelevel
)
evalcv_lasso$cvroc
evalcv_lasso$cvpr
evalcv_lasso$evalcv








# 数据预处理配方
datarecipe_ridge <- recipe(formula = Operation_time ~ ., traindata) %>%
  step_dummy(all_nominal_predictors(), naming = new_dummy_names) %>%
  step_nzv(all_predictors()) %>%
  step_normalize(all_predictors())
datarecipe_ridge


# 设定模型
model_ridge <- logistic_reg(
  mode = "classification",
  engine = "glmnet",
  mixture = 0,  # 岭回归
  penalty = tune()
)
model_ridge

# workflow
wk_ridge <- 
  workflow() %>%
  add_recipe(datarecipe_ridge) %>%
  add_model(model_ridge)
wk_ridge

##############################################################
############################   超参数寻优2选1-网格搜索

# 超参数寻优网格
set.seed(42)
hpgrid_ridge <- parameters(
  penalty(range = c(-5, 0))
) %>%
  grid_regular(levels = c(20)) # 常规网格
# grid_random(size = 5) # 随机网格
# grid_latin_hypercube(size = 10) # 拉丁方网格
# grid_max_entropy(size = 10) # 最大熵网格
hpgrid_ridge
# 网格也可以自己手动生成expand.grid()

# 交叉验证网格搜索过程
set.seed(42)
tune_ridge <- wk_ridge %>%
  tune_grid(
    resamples = folds,
    grid = hpgrid_ridge,
    metrics = metricset_cls2,
    control = control_grid(save_pred = T, 
                           verbose = T,
                           event_level = "second",
                           parallel_over = "everything",
                           save_workflow = T)
  )

#########################  超参数寻优2选1-贝叶斯优化

# 贝叶斯优化超参数
set.seed(42)
tune_ridge <- wk_ridge %>%
  tune_bayes(
    resamples = folds,
    initial = 10,
    iter = 50,
    metrics = metricset_cls2,
    control = control_bayes(save_pred = T, 
                            verbose = T,
                            no_improve = 10,
                            uncertain = 5,
                            event_level = "second",
                            parallel_over = "everything",
                            save_workflow = T)
  )

########################  超参数寻优结束


# 交叉验证结果
eval_tune_ridge <- tune_ridge %>%
  collect_metrics()
eval_tune_ridge

# 图示
autoplot(tune_ridge)

# 经过交叉验证得到的最优超参数
hpbest_ridge <- tune_ridge %>%
  select_by_one_std_err(metric = "roc_auc", desc(penalty))
hpbest_ridge

# 采用最优超参数组合训练最终模型
set.seed(42)
final_ridge <- wk_ridge %>%
  finalize_workflow(hpbest_ridge) %>%
  fit(traindata)
final_ridge

##################################################################

# 训练集预测评估
predtrain_ridge <- eval4cls2(
  model = final_ridge, 
  dataset = traindata, 
  yname = "Operation_time", 
  modelname = "Ridge", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_ridge$prediction
predtrain_ridge$predprobplot
predtrain_ridge$rocplot
predtrain_ridge$prplot
predtrain_ridge$caliplot
predtrain_ridge$cmplot
predtrain_ridge$metrics
predtrain_ridge$diycutoff
predtrain_ridge$ksplot

# pROC包auc值及其置信区间
pROC::auc(predtrain_ridge$proc)
pROC::ci.auc(predtrain_ridge$proc)

# 预测评估测试集预测评估
predtest_ridge <- eval4cls2(
  model = final_ridge, 
  dataset = testdata, 
  yname = "Operation_time", 
  modelname = "Ridge", 
  datasetname = "testdata",
  cutoff = predtrain_ridge$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_ridge$prediction
predtest_ridge$predprobplot
predtest_ridge$rocplot
predtest_ridge$prplot
predtest_ridge$caliplot
predtest_ridge$cmplot
predtest_ridge$metrics
predtest_ridge$diycutoff
predtest_ridge$ksplot

# pROC包auc值及其置信区间
pROC::auc(predtest_ridge$proc)
pROC::ci.auc(predtest_ridge$proc)

# ROC比较检验
pROC::roc.test(predtrain_ridge$proc, predtest_ridge$proc)


# 合并训练集和测试集上ROC曲线
predtrain_ridge$rocresult %>%
  bind_rows(predtest_ridge$rocresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上PR曲线
predtrain_ridge$prresult %>%
  bind_rows(predtest_ridge$prresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上校准曲线
predtrain_ridge$caliresult %>%
  bind_rows(predtest_ridge$caliresult) %>%
  mutate(data = forcats::as_factor(data)) %>%
  ggplot(aes(x = predprobgroup,
             y = Fraction, 
             color = data)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3, pch = 15) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  labs(x = "Bin Midpoint", y = "Event Rate", color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上性能指标
predtrain_ridge$metrics %>%
  bind_rows(predtest_ridge$metrics) %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


# 最优超参数交叉验证的结果
evalcv_ridge <- bestcv4cls2(
  wkflow = wk_ridge,
  tuneresult = tune_ridge,
  hpbest = hpbest_ridge,
  yname = "Operation_time",
  modelname = "Ridge",
  v = 10,
  positivelevel = yourpositivelevel
)
evalcv_ridge$cvroc
evalcv_ridge$cvpr
evalcv_ridge$evalcv









# 数据预处理配方
datarecipe_enet <- recipe(formula = Operation_time ~ ., traindata) %>%
  step_dummy(all_nominal_predictors(), naming = new_dummy_names) %>%
  step_nzv(all_predictors()) %>%
  step_normalize(all_predictors())
datarecipe_enet


# 设定模型
model_enet <- logistic_reg(
  mode = "classification",
  engine = "glmnet",
  mixture = tune(),
  penalty = tune()
)
model_enet

# workflow
wk_enet <- 
  workflow() %>%
  add_recipe(datarecipe_enet) %>%
  add_model(model_enet)
wk_enet

##############################################################
############################   超参数寻优2选1-网格搜索

# 超参数寻优网格
set.seed(42)
hpgrid_enet <- parameters(
  mixture(),
  penalty(range = c(-5, 0))
) %>%
  grid_regular(levels = c(5, 20)) # 常规网格
# grid_random(size = 5) # 随机网格
# grid_latin_hypercube(size = 10) # 拉丁方网格
# grid_max_entropy(size = 10) # 最大熵网格
hpgrid_enet
# 网格也可以自己手动生成expand.grid()

# 交叉验证网格搜索过程
set.seed(42)
tune_enet <- wk_enet %>%
  tune_grid(
    resamples = folds,
    grid = hpgrid_enet,
    metrics = metricset_cls2,
    control = control_grid(save_pred = T, 
                           verbose = T,
                           event_level = "second",
                           parallel_over = "everything",
                           save_workflow = T)
  )

#########################  超参数寻优2选1-贝叶斯优化

# 贝叶斯优化超参数
set.seed(42)
tune_enet <- wk_enet %>%
  tune_bayes(
    resamples = folds,
    initial = 10,
    iter = 50,
    metrics = metricset_cls2,
    control = control_bayes(save_pred = T, 
                            verbose = T,
                            no_improve = 10,
                            uncertain = 5,
                            event_level = "second",
                            parallel_over = "everything",
                            save_workflow = T)
  )

########################  超参数寻优结束


# 交叉验证结果
eval_tune_enet <- tune_enet %>%
  collect_metrics()
eval_tune_enet

# 图示
# autoplot(tune_enet)
eval_tune_enet %>% 
  filter(.metric == "roc_auc") %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'mixture', values = ~mixture),
      list(label = 'penalty', values = ~penalty)
    )
  ) %>%
  plotly::layout(title = "ENet HPO Guided by AUCROC",
                 font = list(family = "serif"))

# 经过交叉验证得到的最优超参数
hpbest_enet <- tune_enet %>%
  select_by_one_std_err(metric = "roc_auc", desc(penalty))
hpbest_enet

# 采用最优超参数组合训练最终模型
set.seed(42)
final_enet <- wk_enet %>%
  finalize_workflow(hpbest_enet) %>%
  fit(traindata)
final_enet

##################################################################

# 训练集预测评估
predtrain_enet <- eval4cls2(
  model = final_enet, 
  dataset = traindata, 
  yname = "Operation_time", 
  modelname = "ENet", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_enet$prediction
predtrain_enet$predprobplot
predtrain_enet$rocplot
predtrain_enet$prplot
predtrain_enet$caliplot
predtrain_enet$cmplot
predtrain_enet$metrics
predtrain_enet$diycutoff
predtrain_enet$ksplot

# pROC包auc值及其置信区间
pROC::auc(predtrain_enet$proc)
pROC::ci.auc(predtrain_enet$proc)

# 预测评估测试集预测评估
predtest_enet <- eval4cls2(
  model = final_enet, 
  dataset = testdata, 
  yname = "Operation_time", 
  modelname = "ENet", 
  datasetname = "testdata",
  cutoff = predtrain_enet$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_enet$prediction
predtest_enet$predprobplot
predtest_enet$rocplot
predtest_enet$prplot
predtest_enet$caliplot
predtest_enet$cmplot
predtest_enet$metrics
predtest_enet$diycutoff
predtest_enet$ksplot

# pROC包auc值及其置信区间
pROC::auc(predtest_enet$proc)
pROC::ci.auc(predtest_enet$proc)

# ROC比较检验
pROC::roc.test(predtrain_enet$proc, predtest_enet$proc)


# 合并训练集和测试集上ROC曲线
predtrain_enet$rocresult %>%
  bind_rows(predtest_enet$rocresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上PR曲线
predtrain_enet$prresult %>%
  bind_rows(predtest_enet$prresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上校准曲线
predtrain_enet$caliresult %>%
  bind_rows(predtest_enet$caliresult) %>%
  mutate(data = forcats::as_factor(data)) %>%
  ggplot(aes(x = predprobgroup,
             y = Fraction, 
             color = data)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3, pch = 15) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  labs(x = "Bin Midpoint", y = "Event Rate", color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上性能指标
predtrain_enet$metrics %>%
  bind_rows(predtest_enet$metrics) %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


# 最优超参数交叉验证的结果
evalcv_enet <- bestcv4cls2(
  wkflow = wk_enet,
  tuneresult = tune_enet,
  hpbest = hpbest_enet,
  yname = "Operation_time",
  modelname = "ENet",
  v = 10,
  positivelevel = yourpositivelevel
)
evalcv_enet$cvroc
evalcv_enet$cvpr
evalcv_enet$evalcv








# 保存评估结果
save(datarecipe_dt,
     model_dt,
     wk_dt,
     hpgrid_dt, # 如果采用贝叶斯优化则删掉这一行
     tune_dt,
     predtrain_dt,
     predtest_dt,
     evalcv_dt,
    # vipdata_dt,
     file = ".\\cls2\\evalresult_dt.RData")

# 保存模型结果供shiny部署之用，本课程不包括shiny内容
final_dt_heart <- final_dt
traindata_heart <- traindata
save(final_dt_heart,
     traindata_heart,
     file = ".\\cls2shiny\\shiny_dt_heart.RData")





# 保存评估结果
save(datarecipe_rf,
     model_rf,
     wk_rf,
     hpgrid_rf,   # 如果采用贝叶斯优化则替换为 param_rf
     tune_rf,
     predtrain_rf,
     predtest_rf,
     evalcv_rf,
    # vipdata_rf,
     file = ".\\cls2\\evalresult_rf.RData")

# 保存模型结果供shiny部署之用，本课程不包括shiny内容
final_rf_heart <- final_rf
traindata_heart <- traindata
save(final_rf_heart,
     traindata_heart,
     file = ".\\cls2shiny\\shiny_rf_heart.RData")






# 保存评估结果
save(datarecipe_xgboost,
     model_xgboost,
     wk_xgboost,
     hpgrid_xgboost,  # 如果采用贝叶斯优化则替换为 param_xgboost
     tune_xgboost,
     predtrain_xgboost,
     predtest_xgboost,
     evalcv_xgboost,
    # vipdata_xgboost,
     file = ".\\cls2\\evalresult_xgboost.RData")

# 保存模型结果供shiny部署之用，本课程不包括shiny内容
final_xgboost_heart <- final_xgboost
traindata_heart <- traindata
save(final_xgboost_heart,
     traindata_heart,
     file = ".\\cls2shiny\\shiny_xgboost_heart.RData")




# 保存评估结果
save(datarecipe_lightgbm,
     model_lightgbm,
     wk_lightgbm,
     hpgrid_lightgbm, # 如果采用贝叶斯优化则替换为 param_lightgbm
     tune_lightgbm,
     predtrain_lightgbm,
     predtest_lightgbm,
     evalcv_lightgbm,
   #  vipdata_lightgbm,
     file = ".\\cls2\\evalresult_lightgbm.RData")

# 保存模型结果供shiny部署之用，本课程不包括shiny内容
final_lightgbm_heart <- final_lightgbm
traindata_heart <- traindata
save(final_lightgbm_heart,
     traindata_heart,
     file = ".\\cls2shiny\\shiny_lightgbm_heart.RData")




# 保存评估结果
save(datarecipe_svm,
     model_svm,
     wk_svm,
     hpgrid_svm, # 如果采用贝叶斯优化则删除这一行
     tune_svm,
     predtrain_svm,
     predtest_svm,
     evalcv_svm,
    # vipdata_svm,
     file = ".\\cls2\\evalresult_svm.RData")

# 保存模型结果供shiny部署之用，本课程不包括shiny内容
final_svm_heart <- final_svm
traindata_heart <- traindata
save(final_svm_heart,
     traindata_heart,
     file = ".\\cls2shiny\\shiny_svm_heart.RData")


# 保存评估结果
save(datarecipe_mlp,
     model_mlp,
     wk_mlp,
     hpgrid_mlp, # 如果采用贝叶斯优化则删除这一行
     tune_mlp,
     predtrain_mlp,
     predtest_mlp,
     evalcv_mlp,
     #vipdata_mlp,
     file = ".\\cls2\\evalresult_mlp.RData")

# 保存模型结果供shiny部署之用，本课程不包括shiny内容
final_mlp_heart <- final_mlp
traindata_heart <- traindata
save(final_mlp_heart,
     traindata_heart,
     file = ".\\cls2shiny\\shiny_mlp_heart.RData")





# 保存评估结果
save(datarecipe_knn,
     model_knn,
     wk_knn,
     hpgrid_knn, # 如果采用贝叶斯优化则删掉这一行
     tune_knn,
     predtrain_knn,
     predtest_knn,
     evalcv_knn,
    # vipdata_knn,
     file = ".\\cls2\\evalresult_knn.RData")

# 保存模型结果供shiny部署之用，本课程不包括shiny内容
final_knn_heart <- final_knn
traindata_heart <- traindata
save(final_knn_heart,
     traindata_heart,
     file = ".\\cls2shiny\\shiny_knn_heart.RData")





# 保存评估结果
save(datarecipe_logistic,
     model_logistic,
     wk_logistic,
     cv_logistic,
     predtrain_logistic,
     predtest_logistic,
     evalcv_logistic,
     #vipdata_logistic,
     file = ".\\cls2\\evalresult_logistic.RData")


# 保存模型结果供shiny部署之用，本课程不包括shiny内容
final_logistic_heart <- final_logistic
traindata_heart <- traindata
save(final_logistic_heart,
     traindata_heart,
     file = ".\\cls2shiny\\shiny_logistic_heart.RData")



# 保存评估结果
save(datarecipe_lasso,
     model_lasso,
     wk_lasso,
     hpgrid_lasso,   # 如果采用贝叶斯优化则删掉这一行
     tune_lasso,
     predtrain_lasso,
     predtest_lasso,
     evalcv_lasso,
     #vipdata_lasso,
     file = ".\\cls2\\evalresult_lasso.RData")

# 保存模型结果供shiny部署之用，本课程不包括shiny内容
final_lasso_heart <- final_lasso
traindata_heart <- traindata
save(final_lasso_heart,
     traindata_heart,
     file = ".\\cls2shiny\\shiny_lasso_heart.RData")



# 保存评估结果
save(datarecipe_ridge,
     model_ridge,
     wk_ridge,
     hpgrid_ridge,   # 如果采用贝叶斯优化则删掉这一行
     tune_ridge,
     predtrain_ridge,
     predtest_ridge,
     evalcv_ridge,
     #vipdata_ridge,
     file = ".\\cls2\\evalresult_ridge.RData")

# 保存模型结果供shiny部署之用，本课程不包括shiny内容
final_ridge_heart <- final_ridge
traindata_heart <- traindata
save(final_ridge_heart,
     traindata_heart,
     file = ".\\cls2shiny\\shiny_ridge_heart.RData")






# 保存评估结果
save(datarecipe_enet,
     model_enet,
     wk_enet,
     hpgrid_enet,   # 如果采用贝叶斯优化则删掉这一行
     tune_enet,
     predtrain_enet,
     predtest_enet,
     evalcv_enet,
     #vipdata_enet,
     file = ".\\cls2\\evalresult_enet.RData")

# 保存模型结果供shiny部署之用，本课程不包括shiny内容
final_enet_heart <- final_enet
traindata_heart <- traindata
save(final_enet_heart,
     traindata_heart,
     file = ".\\cls2shiny\\shiny_enet_heart.RData")






library(stacks)

##############################

load(".\\cls2\\evalresult_knn.RData")
load(".\\cls2\\evalresult_rf.RData")
load(".\\cls2\\evalresult_logistic.RData")
load(".\\cls2\\evalresult_dt.RData")
load(".\\cls2\\evalresult_enet.RData")
load(".\\cls2\\evalresult_lasso.RData")
load(".\\cls2\\evalresult_ridge.RData")
load(".\\cls2\\evalresult_mlp.RData")
load(".\\cls2\\evalresult_svm.RData")
load(".\\cls2\\evalresult_mlp.RData")
load(".\\cls2\\evalresult_svm.RData")
load(".\\cls2\\evalresult_xgboost.RData")
models_stack <- 
  stacks() %>% 
  add_candidates(tune_knn) %>%
  add_candidates(tune_rf) %>%
  add_candidates(tune_dt)%>%
  add_candidates(tune_lasso)%>%
  add_candidates(tune_ridge)%>%
  add_candidates(tune_enet)%>%
  add_candidates(tune_mlp)%>%
  add_candidates(tune_svm)%>%
  add_candidates(tune_xgboost)%>%
  add_candidates(cv_logistic)
models_stack


##############################

# 拟合stacking元模型——stack
set.seed(42)
meta_stack <- blend_predictions(
  models_stack, 
  penalty = 10^seq(-2, -0.5, length = 20),
  control = control_grid(save_pred = T, 
                         verbose = T,
                         event_level = "second",
                         parallel_over = "everything",
                         save_workflow = T)
)
meta_stack
autoplot(meta_stack) +
  theme_bw() +
  theme(text = element_text(family = "serif"))

# 拟合选定的基础模型
set.seed(42)
final_stack <- fit_members(meta_stack)
final_stack

######################################################

# 应用stacking模型预测并评估
# 训练集
predtrain_stack <- eval4cls2(
  model = final_stack, 
  dataset = traindata, 
  yname = "Operation_time", 
  modelname = "Stacking", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_stack$prediction
predtrain_stack$predprobplot
predtrain_stack$rocplot
predtrain_stack$prplot
predtrain_stack$caliplot
predtrain_stack$cmplot
predtrain_stack$metrics
predtrain_stack$diycutoff
predtrain_stack$ksplot

# pROC包auc值及其置信区间
pROC::auc(predtrain_stack$proc)
pROC::ci.auc(predtrain_stack$proc)

# 测试集
predtest_stack <- eval4cls2(
  model = final_stack, 
  dataset = testdata, 
  yname = "Operation_time", 
  modelname = "Stacking", 
  datasetname = "testdata",
  cutoff = predtrain_stack$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_stack$prediction
predtest_stack$predprobplot
predtest_stack$rocplot
predtest_stack$prplot
predtest_stack$caliplot
predtest_stack$cmplot
predtest_stack$metrics
predtest_stack$diycutoff
predtest_stack$ksplot

# pROC包auc值及其置信区间
pROC::auc(predtest_stack$proc)
pROC::ci.auc(predtest_stack$proc)

# ROC比较检验
pROC::roc.test(predtrain_stack$proc, predtest_stack$proc)


# 合并训练集和测试集上ROC曲线
predtrain_stack$rocresult %>%
  bind_rows(predtest_stack$rocresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上PR曲线
predtrain_stack$prresult %>%
  bind_rows(predtest_stack$prresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上校准曲线
predtrain_stack$caliresult %>%
  bind_rows(predtest_stack$caliresult) %>%
  mutate(data = forcats::as_factor(data)) %>%
  ggplot(aes(x = predprobgroup,
             y = Fraction, 
             color = data)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3, pch = 15) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  labs(x = "Bin Midpoint", y = "Event Rate", color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上性能指标
predtrain_stack$metrics %>%
  bind_rows(predtest_stack$metrics) %>%
  select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)





# 保存评估结果
save(predtrain_stack,
     predtest_stack,
     #vipdata_stack,
     file = ".\\cls2\\evalresult_stack.RData")

# 保存模型结果供shiny部署之用，本课程不包括shiny内容
final_stack_heart <- final_stack
traindata_heart <- traindata
save(final_stack_heart,
     traindata_heart,
     file = ".\\cls2shiny\\shiny_stack_heart.RData")









# 模型机器---R语言tidymodels包机器学习分类与回归模型---二分类---模型比较

#############################################################
# remotes::install_github("tidymodels/probably")
library(tidymodels)

# 加载各个模型的评估结果
evalfiles <- list.files(".\\cls2\\", full.names = T)
lapply(evalfiles, load, .GlobalEnv)

# 横向比较的模型个数
nmodels <- 12
cols4model <- rainbow(nmodels)  # 模型统一配色

#############################################################

# 各个模型在训练集上的性能指标
evaltrain <- bind_rows(
  lapply(list(predtrain_logistic, predtrain_dt, 
              predtrain_lasso, predtrain_ridge, predtrain_enet,
              predtrain_knn, predtrain_lightgbm, predtrain_rf,
              predtrain_xgboost, predtrain_svm, predtrain_mlp,
              predtrain_stack), 
         "[[", 
         "metrics")
) %>%
  mutate(model = forcats::as_factor(model))
evaltrain
# 平行线图
evaltrain_max <-   evaltrain %>% 
  filter(!(.metric %in% c("detection_prevalence"))) %>%
  group_by(.metric) %>%
  slice_max(.estimate)
evaltrain_min <-   evaltrain %>% 
  filter(!(.metric %in% c("detection_prevalence"))) %>%
  group_by(.metric) %>%
  slice_min(.estimate)

evaltrain %>%
  filter(!(.metric %in% c("detection_prevalence"))) %>%
  ggplot(aes(x = .metric, y = .estimate, color = model)) +
  geom_point() +
  geom_line(aes(group = model)) +
  # ggrepel::geom_text_repel(evaltrain_max,
  #                          mapping = aes(label = model),
  #                          nudge_y = 0.05,
  #                          angle = 90,
  #                          show.legend = F) +
  # ggrepel::geom_text_repel(evaltrain_min,
  #                          mapping = aes(label = model),
  #                          nudge_y = -0.05,
  #                          angle = 90,
  #                          show.legend = F) +
  scale_color_manual(values = cols4model) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(x = "", y = "", color = "") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(),
        panel.grid.minor.y = element_blank(),
        panel.grid.major.y = element_blank(),
        text = element_text(family = "serif"))

# 指标热图
evaltrain %>%
  filter(!(.metric %in% c("detection_prevalence"))) %>%
  dplyr::select(model, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  mutate(model = reorder(model, roc_auc)) %>%
  pivot_longer(cols = -1) %>%
  group_by(name) %>%
  mutate(valuescale = (value-min(value)) / (max(value)-min(value))) %>%
  ungroup() %>%
  ggplot(aes(x = name, y = model, fill = valuescale)) +
  geom_tile(color = "white", show.legend = F) +
  geom_text(aes(label = round(value, 2))) +
  scale_fill_gradient(low = "green", high = "red") +
  labs(x = "", y = "", fill = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
        text = element_text(family = "serif"))

# 各个模型在训练集上的性能指标表格
evaltrain2 <- evaltrain %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)
evaltrain2

# 各个模型在训练集上的性能指标图示
# ROCAUC
evaltrain2 %>%
  ggplot(aes(x = model, y = roc_auc, fill = model)) +
  geom_col(width = 0.5, show.legend = F) +
  geom_text(aes(label = round(roc_auc, 2)), 
            nudge_y = -0.03) +
  scale_fill_manual(values = cols4model) +
  labs(x = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        text = element_text(family = "serif"))

#############################

# 各个模型在训练集上的ROC
roctrain <- bind_rows(
  lapply(list(predtrain_logistic, predtrain_dt, 
              predtrain_lasso, predtrain_ridge, predtrain_enet,
              predtrain_knn, predtrain_lightgbm, predtrain_rf,
              predtrain_xgboost, predtrain_svm, predtrain_mlp,
              predtrain_stack), 
         "[[", 
         "rocresult")
) %>%
  mutate(model = forcats::as_factor(model))
roctrain

roctrain %>%
  mutate(modelauc = paste(model,  curvelab),
         modelauc = forcats::as_factor(modelauc)) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = modelauc)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_color_manual(values = cols4model) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
  labs(color = "", title = paste0("ROCs on traindata")) +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(),
        text = element_text(family = "serif"))

# 各个模型在训练集上的PR
prtrain <- bind_rows(
  lapply(list(predtrain_logistic, predtrain_dt, 
              predtrain_lasso, predtrain_ridge, predtrain_enet,
              predtrain_knn, predtrain_lightgbm, predtrain_rf,
              predtrain_xgboost, predtrain_svm, predtrain_mlp,
              predtrain_stack), 
         "[[", 
         "prresult")
) %>%
  mutate(model = forcats::as_factor(model))
prtrain

prtrain %>%
  mutate(modelauc = paste(model, curvelab),
         modelauc = forcats::as_factor(modelauc)) %>%
  ggplot(aes(x = recall, y = precision, color = modelauc)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_color_manual(values = cols4model) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
  labs(color = "", title = paste0("PRs on traindata")) +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank(),
        text = element_text(family = "serif"))

############################

# 各个模型在训练集上的预测概率
predtrain <- bind_rows(
  lapply(list(predtrain_logistic, predtrain_dt, 
              predtrain_lasso, predtrain_ridge, predtrain_enet,
              predtrain_knn, predtrain_lightgbm, predtrain_rf,
              predtrain_xgboost, predtrain_svm, predtrain_mlp,
              predtrain_stack), 
         "[[", 
         "prediction")
) %>%
  mutate(model = forcats::as_factor(model))
predtrain

# 各个模型在训练集上的预测概率---宽数据
predtrain2 <- predtrain %>%
  dplyr::select(-.pred_No) %>%
  mutate(id = rep(1:nrow(predtrain_logistic$prediction), 
                  length(unique(predtrain$model)))) %>%
  pivot_wider(id_cols = c(id, .obs), 
              names_from = model, 
              values_from = .pred_Yes) %>%
  dplyr::select(id, .obs, sort(unique(predtrain$model)))
predtrain2

############################


# 各个模型在训练集上的校准曲线
# 校准曲线附加置信区间
predtrain %>%
  probably::cal_plot_breaks(.obs, 
                            .pred_Yes, 
                            event_level = "second", 
                            num_breaks = 5,  # 可以改大改小
                            .by = model) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
  scale_color_manual(values = cols4model) +
  theme_bw() +
  theme(legend.position = "none",
        panel.grid = element_blank(),
        text = element_text(family = "serif"))

# 各个模型在训练集上的校准曲线
calitrain <- bind_rows(
  lapply(list(predtrain_logistic, predtrain_dt, 
              predtrain_lasso, predtrain_ridge, predtrain_enet,
              predtrain_knn, predtrain_lightgbm, predtrain_rf,
              predtrain_xgboost, predtrain_svm, predtrain_mlp,
              predtrain_stack), 
         "[[", 
         "caliresult")
) %>%
  mutate(model = forcats::as_factor(model))
calitrain

calitrain %>%
  mutate(model = forcats::as_factor(model)) %>%
  ggplot(aes(x = predprobgroup, y = Fraction, color = model)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3, pch = 15) +
  geom_abline(linetype = "dashed") +
  scale_color_manual(values = cols4model) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
  labs(color = "", x = "Bin Midpoint", y = "Event Rate",
       title = paste0("calibration on traindata")) +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(),
        text = element_text(family = "serif"))


############################

# 各个模型在训练集上的DCA
traindca_obj <- dcurves::dca(as.formula(
  paste0(".obs ~ ", 
         paste(colnames(predtrain2)[3:ncol(predtrain2)], 
               collapse = " + "))
),
data = predtrain2,
thresholds = seq(0, 1, by = 0.01)
)
plot(traindca_obj, smooth = T, span = 0.5) +
  scale_color_manual(values = c("black", "grey", cols4model)) +
  labs(title = "DCA on traindata") +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,1),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

#############################################################

# 各个模型在测试集上的性能指标
evaltest <- bind_rows(
  lapply(list(predtest_logistic, predtest_dt, 
              predtest_lasso, predtest_ridge, predtest_enet,
              predtest_knn, predtest_lightgbm, predtest_rf,
              predtest_xgboost, predtest_svm, predtest_mlp,
              predtest_stack), 
         "[[", 
         "metrics")
) %>%
  mutate(model = forcats::as_factor(model))
evaltest
# 平行线图
evaltest_max <-   evaltest %>% 
  filter(!(.metric %in% c("detection_prevalence"))) %>%
  group_by(.metric) %>%
  slice_max(.estimate)
evaltest_min <-   evaltest %>% 
  filter(!(.metric %in% c("detection_prevalence"))) %>%
  group_by(.metric) %>%
  slice_min(.estimate)

evaltest %>%
  filter(!(.metric %in% c("detection_prevalence"))) %>%
  ggplot(aes(x = .metric, y = .estimate, color = model)) +
  geom_point() +
  geom_line(aes(group = model)) +
  # ggrepel::geom_text_repel(evaltest_max, 
  #                          mapping = aes(label = model), 
  #                          nudge_y = 0.05,
  #                          angle = 90,
  #                          show.legend = F) +
  # ggrepel::geom_text_repel(evaltest_min, 
  #                          mapping = aes(label = model), 
  #                          nudge_y = -0.05,
  #                          angle = 90,
  #                          show.legend = F) +
  scale_y_continuous(limits = c(0, 1)) +
  scale_color_manual(values = cols4model) +
  labs(x = "", y = "", color = "") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(),
        panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank(),
        text = element_text(family = "serif"))

# 指标热图
evaltest %>%
  filter(!(.metric %in% c("detection_prevalence"))) %>%
  dplyr::select(model, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  mutate(model = reorder(model, roc_auc)) %>%
  pivot_longer(cols = -1) %>%
  group_by(name) %>%
  mutate(valuescale = (value-min(value)) / (max(value)-min(value))) %>%
  ungroup() %>%
  ggplot(aes(x = name, y = model, fill = valuescale)) +
  geom_tile(color = "white", show.legend = F) +
  geom_text(aes(label = round(value, 2))) +
  scale_fill_gradient(low = "green", high = "red") +
  labs(x = "", y = "", fill = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
        text = element_text(family = "serif"))

# 各个模型在测试集上的性能指标表格
evaltest2 <- evaltest %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)
evaltest2

# 各个模型在测试集上的性能指标图示
# ROCAUC
evaltest2 %>%
  ggplot(aes(x = model, y = roc_auc, fill = model)) +
  geom_col(width = 0.5, show.legend = F) +
  geom_text(aes(label = round(roc_auc, 2)), 
            nudge_y = -0.03) +
  scale_fill_manual(values = cols4model) +
  labs(x = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        text = element_text(family = "serif"))

#############################

# 各个模型在测试集上的ROC
roctest <- bind_rows(
  lapply(list(predtest_logistic, predtest_dt,
              predtest_lasso, predtest_ridge, predtest_enet,
              predtest_knn, predtest_lightgbm, predtest_rf,
              predtest_xgboost, predtest_svm, predtest_mlp,
              predtest_stack), 
         "[[", 
         "rocresult")
) %>%
  mutate(model = forcats::as_factor(model))
roctest

roctest %>%
  mutate(modelauc = paste(model,  curvelab),
         modelauc = forcats::as_factor(modelauc)) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = modelauc)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_color_manual(values = cols4model) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
  labs(color = "", title = paste0("ROCs on testdata")) +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(),
        text = element_text(family = "serif"))

# 各个模型在测试集上的PR
prtest <- bind_rows(
  lapply(list(predtest_logistic, predtest_dt,
              predtest_lasso, predtest_ridge, predtest_enet,
              predtest_knn, predtest_lightgbm, predtest_rf,
              predtest_xgboost, predtest_svm, predtest_mlp,
              predtest_stack), 
         "[[", 
         "prresult")
) %>%
  mutate(model = forcats::as_factor(model))
prtest

prtest %>%
  mutate(modelauc = paste(model, curvelab),
         modelauc = forcats::as_factor(modelauc)) %>%
  ggplot(aes(x = recall, y = precision, color = modelauc)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_color_manual(values = cols4model) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
  labs(color = "", title = paste0("PRs on testdata")) +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank(),
        text = element_text(family = "serif"))

############################

# 各个模型在测试集上的预测概率
predtest <- bind_rows(
  lapply(list(predtest_logistic, predtest_dt,
              predtest_lasso, predtest_ridge, predtest_enet,
              predtest_knn, predtest_lightgbm, predtest_rf,
              predtest_xgboost, predtest_svm, predtest_mlp,
              predtest_stack), 
         "[[", 
         "prediction")
) %>%
  mutate(model = forcats::as_factor(model))
predtest

# 各个模型在测试集上的预测概率---宽数据
predtest2 <- predtest %>%
  dplyr::select(-.pred_No) %>%
  mutate(id = rep(1:nrow(predtest_logistic$prediction), 
                  length(unique(predtest$model)))) %>%
  pivot_wider(id_cols = c(id, .obs), 
              names_from = model, 
              values_from = .pred_Yes) %>%
  dplyr::select(id, .obs, sort(unique(predtest$model)))
predtest2

############################


# 各个模型在测试集上的校准曲线
# 校准曲线附加置信区间
predtest %>%
  probably::cal_plot_breaks(.obs, 
                            .pred_Yes, 
                            event_level = "second", 
                            num_breaks = 5,  # 可以改大改小
                            .by = model) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
  scale_color_manual(values = cols4model) +
  theme_bw() +
  theme(legend.position = "none",
        panel.grid = element_blank(),
        text = element_text(family = "serif"))

# 各个模型在测试集上的校准曲线
calitest <- bind_rows(
  lapply(list(predtest_logistic, predtest_dt, 
              predtest_lasso, predtest_ridge, predtest_enet,
              predtest_knn, predtest_lightgbm, predtest_rf,
              predtest_xgboost, predtest_svm, predtest_mlp,
              predtest_stack), 
         "[[", 
         "caliresult")
) %>%
  mutate(model = forcats::as_factor(model))
calitest

calitest %>%
  mutate(model = forcats::as_factor(model)) %>%
  ggplot(aes(x = predprobgroup, y = Fraction, color = model)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3, pch = 15) +
  geom_abline(linetype = "dashed") +
  scale_color_manual(values = cols4model) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
  labs(color = "", x = "Bin Midpoint", y = "Event Rate",
       title = paste0("calibration on testdata")) +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(),
        text = element_text(family = "serif"))


############################

# 各个模型在测试集上的DCA
testdca_obj <- dcurves::dca(as.formula(
  paste0(".obs ~ ", 
         paste(colnames(predtest2)[3:ncol(predtest2)], 
               collapse = " + "))
),
data = predtest2,
thresholds = seq(0, 1, by = 0.01)
)
plot(testdca_obj, smooth = T, span = 0.5) +
  scale_color_manual(values = c("black", "grey", cols4model)) +
  labs(title = "DCA on testdata") +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,1),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

#############################################################

# 各个模型交叉验证的各折指标点线图
evalcv <- bind_rows(
  lapply(list(evalcv_logistic, evalcv_dt,
              evalcv_lasso, evalcv_ridge, evalcv_enet,
              evalcv_knn, evalcv_lightgbm, evalcv_rf,
              evalcv_xgboost, evalcv_svm, evalcv_mlp), 
         "[[", 
         "evalcv")
) %>%
  mutate(
    model = forcats::as_factor(model),
    modelperf = paste0(model, "(", round(mean, 2),"±",
                       round(sd,2), ")")
  )
evalcv

# ROC
evalcvroc_max <-   evalcv %>% 
  filter(.metric == "roc_auc") %>%
  group_by(id) %>%
  slice_max(.estimate)
evalcvroc_min <-   evalcv %>% 
  filter(.metric == "roc_auc") %>%
  group_by(id) %>%
  slice_min(.estimate)
evalcv %>%
  filter(.metric == "roc_auc") %>%
  ggplot(aes(x = id, y = .estimate, 
             group = modelperf, color = modelperf)) +
  geom_point() +
  geom_line() +
  ggrepel::geom_text_repel(evalcvroc_max, 
                           mapping = aes(label = model), 
                           nudge_y = 0.01,
                           show.legend = F) +
  ggrepel::geom_text_repel(evalcvroc_min, 
                           mapping = aes(label = model), 
                           nudge_y = -0.01,
                           show.legend = F) +
  scale_y_continuous(limits = c(0, 1)) +
  scale_color_manual(values = cols4model) +
  labs(x = "", y = "ROCAUC", color = "") +
  theme_bw() +
  theme(legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(),
        panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank(),
        text = element_text(family = "serif"))

# PR
evalcvpr_max <-   evalcv %>% 
  filter(.metric == "pr_auc") %>%
  group_by(id) %>%
  slice_max(.estimate)
evalcvpr_min <-   evalcv %>% 
  filter(.metric == "pr_auc") %>%
  group_by(id) %>%
  slice_min(.estimate)
evalcv %>%
  filter(.metric == "pr_auc") %>%
  ggplot(aes(x = id, y = .estimate, 
             group = modelperf, color = modelperf)) +
  geom_point() +
  geom_line() +
  ggrepel::geom_text_repel(evalcvpr_max, 
                           mapping = aes(label = model), 
                           nudge_y = 0.01,
                           show.legend = F) +
  ggrepel::geom_text_repel(evalcvpr_min, 
                           mapping = aes(label = model), 
                           nudge_y = -0.01,
                           show.legend = F) +
  scale_y_continuous(limits = c(0.5, 1)) +
  scale_color_manual(values = cols4model) +
  labs(x = "", y = "PRAUC", color = "") +
  theme_bw() +
  theme(legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(),
        panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank(),
        text = element_text(family = "serif"))

# 各个模型交叉验证的指标平均值图(带上下限)
# ROC
evalcv %>%
  filter(.metric == "roc_auc") %>%
  group_by(model) %>%
  sample_n(size = 1) %>%
  ungroup() %>%
  ggplot(aes(x = model, y = mean, color = model)) +
  geom_point(size = 2, show.legend = F) +
  # geom_line(group = 1) +
  geom_errorbar(aes(ymin = mean-sd, 
                    ymax = mean+sd),
                width = 0.1, 
                linewidth = 1.2,
                show.legend = F) +
  scale_y_continuous(limits = c(0.5, 1)) +
  scale_color_manual(values = cols4model) +
  labs(x = "", y = "cv roc_auc") +
  theme_bw() +
  theme(panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank(),
        text = element_text(family = "serif"))

# PR
evalcv %>%
  filter(.metric == "pr_auc") %>%
  group_by(model) %>%
  sample_n(size = 1) %>%
  ungroup() %>%
  ggplot(aes(x = model, y = mean, color = model)) +
  geom_point(size = 2, show.legend = F) +
  # geom_line(group = 1) +
  geom_errorbar(aes(ymin = mean-sd, 
                    ymax = mean+sd),
                width = 0.1, 
                linewidth = 1.2,
                show.legend = F) +
  scale_y_continuous(limits = c(0.5, 1)) +
  scale_color_manual(values = cols4model) +
  labs(x = "", y = "cv pr_auc") +
  theme_bw() +
  theme(panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank(),
        text = element_text(family = "serif"))



write.csv(evaltest, file = "evaltest.csv")
write.csv(evaltrain, file = "evaltrain.csv")
write.csv(predtest, file = "testpredtest.csv")#模型预测结果长数据
write.csv(predtrain, file = "trainpredtest.csv")#模型预测结果长数据
write.csv(traindata, file = "traindata.csv")
write.csv(testdata, file = "testdata.csv")
write.csv(evalcv, file = "evalcv.csv")
