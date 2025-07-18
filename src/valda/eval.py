## data_removal.py
## Evaluate the performance of data valuation by removing one data point
## at a time from the training set

from sklearn.metrics import accuracy_score, auc
from sklearn.linear_model import LogisticRegression as LR

import operator

def data_removal(vals, trnX, trnY, tstX, tstY, clf=None,
                     remove_high_value=True):
    '''
    trnX, trnY - training examples
    tstX, tstY - test examples
    vals - a Python dict that contains data indices and values
    clf - the classifier that will be used for evaluation
    '''
    # Create data indices for data removal
    N = trnX.shape[0]
    Idx_keep = [True]*N

    if clf is None:
        clf = LR(solver="liblinear", max_iter=500, random_state=0)
    # Sorted the data indices with a descreasing order
    sorted_dct = sorted(vals.items(), key=operator.itemgetter(1), reverse=True)
    # Accuracy list
    accs = []
    if remove_high_value:
      lst = range(N)
    else:
      lst = range(N-1, -1, -1)
    # Compute 
    clf.fit(trnX, trnY)
    acc = accuracy_score(clf.predict(tstX), tstY)
    accs.append(acc)
    for k in lst: 
        # print(k)
        Idx_keep[sorted_dct[k][0]] = False
        trnX_k = trnX[Idx_keep, :]
        trnY_k = trnY[Idx_keep]
        try:
            clf.fit(trnX_k, trnY_k)
            # print('trnX_k.shape = {}'.format(trnX_k.shape))
            acc = accuracy_score(clf.predict(tstX), tstY)
            # print('acc = {}'.format(acc))
            accs.append(acc)
        except ValueError:
            # print("Training with data from a single class")
            accs.append(0.0)    
    return accs

# import numpy as np

# def softmax(x):
#     """计算输入数组的softmax."""
#     e_x = np.exp(x - np.max(x))  # 为了数值稳定性，减去最大值
#     return e_x / e_x.sum(axis=0)

# # 假设 sorted_vals 是已经排序的 [(index, value)] 格式的列表
# def select_indices_with_softmax(sorted_vals, threshold, size_sel):
#     # 选取超过阈值的索引
#     idx_sel = [idx for idx, val in sorted_vals if val > threshold]
    
#     if len(idx_sel) > size_sel:
#         # 如果选中的索引超过需要的数量，随机选择其中的 size_sel 个
#         idx_sel = np.random.choice(idx_sel, size=size_sel, replace=False)
#     elif len(idx_sel) < size_sel:
#         # 如果选中的索引不足，则从剩余索引中根据softmax后的概率选取
#         remaining = [(idx, val) for idx, val in sorted_vals if idx not in idx_sel]
        
#         # 将剩余的值转换为数组
#         remaining_vals = np.array([val for idx, val in remaining])
#         remaining_idxs = np.array([idx for idx, val in remaining])
        
#         # 计算剩余值的softmax概率
#         probs = softmax(remaining_vals)
        
#         # 计算还需要多少个索引
#         additional_needed = size_sel - len(idx_sel)
        
#         # 根据softmax概率分布随机选取剩余的索引
#         additional_idx_sel = np.random.choice(remaining_idxs, size=additional_needed, replace=False, p=probs)
        
#         # 将补充的索引加入到选中的索引列表中
#         idx_sel.extend(additional_idx_sel)
    
#     return idx_sel

# def stratified_sampling_with_softmax(trnY, values, sel, strategy='stratified'):
#     """
#     根据给定标签trnY，结合softmax进行分层采样。

#     参数:
#     - trnY: 标签数组
#     - values: 每个样本的值（用于softmax计算）
#     - sel: 每个类要选取的比例或数量
#     - strategy: 采样策略

#     返回:
#     - 选中的索引数组
#     """
#     N = len(trnY)
    
#     if strategy == 'stratified':
#         # 获取唯一类别
#         unique_classes = np.unique(trnY)
#         idx_sel = []
        
#         for cls in unique_classes:
#             # 获取当前类别的所有索引
#             cls_indices = [idx for idx, val in values if trnY[idx] == cls]
#             # 对应的值 (用于 softmax)
#             cls_values = [val for idx, val in values if idx in cls_indices]
            
#             # 如果 sel 是比例，计算要选择的样本数量
#             cls_size = int(np.floor(len(cls_indices) * sel)) if sel < 1 else int(sel)
            
#             # 计算当前类别下的 softmax 概率分布
#             cls_probs = softmax(np.array(cls_values))
            
#             # 根据 softmax 概率从当前类别中随机选择索引
#             cls_sel = np.random.choice(cls_indices, size=cls_size, replace=False, p=cls_probs)
            
#             # 将选择的索引加入 idx_sel 列表
#             idx_sel.extend(cls_sel)
        
#         # 将结果转换为数组返回
#         idx_sel = np.array(idx_sel)
        
#     return idx_sel


# def stratified_top_percent(trnY, values, sel):
#     """
#     对每个类别进行分层取Top sel百分比的样本。

#     参数:
#     - trnY: 标签数组，表示每个样本的类别
#     - values: 每个样本的值，用于确定Top百分比
#     - sel: 每个类别中要选择的百分比（0 < sel <= 1）

#     返回:
#     - 选中的索引数组
#     """
#     if not (0 < sel <= 1):
#         raise ValueError("参数 sel 必须是介于 0 和 1 之间的浮点数，表示百分比。")

#     N = len(trnY)
    
#     # 获取唯一类别
#     unique_classes = np.unique(trnY)
#     idx_sel = []
    
#     for cls in unique_classes:
#         # 获取当前类别的所有索引
#         cls_indices = [idx for idx, val in values if trnY[idx] == cls]
#             # 对应的值 (用于 softmax)
#         cls_values = [val for idx, val in values if idx in cls_indices]
        
#         # 计算要选择的样本数量，确保至少选择一个样本
#         cls_size = max(int(np.floor(len(cls_indices) * sel)), 1)
        
#         # 如果当前类别的样本数量小于或等于 cls_size，则选择所有样本
#         if len(cls_indices) <= cls_size:
#             top_indices = cls_indices
#         else:
#             # 获取Top cls_size的索引（按值从大到小排序）
#             # top_order = np.argsort(cls_values)[-cls_size:][::-1]
#             top_indices = cls_indices[:cls_size]
        
#         # 将选中的索引加入 idx_sel 列表
#         idx_sel.extend(top_indices)
    
#     # 将结果转换为数组返回
#     idx_sel = np.array(idx_sel)
    
#     return idx_sel

# def data_selection(vals, trnX, trnY, tstX, tstY, clf=None, 
#                    sel=0.25, strategy='greedy', temperature=1.0, threshold=0.5):
#     '''
#     Parameters:
#     - vals: A Python dictionary containing data indices and their corresponding values.
#     - trnX, trnY: Training examples and their labels.
#     - tstX, tstY: Test examples and their labels.
#     - clf: The classifier to be used for evaluation. Defaults to Logistic Regression.
#     - sel: The proportion or the number of data points to select.
#     - strategy: The strategy to use for data selection ('greedy', 'prob', 'softmax', 'roulette', 'stratified', 'threshold').
#     - temperature: Used in probabilistic sampling strategies to control randomness.
#     - threshold: Used in threshold-based sampling strategy.
#     '''
#     # Validate 'sel' parameter
#     if sel <= 0:
#         raise ValueError("The selection size cannot be zero or negative.")

#     N = trnX.shape[0]  # Total number of training examples

#     # Determine the number of items to select
#     size_sel = int(sel * N) if sel < 1 else int(sel)

#     if size_sel > N:
#         raise ValueError("Selection size exceeds available data.")

#     if clf is None:
#         clf = LR(solver="liblinear", max_iter=500, random_state=0)

#     threshold = 2 * 1 / len(trnY)
#     # Convert 'vals' dictionary to a sorted list of tuples (index, value) in descending order
#     sorted_vals = sorted(vals.items(), key=operator.itemgetter(1), reverse=True)
#     if strategy == 'random':
#         indices = np.array([idx for idx, val in sorted_vals])
#         idx_sel = np.random.choice(indices, size=size_sel, replace=False)

#     elif strategy == 'greedy':
#         # Select top 'size_sel' indices with highest values
#         idx_sel = [idx for idx, val in sorted_vals[:size_sel]]
#     elif strategy == 's-greedy':
#         idx_sel = stratified_top_percent(trnY, sorted_vals, sel=size_sel/len(trnY))
#     elif strategy == 'prob':
#         # Extract indices and corresponding values
#         indices = np.array([idx for idx, val in sorted_vals])
#         values = np.array([val for idx, val in sorted_vals])
#         values[values<=0] = 0
#         # Normalize values to create a probability distribution
#         probs = values / np.sum(values)

#         # Randomly select indices based on the probability distribution
#         idx_sel = np.random.choice(indices, size=size_sel, replace=False, p=probs)

#     elif strategy == 'softmax':
#         # Extract indices and corresponding values
#         indices = np.array([idx for idx, val in sorted_vals])
#         values = np.array([val for idx, val in sorted_vals])

#         # Apply temperature scaling
#         values_temp = np.exp(values / temperature)
#         probs = values_temp / np.sum(values_temp)  # Normalize to get probabilities

#         # Randomly select indices based on the scaled probability distribution
#         idx_sel = np.random.choice(indices, size=size_sel, replace=False, p=probs)

#     elif strategy == 'roulette':
#         # Extract indices and corresponding values
#         indices = np.array([idx for idx, val in sorted_vals])
#         values = np.array([val for idx, val in sorted_vals])
#         values[values<=0] = 0
#         # Normalize values to create a probability distribution
#         probs = values / np.sum(values)

#         # Randomly select indices based on the probability distribution
#         idx_sel = np.random.choice(indices, size=size_sel, replace=True, p=probs)

#     elif strategy == 'stratified':
#         idx_sel = stratified_sampling_with_softmax(trnY, sorted_vals, sel=size_sel/len(trnY))
#         # # Stratified sampling to maintain the proportion of each class
#         # unique_classes = np.unique(trnY)
#         # idx_sel = []
#         # for cls in unique_classes:
#         #     # Get all indices of the current class
#         #     cls_indices = [idx for idx in range(N) if trnY[idx] == cls]
#         #     # Determine the number of samples to select from this class
#         #     cls_size = int(np.floor(len(cls_indices) * sel)) if sel < 1 else int(sel)
#         #     # Randomly select indices from the current class
#         #     cls_sel = np.random.choice(cls_indices, size=cls_size, replace=False)
#         #     idx_sel.extend(cls_sel)
#         # idx_sel = np.array(idx_sel)

#     elif strategy == 'threshold':
#         idx_sel = select_indices_with_softmax(sorted_vals, threshold, size_sel)
#         # Select indices where the value exceeds the threshold
#         # idx_sel = [idx for idx, val in sorted_vals if val > threshold]
#         # if len(idx_sel) > size_sel:
#             # If more samples are selected than desired, randomly choose among them
#         #     idx_sel = np.random.choice(idx_sel, size=size_sel, replace=False)
#         # elif len(idx_sel) < size_sel:
#             # If fewer samples are selected, fill the rest with highest remaining values
#         #    remaining_indices = [idx for idx, val in sorted_vals if idx not in idx_sel]
#         #    additional_needed = size_sel - len(idx_sel)
#         #    idx_sel.extend(remaining_indices[:additional_needed])

#     else:
#         raise ValueError(f"Unimplemented selection strategy: {strategy}")

#     # Select data based on the chosen indices
#     trnX_sel, trnY_sel = trnX[idx_sel, :], trnY[idx_sel]
#     try:
#         # Train classifier and evaluate performance
#         clf.fit(trnX_sel, trnY_sel)
#         acc = accuracy_score(clf.predict(tstX), tstY)
#     except:
#         acc = 0

#     return acc



import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression as LR
import operator

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def rank_selection(vals, sel):
    """
    Rank-based selection method.
    Parameters:
        vals: A dictionary of data indices and Shapley values.
        sel: Number of data points to select (can be a float for proportion).
    Returns:
        Indices of selected data points.
    """
    # Ensure sel is an integer
    total_items = len(vals)
    sel = int(np.ceil(sel * total_items)) if isinstance(sel, float) else int(sel)
    sorted_vals = sorted(vals.items(), key=operator.itemgetter(1), reverse=True)
    return [idx for idx, _ in sorted_vals[:sel]]

def roulette_selection(vals, sel):
    """
    Roulette selection method.
    Parameters:
        vals: A dictionary of data indices and Shapley values.
        sel: Number of data points to select.
    Returns:
        Indices of selected data points.
    """
    indices = np.array([idx for idx, _ in vals.items()])
    values = np.array([val for _, val in vals.items()])
    values[values < 0] = 0  # Ensure non-negative probabilities
    probs = values / np.sum(values)
    return list(np.random.choice(indices, size=sel, replace=False, p=probs))

def softmax_selection(vals, sel, temperature=1.0):
    """
    Softmax-based selection method.
    Parameters:
        vals: A dictionary of data indices and Shapley values.
        sel: Number of data points to select.
        temperature: Temperature parameter for softmax.
    Returns:
        Indices of selected data points.
    """
    indices = np.array([idx for idx, _ in vals.items()])
    values = np.array([val for _, val in vals.items()])
    values_temp = np.exp(values / temperature)
    probs = values_temp / np.sum(values_temp)
    return list(np.random.choice(indices, size=sel, replace=False, p=probs))

def threshold_selection(vals, sel, threshold=0.5):
    """
    Threshold-based selection method.
    Parameters:
        vals: A dictionary of data indices and Shapley values.
        sel: Number of data points to select.
        threshold: Threshold for selecting data points.
    Returns:
        Indices of selected data points.
    """
    sorted_vals = sorted(vals.items(), key=operator.itemgetter(1), reverse=True)
    idx_sel = [idx for idx, val in sorted_vals if val > threshold]
    if len(idx_sel) > sel:
        idx_sel = np.random.choice(idx_sel, size=sel, replace=False)
    elif len(idx_sel) < sel:
        remaining = [idx for idx, _ in sorted_vals if idx not in idx_sel]
        additional_needed = sel - len(idx_sel)
        idx_sel.extend(remaining[:additional_needed])
    return idx_sel

def dynamic_softmax_selection(vals, sel, initial_temperature=1.0, feedback=None):
    """
    Dynamic Softmax-based selection method.
    Parameters:
        vals: A dictionary of data indices and Shapley values.
        sel: Number of data points to select.
        initial_temperature: Initial temperature for softmax.
        feedback: A callable function that provides feedback to adjust temperature.
    Returns:
        Indices of selected data points.
    """
    temperature = initial_temperature
    indices = np.array([idx for idx, _ in vals.items()])
    values = np.array([val for _, val in vals.items()])
    for _ in range(5):  # Example: Iterate 5 times for feedback adjustment
        values_temp = np.exp(values / temperature)
        probs = values_temp / np.sum(values_temp)
        selected_indices = list(np.random.choice(indices, size=sel, replace=False, p=probs))
        
        if feedback is not None:
            temperature = feedback(temperature)  # Update temperature based on feedback
    return selected_indices

# Example feedback function for dynamic softmax
def feedback_function(temperature):
    """
    Example feedback function that decreases the temperature over time.
    """
    return max(0.5, temperature * 0.9)  # Reduce temperature, but not below 0.5


def random_selection(vals, sel):
    """
    Random selection method.
    Parameters:
        vals: A dictionary of data indices and Shapley values.
        sel: Number of data points to select.
    Returns:
        Indices of selected data points.
    """
    indices = np.array([idx for idx, _ in vals.items()])
    return list(np.random.choice(indices, size=sel, replace=False))


def data_selection(vals, trnX, trnY, tstX, tstY, clf=None, 
                   sel=0.25, strategy='rank', temperature=1.0, threshold=0.5):
    '''
    Parameters:
    - vals: A Python dictionary containing data indices and their corresponding values.
    - trnX, trnY: Training examples and their labels.
    - tstX, tstY: Test examples and their labels.
    - clf: The classifier to be used for evaluation. Defaults to Logistic Regression.
    - sel: The proportion or the number of data points to select.
    - strategy: The strategy to use for data selection ('rank', 'roulette', 'softmax', 'threshold', 'dynamic').
    - temperature: Used in probabilistic sampling strategies to control randomness.
    - threshold: Used in threshold-based sampling strategy.
    '''
    # Validate 'sel' parameter
    if sel <= 0:
        raise ValueError("The selection size cannot be zero or negative.")
    
    if sel > trnX.shape[0]:
        raise ValueError("Selection size exceeds available data.")
    
    if sel < 1:
        sel = int(sel * trnX.shape[0])
    
    if strategy == 'rank':
        selected_indices = rank_selection(vals, sel)
    elif strategy == 'roulette':
        selected_indices = roulette_selection(vals, sel)
    elif strategy == 'softmax':
        selected_indices = softmax_selection(vals, sel, temperature)
    elif strategy == 'threshold':
        selected_indices = threshold_selection(vals, sel, threshold)
    elif strategy == 'dynamic':
        selected_indices = dynamic_softmax_selection(vals, sel, temperature, feedback_function)
    elif strategy == 'random':
        selected_indices = random_selection(vals, sel)
    else:
        raise ValueError(f"Unimplemented selection strategy: {strategy}")
    
    # Select data based on the chosen indices
    if clf is None:
        clf = LR(solver="liblinear", max_iter=500, random_state=0)
    if len(selected_indices) == 0:
        return 0
    else:
        trnX_sel, trnY_sel = trnX[selected_indices, :], trnY[selected_indices]
        try:
            # Train classifier and evaluate performance
            clf.fit(trnX_sel, trnY_sel)
            acc = accuracy_score(clf.predict(tstX), tstY)
        except:
            acc = 0

        return acc
