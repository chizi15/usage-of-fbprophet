import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from statsmodels.tools import eval_measures
from scipy import stats


"""
预测结果评价函数的使用方法:
1. 评价种类、品类群、课别、企业、全体层级的结果时，将企业种类的各对预测金额及理论销售金额传入函数，而不是将各汇总层级的序列对输入函数，避免低层级的偏差信息因正负号不同而消失；
评价单品、品种层级的结果时，将企业单品的各对预测销量及理论销量传入函数。不用单品评价结果去计算种类及更高层级的精度，是因为预测系统算法设计的优化目标是企业种类预测金额，而不是直接优化单品预测销量。
2. 当同一目标有三种序列时：预测2.0、预测1.0、实际销售，应把预测2.0和实际销售、预测1.0和实际销售分别传入函数regression_evaluation，
来评价预测2.0贴近实际销售的程度、预测1.0贴近实际销售的程度。当要评价预测2.0和预测1.0的偏离程度时，应把从regression_evaluation得到的两种值，再传入函数smape得到偏离程度。
因为smape是所有指标里唯一具有无偏性的百分比指标，不是以某一种序列为目标，评价另一种序列趋近该种序列的程度；所以不应把预测2.0和预测1.0传入regression_evaluation直接得到偏离程度。
也不能将预测2.0与预测1.0拼到一起，与理论销售作为一个整体的序列对传入regression_evaluation，因为两种预测结果是由两种不同模型得到的，若拼到一起，
在后续归一化时会相互影响，进而也会影响其后的加权平均，从而不能对每种模型的预测结果得到准确的相互独立的评价结果。
3. 除了最终指标precision是最重要的评价指标外，如果更关心平时的预测结果，参考MAPE更好；如果更关心节假日期间的预测结果，参考RMSE(eval_measures.rmse)更好。
4. 若要评价某个具体的企业种类的预测结果，不能将这个序列对输入regression_evaluation，可以输入分项函数中的任意一个；
若要评价某个种类的预测结果，则将各个企业种类的预测序列和真实序列组成的序列对传入regression_evaluation，对返回结果按序列求平均，得到该种类的预测评价结果；
此时序列对中的y_true根据企业的不同而不同；而不是将该种类的汇总序列对传入regression_evaluation。若要对某个具体的企业单品、某个单品的预测结果进行评价，与4中方法类似。
5. 若要评价某个企业品类群的预测结果，则将该企业品类群下所有企业种类序列对传入regression_evaluation，对返回结果按序列求平均，此时序列对中的y_true是不同的，因为是不同的企业种类的理论销售；
而不是将该品类群的汇总序列对传入regression_evaluation。若要对某个企业课别、某个企业、全体的预测结果进行评价，与5中方法类似；
若要对某个企业品种、某个品种的预测结果进行评价，也与5中方法类似，只是传入regression_evaluation的是由企业单品构成的序列对。
6. 若要评价（优选）不同算法模型或不同中间计算结果对同一企业种类的预测结果，则传入regression_evaluation的序列对中y_true是相同的，因为是同一企业种类的理论销售。
7. 综上，可完成对任一层级、每个层级下任一子集的准确评价。
"""


def dyn_seri_weighted(seri, type=None, w=None, initial=1, r=2, d=1):
    """
    传入一维数组seri，可以是series,array,list,tuple；若type='geometric'或'arithmetic'，且输入了w，则w不起作用；若不输入权重，则根据seri的长度动态计算基于几何级数或算数级数再作归一化的权重，再做算术平均；
    也可人为输入权重做算术平均；若不输入type和w，则进行简单算数平均；因为使用np.dot，则seri索引越小，权重越大；将seri各点与权重相乘再相加，得到一个最终点。
    :param seri: 需要进行加权变成一个点的一维数组
    :param type: 采用几何级数或算数级数进行加权，或人为指定权重，或默认权重相等，type = 'geometric'或'arithmetic'或None；若type='geometric'或'arithmetic'，且输入了w，则w不起作用。
    :param w: 一维的权重系数，可以是series,array,list,tuple；若手动输入，其长度必须和一维数组seri（即序列点数）相等
    :param r: 指定几何级数分母的公比
    :param d: 指定算数级数分母的公差
    :param initial: 指定算数级数分母的初始值
    :return: seri各点与权重w相乘再相加，返回的一个加权后的最终点
    """
    if type not in ['geometric', 'arithmetic', None]:
        raise Exception('type must be one of geometric, arithmetic or None')
    if type is not None:
        w = list()
        if type == 'geometric':
            for i in range(len(seri)):
                w.append(initial * (1 / r) ** i)  # 生成首项是initial，公比是(1/r)的几何级数作权重
        else:
            for i in range(len(seri)):
                w.append(1 / (initial + d * i))  # 生成首项是initial，公差是d的算术级数，再做倒数作为权重
        w = np.array(w) / sum(w)
    elif (type is None) and (w is None):
        w = np.ones(len(seri)) / sum(np.ones(len(seri)))  # 生成均等权重
    elif (type is None) and (w is not None) and (len(w) == len(seri)):
        w = np.array(w) / sum(w)  # 自定义权重
    else:
        raise Exception('手动输入的权重长度必须和一维数组长度（即序列点数）相等')
    if abs(sum(w)-1) > 0.001:
        raise Exception('weights are not useable')
    return np.dot(np.array(seri), w)


def dyn_df_weighted(df, type=None, w=None, initial=1, r=2, d=1):
    """
    传入二维数组df；若type='geometric'或'arithmetic'，且输入了w，则w不起作用；若不输入权重，则根据df的列数动态计算基于几何级数或算数级数再作归一化的权重，再做算术平均；
    也可人为输入权重做算术平均；若不输入type和w，则进行简单算数平均；因为使用np.matmul，则df.columns的索引越小，权重越大；将df的各列与权重相乘再相加，得到一条最终的序列。
    :param df: 需要进行加权变成一条序列的二维数组，df的每列代表一条需要进行加权的序列
    :param type: 采用几何级数或算数级数进行加权，或人为指定权重，或默认权重相等，type = 'geometric'或'arithmetic'或None；若type='geometric'或'arithmetic'，且输入了w，则w不起作用。
    :param w: 一维的权重系数，可以是series,array,list,tuple；若手动输入，其长度必须和一维数组seri（即序列点数）相等
    :param r: 指定几何级数分母的公比
    :param d: 指定算数级数分母的公差
    :param initial: 指定算数级数分母的初始值
    :return: df各列与权重w相乘再相加，返回一条最终的序列
    """
    if type not in ['geometric', 'arithmetic', None]:
        raise Exception('type must be one of geometric, arithmetic or None')
    if type is not None:
        w = list()
        if type == 'geometric':
            for i in range(len(df.columns)):
                w.append(initial * (1 / r) ** i)  # 生成首项是initial，公比是(1/r)的几何级数作权重
        else:
            for i in range(len(df.columns)):
                w.append(1 / (initial + d * i))  # 生成首项是initial，公差是d的算术级数，再做倒数作为权重
        w = np.array(w) / sum(w)
    elif (type is None) and (w is None):
        w = np.ones(len(df.columns)) / sum(np.ones(len(df.columns)))  # 生成均等权重
    elif (type is None) and (w is not None) and (len(w) == len(df.columns)):
        w = np.array(w) / sum(w)  # 自定义权重
    else:
        raise Exception('手动输入的权重长度必须和一维数组长度（即序列点数）相等')
    if abs(sum(w)-1) > 0.001:
        raise Exception('weights are not useable')
    return np.matmul(df.values, w)


def rmse(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean())


# y_true, y_pred无限制条件
def emlae(y_true, y_pred):
    """
    :param y_true: 一条真实值序列，array,series,list,tuple均可，长度要与预测值长度相等
    :param y_pred: 一条预测值序列，array,series,list,tuple均可，长度要与真实值长度相等
    :return: EMLAE，将残差离群值压缩后的一次绝对性指标，对残差的离群值不如MAE敏感
    """
    n = len(y_true)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    emlae = np.exp(sum(np.log(abs(y_pred - y_true) + 1)) / n) - 1
    return emlae


# y_true != 0
def mape(y_true, y_pred):
    """
    :param y_true: 一条真实值序列，array,series,list,tuple均可，长度要与预测值长度相等，且不能为0
    :param y_pred: 一条预测值序列，array,series,list,tuple均可，长度要与真实值长度相等
    :return: MAPE，零次的相对性指标，小数，代表预测值偏离真实值的平均程度；若>1，表示预测值偏离真实值的平均程度超过真实值的1倍，若<1，表示预测值偏离真实值的平均程度小于真实值的1倍
    """
    n = len(y_true)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mape = sum(abs((y_pred - y_true) / y_true)) / n
    return mape


# y_true + y_pred != 0
def smape(y_true, y_pred):
    """
    :param y_true: 一条真实值序列，array,series,list,tuple均可，长度要与预测值长度相等，与对应预测值不能同时为0
    :param y_pred: 一条预测值序列，array,series,list,tuple均可，长度要与真实值长度相等，与对应真实值不能同时为0
    :return: SMAPE，零次的相对性指标，小数，代表预测值偏离真实值的平均程度；不以某类特定的量为基准分母，而是将两类值均作为分母，第一具有对称性，
    第二不会因某类量数值的大小而影响评价结果，所以受离群值影响也比MAPE小；若>1，表示预测值偏离真实值的平均程度超过真实值与预测值均值的1倍，若<1，表示预测值偏离真实值的平均程度小于真实值与预测值均值的1倍。
    """
    n = len(y_true)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    smape = sum(abs(2 * (y_pred - y_true) / (y_pred + y_true))) / n
    return smape


# y_true, y_pred无限制条件
def male(y_true, y_pred):
    """
    param：
        Y:原始序列，array,series,list,tuple均可
        y:拟合序列，array,series,list,tuple均可
    return：
        对数MAE值
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true[y_true < 0] = 0
    y_pred[y_pred == -1] = -0.99
    male = sum(abs(np.log(abs(y_true+1)) - np.log(abs(y_pred+1)))) / len(y_true)
    return male


def regression_accuracy(y_true, y_pred):
    """
    :param y_true: 若干条真实序列组成的一个二维list或array或series，其中的每条真实序列必须是带索引的series，为了能对>0的数值的索引取交集；并与y_pred中的预测序列按顺序一一对应
    :param y_pred: 若干条预测序列组成的一个二维list或array或series，其中的每条预测序列必须是带索引的series，为了能对>0的数值的索引取交集；并与y_true中的真实序列按顺序一一对应
    :return: 精度指标，按顺序分别是：最终精度指标（越接近0越好），MAPE, SMAPE, RMSPE, MTD_p2, EMLAE, MAE, RMSE, MedAE, MTD_p1, MSE, MSLE

    测试几种常用的评价序列对的精度指标
    原则：1.带平方项的指标会放大在正负1之外的残差的影响，而压缩在正负1之内的残差的影响，由于各指标越接近零越好，则会惩罚正负1之外的残差，偏离正负1越远，越受到惩罚；而奖励正负1之内的残差。
    2.作对数变换的指标会降低离群值的影响，相对于不带对数项的指标，会惩罚非离群值。因为在(1,+∞)区间内，y=lnx的一阶导数比y=x的一阶导数小，则y=lnx比y=x递增慢。
    3.ln(1/x)+len(x)=0，即对一个数（如x）取对数，与其倒数（1/x）取对数，互为相反数；当x>0，ln(1/x)+x-1≥0，lnx+(1/x)-1≥0，可由求导证明；在(0,4]的区间内，lnx的增长速度快于x**(1/2)，在[4,+∞)区间内，lnx的增长速度慢于x**(1/2)，可由求导证明。
    """

    MAPE, SMAPE, RMSPE, MTD_p2 = [], [], [], []  # 零次的相对性指标
    EMLAE, MALE, MAE, RMSE, MedAE, MTD_p1 = [], [], [], [], [], []  # 一次的绝对性指标
    MSE, MSLE = [], []  # 二次的绝对性指标

    y_true_trun, y_pred_trun = [], []
    for i in range(len(y_true)):
        # 为了统一下列12个精度指标的条件，在y_true和y_pred的序列对中，取大于0的对应点，即排除≤0的对应点；但不应取>0，可以取>0.01，否则若序列中存在大于0但非常接近0的数做分母，可能产生很大的值，不利于得到有效可用的精度值
        judge = (y_true[i] > 0.01) & (y_pred[i] > 0.01)
        if sum(judge):
            y_true_trun.append(y_true[i][judge])
            y_pred_trun.append(y_pred[i][judge])
        else:
            continue

    if (len(y_true_trun) != len(y_pred_trun)) or (len(y_true_trun) < 2):
        raise Exception('y_true_trun与y_pred_trun中序列条数必须相等且≥2')  # 若序列对的数目小于2，则数值变换后的指标均为1

    for i in range(len(y_true_trun)):
        # 第一组，零次的相对性指标：
        MAPE.append(mape(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true != 0; no bias
        SMAPE.append(smape(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true + y_pred != 0; symmetric MAPE, no bias and more general, less susceptible to outliers than MAPE.
        RMSPE.append(eval_measures.rmspe(np.array(y_true_trun[i]), np.array(y_pred_trun[i])) / 10)  # y_true != 0; susceptible to outliers of deviation ratio, if more, RMSPE will be larger than MAPE.
        MTD_p2.append(metrics.mean_tweedie_deviance(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i]), power=2)) # y_pred > 0, y_true > 0; less susceptible to outliers than MAPE when y_pred[i] / y_true[i] > 1, nevertheless, more susceptible to outliers than MAPE when y_pred[i] / y_true[i] < 1
        # 第二组，一次的绝对性指标：
        EMLAE.append(emlae(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件; less susceptible to outliers of error than MAE, so this will penalize small deviation and award large deviation relative to MAE.
        MALE.append(male(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件;
        MAE.append(metrics.mean_absolute_error(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件； this metric has no penalty, no bias
        RMSE.append(eval_measures.rmse(np.array(y_true_trun[i]), np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件；susceptible to outliers of error than MAE, so this will penalize large deviation and award small deviation relative to MAE.
        MedAE.append(metrics.median_absolute_error(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件； if len(y) is slightly large; won't be affected by outliers completely
        MTD_p1.append(metrics.mean_tweedie_deviance(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i]), power=1))  # y_pred > 0, y_true ≥ 0; The higher `p` the less weight is given to extreme deviations between true and predicted targets.
        # 第三组，二次的绝对性指标：
        MSE.append(metrics.mean_squared_error(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件； this metric penalizes a large residual greater than a small residual because of square
        MSLE.append(metrics.mean_squared_log_error(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true≥0, y_pred≥0； this metric penalizes an under-predicted estimate greater than an over-predicted estimate because of logarithm

    # 将各序列对的若干精度指标整合成各序列对的最终单一评价指标；序列对的数目必须≥2，否则归一化后各指标值均为1。
    # 将各精度指标在各自维度内进行数值变换：1.对各指标除以其均值，将任意数量级的指标转化为在1上下波动的数值。
    # 2.再对抹平数量级后的指标取幂函数，进一步缩小指标内数值的差距，保留代表优劣的方向性即可；原始指标内数值差异越大，所开次方根数越大，反之越小，可以避免指标间离群值的出现。
    # 3.再对list作归一化，将所有结果都转化为(0,1)之间的数，越趋近0越好，代表预测列越趋近真实序列；最终精度presion经过有偏向的加权后，也是(0,1)之间的数值。
    MAPE_1 = (MAPE / np.mean(MAPE)) / sum(MAPE / np.mean(MAPE))
    SMAPE_1 = (SMAPE / np.mean(SMAPE)) / sum(SMAPE / np.mean(SMAPE))
    RMSPE_1 = (RMSPE / np.mean(RMSPE)) / sum(RMSPE / np.mean(RMSPE))
    MTD_p2_1 = np.sqrt(MTD_p2 / np.mean(MTD_p2)) / sum(np.sqrt(MTD_p2 / np.mean(MTD_p2)))

    EMLAE_1 = np.sqrt(EMLAE / np.mean(EMLAE)) / sum(np.sqrt(EMLAE / np.mean(EMLAE)))
    MALE_1 = (MALE / np.mean(MALE)) / sum(MALE / np.mean(MALE))
    MAE_1 = np.sqrt(MAE / np.mean(MAE)) / sum(np.sqrt(MAE / np.mean(MAE)))
    RMSE_1 = np.sqrt(RMSE / np.mean(RMSE)) / sum(np.sqrt(RMSE / np.mean(RMSE)))
    MedAE_1 = np.sqrt(MedAE / np.mean(MedAE)) / sum(np.sqrt(MedAE / np.mean(MedAE)))
    MTD_p1_1 = np.sqrt(MTD_p1 / np.mean(MTD_p1)) / sum(np.sqrt(MTD_p1 / np.mean(MTD_p1)))

    MSE_1 = (MSE / np.mean(MSE))**(1/4) / sum((MSE / np.mean(MSE))**(1/4))
    MSLE_1 = np.sqrt(MSLE / np.mean(MSLE)) / sum(np.sqrt(MSLE / np.mean(MSLE)))

    precision = []
    for i in range(len(y_true_trun)):
        # 不用调和平均、几何平均，避免结果向极小值趋近；不用均方根，避免结果向极大值趋近；使用算术平均加权，权重可根据实际需求手动调整。
        precision.append(dyn_seri_weighted([MAPE_1[i], SMAPE_1[i], RMSPE_1[i], MTD_p2_1[i],
                                      EMLAE_1[i], MALE_1[i], MAE_1[i], RMSE_1[i], MedAE_1[i], MTD_p1_1[i],
                                      MSE_1[i], MSLE_1[i]], w=[3,2,2,1,  1,1,1,3,1,1,  1,1]))

    return precision, MAPE, SMAPE, RMSPE, MTD_p2, EMLAE, MALE, MAE, RMSE, MedAE, MTD_p1, MSE, MSLE  # 注意返回的各分量精度指标是未归一化前的数值，而最终precision是由各分量精度指标归一化后的数值算出的


def regression_evaluation(y_true, y_pred):
    """
    :param y_true: 若干条真实序列组成的一个二维list或array或series，其中的每条真实序列必须是带索引的series，为了能对>0的数值的索引取交集；
    并与y_pred中的预测序列按顺序一一对应；y_true是历史上进模型之前的可能经过处理的真实值。
    :param y_pred: 若干条预测序列组成的一个二维list或array或series，其中的每条预测序列必须是带索引的series，为了能对>0的数值的索引取交集；
    并与y_true中的真实序列按顺序一一对应；y_pred是历史上该模型输出的预测值，或者经过补偿的预测值，总之是最终用于订货的预测值。
    :return: 精度指标，按顺序分别是：最终评估指标（越接近0越好），MAPE, SMAPE, RMSPE, MTD_p2, EMLAE, MAE, RMSE, MedAE, MTD_p1, MSE, MSLE, VAR

    测试几种常用的评价序列对的精度指标
    原则：1.带平方项的指标会放大在正负1之外的残差的影响，而压缩在正负1之内的残差的影响，由于各指标越接近零越好，则会惩罚正负1之外的残差，偏离正负1越远，越受到惩罚；而奖励正负1之内的残差。
    2.作对数变换的指标会降低离群值的影响，相对于不带对数项的指标，会惩罚非离群值。因为在(1,+∞)区间内，y=lnx的一阶导数比y=x的一阶导数小，则y=lnx比y=x递增慢。
    3.ln(1/x)+len(x)=0，即对一个数（如x）取对数，与其倒数（1/x）取对数，互为相反数；当x>0，ln(1/x)+x-1≥0，lnx+(1/x)-1≥0，可由求导证明；在(0,4]的区间内，lnx的增长速度快于x**(1/2)，在[4,+∞)区间内，lnx的增长速度慢于x**(1/2)，可由求导证明。
    """

    MAPE, SMAPE, RMSPE, MTD_p2  = [], [], [], []  # 零次的相对性指标
    EMLAE, MALE, MAE, RMSE, MedAE, MTD_p1 = [], [], [], [], [], []  # 一次的绝对性指标
    MSE, MSLE = [], []  # 二次的绝对性指标
    VAR, R2, PR, SR, KT, WT, MGC = [], [], [], [], [], [], []  # 相关性指标

    y_true_trun, y_pred_trun = [], []
    for i in range(len(y_true)):
        # 为了统一下列精度指标的条件，在y_true和y_pred的序列对中，取大于0的对应点，即排除≤0的对应点；但不应取>0，可以取>0.01，否则若序列中存在大于0但非常接近0的数做分母，可能产生很大的值，不利于得到有效可用的精度值
        judge = (y_true[i] > 0.01) & (y_pred[i] > 0.01)
        if sum(judge):
            y_true_trun.append(y_true[i][judge])
            y_pred_trun.append(y_pred[i][judge])
        else:
            continue

    if (len(y_true_trun) != len(y_pred_trun)) or (len(y_true_trun) < 2):
        raise Exception('y_true_trun与y_pred_trun中序列条数必须相等且≥2')  # 若序列对的数目小于2，则数值变换后的指标均为1

    for i in range(len(y_true_trun)):
        # 第四组相关性函数使用如图所示形态的数据作为输入值。此for循环不能与上一个for循环合并，否则会错误调用plt。
        plt.figure('the 4th group of correlation fuctions use following scatters as {0}th inputs'.format(i))
        plt.scatter(y=y_true_trun[i], x=y_pred_trun[i])
        plt.xlabel('y_pred_trun[{0}]'.format(i))
        plt.ylabel('y_true_trun[{0}]'.format(i))

    for i in range(len(y_true_trun)):
        if (len(y_true_trun[i]) < 5) or (len(y_pred_trun[i]) < 5):
            raise Exception('实际使用的序列对y_true_trun[{0}]与y_pred_trun[{1}]中，点数过少不具有统计意义，每条序列至少要≥5个点'.format(i, i))
        # 第一组，零次的相对性指标：
        MAPE.append(mape(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true != 0; no bias
        SMAPE.append(smape(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true + y_pred != 0; symmetric MAPE, no bias and more general, less susceptible to outliers than MAPE.
        RMSPE.append(eval_measures.rmspe(np.array(y_true_trun[i]), np.array(y_pred_trun[i])) / 10)  # y_true != 0; susceptible to outliers of deviation ratio, if more, RMSPE will be larger than MAPE.
        MTD_p2.append(metrics.mean_tweedie_deviance(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i]), power=2)) # y_pred > 0, y_true > 0; less susceptible to outliers than MAPE when y_pred[i] / y_true[i] > 1, nevertheless, more susceptible to outliers than MAPE when y_pred[i] / y_true[i] < 1
        # 第二组，一次的绝对性指标：
        EMLAE.append(emlae(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件; less susceptible to outliers of error than MAE, so this will penalize small deviation and award large deviation relative to MAE.
        MALE.append(male(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件;
        MAE.append(metrics.mean_absolute_error(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件； this metric has no penalty, no bias
        RMSE.append(eval_measures.rmse(np.array(y_true_trun[i]), np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件；susceptible to outliers of error than MAE, so this will penalize large deviation and award small deviation relative to MAE.
        MedAE.append(metrics.median_absolute_error(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件； if len(y) is slightly large; won't be affected by outliers completely
        MTD_p1.append(metrics.mean_tweedie_deviance(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i]), power=1))  # y_pred > 0, y_true ≥ 0; The higher `p` the less weight is given to extreme deviations between true and predicted targets.
        # 第三组，二次的绝对性指标：
        MSE.append(metrics.mean_squared_error(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件； this metric penalizes a large residual greater than a small residual because of square
        MSLE.append(metrics.mean_squared_log_error(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true≥0, y_pred≥0； this metric penalizes an under-predicted estimate greater than an over-predicted estimate because of logarithm
        # 第四组，相关性指标：
        VAR.append(metrics.explained_variance_score(y_true=y_true_trun[i], y_pred=y_pred_trun[i]))  # y_true, y_pred无限制条件；但explained_variance_score为极大化目标函数，值域为(-∞, 1]，越趋近1越好；与其余的极小化目标函数相反，它们的因变量是越小越好。
        R2.append(metrics.r2_score(y_true=y_true_trun[i], y_pred=y_pred_trun[i]))  # y_true, y_pred的series中，至少要有≥2个点，否则会返回nan；r2_score也为极大化目标函数，值域为(-∞, 1]，越趋近1越好；与其余的极小化目标函数相反，它们的因变量是越小越好。
        PR.append(stats.pearsonr(x=y_true_trun[i], y=y_pred_trun[i])[0])
        SR.append(stats.spearmanr(a=y_true_trun[i], b=y_pred_trun[i])[0])
        KT.append(stats.kendalltau(x=y_true_trun[i], y=y_pred_trun[i])[0])
        WT.append(stats.weightedtau(x=y_true_trun[i], y=y_pred_trun[i])[0])
        MGC.append(stats.multiscale_graphcorr(x=np.array(y_true_trun[i]), y=np.array(y_pred_trun[i]))[0])  # hardly affected by abnormal scatters (i.e. outliers); x and y must be ndarrays; MGC requires at least 5 samples to give reasonable results

    # 将各序列对的若干精度指标整合成各序列对的最终单一评价指标；序列对的数目必须≥2，否则归一化后各指标值均为1。
    # 将各精度指标在各自维度内进行数值变换：1.对各指标除以其均值，将任意数量级的指标转化为在1上下波动的数值。
    # 2.再对抹平数量级后的指标取幂函数，进一步缩小指标内数值的差距，保留代表优劣的方向性即可；原始指标内数值差异越大，所开次方根数越大，反之越小，可以避免指标间离群值的出现。
    # 3.再对list作归一化，将所有结果都转化为(0,1)之间的数，越趋近0越好，代表预测列越趋近真实序列；最终精度presion经过有偏向的加权后，也是(0,1)之间的数值。
    MAPE_1 = (MAPE / np.mean(MAPE)) / sum(MAPE / np.mean(MAPE))
    SMAPE_1 = (SMAPE / np.mean(SMAPE)) / sum(SMAPE / np.mean(SMAPE))
    RMSPE_1 = (RMSPE / np.mean(RMSPE)) / sum(RMSPE / np.mean(RMSPE))
    MTD_p2_1 = np.sqrt(MTD_p2 / np.mean(MTD_p2)) / sum(np.sqrt(MTD_p2 / np.mean(MTD_p2)))

    EMLAE_1 = np.sqrt(EMLAE / np.mean(EMLAE)) / sum(np.sqrt(EMLAE / np.mean(EMLAE)))
    MALE_1 = (MALE / np.mean(MALE)) / sum(MALE / np.mean(MALE))
    MAE_1 = np.sqrt(MAE / np.mean(MAE)) / sum(np.sqrt(MAE / np.mean(MAE)))
    RMSE_1 = np.sqrt(RMSE / np.mean(RMSE)) / sum(np.sqrt(RMSE / np.mean(RMSE)))
    MedAE_1 = np.sqrt(MedAE / np.mean(MedAE)) / sum(np.sqrt(MedAE / np.mean(MedAE)))
    MTD_p1_1 = np.sqrt(MTD_p1 / np.mean(MTD_p1)) / sum(np.sqrt(MTD_p1 / np.mean(MTD_p1)))

    MSE_1 = (MSE / np.mean(MSE))**(1/4) / sum((MSE / np.mean(MSE))**(1/4))
    MSLE_1 = np.sqrt(MSLE / np.mean(MSLE)) / sum(np.sqrt(MSLE / np.mean(MSLE)))

    VAR_1 = (-np.array(VAR)+1.01 + np.mean(-np.array(VAR)+1.01)) / sum(-np.array(VAR)+1.01 + np.mean(-np.array(VAR)+1.01))  # 因为VAR的取值范围是(-∞, 1]，越趋近1越好，可看作极大化目标函数，与其他指标相反；所以需要对其做数值变换，使其变为极小化目标函数。
    VAR_1 = (VAR_1 / np.mean(VAR_1))**(1/4) / sum((VAR_1 / np.mean(VAR_1))**(1/4))                                      # 但不能+1，可以加比1多一点点的任何数，如1.01，否则当原始VAR均为1时，sum(-np.array(VAR)+1 + np.mean(-np.array(VAR)+1))就会为0，则VAR_1就会为nan。
    R2_1 = (-np.array(R2)+1.01 + np.mean(-np.array(R2)+1.01)) / sum(-np.array(R2)+1.01 + np.mean(-np.array(R2)+1.01))  # 因为R2的取值范围是(-∞, 1]，越趋近1越好，可看作极大化目标函数，与其他指标相反；所以需要对其做数值变换，使其变为极小化目标函数。
    R2_1 = (R2_1 / np.mean(R2_1))**(1/4) / sum((R2_1 / np.mean(R2_1))**(1/4))                                      # 但不能+1，可以加比1多一点点的任何数，如1.01，否则当原始VAR均为1时，sum(-np.array(R2)+1 + np.mean(-np.array(R2)+1))就会为0，则R2_1就会为nan。
    PR_1 = (-np.array(PR)+1.01 + np.mean(-np.array(PR)+1.01)) / sum(-np.array(PR)+1.01 + np.mean(-np.array(PR)+1.01))  # 因为PR的取值范围是[-1, 1]，越趋近1越好，可看作极大化目标函数，与其他指标相反；所以需要对其做数值变换，使其变为极小化目标函数。
    PR_1 = (PR_1 / np.mean(PR_1))**(1/4) / sum((PR_1 / np.mean(PR_1))**(1/4))                                      # 但不能+1，可以加比1多一点点的任何数，如1.01，否则当原始PR均为1时，sum(-np.array(PR)+1 + np.mean(-np.array(PR)+1))就会为0，则PR_1就会为nan。
    SR_1 = (-np.array(SR)+1.01 + np.mean(-np.array(SR)+1.01)) / sum(-np.array(SR)+1.01 + np.mean(-np.array(SR)+1.01))  # 因为SR的取值范围是[-1, 1]，越趋近1越好，可看作极大化目标函数，与其他指标相反；所以需要对其做数值变换，使其变为极小化目标函数。
    SR_1 = (SR_1 / np.mean(SR_1))**(1/4) / sum((SR_1 / np.mean(SR_1))**(1/4))                                      # 但不能+1，可以加比1多一点点的任何数，如1.01，否则当原始SR均为1时，sum(-np.array(SR)+1 + np.mean(-np.array(SR)+1))就会为0，则SR_1就会为nan。
    KT_1 = (-np.array(KT)+1.01 + np.mean(-np.array(KT)+1.01)) / sum(-np.array(KT)+1.01 + np.mean(-np.array(KT)+1.01))  # 因为KT的取值范围是[-1, 1]，越趋近1越好，可看作极大化目标函数，与其他指标相反；所以需要对其做数值变换，使其变为极小化目标函数。
    KT_1 = (KT_1 / np.mean(KT_1))**(1/4) / sum((KT_1 / np.mean(KT_1))**(1/4))                                      # 但不能+1，可以加比1多一点点的任何数，如1.01，否则当原始KT均为1时，sum(-np.array(KT)+1 + np.mean(-np.array(KT)+1))就会为0，则KT_1就会为nan。
    WT_1 = (-np.array(WT)+1.01 + np.mean(-np.array(WT)+1.01)) / sum(-np.array(WT)+1.01 + np.mean(-np.array(WT)+1.01))  # 因为WT的取值范围是[-1, 1]，越趋近1越好，可看作极大化目标函数，与其他指标相反；所以需要对其做数值变换，使其变为极小化目标函数。
    WT_1 = (WT_1 / np.mean(WT_1))**(1/4) / sum((WT_1 / np.mean(WT_1))**(1/4))                                      # 但不能+1，可以加比1多一点点的任何数，如1.01，否则当原始WT均为1时，sum(-np.array(WT)+1 + np.mean(-np.array(WT)+1))就会为0，则WT_1就会为nan。
    MGC_1 = (-np.array(MGC)+1.01 + np.mean(-np.array(MGC)+1.01)) / sum(-np.array(MGC)+1.01 + np.mean(-np.array(MGC)+1.01))  # 因为MGC的取值范围是[-1, 1]，越趋近1越好，可看作极大化目标函数，与其他指标相反；所以需要对其做数值变换，使其变为极小化目标函数。
    MGC_1 = (MGC_1 / np.mean(MGC_1))**(1/4) / sum((MGC_1 / np.mean(MGC_1))**(1/4))                                      # 但不能+1，可以加比1多一点点的任何数，如1.01，否则当原始MGC均为1时，sum(-np.array(MGC)+1 + np.mean(-np.array(MGC)+1))就会为0，则MGC_1就会为nan。

    precision = []
    for i in range(len(y_true_trun)):
        # 不用调和平均、几何平均，避免结果向极小值趋近；不用均方根，避免结果向极大值趋近；使用算术平均加权，权重可根据实际需求手动调整。
        precision.append(dyn_seri_weighted([MAPE_1[i], SMAPE_1[i], RMSPE_1[i], MTD_p2_1[i],
                                      EMLAE_1[i], MALE_1[i], MAE_1[i], RMSE_1[i], MedAE_1[i], MTD_p1_1[i],
                                      MSE_1[i], MSLE_1[i],
                                      VAR_1[i], R2_1[i], PR_1[i], SR_1[i], KT_1[i], WT_1[i], MGC_1[i]],
                                      w=[3,2,2,1,  1,1,1,3,1,1,  1,1,  1,1,1,1,1,1,1]))

    # 注意返回的各分量指标是未数值变换前的结果，而最终precision是由各分量指标经数值变换后的结果加权算出的
    return precision, MAPE, SMAPE, RMSPE, MTD_p2, EMLAE, MALE, MAE, RMSE, MedAE, MTD_p1, MSE, MSLE, VAR, R2, PR, SR, KT, WT, MGC


def train_check(score):
    """
    :param score: 某一模型每周得分组成的一维数组
    :return: 若为'no-train'，则不重新训练；若为're-train'，则重新训练
    """
    score = np.array(score)
    if len(score) <= 4:
        return 'no-train'
    else:
        if score[-1] > score[-13:-1].mean() + score[-13:-1].std():
            return 're-train'
        else:
            return 'no-train'
