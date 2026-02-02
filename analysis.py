import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import f_oneway
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample


class Analysis:
    def __init__(self, data=None, feature='feature', target='target',
                 lot_col='Carrier ID', wafer_col='Wafer ID',
                 toolchamber_col = 'Tool/Chamber', time_col='Start time',
                 timeperiod_col='time_cut'):
        
        self.data = data
        if data is None:
            data = self.make_toy_dataset()
        self.feature = feature
        self.target = target
        self.lot_col = lot_col
        self.wafer_col = wafer_col
        self.toolchamber_col = toolchamber_col
        self.time_col = time_col
        self.timeperiod_col = timeperiod_col

        self.initial_check()

    def initial_check(self):
        # null value check 필요함 / null value 있을 때, 얼마나 살리고 얼마나 제거할지 y기준으로?
        # constant value 제거 필요함
        self.null_value_check()

    def null_value_figure_check(self, threshold=0.2):
        # Step 1: Identify columns with null values and their ratios
        null_ratio = self.data.isnull().mean()
        null_columns = null_ratio[null_ratio > threshold].sort_values(ascending=False)

        # Plot null ratio distribution for overview
        plt.figure(figsize=(10, 6))
        null_columns.head(30).plot(kind='bar')
        plt.title("Top 30 Columns with Highest Null Ratio")
        plt.ylabel("Null Ratio")
        plt.xlabel("Columns")
        plt.tight_layout()
        plt.show()

    def null_value_check(self, threshold=0.2):
        parameter_list = list(filter(lambda x: 'Step' in x, self.data.columns))
        
        null_ratio = self.data[parameter_list].isnull().mean()
        drop_cols = []
        
        for col in parameter_list:
            if null_ratio <= threshold:
                self.data[col] = self.data[col].fillna(method='ffill').fillna(method='bfill')
            
            elif null_ratio == 0.0:
                continue
            
            else:
                drop_cols.append(col)
        
        self.data = self.data.drop(drop_cols)
                
    def create_lot_col(self, data):
        data['lot_identifier'] = data['Tool/Chamber'] + '/' + data['Carrier ID']
        #data['lot_identifier'] = data[self.lot_col]

        data_lot = data.groupby('lot_identifier').agg({
            self.target : 'mean',
            self.feature : 'mean'
        }).reset_index().rename(columns={self.target: 'lot_y',
                                         self.feature: 'lot_x'})

        return data_lot

    def create_within_lot_col(self, data, data_lot):
        data = data.merge(data_lot, on='lot_identifier')
        data['wlot_x'] = data[self.feature] - data['lot_x']
        data['wlot_y'] = data[self.target] - data['lot_y']

        return data

    def corr(self, data, feature, target):
        df = data[[feature, target]].dropna()

        x = df[feature].values.reshape(-1, 1)
        y = df[target].values

        n = len(df)
        if n > 2:
            coeff, pval = stats.pearsonr(df[feature], df[target])
            coeff, pval = float(coeff), float(pval)

            model = LinearRegression()
            model.fit(x, y)

            gradient = float(model.coef_[0])
            intercept = float(model.intercept_)
            r2 = model.score(x, y)

            result = {'corr': coeff,
                      'pval': pval,
                      'slope': gradient,
                      'intercept': intercept,
                      'r2': r2}
        else:
            result = {'corr': 0,
                      'pval': 1,
                      'slope': 0,
                      'intercept': 0,
                      'r2': 0}

        return result
    
    def calculate_score(self, data, data_lot, corr_wf, corr_lot, corr_wlot):
        case_para = 'None'
        
        pval_lot = corr_lot['pval']e
        slope_lot = corr_lot['slope']
        r2_lot = corr_lot['r2']

        pval_wlot = corr_wlot['pval']
        slope_wlot = corr_wlot['slope']
        r2_wlot = corr_wlot['r2']

        pval_wf = corr_wf['pval']
        slope_wf = corr_wf['slope']
        r2_wf = corr_wf['r2']

        sigma_lot = np.sqrt(data_lot['lot_x'].var())
        sigma_wlot = np.sqrt(data['wlot_x'].var())

        pval_thres = 0.2
        wratio = 2 * sigma_lot / (sigma_lot + 2 * sigma_wlot)
        
        # slope는 값이 존재함
        # pval, corr 값은 nan 값이 있음
        if np.isnan(slope_wlot * slope_lot * pval_lot * pval_wlot):
            print("Not valid for correlation matrix")
            return {
                "parameter": self.feature,
                'score': 0,
                'pval': 1,
                'r2': 0,
                'slope': 0,
                'pval_l': 1,
                'r2_l': 0,
                'slope_l': 0,
                'pval_wl': 1,
                'r2_wl': 0,
                'slope_wl': 0,
                'class': ''
            }

        cond1 = slope_wlot * slope_lot > 0
        cond2 = pval_lot < pval_thres
        cond3 = pval_wlot < pval_thres

        if cond1 and cond2 and cond3:
            apval_lot = min(1, pval_lot / pval_thres)
            apval_wlot = min(1, pval_wlot / pval_thres)
            awratio = min(2 / 3, max(1 / 3, wratio))
            sig_score = 1 - (apval_lot * (1 - awratio) + apval_wlot * awratio)
            
        else:
            sig_score = 0
            apval_lot = min(1, pval_lot / 0.1)
            apval_wlot = min(1, pval_wlot / 0.1)

            if wratio < 0.2:
                sig_score = min(1, (1 - apval_lot) * 2 * r2_lot)
            elif wratio > 0.8:
                sig_score = min(1, (1 - apval_wlot) * 5 * r2_wlot)
            
            if pval_lot < 0.02 and pval_wlot < 0.02:
                sig_score = max(sig_score, 1 - max(pval_lot, pval_wlot))

        res = {
            "parameter": self.feature,
            'score': sig_score,
            'pval': pval_wf,
            'r2': r2_wf,
            'slope': slope_wf,
            'pval_l': pval_lot,
            'r2_l': r2_lot,
            'slope_l': slope_lot,
            'pval_wl': pval_wlot,
            'r2_wl': r2_wlot,
            'slope_wl': slope_wlot,
            'class': ''
        }

        return res
    
    def create_data_norm_format(self, data, norm_col=None, stat='mean'):
        data_norm = data.groupby(norm_col, observed=False).agg({
            self.target: stat,
            self.feature: stat
        }).reset_index().rename(columns={
            self.feature: 'norm_x',
            self.target: 'norm_y'
        })

        data = data.merge(data_norm, on=norm_col)
        data[self.feature] -= data['norm_x']
        data[self.target] -= data['norm_y']
        return data

    def make_analysis_form(self, norm_col=None):
        data = self.data.copy()
        
        if norm_col is not None:
            data = self.create_data_norm_format(data, norm_col)
        
        data_lot = self.create_lot_col(data)
        data = self.create_within_lot_col(data, data_lot)
        return data, data_lot

    def get_score(self, norm_col=None):
        data, data_lot = self.make_analysis_form(norm_col)
        
        corr_wf = self.corr(data, self.feature, self.target)
        corr_lot = self.corr(data_lot, 'lot_x', 'lot_y')
        corr_wl = self.corr(data, 'wlot_x', 'wlot_y')
        
        score = self.calculate_score(data, data_lot, corr_wf, corr_lot, corr_wl)
        return score
    
    def get_statistics_weight(self):
        stats = ['Avg', 'Min', 'Max', 'Stddev', 'Area', 'Step length', 'Median', 'Range']
        stat_weights = [1, 0.9, 0.9, 0.5, 0.7, 1, 0.7, 0.8]
        stat_weight_dict = dict(zip(stats, stat_weights))
        
        return stat_weight_dict
    
    def calculate_adjust_score(self):
        toolchamber_col = self.toolchamber_col
        timeperiod_col = self.timeperiod_col
        norm_list = [toolchamber_col, timeperiod_col]

        stat_weight_dict = self.get_statistics_weight()
        
        baseline_result = self.get_score()

        base_score = baseline_result['score']
        adj_score = base_score
        best_score = base_score

        if base_score > 0.5:
            score_class = 'Strong'
        elif 0 < base_score <= 0.5:
            score_class = 'Marginal'
        else:
            score_class = ''

        note = ''

        for norm_col in norm_list:
            factor_result = self.get_score(norm_col)
            factor_score = factor_result['score']
            adj_score = min(adj_score, factor_score)
            best_score = max(best_score, factor_score)

            if factor_score < 0.5 * base_score:
                score_class = f"{score_class}/Confounded" if score_class else "Confounded"
                note += f"Confounded by {norm_col} "
            elif factor_score > 2 * (0.1 + base_score):
                score_class = f"{score_class}/Disguised" if score_class else "Disguised"
                note += f"Disguised by {norm_col} "

        baseline_result['class'] = score_class
        baseline_result['note'] = note
        baseline_result['adj_score'] = adj_score
        baseline_result['best_score'] = best_score
        baseline_result['base_score'] = base_score

        final_score = best_score
        if final_score > 0:
            final_score = (0.5 + final_score * 0.5) ** 2
            chamber_weight = self.calculate_split_weight(toolchamber_col)
            time_weight = self.calculate_split_weight(timeperiod_col)
            final_score = final_score * chamber_weight * time_weight
        else:
            final_score = 0

        baseline_result['score_unweighted'] = final_score

        try:
            baseline_result['score'] = final_score * stat_weight_dict[self.feature.split('_')[-1]]
        except:
            baseline_result['score'] = final_score

        return baseline_result
    
    def calculate_split_weight(self, factor):
        factor_score = self.get_score(norm_col=factor)
        factor_data, _ = self.make_analysis_form(norm_col=factor)

        slp = factor_score['slope']
        lvs = factor_data[factor].unique()

        num_slp = len(lvs) * 3
        num_cons_slp = 0

        for _, df in factor_data.groupby(factor, observed=False):
            corr_wf = self.corr(df, self.feature, self.target)
            corr_wl = self.corr(df, 'wlot_x', 'wlot_y')
            df_lot = df.drop_duplicates(subset='lot_identifier', keep='first', ignore_index=True, inplace=False)
            corr_l = self.corr(df_lot, 'lot_x', 'lot_y')

            if slp * corr_wf['slope'] > 0 and abs(corr_wf['slope']) > 0.05 * abs(slp):
                num_cons_slp += 1
            if slp * corr_wl['slope'] > 0 and abs(corr_wl['slope']) > 0.05 * abs(slp):
                num_cons_slp += 1
            if slp * corr_l['slope'] > 0 and abs(corr_l['slope']) > 0.05 * abs(slp):
                num_cons_slp += 1

        if num_slp > 0:
            return num_cons_slp / num_slp

        return 0.75
        
    def get_result(self, parameters):
        score_table = []
        
        for feature in parameters:
            self.feature = feature
            score = self.calculate_adjust_score()
            score_table.append(score)
            
        return score_table

    def get_color_map(self, color_by, color_type='gist_rainbow'):
        unique_group = self.data[color_by].unique()
        cmap = plt.colormaps[color_type]
        colors = cmap(np.linspace(0, 1, len(unique_group)))
        color_map = {group: colors[i] for i, group in enumerate(unique_group)}
        return color_map

    def draw_trend_chart(self, para, time_col, color_by, ax):
        color_map = self.get_color_map(color_by)
        for group, group_data in self.data.groupby(color_by):
            ax.plot(group_data[time_col], group_data[para],
                    label=f"{group}", color=color_map[group], marker='o')
        ax.set_ylabel(para)
        ax.legend()
        ax.grid()
        ax.set_xticks([])
        return ax

    def draw_corr_chart(self, data, feature, target, corr, color_by, level, ax):
        color_map = self.get_color_map(color_by)
        for group, group_data in data.groupby(color_by):
            ax.scatter(group_data[feature], group_data[target],
                       color=color_map[group], marker='o')

        rsq = corr['r2']
        pval = corr['pval']
        b = corr['slope']
        a = corr['intercept']

        xseq = np.linspace(data[feature].min(), data[feature].max())
        ax.plot(xseq, a + b * xseq, alpha=0.8, color='black', lw=2, linestyle='--')
        ax.grid()
        ax.set_title(f"{level} level Correlation")
        ax.set_xlabel(feature)
        ax.set_ylabel(target)
        ax.legend([f'p-value: {pval:.3f}\n$R^2$: {rsq:.3f}\nslope: {b:.3f}'])
        return ax
    
    def draw_wafer_level_chart(self, data, color_by, ax):
        corr = self.corr(data, self.feature, self.target)
        return self.draw_corr_chart(data, self.feature, self.target, corr, color_by, 'wafer', ax)

    def draw_lot_level_chart(self, data, color_by, ax):
        data = data.drop_duplicates(subset='lot_identifier', keep='first', ignore_index=True)
        corr = self.corr(data, 'lot_x', 'lot_y')
        return self.draw_corr_chart(data, 'lot_x', 'lot_y', corr, color_by, 'lot', ax)

    def draw_wlot_level_chart(self, data, color_by, ax):
        corr = self.corr(data, 'wlot_x', 'wlot_y')
        return self.draw_corr_chart(data, 'wlot_x', 'wlot_y', corr, color_by, 'within-lot', ax)

    def draw_chart(self, color_by, norm_col=None, trend=False):
        data, data_lot = self.make_analysis_form(norm_col)
        fig_list = []
        if trend:
            fig, ax1 = plt.subplots(1, 2, figsize=(18, 3))
            fig_list.append(fig)
            self.draw_trend_chart(self.target, self.time_col, color_by, ax1[1])
            self.draw_trend_chart(self.feature, self.time_col, color_by, ax1[0])
        fig, ax2 = plt.subplots(1, 3, figsize=(18, 5))
        self.draw_wafer_level_chart(data, color_by, ax=ax2[0])
        self.draw_lot_level_chart(data, color_by, ax=ax2[1])
        self.draw_wlot_level_chart(data, color_by, ax=ax2[2])
        fig_list.append(fig)
        plt.show()
        return fig_list

    def draw_multi_corr_chart(self, data, feature, target, color_by, level, ax):
        color_map = self.get_color_map(color_by)
        for group, group_data in data.groupby(color_by):
            ax.scatter(group_data[feature], group_data[target], s=40, alpha=0.7,
                        color=color_map[group], marker='o')
            corr = self.corr(group_data, feature, target)

            rsq = corr['r2']
            pval = corr['pval']
            b = corr['slope']
            a = corr['intercept']

            xax, yax = data[feature], data[target]
            xseq = np.linspace(xax.min(), xax.max())
            if rsq is None:
                continue
            else:
                ax.plot(xseq, a+b*xseq, alpha=0.8, color=color_map[group], lw=1, linestyle='--')
        ax.grid()
        ax.set_title(f"{level} level Correlation")
        ax.set_xlabel(feature)
        ax.set_ylabel(target)

        return ax

    def draw_multi_chart(self, color_by, trend=False):
        data, _ = self.make_analysis_form()
        if trend:
            fig, ax1 = plt.subplots(1,2, figsize=(18,3))
            self.draw_trend_chart(self.target, self.time_col, color_by, ax1[1])
            self.draw_trend_chart(self.feature, self.time_col, color_by, ax1[0])
        fig, ax2 = plt.subplots(1,3, figsize=(18,5))
        self.draw_multi_corr_chart(data, self.target, self.feature, color_by, 'wafer', ax2[0])
        data_lot = data.drop_duplicates(subset='lot_identifier', keep='first', ignore_index=True)
        self.draw_multi_corr_chart(data_lot, 'lot_x', 'lot_y', color_by, 'lot', ax2[1])
        self.draw_multi_corr_chart(data, 'wlot_x', 'wlot_y', color_by, 'within-lot', ax2[2])

        plt.show()

        return fig    
    

    def draw_boxplot(self, data, numeric_col, category_col, ax, anova=True):
        color_dict = self.get_color_map(category_col)
        f_stat, p_val = None, None
        if anova:
            groups = [group[numeric_col].values for name, group in data.groupby(category_col)]
            f_stat, p_val = f_oneway(*groups)
        bp = ax.boxplot(groups, patch_artist=True)
        for patch, group in zip(bp['boxes'], color_dict.keys()):
            patch.set_facecolor(color_dict[group])
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(color_dict.keys()) + 1))
        ax.set_xticklabels(color_dict.keys(), rotation=90)
        if anova:
            ax.set_title(f"Boxplot of {numeric_col} by {category_col}\\nANOVA: F: {f_stat:.3f}, p-value: {p_val:.3f}")
        else:
            ax.set_title(f"Boxplot of {numeric_col} by {category_col}")
        for cat in color_dict.keys():
            ax.plot([], [], label=f"{cat}", color=color_dict[cat])
        ax.grid()
        return ax

    def draw_pair_boxplot(self, color_by):
        data = self.data
        fig, ax = plt.subplots(1, 2, figsize=(18, 3))
        self.draw_boxplot(data, self.feature, color_by, ax[0])
        self.draw_boxplot(data, self.target, color_by, ax[1])
        plt.show()
        return fig
    
    def make_toy_dataset(self):

        dataset_size = 1

        # Generate toy example dataset
        tool_chambers = [
                            "EP00018/PM-1", "EP00018/PM-1", "EP00018/PM-2",
                            "EP00019/PM-3", "EP00019/PM-3", "EP00019/PM-4",
                            "EP00020/PM-5", "EP00020/PM-5", "EP00020/PM-5"
                        ] * dataset_size

        # Generate random features and targets
        # features = [np.random.normal(5.0, 1, 1) for _ in range(len(tool_chambers))][0]
        # targets = [np.random.normal(0.1, 1.0, 1) for _ in range(len(tool_chambers))][0]
        features = np.random.normal(0, 1, len(tool_chambers))
        targets = np.random.normal(0.1, 1, len(tool_chambers))

        # features = np.arange(9)
        # targets = np.arange(10, 19)

        # Generate Carrier IDs and Wafer IDs
        # carrier_ids = [f"PFP0073" for _ in range(len(tool_chambers))]
        carrier_ids = ['PFP0073', 'PFP0073', 'PFP0074',
                        'PFP0075', 'PFP0075', 'PFP0076',
                        'PFP0077', 'PFP0077', 'PFP0077'] * dataset_size
        wafer_ids = [f"{carrier_ids[i]}.{i + 1}" for i in range(len(carrier_ids)) for j in range(1)]

        # Create the DataFrame
        data = {
            "Tool/Chamber": tool_chambers,
            "feature": features,
            "target": targets,
            "Carrier ID": carrier_ids,
            "Wafer ID": wafer_ids
        }

        df = pd.DataFrame(data)
        df['Start time'] = pd.date_range("2024-01-01", periods=df.shape[0], freq="0.1s")
        return df