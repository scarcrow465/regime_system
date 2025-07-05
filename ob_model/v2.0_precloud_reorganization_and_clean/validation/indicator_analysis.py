#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Indicator analysis and validation module
Analyzes indicator correlations, contributions, and redundancies
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple, Optional, Any
import os
import sys
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    INDICATOR_WEIGHTS, REGIME_DIMENSIONS, RESULTS_DIR,
    FIGURE_SIZE, FIGURE_DPI
)
from utils.logger import get_logger

logger = get_logger(__name__)

# =============================================================================
# INDICATOR CORRELATION ANALYSIS
# =============================================================================

class IndicatorCorrelationAnalyzer:
    """Analyze correlations between indicators"""
    
    def __init__(self, correlation_threshold: float = 0.90):
        """
        Initialize analyzer
        
        Args:
            correlation_threshold: Threshold for identifying redundant indicators
        """
        self.correlation_threshold = correlation_threshold
        self.correlation_matrix = None
        self.redundant_pairs = []
        
    def analyze_correlations(self, data: pd.DataFrame, 
                           indicators: Optional[List[str]] = None,
                           method: str = 'pearson') -> pd.DataFrame:
        """
        Calculate correlation matrix for indicators
        
        Args:
            data: DataFrame with indicators
            indicators: List of indicator columns to analyze
            method: Correlation method ('pearson' or 'spearman')
            
        Returns:
            Correlation matrix
        """
        if indicators is None:
            # Get all indicator columns (exclude basic price/volume)
            exclude = ['open', 'high', 'low', 'close', 'volume', 'returns', 
                      'log_returns', 'date', 'timestamp']
            indicators = [col for col in data.columns if col not in exclude]
        
        # Filter to existing columns
        indicators = [col for col in indicators if col in data.columns]
        
        logger.info(f"Analyzing correlations for {len(indicators)} indicators using {method} method")
        
        # Calculate correlation matrix
        if method == 'pearson':
            self.correlation_matrix = data[indicators].corr(method='pearson')
        elif method == 'spearman':
            self.correlation_matrix = data[indicators].corr(method='spearman')
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        # Find redundant pairs
        self._find_redundant_pairs()
        
        return self.correlation_matrix
    
    def _find_redundant_pairs(self):
        """Find highly correlated indicator pairs"""
        self.redundant_pairs = []
        
        if self.correlation_matrix is None:
            return
        
        # Get upper triangle of correlation matrix
        upper_triangle = np.triu(self.correlation_matrix.values, k=1)
        
        # Find pairs with high correlation
        for i in range(len(upper_triangle)):
            for j in range(i+1, len(upper_triangle)):
                corr_value = upper_triangle[i, j]
                
                if abs(corr_value) >= self.correlation_threshold:
                    self.redundant_pairs.append({
                        'indicator1': self.correlation_matrix.index[i],
                        'indicator2': self.correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        logger.info(f"Found {len(self.redundant_pairs)} redundant indicator pairs")
    
    def get_redundancy_report(self) -> Dict[str, Any]:
        """Generate redundancy analysis report"""
        if not self.redundant_pairs:
            return {'redundant_count': 0, 'pairs': []}
        
        # Sort by absolute correlation
        sorted_pairs = sorted(self.redundant_pairs, 
                            key=lambda x: abs(x['correlation']), 
                            reverse=True)
        
        # Group by indicator
        indicator_redundancy = {}
        for pair in sorted_pairs:
            for indicator in [pair['indicator1'], pair['indicator2']]:
                if indicator not in indicator_redundancy:
                    indicator_redundancy[indicator] = []
                other = pair['indicator2'] if indicator == pair['indicator1'] else pair['indicator1']
                indicator_redundancy[indicator].append(other)
        
        return {
            'redundant_count': len(self.redundant_pairs),
            'pairs': sorted_pairs[:10],  # Top 10 most correlated
            'indicators_with_redundancy': indicator_redundancy,
            'recommendation': self._generate_redundancy_recommendations()
        }
    
    def _generate_redundancy_recommendations(self) -> List[str]:
        """Generate recommendations for handling redundant indicators"""
        recommendations = []
        
        # Count redundancies per indicator
        redundancy_count = {}
        for pair in self.redundant_pairs:
            for indicator in [pair['indicator1'], pair['indicator2']]:
                redundancy_count[indicator] = redundancy_count.get(indicator, 0) + 1
        
        # Sort by redundancy count
        most_redundant = sorted(redundancy_count.items(), 
                               key=lambda x: x[1], 
                               reverse=True)[:5]
        
        for indicator, count in most_redundant:
            recommendations.append(
                f"Consider removing {indicator} - correlated with {count} other indicators"
            )
        
        return recommendations
    
    def plot_correlation_heatmap(self, figsize: Tuple[int, int] = FIGURE_SIZE,
                                save_path: Optional[str] = None):
        """Plot correlation heatmap"""
        if self.correlation_matrix is None:
            logger.warning("No correlation matrix to plot")
            return
        
        plt.figure(figsize=figsize)
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(self.correlation_matrix), k=1)
        
        # Plot heatmap
        sns.heatmap(self.correlation_matrix, 
                   mask=mask,
                   cmap='coolwarm',
                   center=0,
                   vmin=-1,
                   vmax=1,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": 0.8})
        
        plt.title('Indicator Correlation Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
            logger.info(f"Correlation heatmap saved to: {save_path}")
        
        plt.close()

# =============================================================================
# INDICATOR IMPORTANCE ANALYSIS
# =============================================================================

class IndicatorImportanceAnalyzer:
    """Analyze indicator importance and contribution"""
    
    def __init__(self):
        self.importance_scores = {}
        self.pca_results = None
        
    def analyze_regime_contribution(self, data: pd.DataFrame, 
                                  regimes: pd.DataFrame,
                                  dimension: str = 'Direction') -> Dict[str, float]:
        """
        Analyze how indicators contribute to regime classification
        
        Args:
            data: DataFrame with indicators
            regimes: DataFrame with regime classifications
            dimension: Regime dimension to analyze
            
        Returns:
            Dictionary of indicator importance scores
        """
        logger.info(f"Analyzing indicator contribution to {dimension} regime")
        
        # Get regime column
        regime_col = f'{dimension}_Regime'
        if regime_col not in regimes.columns:
            logger.error(f"Regime column {regime_col} not found")
            return {}
        
        # Convert regime to numeric
        regime_numeric = pd.Categorical(regimes[regime_col]).codes
        
        # Get indicators for this dimension
        dim_indicators = self._get_dimension_indicators(dimension)
        available_indicators = [ind for ind in dim_indicators if ind in data.columns]
        
        if not available_indicators:
            logger.warning(f"No indicators found for dimension {dimension}")
            return {}
        
        # Calculate mutual information
        mi_scores = mutual_info_regression(
            data[available_indicators].fillna(0),
            regime_numeric
        )
        
        # Normalize scores
        if mi_scores.max() > 0:
            mi_scores = mi_scores / mi_scores.max()
        
        # Create importance dictionary
        importance = dict(zip(available_indicators, mi_scores))
        
        # Sort by importance
        importance = dict(sorted(importance.items(), 
                               key=lambda x: x[1], 
                               reverse=True))
        
        self.importance_scores[dimension] = importance
        
        return importance
    
    def _get_dimension_indicators(self, dimension: str) -> List[str]:
        """Get indicators for a specific dimension"""
        dimension_map = {
            'Direction': ['SMA_Signal', 'EMA_Signal', 'MACD_Signal', 'ADX', 
                         'Aroon_Oscillator', 'CCI', 'PSAR', 'Vortex_Pos', 
                         'SuperTrend_Direction', 'DPO', 'KST'],
            'TrendStrength': ['ADX', 'Aroon_Up', 'CCI', 'MACD_Histogram', 
                            'RSI', 'TSI', 'LinearReg_Slope', 'Correlation'],
            'Velocity': ['ROC', 'RSI', 'TSI', 'MACD_Histogram', 
                        'Acceleration', 'Jerk'],
            'Volatility': ['ATR', 'BB_Width', 'KC_Width', 'DC_Width', 'NATR', 
                         'UI', 'Historical_Vol', 'Parkinson', 'GarmanKlass', 
                         'RogersSatchell', 'YangZhang'],
            'Microstructure': ['Volume', 'OBV', 'CMF', 'MFI', 'ADI', 'EOM', 
                             'FI', 'VPT', 'VWAP', 'CVD', 'Delta']
        }
        
        return dimension_map.get(dimension, [])
    
    def run_pca_analysis(self, data: pd.DataFrame, 
                        n_components: int = 10) -> Dict[str, Any]:
        """
        Run PCA analysis to understand indicator relationships
        
        Args:
            data: DataFrame with indicators
            n_components: Number of PCA components
            
        Returns:
            PCA analysis results
        """
        # Get indicator columns
        exclude = ['open', 'high', 'low', 'close', 'volume', 'returns', 
                  'log_returns', 'date', 'timestamp']
        indicators = [col for col in data.columns if col not in exclude]
        
        # Prepare data
        X = data[indicators].fillna(0)
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Run PCA
        pca = PCA(n_components=min(n_components, len(indicators)))
        X_pca = pca.fit_transform(X_scaled)
        
        # Store results
        self.pca_results = {
            'explained_variance': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'components': pd.DataFrame(
                pca.components_,
                columns=indicators,
                index=[f'PC{i+1}' for i in range(pca.n_components_)]
            ),
            'loadings': self._calculate_loadings(pca, indicators)
        }
        
        logger.info(f"PCA complete: {self.pca_results['cumulative_variance'][4]:.1%} "
                   f"variance explained by first 5 components")
        
        return self.pca_results
    
    def _calculate_loadings(self, pca, feature_names: List[str]) -> pd.DataFrame:
        """Calculate PCA loadings"""
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        
        return pd.DataFrame(
            loadings,
            index=feature_names,
            columns=[f'PC{i+1}' for i in range(pca.n_components_)]
        )
    
    def get_importance_summary(self) -> Dict[str, Any]:
        """Get summary of indicator importance across dimensions"""
        if not self.importance_scores:
            return {}
        
        # Aggregate importance across dimensions
        all_indicators = {}
        
        for dimension, scores in self.importance_scores.items():
            for indicator, score in scores.items():
                if indicator not in all_indicators:
                    all_indicators[indicator] = []
                all_indicators[indicator].append(score)
        
        # Calculate average importance
        avg_importance = {
            ind: np.mean(scores) for ind, scores in all_indicators.items()
        }
        
        # Sort by importance
        sorted_importance = dict(sorted(avg_importance.items(), 
                                      key=lambda x: x[1], 
                                      reverse=True))
        
        return {
            'top_indicators': list(sorted_importance.keys())[:10],
            'importance_scores': sorted_importance,
            'dimension_specific': self.importance_scores,
            'recommendations': self._generate_importance_recommendations(sorted_importance)
        }
    
    def _generate_importance_recommendations(self, 
                                           importance_scores: Dict[str, float]) -> List[str]:
        """Generate recommendations based on importance analysis"""
        recommendations = []
        
        # Find low importance indicators
        low_importance = [ind for ind, score in importance_scores.items() 
                         if score < 0.1]
        
        if low_importance:
            recommendations.append(
                f"Consider removing {len(low_importance)} low-importance indicators: "
                f"{', '.join(low_importance[:5])}"
            )
        
        # Check dimension balance
        for dimension in REGIME_DIMENSIONS:
            if dimension in self.importance_scores:
                dim_scores = self.importance_scores[dimension]
                if len(dim_scores) < 3:
                    recommendations.append(
                        f"Warning: {dimension} dimension has only {len(dim_scores)} "
                        f"active indicators"
                    )
        
        return recommendations

# =============================================================================
# INDICATOR VALIDATION
# =============================================================================

class IndicatorValidator:
    """Validate indicator calculations and data quality"""
    
    def __init__(self):
        self.validation_results = {}
        
    def validate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run comprehensive indicator validation
        
        Args:
            data: DataFrame with indicators
            
        Returns:
            Validation results
        """
        logger.info("Running indicator validation")
        
        results = {
            'missing_data': self._check_missing_data(data),
            'data_quality': self._check_data_quality(data),
            'indicator_ranges': self._check_indicator_ranges(data),
            'calculation_consistency': self._check_calculation_consistency(data),
            'overall_score': 0.0
        }
        
        # Calculate overall score
        scores = []
        
        # Missing data score (lower is better)
        missing_pct = results['missing_data']['overall_missing_pct']
        scores.append(max(0, 1 - missing_pct / 50))  # 50% missing = 0 score
        
        # Quality issues score
        quality_score = results['data_quality']['quality_score']
        scores.append(quality_score)
        
        # Range violations score
        range_issues = len(results['indicator_ranges']['out_of_range'])
        scores.append(max(0, 1 - range_issues / 10))  # 10 issues = 0 score
        
        results['overall_score'] = np.mean(scores)
        
        self.validation_results = results
        
        return results
    
    def _check_missing_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check for missing data in indicators"""
        missing_counts = data.isnull().sum()
        missing_pct = (missing_counts / len(data)) * 100
        
        # Find problematic indicators
        problematic = missing_pct[missing_pct > 10].to_dict()
        
        return {
            'total_missing': missing_counts.sum(),
            'overall_missing_pct': missing_pct.mean(),
            'problematic_indicators': problematic,
            'recommendation': "Consider imputation for indicators with >10% missing"
                            if problematic else "Missing data levels acceptable"
        }
    
    def _check_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check data quality issues"""
        issues = []
        
        # Check for infinite values
        inf_columns = []
        for col in data.columns:
            if data[col].dtype in ['float64', 'int64']:
                if np.isinf(data[col]).any():
                    inf_columns.append(col)
                    issues.append(f"{col} contains infinite values")
        
        # Check for constant columns
        constant_columns = []
        for col in data.columns:
            if data[col].nunique() == 1:
                constant_columns.append(col)
                issues.append(f"{col} is constant")
        
        # Check for duplicate columns
        duplicate_columns = []
        for i, col1 in enumerate(data.columns):
            for col2 in data.columns[i+1:]:
                if data[col1].equals(data[col2]):
                    duplicate_columns.append((col1, col2))
                    issues.append(f"{col1} and {col2} are identical")
        
        quality_score = max(0, 1 - len(issues) / 20)  # 20 issues = 0 score
        
        return {
            'issues': issues[:10],  # Top 10 issues
            'infinite_values': inf_columns,
            'constant_columns': constant_columns,
            'duplicate_columns': duplicate_columns,
            'quality_score': quality_score
        }
    
    def _check_indicator_ranges(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check if indicators are within expected ranges"""
        out_of_range = []
        
        # Define expected ranges for common indicators
        expected_ranges = {
            'RSI': (0, 100),
            'MFI': (0, 100),
            'Stoch_K': (0, 100),
            'Stoch_D': (0, 100),
            'BB_Percent': (0, 1),
            'ADX': (0, 100),
            'Aroon_Up': (0, 100),
            'Aroon_Down': (0, 100),
            'CCI': (-200, 200),  # Typical range
            'CMF': (-1, 1)
        }
        
        for indicator, (min_val, max_val) in expected_ranges.items():
            if indicator in data.columns:
                actual_min = data[indicator].min()
                actual_max = data[indicator].max()
                
                if actual_min < min_val or actual_max > max_val:
                    out_of_range.append({
                        'indicator': indicator,
                        'expected_range': (min_val, max_val),
                        'actual_range': (actual_min, actual_max)
                    })
        
        return {
            'out_of_range': out_of_range,
            'recommendation': f"Review {len(out_of_range)} indicators with "
                            f"unexpected ranges" if out_of_range else 
                            "All indicators within expected ranges"
        }
    
    def _check_calculation_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check calculation consistency between related indicators"""
        inconsistencies = []
        
        # Check MACD components
        if all(col in data.columns for col in ['MACD', 'MACD_Signal_Line']):
            # MACD signal should be smoother than MACD
            macd_std = data['MACD'].std()
            signal_std = data['MACD_Signal_Line'].std()
            
            if signal_std > macd_std:
                inconsistencies.append(
                    "MACD Signal Line more volatile than MACD line"
                )
        
        # Check Bollinger Bands
        if all(col in data.columns for col in ['BB_Upper', 'BB_Lower', 'close']):
            # Close should occasionally touch bands
            touches_upper = (data['close'] >= data['BB_Upper']).sum()
            touches_lower = (data['close'] <= data['BB_Lower']).sum()
            
            touch_pct = (touches_upper + touches_lower) / len(data) * 100
            
            if touch_pct < 1:
                inconsistencies.append(
                    f"Price rarely touches Bollinger Bands ({touch_pct:.1f}%)"
                )
        
        return {
            'inconsistencies': inconsistencies,
            'recommendation': "Review indicator calculations" if inconsistencies
                            else "Indicator calculations appear consistent"
        }

# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def run_indicator_analysis(data: pd.DataFrame, 
                         regimes: pd.DataFrame,
                         save_report: bool = True) -> Dict[str, Any]:
    """
    Run complete indicator analysis
    
    Args:
        data: DataFrame with indicators
        regimes: DataFrame with regime classifications
        save_report: Whether to save analysis report
        
    Returns:
        Complete analysis results
    """
    logger.info("Starting comprehensive indicator analysis")
    
    # Initialize analyzers
    correlation_analyzer = IndicatorCorrelationAnalyzer()
    importance_analyzer = IndicatorImportanceAnalyzer()
    validator = IndicatorValidator()
    
    # Run analyses
    results = {
        'correlation_analysis': {},
        'importance_analysis': {},
        'validation': {},
        'recommendations': []
    }
    
    # 1. Correlation analysis
    correlation_matrix = correlation_analyzer.analyze_correlations(data)
    results['correlation_analysis'] = correlation_analyzer.get_redundancy_report()
    
    # 2. Importance analysis
    for dimension in REGIME_DIMENSIONS:
        importance_analyzer.analyze_regime_contribution(data, regimes, dimension)
    
    results['importance_analysis'] = importance_analyzer.get_importance_summary()
    
    # 3. PCA analysis
    pca_results = importance_analyzer.run_pca_analysis(data)
    results['pca_analysis'] = {
        'variance_explained_5pc': pca_results['cumulative_variance'][4],
        'variance_explained_10pc': pca_results['cumulative_variance'][9] 
                                   if len(pca_results['cumulative_variance']) > 9 else None
    }
    
    # 4. Validation
    results['validation'] = validator.validate_indicators(data)
    
    # 5. Generate overall recommendations
    all_recommendations = []
    
    # From correlation analysis
    all_recommendations.extend(
        results['correlation_analysis'].get('recommendation', [])
    )
    
    # From importance analysis
    all_recommendations.extend(
        results['importance_analysis'].get('recommendations', [])
    )
    
    # From validation
    all_recommendations.append(
        results['validation']['missing_data']['recommendation']
    )
    
    results['recommendations'] = all_recommendations
    
    # 6. Save report if requested
    if save_report:
        save_indicator_analysis_report(results, correlation_analyzer, importance_analyzer)
    
    logger.info("Indicator analysis complete")
    
    return results

def save_indicator_analysis_report(results: Dict[str, Any],
                                 correlation_analyzer: IndicatorCorrelationAnalyzer,
                                 importance_analyzer: IndicatorImportanceAnalyzer):
    """Save indicator analysis report"""
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save plots
    plot_dir = os.path.join(RESULTS_DIR, 'indicator_analysis')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Correlation heatmap
    correlation_analyzer.plot_correlation_heatmap(
        save_path=os.path.join(plot_dir, f'correlation_heatmap_{timestamp}.png')
    )
    
    # Save text report
    report_path = os.path.join(plot_dir, f'indicator_analysis_report_{timestamp}.txt')
    
    with open(report_path, 'w') as f:
        f.write("INDICATOR ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. CORRELATION ANALYSIS\n")
        f.write("-"*40 + "\n")
        f.write(f"Redundant pairs found: {results['correlation_analysis']['redundant_count']}\n")
        
        if results['correlation_analysis']['pairs']:
            f.write("\nTop correlated pairs:\n")
            for pair in results['correlation_analysis']['pairs'][:5]:
                f.write(f"  {pair['indicator1']} <-> {pair['indicator2']}: "
                       f"{pair['correlation']:.3f}\n")
        
        f.write("\n2. IMPORTANCE ANALYSIS\n")
        f.write("-"*40 + "\n")
        f.write("Top 10 most important indicators:\n")
        for i, indicator in enumerate(results['importance_analysis']['top_indicators'], 1):
            score = results['importance_analysis']['importance_scores'][indicator]
            f.write(f"  {i}. {indicator}: {score:.3f}\n")
        
        f.write("\n3. VALIDATION RESULTS\n")
        f.write("-"*40 + "\n")
        f.write(f"Overall validation score: {results['validation']['overall_score']:.2f}/1.0\n")
        f.write(f"Missing data: {results['validation']['missing_data']['overall_missing_pct']:.1f}%\n")
        f.write(f"Quality issues: {len(results['validation']['data_quality']['issues'])}\n")
        
        f.write("\n4. RECOMMENDATIONS\n")
        f.write("-"*40 + "\n")
        for i, rec in enumerate(results['recommendations'], 1):
            f.write(f"{i}. {rec}\n")
    
    logger.info(f"Indicator analysis report saved to: {report_path}")

