#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Performance attribution module
Analyzes which regimes and factors contribute to strategy performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import os
import sys
from scipy import stats

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    REGIME_DIMENSIONS, RESULTS_DIR, 
    FIGURE_SIZE, FIGURE_DPI, PLOT_STYLE
)
from utils.logger import get_logger

logger = get_logger(__name__)

# Set plot style
plt.style.use('default')

# =============================================================================
# PERFORMANCE ATTRIBUTION ANALYZER
# =============================================================================

class PerformanceAttributionAnalyzer:
    """Analyze performance attribution across regimes and factors"""
    
    def __init__(self):
        self.attribution_results = {}
        self.regime_performance = {}
        
    def analyze_regime_performance(self, 
                                 returns: pd.Series,
                                 regimes: pd.DataFrame,
                                 positions: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Analyze performance by regime
        
        Args:
            returns: Strategy returns
            regimes: Regime classifications
            positions: Optional position sizes
            
        Returns:
            Performance attribution by regime
        """
        logger.info("Analyzing performance attribution by regime")
        
        results = {}
        
        for dimension in REGIME_DIMENSIONS:
            regime_col = f'{dimension}_Regime'
            
            if regime_col not in regimes.columns:
                continue
            
            # Calculate performance by regime
            regime_perf = self._calculate_regime_performance(
                returns, regimes[regime_col], positions
            )
            
            # Calculate contribution to total return
            contributions = self._calculate_regime_contributions(
                returns, regimes[regime_col]
            )
            
            # Analyze risk by regime
            risk_metrics = self._calculate_regime_risk(
                returns, regimes[regime_col]
            )
            
            results[dimension] = {
                'performance': regime_perf,
                'contributions': contributions,
                'risk_metrics': risk_metrics,
                'best_regime': max(regime_perf.items(), key=lambda x: x[1]['sharpe_ratio'])[0]
                              if regime_perf else None,
                'worst_regime': min(regime_perf.items(), key=lambda x: x[1]['sharpe_ratio'])[0]
                               if regime_perf else None
            }
            
            self.regime_performance[dimension] = results[dimension]
        
        # Overall attribution summary
        results['overall'] = self._calculate_overall_attribution(results, returns)
        
        return results
    
    def _calculate_regime_performance(self, 
                                    returns: pd.Series,
                                    regime_series: pd.Series,
                                    positions: Optional[pd.Series]) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics for each regime"""
        regime_metrics = {}
        
        for regime in regime_series.unique():
            if regime == 'Undefined':
                continue
            
            # Get returns for this regime
            regime_mask = regime_series == regime
            regime_returns = returns[regime_mask]
            
            if len(regime_returns) == 0:
                continue
            
            # Calculate metrics
            metrics = {
                'total_return': (1 + regime_returns).prod() - 1,
                'avg_return': regime_returns.mean(),
                'volatility': regime_returns.std(),
                'sharpe_ratio': regime_returns.mean() / regime_returns.std() * np.sqrt(252 * 26)
                               if regime_returns.std() > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(regime_returns),
                'win_rate': (regime_returns > 0).sum() / len(regime_returns)
                           if len(regime_returns) > 0 else 0,
                'num_periods': len(regime_returns),
                'pct_time': len(regime_returns) / len(returns) * 100
            }
            
            # Add position analysis if available
            if positions is not None:
                regime_positions = positions[regime_mask]
                metrics['avg_position'] = regime_positions.mean()
                metrics['position_utilization'] = (regime_positions != 0).sum() / len(regime_positions) if len(regime_positions) > 0 else 0
            
            regime_metrics[regime] = metrics
        
        return regime_metrics
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_regime_contributions(self,
                                      returns: pd.Series,
                                      regime_series: pd.Series) -> Dict[str, float]:
        """Calculate contribution of each regime to total return"""
        total_return = (1 + returns).prod() - 1
        
        contributions = {}
        
        for regime in regime_series.unique():
            if regime == 'Undefined':
                continue
            
            regime_mask = regime_series == regime
            regime_returns = returns[regime_mask]
            
            # Calculate cumulative return contribution
            regime_cum_return = (1 + regime_returns).prod() - 1
            
            # Calculate percentage contribution
            if total_return != 0:
                contribution_pct = (regime_cum_return / total_return) * 100
            else:
                contribution_pct = 0
            
            contributions[regime] = {
                'return_contribution': regime_cum_return,
                'contribution_pct': contribution_pct,
                'periods': len(regime_returns)
            }
        
        return contributions
    
    def _calculate_regime_risk(self,
                             returns: pd.Series,
                             regime_series: pd.Series) -> Dict[str, Dict[str, float]]:
        """Calculate risk metrics by regime"""
        risk_metrics = {}
        
        for regime in regime_series.unique():
            if regime == 'Undefined':
                continue
            
            regime_mask = regime_series == regime
            regime_returns = returns[regime_mask]
            
            if len(regime_returns) < 2:
                continue
            
            # Calculate risk metrics
            metrics = {
                'volatility': regime_returns.std() * np.sqrt(252 * 26),
                'downside_volatility': regime_returns[regime_returns < 0].std() * np.sqrt(252 * 26)
                                      if len(regime_returns[regime_returns < 0]) > 0 else 0,
                'var_95': np.percentile(regime_returns, 5),
                'cvar_95': regime_returns[regime_returns <= np.percentile(regime_returns, 5)].mean()
                          if len(regime_returns[regime_returns <= np.percentile(regime_returns, 5)]) > 0 else 0,
                'skewness': stats.skew(regime_returns),
                'kurtosis': stats.kurtosis(regime_returns)
            }
            
            risk_metrics[regime] = metrics
        
        return risk_metrics
    
    def _calculate_overall_attribution(self, 
                                     dimension_results: Dict[str, Any],
                                     returns: pd.Series) -> Dict[str, Any]:
        """Calculate overall attribution summary"""
        # Find most impactful dimension
        dimension_impacts = {}
        
        for dimension, results in dimension_results.items():
            if dimension == 'overall' or 'performance' not in results:
                continue
            
            # Calculate dispersion of performance across regimes
            performances = [r['sharpe_ratio'] for r in results['performance'].values()]
            if performances:
                dispersion = np.std(performances)
                dimension_impacts[dimension] = dispersion
        
        most_impactful = max(dimension_impacts.items(), key=lambda x: x[1])[0] if dimension_impacts else None
        
        # Calculate overall metrics
        total_return = (1 + returns).prod() - 1
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 26) if returns.std() > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'most_impactful_dimension': most_impactful,
            'dimension_impacts': dimension_impacts,
            'recommendation': self._generate_attribution_recommendations(dimension_results)
        }
    
    def _generate_attribution_recommendations(self, 
                                            dimension_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on attribution analysis"""
        recommendations = []
        
        # Find dimensions with poor performing regimes
        for dimension, results in dimension_results.items():
            if dimension == 'overall' or 'performance' not in results:
                continue
            
            # Check for consistently poor regimes
            poor_regimes = [regime for regime, metrics in results['performance'].items()
                           if metrics['sharpe_ratio'] < -0.5]
            
            if poor_regimes:
                recommendations.append(
                    f"Consider avoiding or reducing exposure during "
                    f"{dimension} regimes: {', '.join(poor_regimes)}"
                )
            
            # Check for unutilized regimes
            low_utilization = [regime for regime, metrics in results['performance'].items()
                              if metrics.get('position_utilization', 1) < 0.2]
            
            if low_utilization:
                recommendations.append(
                    f"Low position utilization in {dimension} regimes: "
                    f"{', '.join(low_utilization)}. Consider improving entry signals."
                )
        
        return recommendations

# =============================================================================
# FACTOR ATTRIBUTION
# =============================================================================

class FactorAttributionAnalyzer:
    """Analyze performance attribution by factors"""
    
    def analyze_factor_attribution(self,
                                 returns: pd.Series,
                                 data: pd.DataFrame,
                                 regimes: pd.DataFrame) -> Dict[str, Any]:
        """
        Decompose returns by various factors
        
        Args:
            returns: Strategy returns
            data: DataFrame with indicators
            regimes: Regime classifications
            
        Returns:
            Factor attribution analysis
        """
        logger.info("Analyzing factor attribution")
        
        results = {
            'timing': self._analyze_timing_attribution(returns, regimes),
            'selection': self._analyze_selection_attribution(returns, regimes),
            'volatility': self._analyze_volatility_attribution(returns, data),
            'indicator': self._analyze_indicator_attribution(returns, data, regimes)
        }
        
        # Calculate total attribution
        results['total'] = self._calculate_total_attribution(results)
        
        return results
    
    def _analyze_timing_attribution(self,
                                  returns: pd.Series,
                                  regimes: pd.DataFrame) -> Dict[str, float]:
        """Analyze timing contribution to returns"""
        # Calculate regime timing effectiveness
        timing_scores = {}
        
        for dimension in REGIME_DIMENSIONS:
            regime_col = f'{dimension}_Regime'
            if regime_col not in regimes.columns:
                continue
            
            # Calculate how well we time regime transitions
            regime_changes = regimes[regime_col] != regimes[regime_col].shift()
            
            # Returns around regime changes
            pre_change_returns = returns.shift(1)[regime_changes].mean()
            post_change_returns = returns[regime_changes].mean()
            
            timing_scores[dimension] = {
                'pre_transition_return': pre_change_returns,
                'post_transition_return': post_change_returns,
                'timing_effectiveness': post_change_returns - pre_change_returns
            }
        
        return timing_scores
    
    def _analyze_selection_attribution(self,
                                     returns: pd.Series,
                                     regimes: pd.DataFrame) -> Dict[str, float]:
        """Analyze selection contribution (regime identification)"""
        selection_scores = {}
        
        # Calculate returns in correctly identified vs misidentified regimes
        # This would require known "true" regimes, so we use confidence as proxy
        
        for dimension in REGIME_DIMENSIONS:
            confidence_col = f'{dimension}_Confidence'
            if confidence_col not in regimes.columns:
                continue
            
            # High vs low confidence returns
            high_conf_mask = regimes[confidence_col] > regimes[confidence_col].median()
            
            high_conf_returns = returns[high_conf_mask].mean()
            low_conf_returns = returns[~high_conf_mask].mean()
            
            selection_scores[dimension] = {
                'high_confidence_return': high_conf_returns,
                'low_confidence_return': low_conf_returns,
                'selection_alpha': high_conf_returns - low_conf_returns
            }
        
        return selection_scores
    
    def _analyze_volatility_attribution(self,
                                      returns: pd.Series,
                                      data: pd.DataFrame) -> Dict[str, float]:
        """Analyze volatility timing contribution"""
        if 'ATR' not in data.columns:
            return {}
        
        # Quartile analysis of returns by volatility
        volatility_quartiles = pd.qcut(data['ATR'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        
        vol_attribution = {}
        for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
            mask = volatility_quartiles == quartile
            if mask.sum() > 0:
                vol_attribution[quartile] = {
                    'avg_return': returns[mask].mean(),
                    'sharpe': returns[mask].mean() / returns[mask].std() * np.sqrt(252 * 26)
                             if returns[mask].std() > 0 else 0,
                    'periods': mask.sum()
                }
        
        return vol_attribution
    
    def _analyze_indicator_attribution(self,
                                     returns: pd.Series,
                                     data: pd.DataFrame,
                                     regimes: pd.DataFrame) -> Dict[str, float]:
        """Analyze which indicators contribute most to returns"""
        indicator_scores = {}
        
        # Test key indicators
        test_indicators = ['RSI', 'MACD_Signal', 'ADX', 'ATR', 'MFI']
        
        for indicator in test_indicators:
            if indicator not in data.columns:
                continue
            
            # Simple analysis: returns when indicator is extreme
            indicator_values = data[indicator]
            
            # Define extreme thresholds
            high_threshold = indicator_values.quantile(0.8)
            low_threshold = indicator_values.quantile(0.2)
            
            high_returns = returns[indicator_values > high_threshold].mean()
            low_returns = returns[indicator_values < low_threshold].mean()
            neutral_returns = returns[(indicator_values >= low_threshold) & 
                                    (indicator_values <= high_threshold)].mean()
            
            indicator_scores[indicator] = {
                'high_value_return': high_returns,
                'low_value_return': low_returns,
                'neutral_return': neutral_returns,
                'indicator_edge': max(abs(high_returns - neutral_returns),
                                     abs(low_returns - neutral_returns))
            }
        
        return indicator_scores
    
    def _calculate_total_attribution(self, 
                                   attribution_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate total attribution summary"""
        # Aggregate attribution scores
        total_timing_effect = np.mean([
            score['timing_effectiveness'] 
            for score in attribution_results['timing'].values()
            if 'timing_effectiveness' in score
        ])
        
        total_selection_effect = np.mean([
            score['selection_alpha']
            for score in attribution_results['selection'].values()
            if 'selection_alpha' in score
        ])
        
        return {
            'timing_contribution': total_timing_effect,
            'selection_contribution': total_selection_effect,
            'total_alpha': total_timing_effect + total_selection_effect
        }

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_regime_performance(regime_performance: Dict[str, Any],
                          save_path: Optional[str] = None):
    """Plot performance by regime"""
    n_dims = len([d for d in regime_performance.keys() if d != 'overall'])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (dimension, results) in enumerate(regime_performance.items()):
        if dimension == 'overall' or i >= len(axes):
            continue
        
        ax = axes[i]
        
        if 'performance' not in results or not results['performance']:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(f'{dimension} - No data')
            continue
        
        # Extract data for plotting
        regimes = list(results['performance'].keys())
        sharpe_ratios = [results['performance'][r]['sharpe_ratio'] for r in regimes]
        returns = [results['performance'][r]['total_return'] * 100 for r in regimes]
        
        # Create bar plot
        x = np.arange(len(regimes))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, sharpe_ratios, width, label='Sharpe Ratio', alpha=0.8)
        
        # Use secondary y-axis for returns
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, returns, width, label='Total Return (%)', 
                       alpha=0.8, color='orange')
        
        # Color bars based on performance
        for bar, sharpe in zip(bars1, sharpe_ratios):
            if sharpe < 0:
                bar.set_color('red')
            elif sharpe > 1:
                bar.set_color('green')
            else:
                bar.set_color('steelblue')
        
        ax.set_xlabel('Regime')
        ax.set_ylabel('Sharpe Ratio', color='steelblue')
        ax2.set_ylabel('Total Return (%)', color='orange')
        ax.set_title(f'{dimension} Regime Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(regimes, rotation=45, ha='right')
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_dims, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Performance Attribution by Regime', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Performance attribution plot saved to: {save_path}")
    
    plt.close()

def plot_contribution_breakdown(contributions: Dict[str, Dict[str, Any]],
                              save_path: Optional[str] = None):
    """Plot contribution breakdown by regime"""
    # Aggregate contributions across dimensions
    all_contributions = {}
    
    for dimension, dim_contributions in contributions.items():
        if dimension == 'overall' or 'contributions' not in dim_contributions:
            continue
        
        for regime, contrib in dim_contributions['contributions'].items():
            key = f"{dimension}_{regime}"
            all_contributions[key] = contrib['return_contribution']
    
    # Sort by absolute contribution
    sorted_contrib = sorted(all_contributions.items(), 
                          key=lambda x: abs(x[1]), 
                          reverse=True)[:15]  # Top 15
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    labels = [k for k, _ in sorted_contrib]
    values = [v * 100 for _, v in sorted_contrib]  # Convert to percentage
    
    # Create horizontal bar chart
    bars = ax.barh(labels, values)
    
    # Color positive/negative differently
    for bar, val in zip(bars, values):
        if val < 0:
            bar.set_color('red')
        else:
            bar.set_color('green')
    
    ax.set_xlabel('Return Contribution (%)')
    ax.set_title('Top 15 Regime Contributions to Total Return')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (label, value) in enumerate(zip(labels, values)):
        ax.text(value + (0.1 if value > 0 else -0.1), i, f'{value:.1f}%', 
               ha='left' if value > 0 else 'right', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Contribution breakdown plot saved to: {save_path}")
    
    plt.close()

# =============================================================================
# MAIN ATTRIBUTION FUNCTION
# =============================================================================

def run_performance_attribution(returns: pd.Series,
                              regimes: pd.DataFrame,
                              data: pd.DataFrame,
                              positions: Optional[pd.Series] = None,
                              save_report: bool = True) -> Dict[str, Any]:
    """
    Run complete performance attribution analysis
    
    Args:
        returns: Strategy returns
        regimes: Regime classifications
        data: DataFrame with indicators
        positions: Optional position sizes
        save_report: Whether to save attribution report
        
    Returns:
        Complete attribution analysis
    """
    logger.info("Starting performance attribution analysis")
    
    # Initialize analyzers
    perf_analyzer = PerformanceAttributionAnalyzer()
    factor_analyzer = FactorAttributionAnalyzer()
    
    # Run analyses
    results = {
        'regime_attribution': perf_analyzer.analyze_regime_performance(
            returns, regimes, positions
        ),
        'factor_attribution': factor_analyzer.analyze_factor_attribution(
            returns, data, regimes
        ),
        'summary': {}
    }
    
    # Create summary
    overall_attribution = results['regime_attribution'].get('overall', {})
    
    results['summary'] = {
        'total_return': overall_attribution.get('total_return', 0),
        'sharpe_ratio': overall_attribution.get('sharpe_ratio', 0),
        'most_impactful_dimension': overall_attribution.get('most_impactful_dimension'),
        'recommendations': overall_attribution.get('recommendation', [])
    }
    
    # Save report if requested
    if save_report:
        save_attribution_report(results, perf_analyzer.regime_performance)
    
    logger.info("Performance attribution analysis complete")
    
    return results

def save_attribution_report(results: Dict[str, Any],
                          regime_performance: Dict[str, Any]):
    """Save performance attribution report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create attribution directory
    attribution_dir = os.path.join(RESULTS_DIR, 'performance_attribution')
    os.makedirs(attribution_dir, exist_ok=True)
    
    # Save plots
    plot_regime_performance(
        regime_performance,
        save_path=os.path.join(attribution_dir, f'regime_performance_{timestamp}.png')
    )
    
    plot_contribution_breakdown(
        regime_performance,
        save_path=os.path.join(attribution_dir, f'contribution_breakdown_{timestamp}.png')
    )
    
    # Save text report
    report_path = os.path.join(attribution_dir, f'attribution_report_{timestamp}.txt')
    
    with open(report_path, 'w') as f:
        f.write("PERFORMANCE ATTRIBUTION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("SUMMARY\n")
        f.write("-"*40 + "\n")
        summary = results['summary']
        f.write(f"Total Return: {summary['total_return']:.2%}\n")
        f.write(f"Sharpe Ratio: {summary['sharpe_ratio']:.4f}\n")
        f.write(f"Most Impactful Dimension: {summary['most_impactful_dimension']}\n\n")
        
        f.write("1. REGIME ATTRIBUTION\n")
        f.write("-"*40 + "\n")
        
        for dimension in REGIME_DIMENSIONS:
            if dimension in results['regime_attribution']:
                dim_results = results['regime_attribution'][dimension]
                f.write(f"\n{dimension} Dimension:\n")
                f.write(f"  Best Regime: {dim_results.get('best_regime', 'N/A')}\n")
                f.write(f"  Worst Regime: {dim_results.get('worst_regime', 'N/A')}\n")
                
                if 'performance' in dim_results:
                    f.write("\n  Regime Performance:\n")
                    for regime, metrics in dim_results['performance'].items():
                        f.write(f"    {regime}:\n")
                        f.write(f"      Sharpe: {metrics['sharpe_ratio']:.3f}\n")
                        f.write(f"      Return: {metrics['total_return']:.2%}\n")
                        f.write(f"      Time %: {metrics['pct_time']:.1f}%\n")
        
        f.write("\n2. FACTOR ATTRIBUTION\n")
        f.write("-"*40 + "\n")
        
        factor_attr = results.get('factor_attribution', {})
        
        if 'total' in factor_attr:
            f.write(f"\nTotal Attribution:\n")
            f.write(f"  Timing Contribution: {factor_attr['total']['timing_contribution']:.4f}\n")
            f.write(f"  Selection Contribution: {factor_attr['total']['selection_contribution']:.4f}\n")
            f.write(f"  Total Alpha: {factor_attr['total']['total_alpha']:.4f}\n")
        
        f.write("\n3. RECOMMENDATIONS\n")
        f.write("-"*40 + "\n")
        for i, rec in enumerate(summary.get('recommendations', []), 1):
            f.write(f"{i}. {rec}\n")
    
    logger.info(f"Attribution report saved to: {report_path}")

