#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Regime distribution validation module
Analyzes regime distributions, transitions, and stability
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
    REGIME_DIMENSIONS, RESULTS_DIR, REGIME_SMOOTHING_PERIODS,
    FIGURE_SIZE, FIGURE_DPI, PLOT_STYLE
)
from utils.logger import get_logger

logger = get_logger(__name__)

# Set plot style
plt.style.use(PLOT_STYLE)

# =============================================================================
# REGIME DISTRIBUTION ANALYZER
# =============================================================================

class RegimeDistributionAnalyzer:
    """Analyze regime distributions and validate classifications"""
    
    def __init__(self):
        self.distribution_stats = {}
        self.transition_matrix = {}
        self.persistence_metrics = {}
        
    def analyze_distributions(self, regimes: pd.DataFrame, 
                            min_regime_pct: float = 1.0) -> Dict[str, Any]:
        """
        Analyze regime distributions across all dimensions
        
        Args:
            regimes: DataFrame with regime classifications
            min_regime_pct: Minimum percentage for a regime to be considered significant
            
        Returns:
            Distribution analysis results
        """
        logger.info("Analyzing regime distributions")
        
        results = {}
        
        for dimension in REGIME_DIMENSIONS:
            regime_col = f'{dimension}_Regime'
            confidence_col = f'{dimension}_Confidence'
            
            if regime_col not in regimes.columns:
                logger.warning(f"Regime column {regime_col} not found")
                continue
            
            # Calculate distribution
            distribution = regimes[regime_col].value_counts()
            distribution_pct = (distribution / len(regimes)) * 100
            
            # Get confidence statistics
            confidence_stats = {}
            if confidence_col in regimes.columns:
                confidence_stats = {
                    'mean': regimes[confidence_col].mean(),
                    'std': regimes[confidence_col].std(),
                    'min': regimes[confidence_col].min(),
                    'max': regimes[confidence_col].max()
                }
            
            # Check for rare regimes
            rare_regimes = distribution_pct[distribution_pct < min_regime_pct].to_dict()
            
            # Check for dominant regimes (>80%)
            dominant_regimes = distribution_pct[distribution_pct > 80].to_dict()
            
            results[dimension] = {
                'distribution': distribution.to_dict(),
                'distribution_pct': distribution_pct.to_dict(),
                'confidence_stats': confidence_stats,
                'rare_regimes': rare_regimes,
                'dominant_regimes': dominant_regimes,
                'balance_score': self._calculate_balance_score(distribution_pct)
            }
            
            self.distribution_stats[dimension] = results[dimension]
        
        # Overall assessment
        results['overall'] = self._assess_overall_distributions(results)
        
        return results
    
    def _calculate_balance_score(self, distribution_pct: pd.Series) -> float:
        """Calculate how balanced the regime distribution is (0-1)"""
        # Use entropy as balance measure
        if len(distribution_pct) == 0:
            return 0.0
        
        # Normalize to probabilities
        probs = distribution_pct / 100
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log(len(distribution_pct))
        
        if max_entropy > 0:
            return entropy / max_entropy
        else:
            return 0.0
    
    def _assess_overall_distributions(self, dimension_results: Dict) -> Dict[str, Any]:
        """Assess overall distribution health"""
        issues = []
        
        # Check each dimension
        for dimension, stats in dimension_results.items():
            if dimension == 'overall':
                continue
            
            # Check for imbalance
            if stats['balance_score'] < 0.5:
                issues.append(f"{dimension} has poor regime balance (score: {stats['balance_score']:.2f})")
            
            # Check for dominant regimes
            if stats['dominant_regimes']:
                for regime, pct in stats['dominant_regimes'].items():
                    issues.append(f"{dimension} dominated by {regime} ({pct:.1f}%)")
            
            # Check confidence levels
            if stats['confidence_stats'] and stats['confidence_stats']['mean'] < 0.4:
                issues.append(f"{dimension} has low average confidence ({stats['confidence_stats']['mean']:.2f})")
        
        # Calculate overall health score
        health_score = max(0, 1 - len(issues) / 10)
        
        return {
            'health_score': health_score,
            'issues': issues,
            'recommendation': self._generate_distribution_recommendations(issues)
        }
    
    def _generate_distribution_recommendations(self, issues: List[str]) -> List[str]:
        """Generate recommendations based on distribution issues"""
        recommendations = []
        
        if any('dominated by' in issue for issue in issues):
            recommendations.append(
                "Review threshold parameters - some regimes are dominating"
            )
        
        if any('poor regime balance' in issue for issue in issues):
            recommendations.append(
                "Consider adjusting classification thresholds for better balance"
            )
        
        if any('low average confidence' in issue for issue in issues):
            recommendations.append(
                "Review indicator weights and thresholds to improve confidence"
            )
        
        return recommendations

# =============================================================================
# REGIME TRANSITION ANALYZER
# =============================================================================

class RegimeTransitionAnalyzer:
    """Analyze regime transitions and stability"""
    
    def __init__(self):
        self.transition_matrices = {}
        self.transition_stats = {}
        
    def analyze_transitions(self, regimes: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze regime transitions for all dimensions
        
        Args:
            regimes: DataFrame with regime classifications
            
        Returns:
            Transition analysis results
        """
        logger.info("Analyzing regime transitions")
        
        results = {}
        
        for dimension in REGIME_DIMENSIONS:
            regime_col = f'{dimension}_Regime'
            
            if regime_col not in regimes.columns:
                continue
            
            # Calculate transition matrix
            transition_matrix = self._calculate_transition_matrix(regimes[regime_col])
            self.transition_matrices[dimension] = transition_matrix
            
            # Calculate transition statistics
            transitions = self._calculate_transition_stats(regimes[regime_col])
            
            # Calculate stability metrics
            stability = self._calculate_stability_metrics(regimes[regime_col])
            
            results[dimension] = {
                'transition_matrix': transition_matrix.to_dict(),
                'transition_stats': transitions,
                'stability_metrics': stability,
                'persistence': self._calculate_persistence(regimes[regime_col])
            }
            
            self.transition_stats[dimension] = results[dimension]
        
        # Overall transition assessment
        results['overall'] = self._assess_overall_transitions(results)
        
        return results
    
    def _calculate_transition_matrix(self, regime_series: pd.Series) -> pd.DataFrame:
        """Calculate regime transition probability matrix"""
        # Get unique regimes
        unique_regimes = regime_series.unique()
        
        # Initialize transition counts
        transition_counts = pd.DataFrame(
            0, 
            index=unique_regimes, 
            columns=unique_regimes
        )
        
        # Count transitions
        for i in range(1, len(regime_series)):
            from_regime = regime_series.iloc[i-1]
            to_regime = regime_series.iloc[i]
            transition_counts.loc[from_regime, to_regime] += 1
        
        # Convert to probabilities
        transition_probs = transition_counts.div(
            transition_counts.sum(axis=1), 
            axis=0
        ).fillna(0)
        
        return transition_probs
    
    def _calculate_transition_stats(self, regime_series: pd.Series) -> Dict[str, Any]:
        """Calculate transition statistics"""
        # Count regime changes
        regime_changes = (regime_series != regime_series.shift()).sum() - 1
        
        # Calculate average regime duration
        regime_runs = []
        current_regime = regime_series.iloc[0]
        run_length = 1
        
        for i in range(1, len(regime_series)):
            if regime_series.iloc[i] == current_regime:
                run_length += 1
            else:
                regime_runs.append(run_length)
                current_regime = regime_series.iloc[i]
                run_length = 1
        
        regime_runs.append(run_length)
        
        return {
            'total_transitions': regime_changes,
            'transition_rate': regime_changes / len(regime_series),
            'avg_regime_duration': np.mean(regime_runs),
            'min_regime_duration': np.min(regime_runs),
            'max_regime_duration': np.max(regime_runs),
            'regime_run_distribution': np.histogram(regime_runs, bins=10)[0].tolist()
        }
    
    def _calculate_stability_metrics(self, regime_series: pd.Series) -> Dict[str, float]:
        """Calculate regime stability metrics"""
        # Persistence (probability of staying in same regime)
        persistence_probs = []
        
        transition_matrix = self._calculate_transition_matrix(regime_series)
        for regime in transition_matrix.index:
            if regime in transition_matrix.columns:
                persistence_probs.append(transition_matrix.loc[regime, regime])
        
        # Entropy of transitions
        transition_entropy = 0
        for _, row in transition_matrix.iterrows():
            probs = row[row > 0]
            if len(probs) > 0:
                transition_entropy -= np.sum(probs * np.log(probs + 1e-10))
        
        return {
            'avg_persistence': np.mean(persistence_probs) if persistence_probs else 0,
            'min_persistence': np.min(persistence_probs) if persistence_probs else 0,
            'transition_entropy': transition_entropy,
            'stability_score': np.mean(persistence_probs) if persistence_probs else 0
        }
    
    def _calculate_persistence(self, regime_series: pd.Series) -> float:
        """Calculate overall regime persistence"""
        # Count consecutive periods in same regime
        same_regime = (regime_series == regime_series.shift()).sum()
        total_periods = len(regime_series) - 1
        
        return same_regime / total_periods if total_periods > 0 else 0
    
    def _assess_overall_transitions(self, dimension_results: Dict) -> Dict[str, Any]:
        """Assess overall transition patterns"""
        issues = []
        
        for dimension, stats in dimension_results.items():
            if dimension == 'overall':
                continue
            
            # Check for excessive transitions
            if stats['transition_stats']['transition_rate'] > 0.2:
                issues.append(
                    f"{dimension} has excessive transitions "
                    f"({stats['transition_stats']['transition_rate']:.1%})"
                )
            
            # Check for stuck regimes
            if stats['stability_metrics']['avg_persistence'] > 0.95:
                issues.append(
                    f"{dimension} regimes are too sticky "
                    f"(persistence: {stats['stability_metrics']['avg_persistence']:.1%})"
                )
            
            # Check for whipsaws
            if stats['transition_stats']['avg_regime_duration'] < REGIME_SMOOTHING_PERIODS:
                issues.append(
                    f"{dimension} has regime whipsaws "
                    f"(avg duration: {stats['transition_stats']['avg_regime_duration']:.1f})"
                )
        
        return {
            'issues': issues,
            'recommendation': self._generate_transition_recommendations(issues)
        }
    
    def _generate_transition_recommendations(self, issues: List[str]) -> List[str]:
        """Generate recommendations for transition issues"""
        recommendations = []
        
        if any('excessive transitions' in issue for issue in issues):
            recommendations.append(
                f"Increase regime smoothing periods (current: {REGIME_SMOOTHING_PERIODS})"
            )
        
        if any('too sticky' in issue for issue in issues):
            recommendations.append(
                "Review rolling window size - may be too large"
            )
        
        if any('whipsaws' in issue for issue in issues):
            recommendations.append(
                "Consider increasing confirmation periods for regime changes"
            )
        
        return recommendations

# =============================================================================
# TEMPORAL ANALYSIS
# =============================================================================

class RegimeTemporalAnalyzer:
    """Analyze regime patterns over time"""
    
    def analyze_temporal_patterns(self, regimes: pd.DataFrame,
                                data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze temporal patterns in regime classifications
        
        Args:
            regimes: DataFrame with regime classifications
            data: Original data with timestamps
            
        Returns:
            Temporal analysis results
        """
        logger.info("Analyzing temporal regime patterns")
        
        results = {}
        
        # Add time-based features
        if regimes.index.dtype == 'datetime64[ns]':
            regimes['hour'] = regimes.index.hour
            regimes['day_of_week'] = regimes.index.dayofweek
            regimes['month'] = regimes.index.month
        
        # Analyze patterns by time of day
        if 'hour' in regimes.columns:
            results['hourly_patterns'] = self._analyze_hourly_patterns(regimes)
        
        # Analyze patterns by day of week
        if 'day_of_week' in regimes.columns:
            results['weekly_patterns'] = self._analyze_weekly_patterns(regimes)
        
        # Analyze regime duration patterns
        results['duration_patterns'] = self._analyze_duration_patterns(regimes)
        
        # Check for regime clustering
        results['clustering'] = self._analyze_regime_clustering(regimes)
        
        return results
    
    def _analyze_hourly_patterns(self, regimes: pd.DataFrame) -> Dict[str, Any]:
        """Analyze regime patterns by hour of day"""
        hourly_stats = {}
        
        for dimension in REGIME_DIMENSIONS:
            regime_col = f'{dimension}_Regime'
            if regime_col not in regimes.columns:
                continue
            
            # Calculate regime distribution by hour
            hourly_dist = pd.crosstab(regimes['hour'], regimes[regime_col], normalize='index')
            
            # Find hours with unusual patterns
            unusual_hours = []
            for hour in hourly_dist.index:
                hour_dist = hourly_dist.loc[hour]
                # Check if distribution differs significantly from overall
                overall_dist = regimes[regime_col].value_counts(normalize=True)
                
                for regime in hour_dist.index:
                    if regime in overall_dist.index:
                        diff = abs(hour_dist[regime] - overall_dist[regime])
                        if diff > 0.2:  # 20% difference threshold
                            unusual_hours.append({
                                'hour': hour,
                                'regime': regime,
                                'difference': diff
                            })
            
            hourly_stats[dimension] = {
                'distribution': hourly_dist.to_dict(),
                'unusual_hours': unusual_hours[:5]  # Top 5
            }
        
        return hourly_stats
    
    def _analyze_weekly_patterns(self, regimes: pd.DataFrame) -> Dict[str, Any]:
        """Analyze regime patterns by day of week"""
        weekly_stats = {}
        
        for dimension in REGIME_DIMENSIONS:
            regime_col = f'{dimension}_Regime'
            if regime_col not in regimes.columns:
                continue
            
            # Calculate regime distribution by day
            weekly_dist = pd.crosstab(regimes['day_of_week'], regimes[regime_col], normalize='index')
            
            weekly_stats[dimension] = {
                'distribution': weekly_dist.to_dict(),
                'weekend_effect': self._check_weekend_effect(regimes, regime_col)
            }
        
        return weekly_stats
    
    def _check_weekend_effect(self, regimes: pd.DataFrame, regime_col: str) -> bool:
        """Check if there's a weekend effect in regimes"""
        if 'day_of_week' not in regimes.columns:
            return False
        
        # Weekend = Saturday (5) and Sunday (6)
        weekend_mask = regimes['day_of_week'].isin([5, 6])
        
        if weekend_mask.sum() == 0:
            return False
        
        # Compare weekend vs weekday distributions
        weekend_dist = regimes[weekend_mask][regime_col].value_counts(normalize=True)
        weekday_dist = regimes[~weekend_mask][regime_col].value_counts(normalize=True)
        
        # Check for significant differences
        for regime in weekend_dist.index:
            if regime in weekday_dist.index:
                diff = abs(weekend_dist[regime] - weekday_dist[regime])
                if diff > 0.15:  # 15% difference threshold
                    return True
        
        return False
    
    def _analyze_duration_patterns(self, regimes: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in regime durations"""
        duration_stats = {}
        
        for dimension in REGIME_DIMENSIONS:
            regime_col = f'{dimension}_Regime'
            if regime_col not in regimes.columns:
                continue
            
            # Calculate durations for each regime type
            regime_durations = {}
            
            for regime in regimes[regime_col].unique():
                durations = []
                in_regime = False
                duration = 0
                
                for i in range(len(regimes)):
                    if regimes[regime_col].iloc[i] == regime:
                        if not in_regime:
                            in_regime = True
                            duration = 1
                        else:
                            duration += 1
                    else:
                        if in_regime:
                            durations.append(duration)
                            in_regime = False
                            duration = 0
                
                if in_regime:
                    durations.append(duration)
                
                if durations:
                    regime_durations[regime] = {
                        'mean': np.mean(durations),
                        'std': np.std(durations),
                        'min': np.min(durations),
                        'max': np.max(durations)
                    }
            
            duration_stats[dimension] = regime_durations
        
        return duration_stats
    
    def _analyze_regime_clustering(self, regimes: pd.DataFrame) -> Dict[str, Any]:
        """Check for regime clustering (autocorrelation)"""
        clustering_stats = {}
        
        for dimension in REGIME_DIMENSIONS:
            regime_col = f'{dimension}_Regime'
            if regime_col not in regimes.columns:
                continue
            
            # Convert to numeric for autocorrelation
            regime_numeric = pd.Categorical(regimes[regime_col]).codes
            
            # Calculate autocorrelation at different lags
            autocorr = {}
            for lag in [1, 5, 10, 20]:
                if len(regime_numeric) > lag:
                    autocorr[f'lag_{lag}'] = regime_numeric.autocorr(lag=lag)
            
            clustering_stats[dimension] = {
                'autocorrelation': autocorr,
                'has_clustering': any(abs(v) > 0.5 for v in autocorr.values())
            }
        
        return clustering_stats

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_regime_distributions(distribution_stats: Dict[str, Any],
                            save_path: Optional[str] = None):
    """Plot regime distribution charts"""
    n_dims = len([d for d in distribution_stats.keys() if d != 'overall'])
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (dimension, stats) in enumerate(distribution_stats.items()):
        if dimension == 'overall' or i >= len(axes):
            continue
        
        ax = axes[i]
        
        # Get distribution data
        dist_pct = stats['distribution_pct']
        
        # Create bar plot
        regimes = list(dist_pct.keys())
        percentages = list(dist_pct.values())
        
        bars = ax.bar(regimes, percentages)
        
        # Color bars based on percentage
        for j, (bar, pct) in enumerate(zip(bars, percentages)):
            if pct > 80:
                bar.set_color('red')
            elif pct < 5:
                bar.set_color('orange')
            else:
                bar.set_color('steelblue')
        
        ax.set_title(f'{dimension} Regime Distribution')
        ax.set_ylabel('Percentage (%)')
        ax.set_xlabel('Regime')
        
        # Rotate x labels if needed
        if len(regimes) > 4:
            ax.set_xticklabels(regimes, rotation=45, ha='right')
        
        # Add percentage labels
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{pct:.1f}%', ha='center', va='bottom')
        
        # Add balance score
        balance_score = stats['balance_score']
        ax.text(0.02, 0.98, f'Balance: {balance_score:.2f}',
               transform=ax.transAxes, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide unused subplots
    for i in range(n_dims, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Regime Distributions Across Dimensions', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Regime distribution plot saved to: {save_path}")
    
    plt.close()

def plot_transition_matrices(transition_matrices: Dict[str, pd.DataFrame],
                           save_path: Optional[str] = None):
    """Plot regime transition matrices"""
    n_dims = len(transition_matrices)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (dimension, matrix) in enumerate(transition_matrices.items()):
        if i >= len(axes):
            continue
        
        ax = axes[i]
        
        # Create heatmap
        sns.heatmap(matrix, 
                   annot=True, 
                   fmt='.2f',
                   cmap='YlOrRd',
                   vmin=0,
                   vmax=1,
                   square=True,
                   cbar_kws={'label': 'Transition Probability'},
                   ax=ax)
        
        ax.set_title(f'{dimension} Regime Transitions')
        ax.set_xlabel('To Regime')
        ax.set_ylabel('From Regime')
    
    # Hide unused subplots
    for i in range(n_dims, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Regime Transition Probability Matrices', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Transition matrix plot saved to: {save_path}")
    
    plt.close()

# =============================================================================
# MAIN VALIDATION FUNCTION
# =============================================================================

def validate_regime_distributions(regimes: pd.DataFrame,
                                data: pd.DataFrame,
                                save_report: bool = True) -> Dict[str, Any]:
    """
    Run complete regime distribution validation
    
    Args:
        regimes: DataFrame with regime classifications
        data: Original data with indicators
        save_report: Whether to save validation report
        
    Returns:
        Complete validation results
    """
    logger.info("Starting regime distribution validation")
    
    # Initialize analyzers
    dist_analyzer = RegimeDistributionAnalyzer()
    trans_analyzer = RegimeTransitionAnalyzer()
    temporal_analyzer = RegimeTemporalAnalyzer()
    
    # Run analyses
    results = {
        'distribution_analysis': dist_analyzer.analyze_distributions(regimes),
        'transition_analysis': trans_analyzer.analyze_transitions(regimes),
        'temporal_analysis': temporal_analyzer.analyze_temporal_patterns(regimes, data),
        'validation_passed': True,
        'issues': [],
        'recommendations': []
    }
    
    # Collect all issues
    all_issues = []
    
    # Distribution issues
    dist_issues = results['distribution_analysis']['overall'].get('issues', [])
    all_issues.extend(dist_issues)
    
    # Transition issues
    trans_issues = results['transition_analysis']['overall'].get('issues', [])
    all_issues.extend(trans_issues)
    
    results['issues'] = all_issues
    
    # Determine if validation passed
    if len(all_issues) > 5:
        results['validation_passed'] = False
    
    # Generate recommendations
    all_recommendations = []
    
    # From distribution analysis
    dist_recs = results['distribution_analysis']['overall'].get('recommendation', [])
    all_recommendations.extend(dist_recs)
    
    # From transition analysis
    trans_recs = results['transition_analysis']['overall'].get('recommendation', [])
    all_recommendations.extend(trans_recs)
    
    results['recommendations'] = list(set(all_recommendations))  # Remove duplicates
    
    # Save report if requested
    if save_report:
        save_regime_validation_report(
            results, 
            dist_analyzer.distribution_stats,
            trans_analyzer.transition_matrices
        )
    
    logger.info(f"Regime validation complete. Passed: {results['validation_passed']}")
    
    return results

def save_regime_validation_report(results: Dict[str, Any],
                                distribution_stats: Dict[str, Any],
                                transition_matrices: Dict[str, pd.DataFrame]):
    """Save regime validation report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create validation directory
    validation_dir = os.path.join(RESULTS_DIR, 'regime_validation')
    os.makedirs(validation_dir, exist_ok=True)
    
    # Save plots
    plot_regime_distributions(
        distribution_stats,
        save_path=os.path.join(validation_dir, f'regime_distributions_{timestamp}.png')
    )
    
    plot_transition_matrices(
        transition_matrices,
        save_path=os.path.join(validation_dir, f'transition_matrices_{timestamp}.png')
    )
    
    # Save text report
    report_path = os.path.join(validation_dir, f'regime_validation_report_{timestamp}.txt')
    
    with open(report_path, 'w') as f:
        f.write("REGIME DISTRIBUTION VALIDATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Validation Status: {'PASSED' if results['validation_passed'] else 'FAILED'}\n")
        f.write(f"Total Issues Found: {len(results['issues'])}\n\n")
        
        f.write("1. DISTRIBUTION ANALYSIS\n")
        f.write("-"*40 + "\n")
        
        dist_overall = results['distribution_analysis']['overall']
        f.write(f"Health Score: {dist_overall['health_score']:.2f}/1.0\n")
        
        # Write distribution summaries
        for dimension in REGIME_DIMENSIONS:
            if dimension in results['distribution_analysis']:
                stats = results['distribution_analysis'][dimension]
                f.write(f"\n{dimension} Distribution:\n")
                for regime, pct in stats['distribution_pct'].items():
                    f.write(f"  {regime}: {pct:.1f}%\n")
                f.write(f"  Balance Score: {stats['balance_score']:.2f}\n")
        
        f.write("\n2. TRANSITION ANALYSIS\n")
        f.write("-"*40 + "\n")
        
        # Write transition summaries
        for dimension in REGIME_DIMENSIONS:
            if dimension in results['transition_analysis']:
                stats = results['transition_analysis'][dimension]
                f.write(f"\n{dimension} Transitions:\n")
                f.write(f"  Total Transitions: {stats['transition_stats']['total_transitions']}\n")
                f.write(f"  Transition Rate: {stats['transition_stats']['transition_rate']:.1%}\n")
                f.write(f"  Avg Duration: {stats['transition_stats']['avg_regime_duration']:.1f} periods\n")
                f.write(f"  Persistence: {stats['persistence']:.1%}\n")
        
        f.write("\n3. TEMPORAL PATTERNS\n")
        f.write("-"*40 + "\n")
        
        temporal = results['temporal_analysis']
        
        # Check for clustering
        if 'clustering' in temporal:
            f.write("\nRegime Clustering:\n")
            for dimension, cluster_stats in temporal['clustering'].items():
                if cluster_stats['has_clustering']:
                    f.write(f"  {dimension}: Shows significant clustering\n")
        
        f.write("\n4. ISSUES FOUND\n")
        f.write("-"*40 + "\n")
        for i, issue in enumerate(results['issues'], 1):
            f.write(f"{i}. {issue}\n")
        
        f.write("\n5. RECOMMENDATIONS\n")
        f.write("-"*40 + "\n")
        for i, rec in enumerate(results['recommendations'], 1):
            f.write(f"{i}. {rec}\n")
    
    logger.info(f"Regime validation report saved to: {report_path}")

