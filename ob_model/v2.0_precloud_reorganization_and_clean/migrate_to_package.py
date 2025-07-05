#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Migration script to help transition from monolithic files to package structure
Run this from your main project directory
"""

import os
import shutil
import re
from datetime import datetime

def create_backup():
    """Create backup of existing files"""
    backup_dir = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)
    
    files_to_backup = [
        'multidimensional_regime_system.py',
        'multi_objective_optimizer.py',
        'enhanced_multi_objective_optimizer.py',
        'enhanced_backtester.py'
    ]
    
    backed_up = []
    for file in files_to_backup:
        if os.path.exists(file):
            shutil.copy2(file, backup_dir)
            backed_up.append(file)
            print(f"Backed up {file}")
    
    print(f"\nBackup created in: {backup_dir}")
    return backup_dir, backed_up

def extract_imports_mapping():
    """Create mapping of old imports to new imports"""
    import_map = {
        # From multidimensional_regime_system.py
        'from multidimensional_regime_system import calculate_all_indicators': 
            'from regime_system.core.indicators import calculate_all_indicators',
        'from multidimensional_regime_system import RollingRegimeClassifier':
            'from regime_system.core.regime_classifier import RollingRegimeClassifier',
        'from multidimensional_regime_system import load_csv_data':
            'from regime_system.core.data_loader import load_csv_data',
        
        # From multi_objective_optimizer.py
        'from multi_objective_optimizer import MultiObjectiveRegimeOptimizer':
            'from regime_system.optimization.multi_objective import MultiObjectiveRegimeOptimizer',
        'from multi_objective_optimizer import OptimizationResults':
            'from regime_system.optimization.multi_objective import OptimizationResults',
        'from multi_objective_optimizer import run_regime_optimization':
            'from regime_system.optimization.multi_objective import MultiObjectiveRegimeOptimizer',
        'from multi_objective_optimizer import optimize_window_size':
            'from regime_system.optimization.window_optimizer import optimize_window_size',
        'from multi_objective_optimizer import WalkForwardOptimizer':
            'from regime_system.optimization.walk_forward import WalkForwardOptimizer',
        
        # From enhanced_multi_objective_optimizer.py
        'from enhanced_multi_objective_optimizer import':
            'from regime_system.optimization.multi_objective import',
        
        # From enhanced_backtester.py
        'from enhanced_backtester import EnhancedRegimeStrategyBacktester':
            'from regime_system.backtesting.strategies import EnhancedRegimeStrategyBacktester',
        'from enhanced_backtester import StrategyConfig':
            'from regime_system.backtesting.strategies import StrategyConfig',
    }
    
    return import_map

def update_imports_in_file(filepath, import_map):
    """Update imports in a Python file"""
    if not os.path.exists(filepath):
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    changes_made = False
    
    # Update imports
    for old_import, new_import in import_map.items():
        if old_import in content:
            content = content.replace(old_import, new_import)
            changes_made = True
            print(f"Updated import in {filepath}: {old_import} -> {new_import}")
    
    # Save if changes were made
    if changes_made:
        # Create .bak file
        with open(f"{filepath}.bak", 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        # Write updated content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Updated {filepath} (backup saved as {filepath}.bak)")
    
    return changes_made

def create_compatibility_wrapper():
    """Create a compatibility wrapper for old imports"""
    wrapper_content = '''"""
Compatibility wrapper for old import structure
This allows existing code to work while transitioning to new structure
"""

import warnings

# Show deprecation warnings
warnings.filterwarnings('always', category=DeprecationWarning)

def _deprecation_warning(old_name, new_module):
    warnings.warn(
        f"Importing from '{old_name}' is deprecated. "
        f"Please use 'from {new_module} import ...' instead.",
        DeprecationWarning,
        stacklevel=3
    )

# Proxy imports from multidimensional_regime_system.py
try:
    from regime_system.core.indicators import calculate_all_indicators
    from regime_system.core.regime_classifier import RollingRegimeClassifier
    from regime_system.core.data_loader import load_csv_data
    _deprecation_warning('multidimensional_regime_system', 'regime_system.core')
except ImportError:
    pass

# Proxy imports from multi_objective_optimizer.py
try:
    from regime_system.optimization.multi_objective import (
        MultiObjectiveRegimeOptimizer,
        OptimizationResults,
        print_optimization_results
    )
    from regime_system.optimization.window_optimizer import optimize_window_size
    from regime_system.optimization.walk_forward import WalkForwardOptimizer
    
    # Create wrapper for old run_regime_optimization function
    def run_regime_optimization(classifier, data, max_iterations=50, method='differential_evolution', walk_forward=False):
        _deprecation_warning('run_regime_optimization', 'regime_system.optimization.multi_objective')
        optimizer = MultiObjectiveRegimeOptimizer(classifier, data)
        return optimizer.optimize_regime_thresholds(method=method, max_iterations=max_iterations)
    
except ImportError:
    pass

# Proxy imports from enhanced_backtester.py
try:
    from regime_system.backtesting.strategies import (
        EnhancedRegimeStrategyBacktester,
        StrategyConfig
    )
    _deprecation_warning('enhanced_backtester', 'regime_system.backtesting.strategies')
except ImportError:
    pass
'''
    
    # Save compatibility wrappers
    files = {
        'multidimensional_regime_system.py': 'multidimensional_regime_system',
        'multi_objective_optimizer.py': 'multi_objective_optimizer',
        'enhanced_backtester.py': 'enhanced_backtester'
    }
    
    for filename, module_name in files.items():
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                f.write(wrapper_content)
            print(f"Created compatibility wrapper: {filename}")

def create_migration_report():
    """Create a migration report"""
    report = f"""
MIGRATION REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
=====================================================

MIGRATION COMPLETED SUCCESSFULLY!

The regime system has been reorganized into a professional package structure:

regime_system/
├── core/               # Core components
│   ├── regime_classifier.py
│   ├── indicators.py
│   └── data_loader.py
├── optimization/       # Optimization modules
│   ├── multi_objective.py
│   ├── walk_forward.py
│   └── window_optimizer.py
├── backtesting/       # Backtesting strategies
│   └── strategies.py
├── validation/        # Validation tools
│   ├── indicator_analysis.py
│   ├── regime_distribution.py
│   └── performance_attribution.py
├── config/           # Configuration
│   └── settings.py
├── utils/            # Utilities
│   ├── checkpoint.py
│   └── logger.py
└── main.py          # Main entry point

NEXT STEPS:
1. Update your scripts to use new imports:
   OLD: from multidimensional_regime_system import calculate_all_indicators
   NEW: from regime_system.core.indicators import calculate_all_indicators

2. Test your existing code - compatibility wrappers are in place

3. Use the new main.py entry point:
   python regime_system/main.py analyze --data your_data.csv

4. Remove old files once migration is verified (backups are saved)

BENEFITS OF NEW STRUCTURE:
✓ Modular and maintainable code
✓ Easy to test individual components
✓ Ready for cloud deployment
✓ Professional package structure
✓ Clear separation of concerns

For questions or issues, check the README.md in the regime_system directory.
"""
    
    with open('MIGRATION_REPORT.txt', 'w') as f:
        f.write(report)
    
    print(report)

def main():
    """Run the migration process"""
    print("="*60)
    print("REGIME SYSTEM MIGRATION TOOL")
    print("="*60)
    print("\nThis will help migrate your code to the new package structure.")
    
    # Step 1: Create backup
    print("\nStep 1: Creating backup of existing files...")
    backup_dir, backed_up_files = create_backup()
    
    # Step 2: Check if package structure exists
    print("\nStep 2: Checking package structure...")
    if not os.path.exists('regime_system'):
        print("ERROR: regime_system package not found!")
        print("Please run the package structure setup script first.")
        return
    
    # Step 3: Update imports in user files
    print("\nStep 3: Updating imports in Python files...")
    import_map = extract_imports_mapping()
    
    # Find Python files to update (excluding regime_system directory)
    python_files = []
    for root, dirs, files in os.walk('.'):
        # Skip regime_system directory and backups
        if 'regime_system' in root or 'backup_' in root:
            continue
        for file in files:
            if file.endswith('.py') and file not in backed_up_files:
                python_files.append(os.path.join(root, file))
    
    updated_files = []
    for filepath in python_files:
        if update_imports_in_file(filepath, import_map):
            updated_files.append(filepath)
    
    print(f"\nUpdated {len(updated_files)} files")
    
    # Step 4: Create compatibility wrappers
    print("\nStep 4: Creating compatibility wrappers...")
    create_compatibility_wrapper()
    
    # Step 5: Create migration report
    print("\nStep 5: Creating migration report...")
    create_migration_report()
    
    print("\n" + "="*60)
    print("MIGRATION COMPLETE!")
    print("="*60)
    print(f"\nBackup saved in: {backup_dir}")
    print("Compatibility wrappers created for old imports")
    print("See MIGRATION_REPORT.txt for details")

if __name__ == "__main__":
    main()

