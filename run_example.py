#!/usr/bin/env python3
"""
Example script showing how to use the scPDSI processor
"""

import sys
from pathlib import Path
from scpdsi_processor import Config, ScPDSIProcessor, Logger

def setup_example_directories():
    """Create example directory structure"""
    
    directories = [
        Config.NETCDF_DIR,
        Config.OUTPUT_DIR, 
        Config.TEMP_DIR,
        Config.LOG_DIR,
        "geo"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def main():
    """Run example processing"""
    
    print("scPDSI Processor Example")
    print("=" * 50)
    
    # Setup directories
    print("\n1. Setting up directory structure...")
    setup_example_directories()
    
    # Check for required inputs
    print("\n2. Checking for required inputs...")
    
    netcdf_dir = Path(Config.NETCDF_DIR)
    netcdf_files = list(netcdf_dir.glob("*.nc"))
    
    sa1_shapefile = Path(Config.SA1_SHAPEFILE)
    
    print(f"NetCDF directory: {netcdf_dir}")
    print(f"Found {len(netcdf_files)} NetCDF files")
    print(f"SA1 shapefile: {sa1_shapefile}")
    print(f"SA1 shapefile exists: {sa1_shapefile.exists()}")
    
    if not netcdf_files:
        print("\n❌ No NetCDF files found!")
        print(f"Please place NetCDF files with scPDSI data in: {netcdf_dir}")
        print("Expected variable name: 'scpdsi'")
        print("Expected dimensions: longitude, latitude, time")
        return False
    
    if not sa1_shapefile.exists():
        print("\n❌ SA1 shapefile not found!")
        print(f"Please place SA1 2021 shapefile at: {sa1_shapefile}")
        print("Should contain SA1_CODE21 field")
        return False
    
    print("\n✅ All required inputs found!")
    
    # Run processing
    print("\n3. Starting processing...")
    try:
        processor = ScPDSIProcessor()
        processor.run()
        print("\n✅ Processing completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Processing failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)