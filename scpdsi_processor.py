#!/usr/bin/env python3
"""
SA1 scPDSI Annual Statistics Processor

Generates annual scPDSI statistics for Australian SA1 regions from monthly NetCDF inputs,
producing calendar year (Jan-Dec) and financial year (Jul-Jun) outputs.
"""

import os
import sys
import logging
import warnings
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import json

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.crs import CRS
import pyproj

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

class Config:
    """Configuration parameters for scPDSI processing"""
    
    # Input/Output paths
    NETCDF_DIR = "original_dataset"
    SA1_SHAPEFILE = "geo/SA1_2021_AUST_GDA2020.shp"
    OUTPUT_DIR = "annual_results_SA1"
    TEMP_DIR = "temp"
    LOG_DIR = "logs"
    
    # Variable and dimension names
    VARIABLE_NAME = "scpdsi"
    LON_DIM = "longitude"
    LAT_DIM = "latitude"
    TIME_DIM = "time"
    
    # Spatial reference
    TARGET_CRS = "EPSG:3577"  # GDA94 / Australian Albers
    SA1_KEY = "SA1_CODE21"
    
    # Processing options
    COMPLETE_YEAR_REQUIRED = True
    CONSISTENCY_THRESHOLD = 0.01  # 1%
    MAX_CONSISTENCY_THRESHOLD = 0.02  # 2% policy limit
    
    # Export options
    INCLUDE_NEAREST = True
    EXPORT_ENCODING = "utf-8"
    
    @classmethod
    def create_directories(cls):
        """Create required directories"""
        for dir_path in [cls.OUTPUT_DIR, cls.TEMP_DIR, cls.LOG_DIR]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


class Logger:
    """Centralized logging configuration"""
    
    def __init__(self, log_level=logging.INFO):
        self.setup_logging(log_level)
    
    def setup_logging(self, log_level):
        """Setup comprehensive logging to file and console"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = Path(Config.LOG_DIR) / f"scpdsi_processing_{timestamp}.log"
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Configure root logger
        logging.basicConfig(level=log_level, handlers=[])
        logger = logging.getLogger()
        logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        logger.addHandler(console_handler)
        
        self.log_file = log_file
        logging.info(f"Logging initialized. Log file: {log_file}")


class ProcessingError(Exception):
    """Custom exception for processing errors"""
    pass


class ValidationReport:
    """Handles validation reporting for consistency checks"""
    
    def __init__(self, year_label: str, year_type: str):
        self.year_label = year_label
        self.year_type = year_type
        self.n_shape = 0
        self.n_excel = 0
        self.relative_diff = 0.0
        self.missing_in_export = []
        self.extra_in_export = []
        
    def generate_report(self) -> Dict:
        """Generate validation report dictionary"""
        return {
            'year_label': self.year_label,
            'year_type': self.year_type,
            'timestamp': datetime.now().isoformat(),
            'n_shape': self.n_shape,
            'n_excel': self.n_excel,
            'relative_difference': self.relative_diff,
            'threshold_exceeded': self.relative_diff > Config.CONSISTENCY_THRESHOLD,
            'missing_in_export': self.missing_in_export,
            'extra_in_export': self.extra_in_export
        }
    
    def save_report(self, output_path: Path):
        """Save validation report to JSON file"""
        report = self.generate_report()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logging.info(f"Validation report saved: {output_path}")


def main():
    """Main processing entry point"""
    try:
        # Initialize
        Config.create_directories()
        logger_setup = Logger()
        
        logging.info("="*80)
        logging.info("SA1 scPDSI Annual Statistics Processor - Started")
        logging.info("="*80)
        
        # Log configuration
        logging.info("Configuration:")
        for attr in dir(Config):
            if not attr.startswith('_') and not callable(getattr(Config, attr)):
                logging.info(f"  {attr}: {getattr(Config, attr)}")
        
        processor = ScPDSIProcessor()
        processor.run()
        
        logging.info("="*80)
        logging.info("SA1 scPDSI Annual Statistics Processor - Completed Successfully")
        logging.info("="*80)
        
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}", exc_info=True)
        sys.exit(1)


class SA1DataManager:
    """Manages SA1 shapefile reading and preprocessing"""
    
    def __init__(self):
        self.sa1_gdf = None
        self.sa1_centroids = None
        self.sa1_reprojected = None
        self.n_shape = 0
        
    def load_sa1_data(self) -> gpd.GeoDataFrame:
        """Load and validate SA1 shapefile"""
        shapefile_path = Path(Config.SA1_SHAPEFILE)
        
        if not shapefile_path.exists():
            raise ProcessingError(f"SA1 shapefile not found: {shapefile_path}")
        
        logging.info(f"Loading SA1 shapefile: {shapefile_path}")
        
        try:
            self.sa1_gdf = gpd.read_file(shapefile_path)
            logging.info(f"Loaded {len(self.sa1_gdf)} SA1 polygons")
            
            # Validate SA1_CODE21 column
            if Config.SA1_KEY not in self.sa1_gdf.columns:
                raise ProcessingError(f"Required column {Config.SA1_KEY} not found in shapefile")
            
            # Check for duplicates
            duplicate_codes = self.sa1_gdf[Config.SA1_KEY].duplicated()
            if duplicate_codes.any():
                n_duplicates = duplicate_codes.sum()
                raise ProcessingError(f"Found {n_duplicates} duplicate SA1_CODE21 values")
            
            # Check for missing values
            missing_codes = self.sa1_gdf[Config.SA1_KEY].isna()
            if missing_codes.any():
                n_missing = missing_codes.sum()
                raise ProcessingError(f"Found {n_missing} missing SA1_CODE21 values")
            
            self.n_shape = len(self.sa1_gdf)
            logging.info(f"SA1 validation passed: {self.n_shape} unique regions")
            
            return self.sa1_gdf
            
        except Exception as e:
            raise ProcessingError(f"Failed to load SA1 shapefile: {str(e)}")
    
    def reproject_to_target_crs(self) -> gpd.GeoDataFrame:
        """Reproject SA1 data to target CRS (EPSG:3577)"""
        if self.sa1_gdf is None:
            raise ProcessingError("SA1 data not loaded. Call load_sa1_data() first.")
        
        original_crs = self.sa1_gdf.crs
        logging.info(f"Original SA1 CRS: {original_crs}")
        logging.info(f"Reprojecting to target CRS: {Config.TARGET_CRS}")
        
        try:
            self.sa1_reprojected = self.sa1_gdf.to_crs(Config.TARGET_CRS)
            logging.info("SA1 reprojection completed successfully")
            return self.sa1_reprojected
            
        except Exception as e:
            raise ProcessingError(f"Failed to reproject SA1 data: {str(e)}")
    
    def create_centroids(self) -> gpd.GeoDataFrame:
        """Create centroids for all SA1 polygons"""
        if self.sa1_reprojected is None:
            raise ProcessingError("SA1 data not reprojected. Call reproject_to_target_crs() first.")
        
        logging.info("Computing SA1 centroids...")
        
        try:
            centroids_geom = self.sa1_reprojected.geometry.centroid
            
            self.sa1_centroids = gpd.GeoDataFrame(
                self.sa1_reprojected[[Config.SA1_KEY]], 
                geometry=centroids_geom,
                crs=self.sa1_reprojected.crs
            )
            
            logging.info(f"Created centroids for {len(self.sa1_centroids)} SA1 regions")
            return self.sa1_centroids
            
        except Exception as e:
            raise ProcessingError(f"Failed to create SA1 centroids: {str(e)}")
    
    def get_bounds_info(self) -> Dict:
        """Get bounds information for the reprojected SA1 data"""
        if self.sa1_reprojected is None:
            raise ProcessingError("SA1 data not reprojected.")
        
        bounds = self.sa1_reprojected.total_bounds
        return {
            'minx': bounds[0],
            'miny': bounds[1], 
            'maxx': bounds[2],
            'maxy': bounds[3],
            'width': bounds[2] - bounds[0],
            'height': bounds[3] - bounds[1]
        }


class NetCDFManager:
    """Manages NetCDF reading and time parsing"""
    
    def __init__(self):
        self.netcdf_files = []
        self.monthly_data = {}
        self.template_grid = None
        
    def discover_netcdf_files(self) -> List[Path]:
        """Discover all NetCDF files in the input directory"""
        netcdf_dir = Path(Config.NETCDF_DIR)
        
        if not netcdf_dir.exists():
            raise ProcessingError(f"NetCDF directory not found: {netcdf_dir}")
        
        patterns = ['*.nc', '*.NC', '*.netcdf']
        self.netcdf_files = []
        
        for pattern in patterns:
            self.netcdf_files.extend(netcdf_dir.glob(pattern))
        
        if not self.netcdf_files:
            raise ProcessingError(f"No NetCDF files found in {netcdf_dir}")
        
        self.netcdf_files.sort()
        logging.info(f"Discovered {len(self.netcdf_files)} NetCDF files")
        
        return self.netcdf_files
    
    def parse_time_coordinate(self, ds: xr.Dataset) -> pd.DatetimeIndex:
        """Robustly parse time coordinate into datetime index"""
        if Config.TIME_DIM not in ds.dims:
            raise ProcessingError(f"Time dimension '{Config.TIME_DIM}' not found in dataset")
        
        time_var = ds[Config.TIME_DIM]
        
        try:
            # Try direct conversion first
            dates = pd.to_datetime(time_var.values)
            
        except Exception:
            try:
                # Try using xarray's time parsing
                dates = pd.to_datetime(time_var.values, errors='coerce')
                
                # Remove any NaT values
                valid_dates = dates[~pd.isna(dates)]
                if len(valid_dates) != len(dates):
                    n_invalid = len(dates) - len(valid_dates)
                    logging.warning(f"Removed {n_invalid} invalid time values")
                    dates = valid_dates
                    
            except Exception as e:
                raise ProcessingError(f"Failed to parse time coordinate: {str(e)}")
        
        if len(dates) == 0:
            raise ProcessingError("No valid dates found in time coordinate")
        
        return pd.DatetimeIndex(dates)
    
    def read_monthly_data(self) -> Dict:
        """Read all monthly data and organize by date - optimized for large files"""
        logging.info("Reading monthly NetCDF data...")
        
        self.monthly_data = {}
        total_processed = 0
        total_skipped = 0
        
        for file_idx, nc_file in enumerate(self.netcdf_files):
            try:
                logging.info(f"Processing file {file_idx + 1}/{len(self.netcdf_files)}: {nc_file.name}")
                
                # Use optimal chunking for large files
                chunk_sizes = {'time': 1}  # Process one time slice at a time for large files
                
                # Try to determine optimal spatial chunking based on file size
                try:
                    file_size_mb = nc_file.stat().st_size / (1024 * 1024)
                    if file_size_mb > 1000:  # Files larger than 1GB
                        # Use smaller spatial chunks for very large files
                        chunk_sizes.update({'longitude': 100, 'latitude': 100})
                        logging.info(f"  Large file detected ({file_size_mb:.1f} MB), using chunked processing")
                except:
                    pass
                
                # Open with Dask for lazy loading
                with xr.open_dataset(nc_file, chunks=chunk_sizes, decode_times=False) as ds:
                    
                    # Validate required variable
                    if Config.VARIABLE_NAME not in ds.data_vars:
                        logging.warning(f"Variable '{Config.VARIABLE_NAME}' not found in {nc_file.name}, skipping")
                        continue
                    
                    # Parse time coordinate
                    dates = self.parse_time_coordinate(ds)
                    
                    # Process each time slice with memory management
                    var_data = ds[Config.VARIABLE_NAME]
                    n_time_steps = len(dates)
                    
                    logging.info(f"  Processing {n_time_steps} time steps...")
                    
                    for i, date in enumerate(dates):
                        try:
                            # Progress logging for large files
                            if n_time_steps > 50 and (i + 1) % 12 == 0:
                                logging.info(f"    Progress: {i + 1}/{n_time_steps} time steps processed")
                            
                            # Extract monthly slice with explicit loading for efficiency
                            monthly_slice = var_data.isel({Config.TIME_DIM: i})
                            
                            # For very large arrays, ensure spatial coordinates are loaded
                            if hasattr(monthly_slice, 'load'):
                                monthly_slice = monthly_slice.load()  # Load data into memory
                            
                            # Store with date key
                            date_key = date.strftime('%Y-%m')
                            
                            if date_key in self.monthly_data:
                                logging.warning(f"Duplicate data for {date_key}, using latest from {nc_file.name}")
                            
                            self.monthly_data[date_key] = {
                                'date': date,
                                'data': monthly_slice,
                                'source_file': nc_file.name
                            }
                            
                            total_processed += 1
                            
                        except Exception as e:
                            logging.warning(f"Skipped time slice {i} ({date}) from {nc_file.name}: {str(e)}")
                            total_skipped += 1
                            continue
                    
                    logging.info(f"  Completed file {nc_file.name}: {n_time_steps - (total_skipped - (total_processed - n_time_steps))} successful")
                            
            except Exception as e:
                logging.error(f"Failed to process {nc_file.name}: {str(e)}")
                continue
        
        logging.info(f"Monthly data reading completed:")
        logging.info(f"  Total processed: {total_processed}")
        logging.info(f"  Total skipped: {total_skipped}")
        logging.info(f"  Unique months: {len(self.monthly_data)}")
        
        if not self.monthly_data:
            raise ProcessingError("No valid monthly data found")
        
        # Memory cleanup hint
        import gc
        gc.collect()
        
        return self.monthly_data
    
    def group_by_years(self) -> Tuple[Dict, Dict]:
        """Group monthly data by calendar and financial years"""
        if not self.monthly_data:
            raise ProcessingError("No monthly data available. Call read_monthly_data() first.")
        
        calendar_years = {}
        financial_years = {}
        
        for date_key, month_info in self.monthly_data.items():
            date = month_info['date']
            year = date.year
            month = date.month
            
            # Calendar year grouping (Jan-Dec)
            cal_year_key = str(year)
            if cal_year_key not in calendar_years:
                calendar_years[cal_year_key] = {}
            calendar_years[cal_year_key][date_key] = month_info
            
            # Financial year grouping (Jul-Jun)
            if month >= 7:  # Jul-Dec: belongs to financial year starting this calendar year
                fin_year_start = year
            else:  # Jan-Jun: belongs to financial year that started previous calendar year
                fin_year_start = year - 1
            
            fin_year_key = f"{fin_year_start}-{str(fin_year_start + 1)[-2:]}"
            if fin_year_key not in financial_years:
                financial_years[fin_year_key] = {}
            financial_years[fin_year_key][date_key] = month_info
        
        # Filter complete years if required
        if Config.COMPLETE_YEAR_REQUIRED:
            calendar_years = self._filter_complete_years(calendar_years, "calendar")
            financial_years = self._filter_complete_years(financial_years, "financial")
        
        logging.info(f"Year grouping completed:")
        logging.info(f"  Calendar years: {len(calendar_years)} ({list(calendar_years.keys())})")
        logging.info(f"  Financial years: {len(financial_years)} ({list(financial_years.keys())})")
        
        return calendar_years, financial_years
    
    def _filter_complete_years(self, year_groups: Dict, year_type: str) -> Dict:
        """Filter to keep only years with 12 complete months"""
        complete_years = {}
        
        for year_key, months in year_groups.items():
            if len(months) == 12:
                complete_years[year_key] = months
            else:
                available_months = sorted(months.keys())
                logging.info(f"Skipping incomplete {year_type} year {year_key}: "
                           f"{len(months)}/12 months available ({', '.join(available_months)})")
        
        return complete_years


class GridManager:
    """Manages template grid creation and raster alignment"""
    
    def __init__(self):
        self.template_grid = None
        self.template_transform = None
        self.template_shape = None
        self.template_crs = None
        
    def create_template_from_first_slice(self, first_monthly_data: xr.DataArray) -> Dict:
        """Create template grid from first valid monthly slice"""
        logging.info("Creating template grid from first monthly slice...")
        
        try:
            # Get spatial dimensions
            if Config.LON_DIM not in first_monthly_data.dims:
                raise ProcessingError(f"Longitude dimension '{Config.LON_DIM}' not found")
            if Config.LAT_DIM not in first_monthly_data.dims:
                raise ProcessingError(f"Latitude dimension '{Config.LAT_DIM}' not found")
            
            # Extract coordinates
            lon_coords = first_monthly_data[Config.LON_DIM].values
            lat_coords = first_monthly_data[Config.LAT_DIM].values
            
            # Calculate grid parameters
            lon_res = abs(lon_coords[1] - lon_coords[0]) if len(lon_coords) > 1 else 1.0
            lat_res = abs(lat_coords[1] - lat_coords[0]) if len(lat_coords) > 1 else 1.0
            
            # Create transform (assuming regular grid)
            transform = Affine.translation(lon_coords[0] - lon_res/2, lat_coords[0] - lat_res/2) * \
                       Affine.scale(lon_res, lat_res)
            
            # Get CRS from data if available
            crs = first_monthly_data.rio.crs if hasattr(first_monthly_data, 'rio') and first_monthly_data.rio.crs else None
            
            self.template_grid = {
                'transform': transform,
                'width': len(lon_coords),
                'height': len(lat_coords),
                'crs': crs,
                'lon_coords': lon_coords,
                'lat_coords': lat_coords,
                'lon_res': lon_res,
                'lat_res': lat_res,
                'bounds': {
                    'left': lon_coords.min() - lon_res/2,
                    'bottom': lat_coords.min() - lat_res/2,
                    'right': lon_coords.max() + lon_res/2,
                    'top': lat_coords.max() + lat_res/2
                }
            }
            
            self.template_transform = transform
            self.template_shape = (len(lat_coords), len(lon_coords))
            self.template_crs = crs
            
            logging.info(f"Template grid created:")
            logging.info(f"  Shape: {self.template_shape}")
            logging.info(f"  Transform: {transform}")
            logging.info(f"  CRS: {crs}")
            logging.info(f"  Resolution: {lon_res:.6f} x {lat_res:.6f}")
            logging.info(f"  Bounds: {self.template_grid['bounds']}")
            
            return self.template_grid
            
        except Exception as e:
            raise ProcessingError(f"Failed to create template grid: {str(e)}")
    
    def validate_alignment(self, data_array: xr.DataArray) -> bool:
        """Validate that a data array aligns with the template grid - optimized"""
        if self.template_grid is None:
            raise ProcessingError("Template grid not created. Call create_template_from_first_slice() first.")
        
        try:
            # Fast dimension check first
            if (Config.LON_DIM not in data_array.dims or 
                Config.LAT_DIM not in data_array.dims):
                logging.error(f"Required dimensions missing: {list(data_array.dims)}")
                return False
            
            # Quick shape comparison
            current_lon_size = data_array.dims[Config.LON_DIM]
            current_lat_size = data_array.dims[Config.LAT_DIM]
            current_shape = (current_lat_size, current_lon_size)
            
            if current_shape != self.template_shape:
                logging.error(f"Shape mismatch: expected {self.template_shape}, got {current_shape}")
                return False
            
            # Optimized coordinate comparison (sample-based for large arrays)
            lon_coords = data_array[Config.LON_DIM].values
            lat_coords = data_array[Config.LAT_DIM].values
            
            template_lon = self.template_grid['lon_coords']
            template_lat = self.template_grid['lat_coords']
            
            # For large arrays, sample coordinates instead of comparing all
            if len(lon_coords) > 1000:
                # Sample first, middle, and last coordinates
                sample_indices = [0, len(lon_coords)//2, -1]
                lon_sample = lon_coords[sample_indices]
                template_lon_sample = template_lon[sample_indices]
                lat_sample_indices = [0, len(lat_coords)//2, -1]
                lat_sample = lat_coords[lat_sample_indices]
                template_lat_sample = template_lat[lat_sample_indices]
                
                lon_diff = np.max(np.abs(lon_sample - template_lon_sample))
                lat_diff = np.max(np.abs(lat_sample - template_lat_sample))
            else:
                # Full comparison for smaller arrays
                lon_diff = np.max(np.abs(lon_coords - template_lon))
                lat_diff = np.max(np.abs(lat_coords - template_lat))
            
            tolerance = 1e-6  # Coordinate tolerance
            
            if lon_diff > tolerance or lat_diff > tolerance:
                logging.error(f"Coordinate mismatch: lon_diff={lon_diff:.8f}, lat_diff={lat_diff:.8f}")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Alignment validation failed: {str(e)}")
            return False


class AnnualAggregator:
    """Handles annual mean computation for grouped monthly data"""
    
    def __init__(self, grid_manager: GridManager):
        self.grid_manager = grid_manager
        
    def compute_annual_means(self, year_groups: Dict, year_type: str) -> Dict:
        """Compute annual means for grouped monthly data - memory optimized"""
        logging.info(f"Computing annual means for {year_type} years...")
        
        annual_means = {}
        
        for year_idx, (year_key, months_data) in enumerate(year_groups.items()):
            try:
                logging.info(f"Processing {year_type} year {year_key} ({year_idx + 1}/{len(year_groups)})")
                
                # Collect monthly data arrays with memory management
                monthly_arrays = []
                valid_months = []
                alignment_failed = False
                
                for month_key, month_info in months_data.items():
                    data_array = month_info['data']
                    
                    # Validate alignment with optimized check
                    if not self.grid_manager.validate_alignment(data_array):
                        logging.error(f"Alignment validation failed for {month_key}")
                        alignment_failed = True
                        break
                    
                    monthly_arrays.append(data_array)
                    valid_months.append(month_key)
                
                if alignment_failed:
                    logging.error(f"Skipping {year_type} year {year_key} due to alignment failure")
                    continue
                
                if len(monthly_arrays) == 0:
                    logging.warning(f"No valid monthly data for {year_type} year {year_key}, skipping")
                    continue
                
                logging.info(f"  Computing mean from {len(monthly_arrays)} months: {', '.join(valid_months)}")
                
                # Optimized computation for large datasets
                try:
                    # Use chunked computation if arrays are large
                    if hasattr(monthly_arrays[0], 'chunks') and monthly_arrays[0].chunks:
                        logging.info("  Using chunked computation for large arrays")
                        
                        # Stack with optimal chunks
                        stacked = xr.concat(monthly_arrays, dim='month')
                        # Use Dask for computation
                        annual_mean = stacked.mean(dim='month', skipna=True, keep_attrs=True)
                        
                        # Load result to avoid keeping large computation graphs
                        if hasattr(annual_mean, 'load'):
                            annual_mean = annual_mean.load()
                    else:
                        # Regular computation for smaller arrays
                        stacked = xr.concat(monthly_arrays, dim='month')
                        annual_mean = stacked.mean(dim='month', skipna=True)
                    
                    # Preserve important attributes
                    if monthly_arrays[0].attrs:
                        annual_mean.attrs.update(monthly_arrays[0].attrs)
                    
                    # Clean up intermediate arrays to save memory
                    del stacked
                    del monthly_arrays
                    
                    annual_means[year_key] = {
                        'annual_mean': annual_mean,
                        'n_months': len(valid_months),
                        'valid_months': valid_months,
                        'year_type': year_type
                    }
                    
                    logging.info(f"  Annual mean computed successfully")
                    
                    # Force garbage collection between years for memory management
                    import gc
                    gc.collect()
                    
                except Exception as e:
                    logging.error(f"Failed annual mean computation for {year_type} year {year_key}: {str(e)}")
                    continue
                
            except Exception as e:
                logging.error(f"Failed to process {year_type} year {year_key}: {str(e)}")
                continue
        
        logging.info(f"Annual mean computation completed: {len(annual_means)} {year_type} years processed")
        return annual_means


class ReprojectionManager:
    """Handles reprojection and zonal statistics"""
    
    def __init__(self, sa1_manager: SA1DataManager):
        self.sa1_manager = sa1_manager
        
    def reproject_to_target_crs(self, annual_mean: xr.DataArray, year_key: str) -> xr.DataArray:
        """Reproject annual mean to target CRS (EPSG:3577)"""
        logging.info(f"Reprojecting {year_key} to {Config.TARGET_CRS}")
        
        try:
            # Set CRS if not already set
            if not hasattr(annual_mean, 'rio') or annual_mean.rio.crs is None:
                # Assume WGS84 if no CRS info
                annual_mean = annual_mean.rio.write_crs("EPSG:4326")
                logging.info("  No CRS found, assumed EPSG:4326")
            
            # Reproject to target CRS
            reprojected = annual_mean.rio.reproject(
                Config.TARGET_CRS,
                resampling=Resampling.bilinear
            )
            
            logging.info(f"  Reprojection completed: shape {reprojected.shape}")
            return reprojected
            
        except Exception as e:
            raise ProcessingError(f"Failed to reproject {year_key}: {str(e)}")
    
    def compute_zonal_statistics(self, reprojected_data: xr.DataArray, year_key: str) -> pd.DataFrame:
        """Compute zonal statistics for SA1 regions - ensures ALL regions are included"""
        logging.info(f"Computing zonal statistics for {year_key}")
        
        try:
            # Import rasterstats for zonal statistics
            try:
                from rasterstats import zonal_stats
            except ImportError:
                raise ProcessingError("rasterstats package is required for zonal statistics")
            
            # Get all SA1 codes first to ensure complete coverage
            all_sa1_codes = self.sa1_manager.sa1_reprojected[Config.SA1_KEY].tolist()
            n_total_regions = len(all_sa1_codes)
            
            logging.info(f"  Processing zonal statistics for {n_total_regions} SA1 regions")
            
            # Convert xarray to rasterio-compatible format
            if hasattr(reprojected_data, 'rio') and reprojected_data.rio.transform() is not None:
                try:
                    # Perform zonal statistics with proper parameters
                    stats_list = zonal_stats(
                        self.sa1_manager.sa1_reprojected.geometry,  # Use geometry directly
                        reprojected_data.values,
                        affine=reprojected_data.rio.transform(),
                        stats=['mean', 'count'],  # Include count to detect regions with no data
                        nodata=reprojected_data.rio.nodata if reprojected_data.rio.nodata is not None else np.nan,
                        all_touched=True  # Include pixels that touch polygon boundary
                    )
                    
                    # Create complete DataFrame ensuring all SA1 codes are present
                    zonal_results = []
                    for i, sa1_code in enumerate(all_sa1_codes):
                        if i < len(stats_list) and stats_list[i] is not None:
                            stat = stats_list[i]
                            # Check if we have valid data
                            if stat.get('count', 0) > 0 and stat.get('mean') is not None:
                                zonal_results.append({
                                    Config.SA1_KEY: sa1_code,
                                    'scpdsi_mean': stat['mean'],
                                    'pixel_count': stat['count'],
                                    'fill_method': 'zonal_mean'
                                })
                            else:
                                # No data in this region - will be filled by nearest
                                zonal_results.append({
                                    Config.SA1_KEY: sa1_code,
                                    'scpdsi_mean': np.nan,
                                    'pixel_count': 0,
                                    'fill_method': 'zonal_mean'
                                })
                        else:
                            # Region not covered by zonal_stats - will be filled by nearest
                            zonal_results.append({
                                Config.SA1_KEY: sa1_code,
                                'scpdsi_mean': np.nan,
                                'pixel_count': 0,
                                'fill_method': 'zonal_mean'
                            })
                    
                    zonal_df = pd.DataFrame(zonal_results)
                    
                    # Log statistics about zonal computation
                    valid_zonal = zonal_df['scpdsi_mean'].notna().sum()
                    logging.info(f"  Zonal statistics: {valid_zonal}/{n_total_regions} regions have valid zonal means")
                    
                    if valid_zonal == 0:
                        logging.warning("  No regions have valid zonal means - all will use nearest values")
                    
                except Exception as e:
                    logging.warning(f"  Zonal statistics failed ({str(e)}), using fallback approach")
                    # Fallback: create complete DataFrame with NaN values
                    zonal_df = pd.DataFrame({
                        Config.SA1_KEY: all_sa1_codes,
                        'scpdsi_mean': [np.nan] * n_total_regions,
                        'pixel_count': [0] * n_total_regions,
                        'fill_method': ['zonal_mean'] * n_total_regions
                    })
            
            else:
                logging.warning("  No rio accessor or transform found, using fallback approach")
                # Fallback approach - create complete DataFrame
                zonal_df = pd.DataFrame({
                    Config.SA1_KEY: all_sa1_codes,
                    'scpdsi_mean': [np.nan] * n_total_regions,
                    'pixel_count': [0] * n_total_regions,
                    'fill_method': ['zonal_mean'] * n_total_regions
                })
            
            # Verify we have all regions
            if len(zonal_df) != n_total_regions:
                raise ProcessingError(f"Zonal statistics missing regions: expected {n_total_regions}, got {len(zonal_df)}")
            
            logging.info(f"  Zonal statistics completed for {len(zonal_df)} regions")
            return zonal_df
            
        except Exception as e:
            raise ProcessingError(f"Failed to compute zonal statistics for {year_key}: {str(e)}")
    
    def compute_nearest_values(self, reprojected_data: xr.DataArray, year_key: str) -> pd.DataFrame:
        """Compute nearest-cell values at SA1 centroids - ensures ALL regions are included"""
        logging.info(f"Computing nearest values for {year_key}")
        
        try:
            # Get all SA1 codes and centroids to ensure complete coverage
            centroids = self.sa1_manager.sa1_centroids
            all_sa1_codes = centroids[Config.SA1_KEY].tolist()
            n_total_regions = len(all_sa1_codes)
            
            logging.info(f"  Processing nearest values for {n_total_regions} SA1 centroids")
            
            # Extract point values at centroids
            if hasattr(reprojected_data, 'rio') and reprojected_data.rio.crs is not None:
                try:
                    # Get centroid coordinates
                    x_coords = centroids.geometry.x.values
                    y_coords = centroids.geometry.y.values
                    
                    # Sample values at points with better error handling
                    sampled_values = []
                    successful_samples = 0
                    
                    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                        try:
                            # Find nearest grid cell with bounds checking
                            if (hasattr(reprojected_data, 'x') and hasattr(reprojected_data, 'y') and
                                len(reprojected_data.x) > 0 and len(reprojected_data.y) > 0):
                                
                                # Check if point is within raster bounds (with some tolerance)
                                x_min, x_max = float(reprojected_data.x.min()), float(reprojected_data.x.max())
                                y_min, y_max = float(reprojected_data.y.min()), float(reprojected_data.y.max())
                                
                                # Allow some tolerance for coordinates near edges
                                tolerance = max(abs(x_max - x_min), abs(y_max - y_min)) * 0.01
                                
                                if (x_min - tolerance <= x <= x_max + tolerance and 
                                    y_min - tolerance <= y <= y_max + tolerance):
                                    
                                    value = reprojected_data.sel(
                                        x=x, y=y, 
                                        method='nearest'
                                    ).item()
                                    
                                    # Check if value is valid (not NaN or nodata)
                                    if np.isfinite(value):
                                        sampled_values.append(float(value))
                                        successful_samples += 1
                                    else:
                                        sampled_values.append(np.nan)
                                else:
                                    # Point outside raster bounds
                                    sampled_values.append(np.nan)
                            else:
                                sampled_values.append(np.nan)
                                
                        except Exception as e:
                            logging.debug(f"    Failed to sample at point {i} ({x:.2f}, {y:.2f}): {str(e)}")
                            sampled_values.append(np.nan)
                    
                    # Ensure we have values for all regions
                    if len(sampled_values) != n_total_regions:
                        logging.warning(f"  Mismatch in sampled values: expected {n_total_regions}, got {len(sampled_values)}")
                        # Pad or trim to match
                        while len(sampled_values) < n_total_regions:
                            sampled_values.append(np.nan)
                        sampled_values = sampled_values[:n_total_regions]
                    
                    nearest_df = pd.DataFrame({
                        Config.SA1_KEY: all_sa1_codes,
                        'scpdsi_nearest': sampled_values
                    })
                    
                    logging.info(f"  Nearest sampling: {successful_samples}/{n_total_regions} successful samples")
                    
                    if successful_samples == 0:
                        logging.warning("  No successful nearest samples - all centroids outside raster or invalid data")
                    
                except Exception as e:
                    logging.warning(f"  Nearest sampling failed ({str(e)}), using fallback approach")
                    # Fallback: create DataFrame with NaN values for all regions
                    nearest_df = pd.DataFrame({
                        Config.SA1_KEY: all_sa1_codes,
                        'scpdsi_nearest': [np.nan] * n_total_regions
                    })
            
            else:
                logging.warning("  No rio accessor or CRS found, using fallback approach")
                # Fallback approach - create complete DataFrame
                nearest_df = pd.DataFrame({
                    Config.SA1_KEY: all_sa1_codes,
                    'scpdsi_nearest': [np.nan] * n_total_regions
                })
            
            # Verify we have all regions
            if len(nearest_df) != n_total_regions:
                raise ProcessingError(f"Nearest values missing regions: expected {n_total_regions}, got {len(nearest_df)}")
            
            logging.info(f"  Nearest values completed for {len(nearest_df)} regions")
            return nearest_df
            
        except Exception as e:
            raise ProcessingError(f"Failed to compute nearest values for {year_key}: {str(e)}")
    
    def merge_statistics(self, zonal_df: pd.DataFrame, nearest_df: pd.DataFrame) -> pd.DataFrame:
        """Merge zonal and nearest statistics, filling missing values"""
        logging.info("Merging zonal and nearest statistics...")
        
        try:
            # Merge dataframes
            merged_df = pd.merge(
                zonal_df, nearest_df,
                on=Config.SA1_KEY,
                how='outer'
            )
            
            # Fill missing zonal means with nearest values
            missing_zonal = merged_df['scpdsi_mean'].isna()
            if missing_zonal.any():
                n_filled = missing_zonal.sum()
                logging.info(f"  Filling {n_filled} missing zonal means with nearest values")
                
                merged_df.loc[missing_zonal, 'scpdsi_mean'] = merged_df.loc[missing_zonal, 'scpdsi_nearest']
                merged_df.loc[missing_zonal, 'fill_method'] = 'nearest_cell'
            
            # Sort by SA1_CODE21
            merged_df = merged_df.sort_values(Config.SA1_KEY).reset_index(drop=True)
            
            logging.info(f"  Merged statistics: {len(merged_df)} regions")
            return merged_df
            
        except Exception as e:
            raise ProcessingError(f"Failed to merge statistics: {str(e)}")


class ConsistencyChecker:
    """Handles region count consistency validation"""
    
    def __init__(self, sa1_manager: SA1DataManager):
        self.sa1_manager = sa1_manager
        
    def validate_consistency(self, export_df: pd.DataFrame, year_key: str, year_type: str) -> Tuple[bool, ValidationReport]:
        """Validate region count consistency between SA1 and export data"""
        logging.info(f"Validating region count consistency for {year_type} year {year_key}")
        
        # Create validation report
        report = ValidationReport(year_key, year_type)
        
        try:
            # Get unique SA1 codes from both datasets
            shape_codes = set(self.sa1_manager.sa1_reprojected[Config.SA1_KEY].unique())
            export_codes = set(export_df[Config.SA1_KEY].unique())
            
            report.n_shape = len(shape_codes)
            report.n_excel = len(export_codes)
            
            # Calculate relative difference
            if report.n_shape > 0:
                report.relative_diff = abs(report.n_excel - report.n_shape) / report.n_shape
            else:
                report.relative_diff = float('inf')
            
            # Find missing and extra IDs
            report.missing_in_export = sorted(list(shape_codes - export_codes))
            report.extra_in_export = sorted(list(export_codes - shape_codes))
            
            # Log counts
            logging.info(f"  N_shape: {report.n_shape}")
            logging.info(f"  N_excel: {report.n_excel}")
            logging.info(f"  Relative difference: {report.relative_diff:.4%}")
            logging.info(f"  Missing in export: {len(report.missing_in_export)}")
            logging.info(f"  Extra in export: {len(report.extra_in_export)}")
            
            # Check against threshold
            is_valid = report.relative_diff <= Config.CONSISTENCY_THRESHOLD
            
            if is_valid:
                logging.info(f"✓ Consistency check PASSED (≤{Config.CONSISTENCY_THRESHOLD:.1%})")
            else:
                logging.error(f"✗ Consistency check FAILED (>{Config.CONSISTENCY_THRESHOLD:.1%})")
                
                # Log details about discrepancies
                if report.missing_in_export:
                    logging.error(f"  Missing IDs (first 10): {report.missing_in_export[:10]}")
                if report.extra_in_export:
                    logging.error(f"  Extra IDs (first 10): {report.extra_in_export[:10]}")
            
            return is_valid, report
            
        except Exception as e:
            logging.error(f"Consistency validation failed: {str(e)}")
            report.relative_diff = float('inf')
            return False, report


class ExcelExporter:
    """Handles Excel file export functionality"""
    
    def __init__(self):
        self.output_dir = Path(Config.OUTPUT_DIR)
        
    def export_to_excel(self, df: pd.DataFrame, year_key: str, year_type: str) -> Path:
        """Export dataframe to Excel file"""
        
        # Generate filename
        if year_type == "calendar":
            filename = f"calendar_SA1_{year_key}.xlsx"
        elif year_type == "financial":
            filename = f"financial_SA1_{year_key}.xlsx"
        else:
            raise ProcessingError(f"Unknown year type: {year_type}")
        
        output_path = self.output_dir / filename
        
        logging.info(f"Exporting to Excel: {output_path}")
        
        try:
            # Prepare final dataframe
            export_df = df.copy()
            
            # Ensure required columns are present
            required_cols = [Config.SA1_KEY, 'scpdsi_mean', 'fill_method']
            for col in required_cols:
                if col not in export_df.columns:
                    raise ProcessingError(f"Required column missing: {col}")
            
            # Add nearest values if configured
            if Config.INCLUDE_NEAREST and 'scpdsi_nearest' not in export_df.columns:
                logging.warning("scpdsi_nearest column not found, adding NaN values")
                export_df['scpdsi_nearest'] = np.nan
            
            # Select and order columns
            if Config.INCLUDE_NEAREST:
                columns = [Config.SA1_KEY, 'scpdsi_mean', 'scpdsi_nearest', 'fill_method']
            else:
                columns = [Config.SA1_KEY, 'scpdsi_mean', 'fill_method']
            
            export_df = export_df[columns]
            
            # Sort by SA1_CODE21
            export_df = export_df.sort_values(Config.SA1_KEY).reset_index(drop=True)
            
            # Export to Excel
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                export_df.to_excel(
                    writer,
                    sheet_name='SA1_scPDSI',
                    index=False,
                    encoding=Config.EXPORT_ENCODING
                )
            
            logging.info(f"  Excel export completed: {len(export_df)} records")
            logging.info(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
            
            return output_path
            
        except Exception as e:
            raise ProcessingError(f"Failed to export Excel file: {str(e)}")
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        temp_dir = Path(Config.TEMP_DIR)
        if temp_dir.exists():
            try:
                temp_files = list(temp_dir.glob('*'))
                for temp_file in temp_files:
                    if temp_file.is_file():
                        temp_file.unlink()
                logging.info(f"Cleaned up {len(temp_files)} temporary files")
            except Exception as e:
                logging.warning(f"Failed to cleanup temp files: {str(e)}")


class ScPDSIProcessor:
    """Main processor class that orchestrates the entire workflow"""
    
    def __init__(self):
        self.sa1_manager = SA1DataManager()
        self.netcdf_manager = NetCDFManager()
        self.grid_manager = GridManager()
        self.reprojection_manager = ReprojectionManager(self.sa1_manager)
        self.consistency_checker = ConsistencyChecker(self.sa1_manager)
        self.excel_exporter = ExcelExporter()
        
    def run(self):
        """Execute the complete processing workflow - memory optimized"""
        logging.info("Starting scPDSI processing workflow...")
        
        # Track processing statistics
        start_time = time.time()
        processed_years = {'calendar': 0, 'financial': 0}
        failed_years = {'calendar': 0, 'financial': 0}
        
        try:
            # Step 1: Load and prepare SA1 data
            logging.info("Step 1: Loading SA1 data...")
            step_start = time.time()
            
            self.sa1_manager.load_sa1_data()
            self.sa1_manager.reproject_to_target_crs()
            self.sa1_manager.create_centroids()
            
            step_duration = time.time() - step_start
            logging.info(f"Step 1 completed in {step_duration:.2f} seconds")
            
            # Step 2: Read NetCDF data
            logging.info("Step 2: Reading NetCDF data...")
            step_start = time.time()
            
            self.netcdf_manager.discover_netcdf_files()
            monthly_data = self.netcdf_manager.read_monthly_data()
            
            step_duration = time.time() - step_start
            logging.info(f"Step 2 completed in {step_duration:.2f} seconds")
            
            # Step 3: Create template grid from first valid slice
            logging.info("Step 3: Creating template grid...")
            step_start = time.time()
            
            first_data = next(iter(monthly_data.values()))['data']
            self.grid_manager.create_template_from_first_slice(first_data)
            
            step_duration = time.time() - step_start
            logging.info(f"Step 3 completed in {step_duration:.2f} seconds")
            
            # Step 4: Group monthly data by years
            logging.info("Step 4: Grouping by years...")
            step_start = time.time()
            
            calendar_years, financial_years = self.netcdf_manager.group_by_years()
            
            step_duration = time.time() - step_start
            logging.info(f"Step 4 completed in {step_duration:.2f} seconds")
            
            # Step 5: Process each year type with progress tracking
            year_type_pairs = [("calendar", calendar_years), ("financial", financial_years)]
            
            for year_type, year_groups in year_type_pairs:
                if not year_groups:
                    logging.info(f"No {year_type} years to process")
                    continue
                    
                logging.info(f"Step 5: Processing {len(year_groups)} {year_type} years...")
                year_type_start = time.time()
                
                # Compute annual means with memory management
                aggregator = AnnualAggregator(self.grid_manager)
                annual_means = aggregator.compute_annual_means(year_groups, year_type)
                
                # Process each year individually to manage memory
                for year_idx, (year_key, year_data) in enumerate(annual_means.items()):
                    year_start = time.time()
                    
                    try:
                        self.process_single_year(year_key, year_data, year_type)
                        processed_years[year_type] += 1
                        
                        year_duration = time.time() - year_start
                        logging.info(f"  Completed {year_type} {year_key} in {year_duration:.2f}s "
                                   f"({year_idx + 1}/{len(annual_means)})")
                        
                    except Exception as e:
                        failed_years[year_type] += 1
                        logging.error(f"  Failed {year_type} {year_key}: {str(e)}")
                        continue
                
                # Clean up annual means to free memory
                del annual_means
                del aggregator
                import gc
                gc.collect()
                
                year_type_duration = time.time() - year_type_start
                logging.info(f"Completed {year_type} processing in {year_type_duration:.2f} seconds")
            
            # Step 6: Cleanup
            logging.info("Step 6: Cleanup...")
            self.excel_exporter.cleanup_temp_files()
            
            # Final summary
            total_duration = time.time() - start_time
            total_processed = sum(processed_years.values())
            total_failed = sum(failed_years.values())
            
            logging.info("="*80)
            logging.info("PROCESSING COMPLETED SUCCESSFULLY!")
            logging.info(f"Total processing time: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)")
            logging.info(f"Years processed: {total_processed}")
            logging.info(f"Years failed: {total_failed}")
            logging.info(f"Calendar years: {processed_years['calendar']} processed, {failed_years['calendar']} failed")
            logging.info(f"Financial years: {processed_years['financial']} processed, {failed_years['financial']} failed")
            logging.info("="*80)
            
        except Exception as e:
            total_duration = time.time() - start_time
            logging.error(f"Processing workflow failed after {total_duration:.2f} seconds: {str(e)}", exc_info=True)
            raise
    
    def process_single_year(self, year_key: str, year_data: Dict, year_type: str):
        """Process a single year (calendar or financial) - with comprehensive error handling"""
        logging.info(f"Processing {year_type} year {year_key}...")
        
        # Track processing steps for better error reporting
        completed_steps = []
        temp_files = []
        
        try:
            annual_mean = year_data['annual_mean']
            
            # Step 1: Reproject to target CRS
            step_start = time.time()
            try:
                reprojected = self.reprojection_manager.reproject_to_target_crs(annual_mean, year_key)
                completed_steps.append("reprojection")
                step_duration = time.time() - step_start
                logging.info(f"  Reprojection completed in {step_duration:.2f}s")
            except Exception as e:
                raise ProcessingError(f"Reprojection failed: {str(e)}")
            
            # Step 2: Compute zonal statistics
            step_start = time.time()
            try:
                zonal_df = self.reprojection_manager.compute_zonal_statistics(reprojected, year_key)
                completed_steps.append("zonal_statistics")
                step_duration = time.time() - step_start
                logging.info(f"  Zonal statistics completed in {step_duration:.2f}s")
            except Exception as e:
                logging.error(f"Zonal statistics failed, creating fallback: {str(e)}")
                # Create fallback zonal dataframe with all SA1 codes
                all_sa1_codes = self.sa1_manager.sa1_reprojected[Config.SA1_KEY].tolist()
                zonal_df = pd.DataFrame({
                    Config.SA1_KEY: all_sa1_codes,
                    'scpdsi_mean': [np.nan] * len(all_sa1_codes),
                    'pixel_count': [0] * len(all_sa1_codes),
                    'fill_method': ['fallback'] * len(all_sa1_codes)
                })
                completed_steps.append("zonal_statistics_fallback")
            
            # Step 3: Compute nearest values
            step_start = time.time()
            try:
                nearest_df = self.reprojection_manager.compute_nearest_values(reprojected, year_key)
                completed_steps.append("nearest_values")
                step_duration = time.time() - step_start
                logging.info(f"  Nearest values completed in {step_duration:.2f}s")
            except Exception as e:
                logging.error(f"Nearest values failed, creating fallback: {str(e)}")
                # Create fallback nearest dataframe with all SA1 codes
                all_sa1_codes = self.sa1_manager.sa1_centroids[Config.SA1_KEY].tolist()
                nearest_df = pd.DataFrame({
                    Config.SA1_KEY: all_sa1_codes,
                    'scpdsi_nearest': [np.nan] * len(all_sa1_codes)
                })
                completed_steps.append("nearest_values_fallback")
            
            # Step 4: Merge statistics with enhanced validation
            step_start = time.time()
            try:
                merged_df = self.reprojection_manager.merge_statistics(zonal_df, nearest_df)
                completed_steps.append("merge_statistics")
                step_duration = time.time() - step_start
                logging.info(f"  Statistics merged in {step_duration:.2f}s")
                
                # Enhanced data validation
                expected_regions = len(self.sa1_manager.sa1_reprojected)
                actual_regions = len(merged_df)
                
                if actual_regions != expected_regions:
                    logging.warning(f"Region count mismatch: expected {expected_regions}, got {actual_regions}")
                    # Attempt to fix by ensuring all SA1 codes are present
                    all_sa1_codes = self.sa1_manager.sa1_reprojected[Config.SA1_KEY].tolist()
                    missing_codes = set(all_sa1_codes) - set(merged_df[Config.SA1_KEY])
                    
                    if missing_codes:
                        logging.info(f"Adding {len(missing_codes)} missing SA1 regions with NaN values")
                        missing_rows = []
                        for code in missing_codes:
                            missing_rows.append({
                                Config.SA1_KEY: code,
                                'scpdsi_mean': np.nan,
                                'pixel_count': 0,
                                'scpdsi_nearest': np.nan,
                                'fill_method': 'missing_region'
                            })
                        
                        missing_df = pd.DataFrame(missing_rows)
                        merged_df = pd.concat([merged_df, missing_df], ignore_index=True)
                        merged_df = merged_df.sort_values(Config.SA1_KEY).reset_index(drop=True)
                
            except Exception as e:
                raise ProcessingError(f"Statistics merge failed: {str(e)}")
            
            # Step 5: Validate consistency
            step_start = time.time()
            try:
                is_valid, report = self.consistency_checker.validate_consistency(merged_df, year_key, year_type)
                completed_steps.append("consistency_check")
                step_duration = time.time() - step_start
                logging.info(f"  Consistency check completed in {step_duration:.2f}s")
            except Exception as e:
                logging.error(f"Consistency check failed: {str(e)}")
                # Create dummy report and proceed with export (with warning)
                is_valid = False
                report = ValidationReport(year_key, year_type)
                report.relative_diff = float('inf')
                completed_steps.append("consistency_check_failed")
            
            # Step 6: Export or save validation report
            step_start = time.time()
            if is_valid:
                try:
                    output_path = self.excel_exporter.export_to_excel(merged_df, year_key, year_type)
                    completed_steps.append("excel_export")
                    step_duration = time.time() - step_start
                    logging.info(f"✓ Successfully exported {year_type} year {year_key} in {step_duration:.2f}s: {output_path}")
                    
                    # Log data quality summary
                    valid_zonal = merged_df['scpdsi_mean'].notna().sum()
                    valid_nearest = merged_df['scpdsi_nearest'].notna().sum() if 'scpdsi_nearest' in merged_df.columns else 0
                    logging.info(f"  Data quality: {valid_zonal}/{len(merged_df)} valid zonal, {valid_nearest}/{len(merged_df)} valid nearest")
                    
                except Exception as e:
                    raise ProcessingError(f"Excel export failed: {str(e)}")
                
            else:
                # Save validation report but don't fail the year completely
                try:
                    report_filename = f"validation_report_{year_type}_{year_key}.json"
                    report_path = Path(Config.LOG_DIR) / report_filename
                    report.save_report(report_path)
                    completed_steps.append("validation_report_saved")
                    
                    logging.warning(f"⚠ Consistency check failed for {year_type} year {year_key}")
                    logging.warning(f"  Validation report saved: {report_path}")
                    
                    # Option: still export with warning (configurable behavior)
                    if hasattr(Config, 'EXPORT_ON_CONSISTENCY_FAIL') and Config.EXPORT_ON_CONSISTENCY_FAIL:
                        try:
                            output_path = self.excel_exporter.export_to_excel(merged_df, year_key, year_type)
                            logging.warning(f"⚠ Exported {year_type} year {year_key} despite consistency failure: {output_path}")
                            completed_steps.append("excel_export_with_warning")
                        except Exception as e:
                            logging.error(f"Excel export failed even with warning: {str(e)}")
                    else:
                        raise ProcessingError(f"Consistency check failed for {year_type} year {year_key}")
                        
                except Exception as e:
                    logging.error(f"Failed to save validation report: {str(e)}")
                    raise ProcessingError(f"Consistency check and report saving failed for {year_type} year {year_key}")
            
            # Clean up temporary variables
            del reprojected
            del zonal_df
            del nearest_df  
            del merged_df
            
        except Exception as e:
            # Enhanced error reporting
            logging.error(f"Failed to process {year_type} year {year_key} after completing steps: {completed_steps}")
            logging.error(f"Error details: {str(e)}")
            
            # Clean up any temporary files if tracking them
            for temp_file in temp_files:
                try:
                    if Path(temp_file).exists():
                        Path(temp_file).unlink()
                except:
                    pass
            
            raise ProcessingError(f"Processing failed for {year_type} year {year_key}: {str(e)}")


if __name__ == "__main__":
    main()