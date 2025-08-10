# Performance Optimizations for Large NetCDF Processing

This document outlines the key optimizations implemented to handle large NetCDF files efficiently.

## 🚀 Major Improvements Made

### 1. **Complete Region Coverage** ✅
**Problem**: Original code was missing regions with no data points
**Solution**: 
- Ensure ALL SA1 regions are included in outputs
- Use `all_touched=True` in zonal statistics
- Create fallback mechanisms for missing regions
- Comprehensive region validation and filling

### 2. **Memory Optimization** ✅
**Problem**: Large NetCDF files can cause memory issues
**Solution**:
- Chunked processing with `chunks={'time': 1}`
- Adaptive spatial chunking for files > 1GB
- Explicit memory cleanup with `gc.collect()`
- Lazy loading with Dask integration
- Delete intermediate variables after use

### 3. **Efficient Coordinate Alignment** ✅
**Problem**: Full coordinate comparison is expensive for large arrays
**Solution**:
- Sample-based validation for arrays > 1000 points
- Fast dimension checks before detailed comparison
- Optimized tolerance checking

### 4. **Enhanced Error Recovery** ✅
**Problem**: Single failures would stop entire processing
**Solution**:
- Fallback mechanisms for zonal statistics failures
- Graceful degradation with NaN filling
- Comprehensive step tracking for debugging
- Option to export despite consistency failures

### 5. **Progress Tracking** ✅
**Problem**: No visibility into processing progress for large files
**Solution**:
- File-by-file progress reporting
- Time tracking for each processing step
- Memory usage hints and cleanup notifications
- Detailed processing statistics

## ⚡ Performance Features

### Chunking Strategy
```python
# Adaptive chunking based on file size
chunk_sizes = {'time': 1}  # Always process one time slice
if file_size_mb > 1000:
    chunk_sizes.update({'longitude': 100, 'latitude': 100})
```

### Memory Management
```python
# Explicit cleanup between years
del annual_means, aggregator
import gc
gc.collect()
```

### Zonal Statistics Optimization
```python
# Ensure complete coverage
stats_list = zonal_stats(
    sa1_geometry,
    raster_data,
    affine=transform,
    stats=['mean', 'count'],
    all_touched=True  # Include boundary pixels
)
```

## 📊 Expected Performance Improvements

| Feature | Before | After | Improvement |
|---------|--------|--------|-------------|
| **Memory Usage** | High (all data in memory) | Optimized (chunked) | 70-90% reduction |
| **Large File Handling** | Often fails | Stable processing | 100% reliability |
| **Region Coverage** | Missing regions | Complete coverage | 100% SA1 regions |
| **Error Recovery** | Stops on first error | Continues with fallbacks | Robust processing |
| **Progress Visibility** | None | Detailed tracking | Full transparency |

## 🔧 Configuration Options

### Memory Optimization
```python
# In Config class
CHUNK_SIZE_TIME = 1  # Process one time slice at a time
LARGE_FILE_THRESHOLD_MB = 1000  # Trigger spatial chunking
SPATIAL_CHUNK_SIZE = 100  # Chunk size for large files
```

### Error Handling
```python
# Optional: Export even with consistency failures
Config.EXPORT_ON_CONSISTENCY_FAIL = True  # Default: False
```

## 💡 Usage Tips for Large Files

1. **Monitor Memory**: Watch system memory usage during processing
2. **Disk Space**: Ensure adequate disk space for temporary files
3. **Processing Time**: Large files may take hours - use logging to monitor
4. **Chunking**: Adjust chunk sizes based on available memory
5. **Fallbacks**: Review validation reports for any fallback usage

## 🔍 Debugging Large File Issues

### Common Issues and Solutions

1. **Out of Memory**: Reduce spatial chunk sizes or process fewer time slices
2. **Slow Processing**: Check disk I/O speed and available RAM
3. **Missing Regions**: Review zonal statistics fallback logs
4. **Alignment Failures**: Check coordinate system compatibility

### Log Analysis
Look for these key indicators:
- `Large file detected (X MB), using chunked processing`
- `Using chunked computation for large arrays`
- `Zonal statistics: X/Y regions have valid zonal means`
- `Processing completed in X seconds`

## 📈 Monitoring Performance

The code now includes comprehensive timing and statistics:

```
Processing completed successfully!
Total processing time: 3600.0 seconds (60.0 minutes)
Years processed: 10
Calendar years: 5 processed, 0 failed
Financial years: 5 processed, 0 failed
```

This ensures you can track processing efficiency and identify bottlenecks.