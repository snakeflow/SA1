Project Requirements — Open-Source Python Stack (rasterio / xarray / rioxarray)
Requirements only. No implementation details or specific function names.

1) Objective
Generate annual scPDSI statistics for Australian SA1 regions from monthly NetCDF inputs, producing two parallel outputs:

Calendar year (Jan–Dec) means

Financial year (Jul–Jun) means, labeled as YYYY-YY (e.g., 2020-21)

Before exporting, enforce a region count consistency check between the SA1 shapefile and the to-be-exported table.

2) Data Scope & Definitions
Indicator: scPDSI (monthly)

Spatial unit: SA1 2021 regions (Australia)

Primary key: SA1_CODE21

Coordinate reference system: GDA94 / Australian Albers (EPSG:3577)

Complete-year rule: A year (calendar or financial) is eligible only if it contains 12 monthly slices. Otherwise, skip and log the reason.

3) Inputs & Assumptions
NetCDF files containing:

Variable: scpdsi

Dimensions: longitude (x), latitude (y), and time

Time axis may use varying units/calendars; must be robustly parsed into actual dates.

SA1 vector data (SA1_2021 in GDA2020) with SA1_CODE21 as the unique identifier.

Environment: Python 3.x with rasterio, xarray, rioxarray, GeoPandas, PyProj, pandas, numpy; optional Dask for chunked/lazy computation.

4) Outputs
Directory: annual_results_SA1/

Per-year Excel files (UTF-8, no index):

Calendar: calendar_SA1_YYYY.xlsx

Financial: financial_SA1_YYYY-YY.xlsx

Columns (recommended):

SA1_CODE21

scpdsi_mean — SA1 annual mean (zonal statistic or filled)

scpdsi_nearest — nearest-cell value at the SA1 centroid (for fill and reference)

fill_method — zonal_mean or nearest_cell

5) Processing Workflow (Requirements Level)
Initialization

Define input/output/work/temp/log directories, variable/dimension names, CRS, and toggles (e.g., complete-year requirement).

Enable detailed logging (file + console).

Read & Prepare SA1

Read SA1 polygons and reproject to EPSG:3577.

Create centroids for all SA1 polygons for nearest-cell sampling.

Validate uniqueness and completeness of SA1_CODE21.

Read NetCDF & Parse Time

Iterate all NetCDF files; parse the time axis into dates robustly (handling units/calendar variations).

For each monthly slice, associate:

Calendar year label: YYYY

Financial year label: YYYY-YY (Jul–Jun)

Template Grid & Alignment

Derive a template grid (extent, resolution, transform) from the first successfully read monthly slice.

All subsequent rasters (including annual means) must be pixel-aligned to this template.

If alignment cannot be guaranteed, abort that year and log as a fatal error.

Annual Aggregation

Group monthly slices by calendar year and by financial year.

For each group with 12 months, compute the annual mean raster over valid pixels only.

Reprojection

Reproject each annual mean to EPSG:3577 and resample to the template grid parameters so that extent, resolution, and alignment are identical to the template.

Zonal Statistics & Nearest Fallback

Compute zonal means for SA1 polygons over the annual mean raster.

Compute nearest-cell values at SA1 centroids over the annual mean raster.

Join both results by SA1_CODE21. For any missing zonal mean, fill with the nearest value and set fill_method accordingly.

Region Count Consistency Check (Mandatory)

Let N_shape = count of unique SA1_CODE21 in the (reprojected) SA1 dataset.

Let N_excel = count of unique SA1_CODE21 in the table prepared for export.

Compute relative difference: abs(N_excel - N_shape) / N_shape.

Threshold: If the relative difference > 1%:

Do not export the Excel for that year.

Record an error with:

the computed percentage,

IDs missing in export (present in shapefile; absent in export),

IDs extra in export (absent in shapefile; present in export).

Exit with a non-success status or raise an exception for that year.

If the difference ≤ 1%:

Proceed to export,

Log the measured difference and a note on alignment.

Export & Cleanup

Sort by SA1_CODE21 and export Excel.

Remove transient rasters and intermediates; optionally retain QA artifacts (see §10).

6) Data Quality & Error Handling
Time axis tolerance: Skip unparseable time steps; log original value and index.

Complete-year enforcement: If a group has <12 months, skip that year; log available months.

Alignment rule: All annual rasters must match the template grid exactly; on mismatch, abort that year.

NoData policy: Means and zonal statistics must ignore invalid pixels.

Consistency artifacts (when threshold is exceeded):

missing_in_export: IDs in SA1 but not in the table.

extra_in_export: IDs in the table but not in SA1.

Severity levels:

Fatal: alignment failure; region difference >1%; unwritable output path.

Recoverable: some time steps skipped; missing zonal mean filled by nearest value.

7) Performance & Scale
Optional chunked/lazy computation (e.g., with Dask):

Configure chunk sizes,

Avoid loading full time series into memory at once,

Pipeline I/O to minimize repeated reprojection.

Prefer streaming: read → compute → write, avoiding large in-memory accumulations.

8) Configurable Parameters
Variable name: scpdsi

Dimension names: longitude, latitude, time

Region key: SA1_CODE21

Target CRS: EPSG:3577

Complete-year requirement: on/off (default: on)

Consistency threshold: default 1% (must not exceed 2% by policy)

Export fields: include/exclude scpdsi_nearest

Template grid source: first valid slice vs. pre-defined template

9) Suggested Directory Layout
perl
Copy
Edit
<project_root>/
  original_dataset/      # NetCDF sources (*.nc)
  temp/                  # Intermediates (optional retention)
  annual_results_SA1/    # Yearly Excel outputs
  geo/                   # SA1 vector data
  logs/                  # Run logs and validation reports
10) Logging & Reports
Log file must include:

processed files, time indices, skipped reasons,

grid parameters (extent, resolution, transform),

counts for N_shape, N_excel, and the relative difference,

acceptance or rejection note per year.

On consistency failure (>1%), produce a validation report per affected year (CSV/JSON) with the difference and ID lists.

11) Acceptance Criteria
For every eligible year (calendar and financial), exactly one Excel is produced.

Each export contains all SA1 records with correct fill_method semantics.

All annual rasters are in EPSG:3577 and perfectly pixel-aligned to the template grid.

Region count consistency satisfied: abs(N_excel - N_shape) / N_shape ≤ 1%.
If exceeded, the year is not exported and an error + ID lists are produced.

Logs are complete and support full traceability.
