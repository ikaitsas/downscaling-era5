#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 15:55:29 2025

@author: ykaitsas

"""

import os
import gc
import dask
from numpy import ceil as numpy_ceil
from preprocessing_tabular_functions import add_time_features
from dask.diagnostics import ProgressBar, ResourceProfiler


def save_partition(df, filename, engine="pyarrow", write_index=False):
    
    # df will be a pandas DataFrame when this runs
    df.to_parquet(filename, engine=engine, index=write_index)
    return filename

def save_ddf_via_delayed(ddf, outdir, engine="pyarrow", write_index=False, sequential=True,
                         show_progress=True, resource_profiler=False):
    
    os.makedirs(outdir, exist_ok=True)

    delayed_parts = ddf.to_delayed()  # list of delayed pandas DataFrames
    writes = []
    for i, part in enumerate(delayed_parts):
        fn = os.path.join(outdir, f"part-{i:05d}.parquet")
        writes.append(dask.delayed(save_partition)(part, fn, engine, write_index))

    # choose contexts
    pbar_ctx = ProgressBar() if show_progress else None
    rp_ctx = ResourceProfiler() if resource_profiler else None

    if sequential:
        if pbar_ctx:
            pbar_ctx.__enter__()
        if rp_ctx:
            rp_ctx.__enter__()
        try:
            for w in writes:
                w.compute()
        finally:
            if rp_ctx:
                rp_ctx.__exit__(None, None, None)
            if pbar_ctx:
                pbar_ctx.__exit__(None, None, None)
    else:
        # schedule all writes together
        if rp_ctx:
            rp_ctx.__enter__()
        try:
            if pbar_ctx:
                with pbar_ctx:
                    dask.compute(*writes)
            else:
                dask.compute(*writes)
        finally:
            if rp_ctx:
                rp_ctx.__exit__(None, None, None)


# ------------------------------------------------------------------
# Batch-processing driver: slice dataset by time and write each batch
# ------------------------------------------------------------------
def process_and_save_in_time_batches(
    ds,                 # merged ds (pl_wide + sl) or ds_stacked? We expect unstacked ds
    out_base_dir,
    time_dim="valid_time",
    time_batch_size=500,
    sample_chunk_static=200_000,
    dynamic_chunk_size=False,
    n_target_partitions=20,
    parquet_engine="pyarrow",
    write_index=False,
    sequential_writes=True,
    cast_to_float32=False,
):
    """
    Process ds in small time batches to avoid building one giant graph.
    - ds: merged xarray Dataset (NOT yet stacked or wide)
    - out_base_dir: parent directory where batch parquet subfolders will be created
    """
    os.makedirs(out_base_dir, exist_ok=True)

    total_time = ds.sizes[time_dim]
    print(f"Total {time_dim} steps: {total_time}. Batch size: {time_batch_size}")

    for start in range(0, total_time, time_batch_size):
        stop = min(start + time_batch_size, total_time)
        print(f"\nProcessing time slice: {start} .. {stop-1} (size {stop-start})")

        # 1) take time slice (this is lazy)
        ds_slice = ds.isel({time_dim: slice(start, stop)})

        # 2) Optionally cast to float32 early to reduce memory & graph cost
        if cast_to_float32:
            for var in ds_slice.data_vars:
                # only cast numeric variables; xarray will handle dtype
                try:
                    ds_slice[var] = ds_slice[var].astype("float32")
                except Exception:
                    pass

        # 3) stack and chunk for this small slice
        ds_slice_stacked = ds_slice.stack(sample=("valid_time", "latitude", "longitude"))
        if dynamic_chunk_size:
            chunk_size = int(numpy_ceil(ds_slice_stacked.sizes["sample"] / n_target_partitions))
        else:
            chunk_size = sample_chunk_static
        ds_slice_stacked = ds_slice_stacked.chunk({"sample": chunk_size})

        # 4) convert to ddf (graph is small because ds_slice is small)
        ddf_slice = ds_slice_stacked.to_dask_dataframe()
        ddf_slice = add_time_features(ddf_slice)  # your existing helper

        # 5) write this batch using delayed partition writes
        batch_out = os.path.join(out_base_dir, f"batch_{start:06d}_{stop:06d}")
        print(f"Saving batch to {batch_out} ... (npartitions = {ddf_slice.npartitions})")
        save_ddf_via_delayed(
            ddf_slice,
            batch_out,
            engine=parquet_engine,
            write_index=write_index,
            sequential=sequential_writes,
            show_progress=True,
            resource_profiler=False,
        )

        # 6) cleanup to free memory & break graph
        del ds_slice, ds_slice_stacked, ddf_slice
        gc.collect()

    print("All batches processed.")


# ---------------------------
# How to call from main code
# ---------------------------
# Suppose earlier you built:
# pl_wide = widen_pressure_levels(pl)
# sl = sl.reset_coords("number", drop=True)
# ds = xr.merge([pl_wide, sl], compat="no_conflicts")

# Then call:
# process_and_save_in_time_batches(
#     ds,
#     out_base_dir="dataframe-dump/parquet_batches",
#     time_batch_size=500,                # tune smaller if necessary
#     sample_chunk_static=200_000,
#     dynamic_chunk_size=False,
#     n_target_partitions=20,
#     parquet_engine="pyarrow",
#     write_index=False,
#     sequential_writes=True,
#     cast_to_float32=True                # reduces memory footprint
# )