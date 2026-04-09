#!/usr/bin/env python3
"""
Headless CLI interface for StereoCrafter splatting operations.

This script provides command-line access to the splatting functionality
without requiring the Tkinter GUI.
"""

import argparse
import logging
import os
import sys

# Add the current directory to the path so we can import core modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.splatting import SplattingController, ProcessingSettings
from core.common.sidecar_manager import SidecarConfigManager


def setup_logging():
    """Configure logging for the CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="StereoCrafter CLI - Headless splatting operations")

    # Required arguments
    parser.add_argument("--source", required=True, help="Path to source videos directory or file")
    parser.add_argument("--depth", required=True, help="Path to depth maps directory or file")
    parser.add_argument("--output", required=True, help="Output directory for splatted videos")

    # Processing options
    parser.add_argument("--max-disp", type=float, default=20.0, help="Maximum disparity percentage")
    parser.add_argument("--convergence", type=float, default=0.5, help="Convergence plane value (0.0-1.0)")
    parser.add_argument("--gamma", type=float, default=1.0, help="Depth gamma correction")
    parser.add_argument("--process-length", type=int, default=-1, help="Number of frames to process (-1 for all)")

    # Resolution options
    parser.add_argument("--full-res", action="store_true", help="Enable full resolution output")
    parser.add_argument("--low-res", action="store_true", help="Enable low resolution output")
    parser.add_argument("--low-width", type=int, default=640, help="Low resolution width")
    parser.add_argument("--low-height", type=int, default=360, help="Low resolution height")

    # Batch options
    parser.add_argument("--full-batch-size", type=int, default=10, help="Batch size for full resolution")
    parser.add_argument("--low-batch-size", type=int, default=15, help="Batch size for low resolution")

    # Other options
    parser.add_argument("--dual-output", action="store_true", help="Output both left and right eyes")
    parser.add_argument("--move-finished", action="store_true", help="Move processed files to finished folder")
    parser.add_argument(
        "--auto-conv", choices=["Off", "Average", "Peak", "Hybrid"], default="Off", help="Auto-convergence mode"
    )

    return parser.parse_args()


def print_progress(message):
    """Print progress messages to console."""
    if isinstance(message, tuple):
        msg_type, content = message
        if msg_type == "status":
            print(f"STATUS: {content}")
        elif msg_type == "total":
            print(f"TOTAL TASKS: {content}")
        elif msg_type == "processed":
            print(f"PROCESSED: {content}")
        elif msg_type == "update_info":
            if "filename" in content:
                print(f"PROCESSING: {content['filename']}")
    elif message == "finished":
        print("PROCESSING FINISHED")
    else:
        print(f"UNKNOWN MESSAGE: {message}")


def main():
    """Main CLI entry point."""
    setup_logging()
    args = parse_args()

    # Initialize controller
    sidecar_manager = SidecarConfigManager()
    controller = SplattingController(sidecar_manager=sidecar_manager)

    # Find matching pairs
    print("Scanning for matching source/depth pairs...")
    video_list = controller.find_matching_pairs(args.source, args.depth)

    if not video_list:
        print("No matching source/depth pairs found!")
        return 1

    print(f"Found {len(video_list)} matching video/depth pairs")

    # Create processing settings
    settings = ProcessingSettings(
        # Input Settings
        input_source_clips=args.source,
        input_depth_maps=args.depth,
        output_splatted=args.output,
        # Resolution Settings
        enable_full_resolution=args.full_res,
        full_res_batch_size=args.full_batch_size,
        dual_output=args.dual_output,
        enable_low_resolution=args.low_res,
        low_res_batch_size=args.low_batch_size,
        # missing - strict_ffmpeg_decode_var
        low_res_width=args.low_width,
        low_res_height=args.low_height,
        process_length=args.process_length,
        auto_convergence_mode=args.auto_conv,
        # missing - mask_mode_var
        # border_mode="Off",
        zero_disparity_anchor=args.convergence,
        match_depth_res=False,
        move_to_finished=args.move_finished,
        output_crf=18,
        output_crf_full=18,
        output_crf_low=18,
        # Depth Processing Settings
        depth_dilate_size_x=6,
        depth_dilate_size_y=3,
        depth_blur_size_x=1,
        depth_blur_size_y=1,
        depth_dilate_left=0,
        depth_blur_left=0,
        # Stereo Projection Settings
        depth_gamma=args.gamma,
        max_disp=args.max_disp,
        enable_sidecar_gamma=False,
        enable_sidecar_blur_dilate=False,
        border_width=0.0,
        border_bias=0.0,
        enable_global_norm=False,
        # missing - move_to_finished_var
        # missing - map_test_var
        # missing - splat_test_var
        # missing - track_dp_total_true_on_render_var
    )

    # Start processing
    print("Starting batch processing...")
    controller.start_batch(settings)

    # Monitor progress
    try:
        while True:
            message = controller.get_progress()
            if message is not None:
                print_progress(message)
                if message == "finished":
                    break

            # Check if processing thread is still alive
            if not controller.processing_thread or not controller.processing_thread.is_alive():
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user, stopping...")
        controller.stop()

    print("CLI execution completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
