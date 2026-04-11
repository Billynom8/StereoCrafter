The following summary outlines the **Bidirectional Flow-Guided Inpainting** solution developed to address the limitations of standard video inpainting in 3D Side-by-Side (SBS) generation.

1. The Core Problem: Temporal Context and "Smudgy" Artifacts

When generating a 3D right-eye view from a 2D source, spatial "occlusion holes" are created behind foreground objects. Standard inpainting often results in low-quality "smudges" because the model has no prior knowledge of what exists behind an object before it is revealed by camera movement. In a chronological (forward) pass, the AI must "guess" (hallucinate) texture, leading to temporal instability and flickering.

2. The Solution: Bidirectional Video Inpainting

The solution leverages the fact that background information revealed at the end of a clip can be used to fill holes at the beginning. By processing the video in two directions, the system provides the AI with "the answers to the test" upfront.

- **Reverse Temporal Propagation:** When a video is reversed, the background starts fully visible and is gradually covered by the foreground. This allows the AI to use its "past" memory (the video's future) to fill holes with actual, high-fidelity texture instead of guessing.

- **Directional Logic:**
  
  - **Forward Inpainting** is superior for areas where the background is being **covered** (object moving right or camera panning left).
  - **Reverse Inpainting** is superior for areas where the background is being **revealed** (object moving left or camera panning right).
3. The Technical Switchboard: Optical Flow

To automate the selection between forward and reverse inpainting, the system utilizes an **Optical Flow map** (generated via models like RAFT or Megaflow).

- **The X-Channel Decision:** The horizontal movement (X-axis) of pixels determines the flow direction.
  
  - **Positive Flow (>0):** Indicates rightward movement; the system uses the **Forward Inpaint**.
  - **Negative Flow (<0):** Indicates leftward movement; the system uses the **Reverse Inpaint**.

- **Confidence Blending:** Rather than a binary switch, the system uses a **"Soft Switch" (Linear Interpolation)** based on speed. Near-zero movement results in a 50/50 blend to cancel out AI noise, while higher speeds ramp up the weight of the preferred directional pass to 100%.
4. Critical Refinement: Solving the "Hole Paradox"

A major challenge occurs when warping the 2D flow map to the right-eye view, which creates its own "occlusion holes" where no movement data exists.

- **The Failure of Standard Blending:** Traditional dilation or smooth inpainting (like Telea) is "mathematically wrong" here because it bleeds the **foreground object's** movement into the background hole.

- **The "Right-to-Left Stretch" Innovation:** Since the hole represents hidden background, it must inherit the movement of the background to its right. The solution uses an **asymmetric 1D horizontal stretch**, grabbing the background flow value from the right edge and dragging it left across the hole. This ensures the flow map accurately reflects background movement only, preventing mask errors.
5. Final Modular Pipeline Architecture

The solution is implemented in three distinct phases to manage memory and logic:

1. **Splatting Phase:**
   - Generate depth and 2D flow maps from the source.
   - Forward-warp both the RGB frames and the flow maps to the right eye.
   - Apply the **Right-to-Left stretch** to fill the warped flow map holes.
2. **Inpainting Phase:**
   - Run a standard forward SVD inpaint.
   - Run a reverse SVD inpaint, then **re-reverse the output** timeline to ensure chronological alignment with the forward pass.
3. **Merging Phase:**
   - Load the occlusion mask to constrain the blending area.
   - Use the filled flow map to perform **Bidirectional Flow Guided Blending** (Speed-Based) strictly inside the mask.
   - Composite the high-res splatted image over the "smart inpaint" result.