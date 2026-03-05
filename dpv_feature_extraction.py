import os
import glob
import numpy as np
import json
import pandas as pd

# =============================================================================
# Configuration & Constants
# =============================================================================
# BLANK Current Values for Delta I Calculation (User must set these)
# Delta I = I_blank - I_peak
# Please replace None with your experimental blank current values (in uA)
I_BLANK_MIR92A = None 
I_BLANK_MIR21 = None

def check_constants():
    if I_BLANK_MIR92A is None or I_BLANK_MIR21 is None:
        raise ValueError(
            "Critical Error: Blank current constants (I_BLANK_MIR92A, I_BLANK_MIR21) "
            "are not set. Please define them in the script configuration section."
        )

# =============================================================================
# Helper Functions
# =============================================================================

def read_chi_txt(filepath):
    """
    Reads Voltage and Current from CHI660E .txt file format.
    Expects 'Potential/V' header line.
    """
    voltage = []
    current = []
    start_reading = False
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("Potential/V"):
                    start_reading = True
                    continue
                
                if start_reading:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        try:
                            v = float(parts[0])
                            c = float(parts[1])
                            voltage.append(v)
                            current.append(c)
                        except ValueError:
                            continue
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return np.array([]), np.array([])
    
    return np.array(voltage), np.array(current)

def process_file(filepath, mir_type):
    """
    Process a single DPV file to extract 4 features using valley-to-valley baseline correction:
    1. I_peak (Baseline Corrected Peak Current)
    2. Ep (Peak Potential)
    3. Ah (Peak Area via integration)
    4. FWHM (Full Width at Half Maximum)
    """
    
    voltage, current = read_chi_txt(filepath)
    if len(voltage) == 0:
        print("Warning: No valid data found in file.")
        return None
    
    # --- Valley-to-Valley Baseline Correction & Feature Extraction ---
    
    # 1. Find Peak (Global Max in this simplified implementation)
    # Ideally, search in a specific window, but global max works for clean DPV
    max_idx = np.argmax(current)
    ep = float(voltage[max_idx])
    
    # 2. Find Valleys (Left and Right local minima)
    # Search left of peak
    left_region_I = current[:max_idx]
    if len(left_region_I) > 0:
        left_valley_idx = np.argmin(left_region_I)
    else:
        left_valley_idx = 0
        
    # Search right of peak
    right_region_I = current[max_idx:]
    if len(right_region_I) > 0:
        right_valley_idx = max_idx + np.argmin(right_region_I)
    else:
        right_valley_idx = len(current) - 1
        
    # 3. Construct Linear Baseline
    v_left = voltage[left_valley_idx]
    i_left = current[left_valley_idx]
    v_right = voltage[right_valley_idx]
    i_right = current[right_valley_idx]
    
    # Avoid division by zero
    if v_right != v_left:
        slope = (i_right - i_left) / (v_right - v_left)
        baseline = i_left + slope * (voltage - v_left)
    else:
        baseline = np.full_like(current, i_left)
        
    # 4. Calculate Corrected Current
    i_corr = current - baseline
    
    # Extract Features from Corrected Current
    
    # I_peak (Corrected)
    # This is the height of the peak above the linear baseline
    i_peak_corrected = float(i_corr[max_idx])
    
    # Ah (Area) - Integrate corrected current between valleys
    # Only integrate positive part within the valley range
    valley_range_mask = (np.arange(len(current)) >= left_valley_idx) & (np.arange(len(current)) <= right_valley_idx)
    i_corr_window = i_corr[valley_range_mask]
    v_window = voltage[valley_range_mask]
    
    # Ensure positive area (DPV peaks usually positive after correction)
    # If V is decreasing, trapz might be negative, take abs
    try:
        ah = float(np.abs(np.trapezoid(i_corr_window, v_window)))
    except AttributeError:
        ah = float(np.abs(np.trapz(i_corr_window, v_window)))
        
    # FWHM on Corrected Signal
    try:
        half_max = i_peak_corrected / 2.0
        
        # Find crossings in the corrected signal within the window
        # Left crossing
        left_cross_candidates = np.where((i_corr[:max_idx] < half_max) & (np.arange(max_idx) >= left_valley_idx))[0]
        if len(left_cross_candidates) > 0:
            l_idx = left_cross_candidates[-1]
        else:
            l_idx = left_valley_idx
            
        # Right crossing
        right_cross_candidates = np.where((i_corr[max_idx:] < half_max) & (np.arange(max_idx, len(current)) <= right_valley_idx))[0]
        if len(right_cross_candidates) > 0:
            r_idx = max_idx + right_cross_candidates[0]
        else:
            r_idx = right_valley_idx
            
        fwhm = float(abs(voltage[r_idx] - voltage[l_idx]))
    except:
        fwhm = 0.0

    # Feature 4: Delta I
    # Delta I = I_blank - I_peak (Corrected)
    # As per paper definition, I_peak is the background-subtracted peak current
    if I_BLANK_MIR92A is not None and mir_type == "miR-92a":
        delta_i = I_BLANK_MIR92A - i_peak_corrected
    elif I_BLANK_MIR21 is not None and mir_type == "miR-21":
        delta_i = I_BLANK_MIR21 - i_peak_corrected
    else:
        # Should be caught by check_constants() at start, but double check here
        raise ValueError(f"I_BLANK for {mir_type} is not set!")
        
    return {
        "I_peak": i_peak_corrected,
        "Ep": ep,
        "Ah": ah,
        "FWHM": fwhm,
        "Delta_I": delta_i
    }

def main():
    # Ensure constants are set
    check_constants()
    
    # Define input directory structure
    # Users should organize their data in this structure or modify paths
    # Assuming data is in a 'data' folder in the root
    base_dir = "data" 
    paths = {
        "miR-92a": {
            "Healthy": os.path.join(base_dir, "miR-92a", "Healthy"),
            "CRC": os.path.join(base_dir, "miR-92a", "CRC")
        },
        "miR-21": {
            "Healthy": os.path.join(base_dir, "miR-21", "Healthy"),
            "CRC": os.path.join(base_dir, "miR-21", "CRC")
        }
    }
    
    results = {
        "miR-92a": {"Healthy": [], "CRC": []},
        "miR-21": {"Healthy": [], "CRC": []}
    }
    
    print("Starting DPV Feature Extraction...")
    
    for mir_type, categories in paths.items():
        for category, folder in categories.items():
            if not os.path.exists(folder):
                # print(f"Skipping: Folder not found: {folder}")
                continue
                
            files = glob.glob(os.path.join(folder, "*.txt"))
            print(f"Processing {mir_type} - {category}: {len(files)} files")
            
            for f in files:
                feat = process_file(f, mir_type)
                if feat:
                    results[mir_type][category].append(feat)
                    
    # Save extracted features to JSON
    output_file = "extracted_features.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nFeature extraction complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()
