
import os
import pandas as pd
import numpy as np
import time

def generate_dataset(output_dir="dataset", num_records=10000):
    """Generates a synthetic 6G-enabled IoT dataset."""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Generating {num_records} records for 6G IoT Dataset...")
    
    # 1. Define Attributes
    # Context: IoT devices in a smart city / healthcare setting 6G network
    
    ids = [f"DEV_{i:05d}" for i in range(num_records)]
    timestamps = pd.date_range(start="2025-01-01", periods=num_records, freq="T")
    
    # Attributes for Federated Learning
    # e.g., Network parameters, Device Status, Sensitive Data (Heart Rate, Location)
    
    data = {
        "Device_ID": ids,
        "Timestamp": timestamps,
        "Device_Type": np.random.choice(["SmartSensor", "Wearable", "AutonomousVehicle", "IndustrialController"], num_records),
        "Battery_Level": np.random.uniform(10.0, 100.0, num_records).round(2),
        "Signal_Strength_dBm": np.random.normal(-60, 10, num_records).round(1), # 6G Signal
        "Latency_ms": np.random.gamma(2, 2, num_records).round(3), # Ultra-low latency
        "Data_Volume_MB": np.random.exponential(50, num_records).round(2),
        "Local_Model_Accuracy": np.random.uniform(0.70, 0.95, num_records).round(4),
        "Privacy_Budget_Used": np.random.uniform(0.1, 5.0, num_records).round(2),
        # Sensitive Feature (Target for attacks)
        "User_Heart_Rate": np.random.normal(75, 12, num_records).astype(int),
        "Location_X": np.random.uniform(0, 1000, num_records).round(2),
        "Location_Y": np.random.uniform(0, 1000, num_records).round(2),
    }
    
    df = pd.DataFrame(data)
    
    # Save Main Dataset
    csv_path = os.path.join(output_dir, "6G_IoT_Federated_Data.csv")
    df.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Dataset saved to {csv_path}")
    
    # Save Attribute Description
    desc = """
# 6G IoT Federated Learning Dataset - Attribute Description

1. **Device_ID**: Unique identifier for the edge device (client).
2. **Timestamp**: Time of data recording.
3. **Device_Type**: Category of the IoT device (Wearable, Vehicle, etc.).
4. **Battery_Level**: Remaining energy percentage (Critical for FL participation).
5. **Signal_Strength_dBm**: 6G channel quality indicator.
6. **Latency_ms**: Network latency, critical for 6G ultra-reliable low-latency communication (URLLC).
7. **Data_Volume_MB**: Amount of local data available for training.
8. **Local_Model_Accuracy**: Pre-aggregation accuracy of the local model on private data.
9. **Privacy_Budget_Used**: Cumulative epsilon consumed by the device.
10. **User_Heart_Rate**: Sensitive health metric (Protected attribute).
11. **Location_X/Y**: Geo-spatial coordinates (Sensitive attribute).
    """
    
    desc_path = os.path.join(output_dir, "Dataset_Description.txt")
    with open(desc_path, "w") as f:
        f.write(desc.strip())
    print(f"[SUCCESS] Descriptions saved to {desc_path}")

if __name__ == "__main__":
    generate_dataset()
