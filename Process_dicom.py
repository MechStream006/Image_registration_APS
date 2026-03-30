from APS_all_1 import process_dicom_dsa

# Process your DICOM file
result = process_dicom_dsa(
    dicom_path='D:/Rohith/RAW_AUTOPIXEL/Neuro_RAW/1.2.826.0.1.3680043.2.1330.2640165.2408301133290002.5.445_Raw_anon.dcm',  # Your DICOM file path
    mask_frame_index=0,           # Which frame is the mask (usually 0)
    output_path='D:/Rohith/Auto_pixel_shift/APS/ECC/Results/corrected.dcm',  # Where to save corrected DICOM
    save_visualization=True       # Create comparison images
)

print(f"\nSuccess rate: {result['success_rate']:.1%}")