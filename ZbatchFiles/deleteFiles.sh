
directory="/homes/gustavo/Code/MIAM/nnUNet/data/nnUnet_raw/nnUNet_raw_data/Task2206_picaiT2/imagesTr"

for file in "$directory"/*; do
  # Check if the file contains the substring "0003.nii.gz"
  if [[ $file == *0002.nii.gz* || $file == *0003.nii.gz* ]]; then
    # Remove the file
    rm "$file"
    echo "Removed file: $file"
  fi
done


exit


