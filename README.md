# VPRUnderwater Data Placement Guide

## Introduction
This guide provides instructions on how to properly place data in the `data` folder for the VPRUnderwater project. Proper data organization is crucial for the smooth functioning of the project.

## Data Folder Structure
The `data` folder should have the following structure:

```
data/
├── raw/
│   ├── images/
│   └── videos/
├── processed/
│   ├── images/
│   └── videos/
└── metadata/
```

### Raw Data
- **images/**: Place all raw image files here.
- **videos/**: Place all raw video files here.

### Processed Data
- **images/**: Place all processed image files here.
- **videos/**: Place all processed video files here.

### Metadata
- **metadata/**: Place all metadata files related to the raw and processed data here.

## Alternative Data Location
If you choose to place the data in a location other than the `data` folder, you must specify the path in the configuration file `config.yaml` as follows:

```yaml
data_path: /path/to/your/data
```

Ensure that the specified path follows the same folder structure as described above.

## Conclusion
Following this guide will help maintain a consistent and organized data structure, facilitating easier data management and processing for the VPRUnderwater project.
