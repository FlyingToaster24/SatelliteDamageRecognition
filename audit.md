# SatelliteInageRecognition - Audit

1. Moved pytorch device type to a separate file to accomodate for different devices (especially Apple Silicon)

1. Added a new file .gitignore to ignore some files

1. create_datasets on training_model is too slow, due to filesystem access. Let's use multiprocessing to speed it up.
    - Here, we can also stop the for loop early to process the json files faster