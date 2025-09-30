#!/bin/bash
curl -X POST "http://localhost:8000/optimize" -H "Content-Type: application/json" -d '{
    "dataset_name":"mnist",
    "trials":30,
    "min_epochs_per_trial":15,
    "max_epochs_per_trial":30,
    "use_runpod_service":true,
    "concurrent":true,
    "target_gpus_per_worker":2
}'