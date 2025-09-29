#!/bin/bash
curl -X POST "http://localhost:8000/optimize" -H "Content-Type: application/json" -d '{
    "dataset_name":"mnist",
    "trials":4,
    "min_epochs_per_trial":5,
    "max_epochs_per_trial":7,
    "use_runpod_service":true,
    "concurrent":true,
    "target_gpus_per_worker":2
}'