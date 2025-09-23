"""
Direct file download from RunPod workers via API endpoints

This module provides functions to download files directly from RunPod workers
using their API endpoints, eliminating the need for S3 intermediate storage.
"""

import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from utils.logger import logger
import time
import base64
import os


def download_files_from_runpod_worker(
    runpod_endpoint: str,
    run_name: str,
    local_dir: str,
    file_patterns: Optional[List[str]] = None,
    worker_id: Optional[str] = None,
    timeout: int = 300
) -> bool:
    """
    Download files directly from RunPod worker via API endpoints.

    Args:
        runpod_endpoint: RunPod worker endpoint URL
        run_name: Run name to identify files
        local_dir: Local directory to download files to
        file_patterns: Optional list of file patterns to filter (e.g., ['*.png', '*.json'])
        timeout: Timeout in seconds for each request

    Returns:
        True if download successful, False otherwise
    """
    try:
        logger.info(f"📥 Starting direct download from RunPod worker: {runpod_endpoint}")
        logger.info(f"🏷️ Run name: {run_name}")
        logger.info(f"📁 Local directory: {local_dir}")

        # Use worker-specific endpoint if worker_id is provided
        if worker_id:
            worker_endpoint = f"https://{worker_id}-80.proxy.runpod.net"
            logger.info(f"🎯 Using worker-specific endpoint: {worker_endpoint}")
        else:
            worker_endpoint = runpod_endpoint
            logger.info(f"🎯 Using provided endpoint: {worker_endpoint}")

        # Create local directory if it doesn't exist
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)

        # Get list of available files
        files_url = f"{worker_endpoint}/files/{run_name}"
        logger.debug(f"📋 Fetching file list from: {files_url}")

        response = requests.get(files_url, timeout=timeout)
        response.raise_for_status()

        file_data = response.json()
        available_files = file_data.get('files', [])

        logger.info(f"📋 Found {len(available_files)} files on RunPod worker")

        if not available_files:
            logger.warning(f"📋 No files found for run_name: {run_name}")
            return False

        # Filter files by patterns if specified
        files_to_download = available_files
        if file_patterns:
            files_to_download = []
            for file_info in available_files:
                file_path = file_info['path']
                if any(Path(file_path).match(pattern) for pattern in file_patterns):
                    files_to_download.append(file_info)

            logger.info(f"📋 Filtered to {len(files_to_download)} files matching patterns: {file_patterns}")

        downloaded_count = 0
        downloaded_files = []

        # Download each file
        for file_info in files_to_download:
            file_path = file_info['path']
            filename = file_info['filename']
            file_size = file_info['size']

            try:
                # Construct download URL
                download_url = f"{worker_endpoint}/download/{run_name}/{file_path}"
                logger.debug(f"📥 Downloading: {download_url}")

                # Download file
                download_response = requests.get(download_url, timeout=timeout, stream=True)
                download_response.raise_for_status()

                # Save to local file
                local_file_path = local_path / filename

                with open(local_file_path, 'wb') as f:
                    for chunk in download_response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                # Verify file size
                actual_size = local_file_path.stat().st_size
                if actual_size != file_size:
                    logger.warning(f"📥 Size mismatch for {filename}: expected {file_size}, got {actual_size}")

                downloaded_files.append(str(local_file_path))
                downloaded_count += 1

                logger.debug(f"✅ Downloaded: {filename} ({actual_size} bytes)")

            except Exception as e:
                logger.error(f"❌ Failed to download {file_path}: {e}")
                return False

        if downloaded_count > 0:
            logger.info(f"✅ Successfully downloaded {downloaded_count} files from RunPod worker")
            logger.info(f"📁 Files saved to: {local_dir}")
            logger.debug(f"📋 Downloaded files: {downloaded_files}")
            return True
        else:
            logger.warning(f"❌ No files were downloaded")
            return False

    except Exception as e:
        logger.error(f"❌ Failed to download files from RunPod worker: {e}")
        return False


def download_specific_files_from_runpod_worker(
    runpod_endpoint: str,
    run_name: str,
    file_list: List[str],
    local_dir: str,
    worker_id: Optional[str] = None,
    timeout: int = 300
) -> bool:
    """
    Download specific files from RunPod worker by exact file paths.

    Args:
        runpod_endpoint: RunPod worker endpoint URL
        run_name: Run name to identify files
        file_list: List of specific file paths to download
        local_dir: Local directory to download files to
        timeout: Timeout in seconds for each request

    Returns:
        True if all files downloaded successfully, False otherwise
    """
    try:
        logger.info(f"📥 Starting specific file download from RunPod worker")
        logger.info(f"🏷️ Run name: {run_name}")
        logger.info(f"📋 Files to download: {len(file_list)}")

        # Use worker-specific endpoint if worker_id is provided
        if worker_id:
            worker_endpoint = f"https://{worker_id}-80.proxy.runpod.net"
            logger.info(f"🎯 Using worker-specific endpoint: {worker_endpoint}")
        else:
            worker_endpoint = runpod_endpoint
            logger.info(f"🎯 Using provided endpoint: {worker_endpoint}")

        # === DEBUG: List all actual files on RunPod worker ===
        logger.info(f"🔍 DEBUG: Listing all files available on RunPod worker for run: {run_name}")
        try:
            files_endpoint = f"{worker_endpoint}/files/{run_name}"
            logger.info(f"🔍 DEBUG: Calling files endpoint: {files_endpoint}")

            files_response = requests.get(files_endpoint, timeout=timeout)
            if files_response.status_code == 200:
                response_data = files_response.json()
                available_files_data = response_data.get('files', [])
                available_file_paths = [f['path'] for f in available_files_data]

                logger.info(f"🔍 DEBUG: Found {len(available_files_data)} files on RunPod worker:")

                # Log all files in organized way
                for i, file_data in enumerate(available_files_data):
                    file_path = file_data['path']
                    file_size = file_data['size']
                    logger.info(f"🔍 DEBUG: [{i+1:3d}] {file_path} ({file_size} bytes)")

                # Compare with requested files
                logger.info(f"🔍 DEBUG: Comparing requested vs available files:")
                for file_path in file_list:
                    if file_path in available_file_paths:
                        logger.info(f"✅ MATCH: {file_path}")
                    else:
                        logger.error(f"❌ MISSING: {file_path}")
                        # Look for similar files
                        filename_only = Path(file_path).name
                        similar_files = [f for f in available_file_paths if filename_only in f]
                        if similar_files:
                            logger.info(f"🔍 DEBUG: Similar files found: {similar_files}")
            else:
                logger.error(f"🔍 DEBUG: Files endpoint failed: {files_response.status_code} - {files_response.text}")

        except Exception as e:
            logger.error(f"🔍 DEBUG: Failed to list files: {e}")

        logger.info(f"🔍 DEBUG: File listing complete, proceeding with downloads...")
        # === END DEBUG ===

        # Create local directory if it doesn't exist
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)

        downloaded_count = 0
        downloaded_files = []

        # Download each specific file
        for file_path in file_list:
            try:
                # Extract filename from path
                filename = Path(file_path).name

                # Construct download URL
                download_url = f"{worker_endpoint}/download/{run_name}/{file_path}"
                logger.debug(f"📥 LOCAL DOWNLOAD: Requesting URL: {download_url}")
                logger.debug(f"📥 LOCAL DOWNLOAD: Breakdown - Endpoint: {worker_endpoint}")
                logger.debug(f"📥 LOCAL DOWNLOAD: Breakdown - Run name: {run_name}")
                logger.debug(f"📥 LOCAL DOWNLOAD: Breakdown - File path: {file_path}")
                logger.debug(f"📥 LOCAL DOWNLOAD: Expected RunPod path: /app/optimization_results/{run_name}/{file_path}")

                # Download file
                download_response = requests.get(download_url, timeout=timeout, stream=True)
                download_response.raise_for_status()

                # Save to local file
                local_file_path = local_path / filename

                with open(local_file_path, 'wb') as f:
                    for chunk in download_response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                downloaded_files.append(str(local_file_path))
                downloaded_count += 1

                file_size = local_file_path.stat().st_size
                logger.debug(f"✅ Downloaded: {filename} ({file_size} bytes)")

            except Exception as e:
                logger.error(f"❌ Failed to download {file_path}: {e}")
                return False

        if downloaded_count > 0:
            logger.info(f"✅ Successfully downloaded {downloaded_count} specific files from RunPod worker")
            logger.info(f"📁 Files saved to: {local_dir}")
            return True
        else:
            logger.warning(f"❌ No files were downloaded")
            return False

    except Exception as e:
        logger.error(f"❌ Failed to download specific files from RunPod worker: {e}")
        return False


def get_runpod_worker_endpoint(runpod_api_url: str) -> str:
    """
    Extract the RunPod worker endpoint URL from the API URL.

    Args:
        runpod_api_url: RunPod API URL (e.g., https://api.runpod.ai/v2/endpoint_id/run)

    Returns:
        Worker endpoint URL (e.g., https://endpoint_id-80.proxy.runpod.net)
    """
    try:
        # Extract endpoint ID from RunPod API URL
        # Format: https://api.runpod.ai/v2/{endpoint_id}/run
        if '/v2/' in runpod_api_url and '/run' in runpod_api_url:
            endpoint_id = runpod_api_url.split('/v2/')[1].split('/run')[0]
            worker_endpoint = f"https://{endpoint_id}-80.proxy.runpod.net"
            logger.debug(f"🔗 Converted API URL to worker endpoint: {worker_endpoint}")
            return worker_endpoint
        else:
            logger.error(f"❌ Invalid RunPod API URL format: {runpod_api_url}")
            return runpod_api_url

    except Exception as e:
        logger.error(f"❌ Failed to extract worker endpoint: {e}")
        return runpod_api_url


def wait_for_runpod_worker_ready(runpod_endpoint: str, timeout: int = 60) -> bool:
    """
    Wait for RunPod worker to be ready to serve files.

    Args:
        runpod_endpoint: RunPod worker endpoint URL
        timeout: Timeout in seconds

    Returns:
        True if worker is ready, False if timeout
    """
    try:
        health_url = f"{runpod_endpoint}/health"
        start_time = time.time()

        logger.info(f"🔍 Waiting for RunPod worker to be ready: {health_url}")

        while time.time() - start_time < timeout:
            try:
                response = requests.get(health_url, timeout=10)
                if response.status_code == 200:
                    logger.info(f"✅ RunPod worker is ready")
                    return True
            except requests.exceptions.RequestException:
                pass

            time.sleep(2)

        logger.warning(f"⏰ Timeout waiting for RunPod worker to be ready")
        return False

    except Exception as e:
        logger.error(f"❌ Error checking RunPod worker readiness: {e}")
        return False


def download_files_via_runpod_api(
    runpod_api_url: str,
    runpod_api_key: str,
    run_name: str,
    local_dir: str,
    file_list: Optional[List[str]] = None,
    timeout: int = 300
) -> bool:
    """
    Download files from RunPod worker using the new RunPod API approach.

    Args:
        runpod_api_url: RunPod API URL (e.g., https://api.runpod.ai/v2/endpoint_id/run)
        runpod_api_key: RunPod API key for authentication
        run_name: Run name to identify files
        local_dir: Local directory to download files to
        file_list: Optional list of specific files to download
        timeout: Timeout in seconds for each request

    Returns:
        True if download successful, False otherwise
    """
    try:
        logger.info(f"📥 Starting download via RunPod API: {runpod_api_url}")
        logger.info(f"🏷️ Run name: {run_name}")
        logger.info(f"📁 Local directory: {local_dir}")

        # Create local directory if it doesn't exist
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)

        # First, get list of available files
        files_response = _call_runpod_api(
            runpod_api_url=runpod_api_url,
            runpod_api_key=runpod_api_key,
            command="list_files",
            run_name=run_name,
            timeout=timeout
        )

        if not files_response or files_response.get("error"):
            logger.error(f"❌ Failed to list files: {files_response.get('error', 'Unknown error')}")
            return False

        available_files = files_response.get("files", [])
        logger.info(f"📋 Found {len(available_files)} files on RunPod worker")

        if not available_files:
            logger.warning(f"📋 No files found for run_name: {run_name}")
            return False

        # Determine which files to download
        files_to_download = file_list if file_list else available_files

        downloaded_count = 0
        downloaded_files = []

        # Download each file
        for file_path in files_to_download:
            try:
                logger.debug(f"📥 Downloading file: {file_path}")

                # Download file via RunPod API
                download_response = _call_runpod_api(
                    runpod_api_url=runpod_api_url,
                    runpod_api_key=runpod_api_key,
                    command="download_file",
                    run_name=run_name,
                    file_path=file_path,
                    timeout=timeout
                )

                if not download_response or download_response.get("error"):
                    logger.error(f"❌ Failed to download {file_path}: {download_response.get('error', 'Unknown error')}")
                    continue

                # Decode base64 content and save to local file
                file_content = download_response.get("content")
                filename = download_response.get("filename") or Path(file_path).name

                if not file_content:
                    logger.error(f"❌ No content received for {file_path}")
                    continue

                # Decode base64 content
                try:
                    decoded_content = base64.b64decode(file_content)
                except Exception as e:
                    logger.error(f"❌ Failed to decode base64 content for {file_path}: {e}")
                    continue

                # Save to local file
                local_file_path = local_path / filename
                with open(local_file_path, 'wb') as f:
                    f.write(decoded_content)

                downloaded_files.append(str(local_file_path))
                downloaded_count += 1

                file_size = local_file_path.stat().st_size
                logger.debug(f"✅ Downloaded: {filename} ({file_size} bytes)")

            except Exception as e:
                logger.error(f"❌ Failed to download {file_path}: {e}")
                continue

        if downloaded_count > 0:
            logger.info(f"✅ Successfully downloaded {downloaded_count} files via RunPod API")
            logger.info(f"📁 Files saved to: {local_dir}")
            return True
        else:
            logger.warning(f"❌ No files were downloaded")
            return False

    except Exception as e:
        logger.error(f"❌ Failed to download files via RunPod API: {e}")
        return False


def _call_runpod_api(
    runpod_api_url: str,
    runpod_api_key: str,
    command: str,
    timeout: int = 300,
    **kwargs
) -> Optional[Dict[str, Any]]:
    """
    Call RunPod API with a command and parameters.

    Args:
        runpod_api_url: RunPod API URL
        runpod_api_key: RunPod API key
        command: Command to execute
        timeout: Request timeout
        **kwargs: Additional parameters for the command

    Returns:
        API response or None if failed
    """
    try:
        payload = {
            "input": {
                "command": command,
                **kwargs
            }
        }

        headers = {
            "Authorization": f"Bearer {runpod_api_key}",
            "Content-Type": "application/json"
        }

        logger.debug(f"🔧 Calling RunPod API: {command}")
        response = requests.post(runpod_api_url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()

        result = response.json()
        job_id = result.get("id")

        if not job_id:
            logger.error(f"❌ No job ID returned from RunPod API")
            return None

        # Poll for job completion
        status_url = runpod_api_url.replace("/run", f"/status/{job_id}")

        logger.debug(f"⏳ Polling job status: {job_id}")
        max_polls = 60  # 5 minutes at 5-second intervals
        poll_count = 0

        while poll_count < max_polls:
            try:
                status_response = requests.get(status_url, headers=headers, timeout=30)
                status_response.raise_for_status()

                status_data = status_response.json()
                job_status = status_data.get("status")

                if job_status == "COMPLETED":
                    logger.debug(f"✅ Job completed: {job_id}")
                    return status_data.get("output", {})
                elif job_status == "FAILED":
                    error_msg = status_data.get("error", "Unknown error")
                    logger.error(f"❌ Job failed: {job_id} - {error_msg}")
                    return {"error": error_msg}
                elif job_status in ["IN_QUEUE", "IN_PROGRESS"]:
                    logger.debug(f"⏳ Job {job_status}: {job_id}")
                    time.sleep(5)
                    poll_count += 1
                else:
                    logger.warning(f"🔍 Unknown job status: {job_status}")
                    time.sleep(5)
                    poll_count += 1

            except Exception as e:
                logger.error(f"❌ Error polling job status: {e}")
                time.sleep(5)
                poll_count += 1

        logger.error(f"⏰ Timeout waiting for job completion: {job_id}")
        return {"error": "Timeout waiting for job completion"}

    except Exception as e:
        logger.error(f"❌ Failed to call RunPod API: {e}")
        return {"error": str(e)}


def download_directory_via_runpod_api(
    runpod_api_url: str,
    runpod_api_key: str,
    run_name: str,
    local_dir: str,
    timeout: int = 300
) -> bool:
    """
    Download entire directory from RunPod worker as a single zip file via RunPod API.
    This is much faster and more reliable than downloading individual files.

    Args:
        runpod_api_url: RunPod API URL (e.g., https://api.runpod.ai/v2/endpoint_id/run)
        runpod_api_key: RunPod API key for authentication
        run_name: Run name to identify directory
        local_dir: Local directory to extract files to
        timeout: Timeout in seconds for the request

    Returns:
        True if download successful, False otherwise
    """
    try:
        logger.info(f"📥 Starting batch directory download via RunPod API")
        logger.info(f"🏷️ Run name: {run_name}")
        logger.info(f"📁 Local directory: {local_dir}")

        # Create local directory if it doesn't exist
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)

        # Download entire directory as zip
        download_response = _call_runpod_api(
            runpod_api_url=runpod_api_url,
            runpod_api_key=runpod_api_key,
            command="download_directory",
            run_name=run_name,
            timeout=timeout
        )

        if not download_response or download_response.get("error"):
            logger.error(f"❌ Failed to download directory: {download_response.get('error', 'Unknown error')}")
            logger.error(f"❌ RunPod download attempt details:")
            logger.error(f"   - API URL: {runpod_api_url}")
            logger.error(f"   - Run name: {run_name}")
            logger.error(f"   - Expected RunPod path: /tmp/plots/{run_name}")
            logger.error(f"   - Local destination: {local_dir}")
            return False

        # Extract response data
        zip_content = download_response.get("content")
        filename = download_response.get("filename", f"{run_name}_plots.zip")
        file_count = download_response.get("file_count", 0)
        zip_size = download_response.get("size", 0)

        if not zip_content:
            logger.error(f"❌ No zip content received")
            return False

        # Decode base64 content
        try:
            decoded_content = base64.b64decode(zip_content)
        except Exception as e:
            logger.error(f"❌ Failed to decode base64 zip content: {e}")
            return False

        # Save zip temporarily and extract
        import tempfile
        import zipfile

        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_zip:
            tmp_zip.write(decoded_content)
            temp_zip_path = tmp_zip.name

        try:
            # Extract zip to local directory
            with zipfile.ZipFile(temp_zip_path, 'r') as zipf:
                zipf.extractall(local_path)
                extracted_files = zipf.namelist()

            # Clean up temporary zip file
            os.unlink(temp_zip_path)

            logger.info(f"✅ Successfully downloaded and extracted directory via RunPod API")
            logger.info(f"📊 Downloaded {file_count} files ({zip_size} bytes compressed)")
            logger.info(f"📁 Files extracted to: {local_dir}")
            logger.debug(f"📋 Extracted files: {extracted_files}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to extract zip file: {e}")
            # Clean up temporary file
            if os.path.exists(temp_zip_path):
                os.unlink(temp_zip_path)
            return False

    except Exception as e:
        logger.error(f"❌ Failed to download directory via RunPod API: {e}")
        return False