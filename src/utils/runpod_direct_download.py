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


def get_runpod_worker_endpoint(api_url_runpod: str) -> str:
    """
    Extract the RunPod worker endpoint URL from the API URL.

    Args:
        api_url_runpod: RunPod API URL (e.g., https://api.runpod.ai/v2/endpoint_id/run)

    Returns:
        Worker endpoint URL (e.g., https://endpoint_id-80.proxy.runpod.net)
    """
    try:
        # Extract endpoint ID from RunPod API URL
        # Format: https://api.runpod.ai/v2/{endpoint_id}/run
        if '/v2/' in api_url_runpod and '/run' in api_url_runpod:
            endpoint_id = api_url_runpod.split('/v2/')[1].split('/run')[0]
            worker_endpoint = f"https://{endpoint_id}-80.proxy.runpod.net"
            logger.debug(f"🔗 Converted API URL to worker endpoint: {worker_endpoint}")
            return worker_endpoint
        else:
            logger.error(f"❌ Invalid RunPod API URL format: {api_url_runpod}")
            return api_url_runpod

    except Exception as e:
        logger.error(f"❌ Failed to extract worker endpoint: {e}")
        return api_url_runpod


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
    api_url_runpod: str,
    api_key_runpod: str,
    run_name: str,
    local_dir: str,
    file_list: Optional[List[str]] = None,
    timeout: int = 300
) -> bool:
    """
    Download files from RunPod worker using the new RunPod API approach.

    Args:
        api_url_runpod: RunPod API URL (e.g., https://api.runpod.ai/v2/endpoint_id/run)
        api_key_runpod: RunPod API key for authentication
        run_name: Run name to identify files
        local_dir: Local directory to download files to
        file_list: Optional list of specific files to download
        timeout: Timeout in seconds for each request

    Returns:
        True if download successful, False otherwise
    """
    try:
        logger.info(f"📥 Starting download via RunPod API: {api_url_runpod}")
        logger.info(f"🏷️ Run name: {run_name}")
        logger.info(f"📁 Local directory: {local_dir}")

        # Create local directory if it doesn't exist
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)

        # First, get list of available files
        files_response = _call_runpod_api(
            api_url_runpod=api_url_runpod,
            api_key_runpod=api_key_runpod,
            command="list_files",
            run_name=run_name,
            timeout=timeout
        )

        if not files_response or files_response.get("error"):
            error_msg = files_response.get('error', 'Unknown error') if files_response else 'No response from RunPod'
            logger.error(f"❌ Failed to list files: {error_msg}")
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
                    api_url_runpod=api_url_runpod,
                    api_key_runpod=api_key_runpod,
                    command="download_file",
                    run_name=run_name,
                    file_path=file_path,
                    timeout=timeout
                )

                if not download_response or download_response.get("error"):
                    error_msg = download_response.get('error', 'Unknown error') if download_response else 'No response from RunPod'
                    logger.error(f"❌ Failed to download {file_path}: {error_msg}")
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
    api_url_runpod: str,
    api_key_runpod: str,
    command: str,
    timeout: int = 300,
    **kwargs
) -> Optional[Dict[str, Any]]:
    """
    Call RunPod API with a command and parameters.

    Args:
        api_url_runpod: RunPod API URL
        api_key_runpod: RunPod API key
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
            "Authorization": f"Bearer {api_key_runpod}",
            "Content-Type": "application/json"
        }

        logger.debug(f"🔧 Calling RunPod API: {command}")
        response = requests.post(api_url_runpod, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()

        result = response.json()
        job_id = result.get("id")

        if not job_id:
            logger.error(f"❌ No job ID returned from RunPod API")
            return None

        # Poll for job completion
        status_url = api_url_runpod.replace("/run", f"/status/{job_id}")

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
    api_url_runpod: str,
    api_key_runpod: str,
    run_name: str,
    local_dir: str,
    timeout: int = 300,
    trial_id: Optional[str] = None,
    trial_number: Optional[int] = None,
    worker_id: Optional[str] = None
) -> bool:
    """
    Download entire directory from RunPod worker using multiple smaller zip files to avoid response size limits.
    Downloads plots and models separately to prevent 400 Bad Request errors from large responses.

    Args:
        api_url_runpod: RunPod API URL (e.g., https://api.runpod.ai/v2/endpoint_id/run)
        api_key_runpod: RunPod API key for authentication
        run_name: Run name to identify directory
        local_dir: Local directory to extract files to
        timeout: Timeout in seconds for the request
        trial_id: Trial identifier for logging
        trial_number: Trial number for logging
        worker_id: Specific RunPod worker ID to pin downloads to

    Returns:
        True if download successful, False otherwise
    """
    # Format trial information for logs (initialize before try block for exception handler)
    trial_info = ""
    if trial_number is not None:
        trial_info = f" (trial_{trial_number}"
        if trial_id:
            trial_info += f", {trial_id}"
        trial_info += ")"

    try:
        logger.info(f"running download_directory_via_runpod_api{trial_info} ... Starting multi-part directory download")
        logger.info(f"running download_directory_via_runpod_api{trial_info} ... Run name: {run_name}")
        logger.info(f"running download_directory_via_runpod_api{trial_info} ... Local directory: {local_dir}")

        # Determine API endpoint based on worker pinning
        if worker_id:
            # Use worker-specific endpoint for guaranteed same-worker routing
            api_url = f"https://{worker_id}-80.proxy.runpod.net"
            logger.info(f"running download_directory_via_runpod_api{trial_info} ... Using worker-specific endpoint for guaranteed same-worker routing")
            logger.info(f"running download_directory_via_runpod_api{trial_info} ... Worker ID: {worker_id}")
            logger.info(f"running download_directory_via_runpod_api{trial_info} ... Worker endpoint: {api_url}")
        else:
            # Use load-balanced endpoint (may hit different workers)
            api_url = api_url_runpod
            logger.info(f"running download_directory_via_runpod_api{trial_info} ... Using load-balanced endpoint (may hit different workers)")
            logger.info(f"running download_directory_via_runpod_api{trial_info} ... Load balancer endpoint: {api_url}")
            logger.warning(f"running download_directory_via_runpod_api{trial_info} ... No worker ID provided - downloads may fail due to worker isolation")

        # Create local directory if it doesn't exist
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)

        # Download plots and models separately to avoid response size limits
        download_types = ['plots', 'models']
        success_count = 0

        for download_type in download_types:
            logger.info(f"📦 Downloading {download_type} for run: {run_name}{trial_info}")

            logger.info(f"running download_directory_via_runpod_api{trial_info} ... Calling RunPod API for {download_type} download")
            if worker_id:
                logger.info(f"running download_directory_via_runpod_api{trial_info} ... Using worker-specific endpoint for {download_type}: {api_url}")
            else:
                logger.info(f"running download_directory_via_runpod_api{trial_info} ... Using load-balanced endpoint for {download_type}: {api_url}")

            download_response = _call_runpod_api(
                api_url_runpod=api_url,
                api_key_runpod=api_key_runpod,
                command="download_directory",
                run_name=run_name,
                download_type=download_type,
                trial_id=trial_id,
                trial_number=trial_number,
                timeout=timeout
            )

            # Check for 502 Bad Gateway errors with worker-specific URLs and retry with load-balanced URL
            if (worker_id and download_response and download_response.get("error") and
                "502" in str(download_response.get("error")) and "Bad Gateway" in str(download_response.get("error"))):
                logger.warning(f"⚠️ Worker-specific URL failed with 502 Bad Gateway for {download_type}{trial_info}")
                logger.info(f"🔄 Retrying {download_type} download with load-balanced URL instead of worker-specific URL")

                # Retry with load-balanced endpoint
                fallback_url = api_url_runpod
                logger.info(f"running download_directory_via_runpod_api{trial_info} ... Fallback to load-balanced endpoint: {fallback_url}")

                download_response = _call_runpod_api(
                    api_url_runpod=fallback_url,
                    api_key_runpod=api_key_runpod,
                    command="download_directory",
                    run_name=run_name,
                    download_type=download_type,
                    trial_id=trial_id,
                    trial_number=trial_number,
                    timeout=timeout
                )

                if download_response and not download_response.get("error"):
                    logger.info(f"✅ Fallback to load-balanced URL succeeded for {download_type}{trial_info}")
                else:
                    logger.error(f"❌ Fallback to load-balanced URL also failed for {download_type}{trial_info}")

            logger.info(f"running download_directory_via_runpod_api{trial_info} ... RunPod API call completed for {download_type} download")
            if worker_id and download_response:
                logger.info(f"running download_directory_via_runpod_api{trial_info} ... Worker pinning successful for {download_type} with worker {worker_id}")
            elif not worker_id and download_response:
                logger.info(f"running download_directory_via_runpod_api{trial_info} ... Load balancer routing successful for {download_type}")
            elif worker_id and not download_response:
                logger.error(f"running download_directory_via_runpod_api{trial_info} ... Worker pinning failed for {download_type} with worker {worker_id}")
            else:
                logger.error(f"running download_directory_via_runpod_api{trial_info} ... Load balancer routing failed for {download_type}")

            if not download_response or download_response.get("error"):
                error_msg = download_response.get('error', 'Unknown error') if download_response else 'No response from RunPod'
                logger.warning(f"⚠️ Failed to download {download_type}{trial_info}: {error_msg}")
                logger.warning(f"⚠️ This may be normal if no {download_type} files exist for this run{trial_info}")
                continue  # Skip this download type but continue with others

            # Extract response data
            zip_content = download_response.get("content")
            filename = download_response.get("filename", f"{run_name}_{download_type}.zip")
            file_count = download_response.get("file_count", 0)
            zip_size = download_response.get("size", 0)

            if not zip_content or file_count == 0:
                logger.info(f"📭 No {download_type} files to download for run: {run_name}{trial_info}")
                continue

            # Decode base64 content
            try:
                decoded_content = base64.b64decode(zip_content)
                logger.info(f"✅ Downloaded {download_type}{trial_info}: {file_count} files, {zip_size} bytes")
            except Exception as e:
                logger.error(f"❌ Failed to decode base64 {download_type} content{trial_info}: {e}")
                continue

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
                    logger.info(f"📂 Extracted {len(extracted_files)} {download_type} files to: {local_path}")

                success_count += 1

            except Exception as e:
                logger.error(f"❌ Failed to extract {download_type} zip: {e}")
                continue
            finally:
                # Clean up temporary zip file
                try:
                    os.unlink(temp_zip_path)
                except:
                    pass

        # Consider success if we downloaded at least one type successfully
        if success_count > 0:
            logger.info(f"running download_directory_via_runpod_api{trial_info} ... Multi-part download completed successfully")
            logger.info(f"running download_directory_via_runpod_api{trial_info} ... Success rate: {success_count}/{len(download_types)} download types")
            if worker_id:
                logger.info(f"running download_directory_via_runpod_api{trial_info} ... Worker pinning successful - all downloads from worker {worker_id}")
            return True
        else:
            logger.error(f"running download_directory_via_runpod_api{trial_info} ... All download types failed for run: {run_name}")
            if worker_id:
                logger.error(f"running download_directory_via_runpod_api{trial_info} ... Worker pinning failed - downloads failed even with specific worker {worker_id}")
            else:
                logger.error(f"running download_directory_via_runpod_api{trial_info} ... Load balancer routing failed - consider implementing worker pinning")
            return False

    except Exception as e:
        logger.error(f"running download_directory_via_runpod_api{trial_info} ... Exception during directory download: {e}")
        if worker_id:
            logger.error(f"running download_directory_via_runpod_api{trial_info} ... Worker pinning exception with worker {worker_id}")
        else:
            logger.error(f"running download_directory_via_runpod_api{trial_info} ... Load balancer exception - no worker pinning used")
        return False


def download_directory_multipart_via_runpod_api(
    api_url_runpod: str,
    api_key_runpod: str,
    run_name: str,
    local_dir: str,
    timeout: int = 300,
    trial_id: Optional[str] = None,
    trial_number: Optional[int] = None,
    worker_id: Optional[str] = None,
    max_part_size_mb: int = 8
) -> bool:
    """
    Download entire directory from RunPod worker using multi-part single response to avoid worker isolation issues.
    Makes a single request that returns all file types in separate parts within the response.
    This solves the worker isolation problem where multiple requests might hit different workers.

    Args:
        api_url_runpod: RunPod API URL (e.g., https://api.runpod.ai/v2/endpoint_id/run)
        api_key_runpod: RunPod API key for authentication
        run_name: Run name to identify directory
        local_dir: Local directory to extract files to
        timeout: Timeout in seconds for the request
        trial_id: Optional trial ID for logging
        trial_number: Optional trial number for logging
        worker_id: Optional worker ID for pinning requests to specific worker
        max_part_size_mb: Maximum size in MB for each part (sent to server as hint)

    Returns:
        bool: True if download succeeded, False otherwise
    """
    # Format trial information for logs (initialize before try block for exception handler)
    trial_info = ""
    if trial_number is not None:
        trial_info = f" (trial_{trial_number}"
        if trial_id:
            trial_info += f", {trial_id}"
        trial_info += ")"

    try:
        logger.info(f"running download_directory_multipart_via_runpod_api{trial_info} ... Starting multipart download for run: {run_name}")

        # Use load-balanced endpoint directly (worker-specific URLs consistently fail with 502 errors)
        api_url = api_url_runpod
        logger.info(f"running download_directory_multipart_via_runpod_api{trial_info} ... Using load-balanced endpoint with multipart approach to avoid worker isolation")
        logger.info(f"running download_directory_multipart_via_runpod_api{trial_info} ... Load balancer endpoint: {api_url}")
        if worker_id:
            logger.info(f"running download_directory_multipart_via_runpod_api{trial_info} ... Worker ID {worker_id} available but skipping worker-specific URL (502 errors)")
        else:
            logger.info(f"running download_directory_multipart_via_runpod_api{trial_info} ... No worker ID provided - using load-balanced multipart approach")

        # Create local directory if it doesn't exist
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"📦 Starting multipart download for run: {run_name}{trial_info}")
        logger.info(f"running download_directory_multipart_via_runpod_api{trial_info} ... Calling RunPod API for multipart download")
        logger.info(f"running download_directory_multipart_via_runpod_api{trial_info} ... Using load-balanced endpoint for multipart download: {api_url}")

        # Make single multipart request
        download_response = _call_runpod_api(
            api_url_runpod=api_url,
            api_key_runpod=api_key_runpod,
            command="download_directory_multipart",
            run_name=run_name,
            trial_id=trial_id,
            trial_number=trial_number,
            max_part_size_mb=max_part_size_mb,
            timeout=timeout
        )

        logger.info(f"running download_directory_multipart_via_runpod_api{trial_info} ... RunPod API call completed for multipart download")

        # Extract download worker ID for definitive worker isolation analysis
        download_worker_id = download_response.get("download_worker_id", "unknown_download_worker") if download_response else "no_response"

        # Log definitive worker comparison - this is the smoking gun evidence
        if worker_id and download_worker_id:
            if worker_id == download_worker_id:
                logger.info(f"🏗️ WORKER_ISOLATION_TRACKING: ✅ SAME WORKER - Training worker_id={worker_id} == Download worker_id={download_worker_id}")
            else:
                logger.error(f"🏗️ WORKER_ISOLATION_TRACKING: ❌ WORKER MISMATCH - Training worker_id={worker_id} != Download worker_id={download_worker_id}")
                logger.error(f"🏗️ WORKER_ISOLATION_TRACKING: DEFINITIVE PROOF of worker isolation issue")
        else:
            logger.warning(f"🏗️ WORKER_ISOLATION_TRACKING: ⚠️ Missing worker IDs - training={worker_id}, download={download_worker_id}")

        if not download_response or download_response.get("error"):
            error_msg = download_response.get('error', 'Unknown error') if download_response else 'No response from RunPod'
            logger.warning(f"⚠️ Failed to download multipart{trial_info}: {error_msg}")
            return False

        # Process multipart response
        parts = download_response.get("parts", [])
        total_parts = download_response.get("total_parts", 0)
        total_file_count = download_response.get("total_file_count", 0)
        total_size = download_response.get("total_size", 0)

        if not parts or total_parts == 0:
            logger.info(f"📭 No parts to download for run: {run_name}{trial_info}")
            return True

        logger.info(f"📦 Processing {total_parts} parts for multipart download{trial_info}: {total_file_count} total files, {total_size} total bytes")

        # Extract each part
        import tempfile
        import zipfile
        import base64

        successful_parts = 0
        for i, part in enumerate(parts):
            part_type = part.get("type", f"part_{i}")
            zip_content = part.get("content")
            filename = part.get("filename", f"{run_name}_{part_type}.zip")
            file_count = part.get("file_count", 0)
            part_size = part.get("size", 0)

            if not zip_content or file_count == 0:
                logger.info(f"📭 No {part_type} files in part {i+1}/{total_parts}{trial_info}")
                continue

            # Decode base64 content
            try:
                decoded_content = base64.b64decode(zip_content)
                logger.info(f"✅ Downloaded part {i+1}/{total_parts} ({part_type}){trial_info}: {file_count} files, {part_size} bytes")
            except Exception as e:
                logger.error(f"❌ Failed to decode base64 content for part {i+1}/{total_parts} ({part_type}){trial_info}: {e}")
                continue

            # Save zip temporarily and extract
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_zip:
                tmp_zip.write(decoded_content)
                temp_zip_path = tmp_zip.name

            try:
                # Extract zip to local directory
                with zipfile.ZipFile(temp_zip_path, 'r') as zipf:
                    zipf.extractall(local_path)
                    extracted_files = zipf.namelist()
                    logger.info(f"📂 Extracted {len(extracted_files)} {part_type} files from part {i+1}/{total_parts} to: {local_path}")

                successful_parts += 1

            except Exception as e:
                logger.error(f"❌ Failed to extract part {i+1}/{total_parts} ({part_type}){trial_info}: {e}")
                continue
            finally:
                # Clean up temporary zip file
                try:
                    import os
                    os.unlink(temp_zip_path)
                except Exception as cleanup_e:
                    logger.warning(f"⚠️ Failed to cleanup temporary zip file: {cleanup_e}")

        # Report results
        if successful_parts == total_parts:
            logger.info(f"✅ Multipart download completed successfully{trial_info}: {successful_parts}/{total_parts} parts")
            logger.info(f"running download_directory_multipart_via_runpod_api{trial_info} ... Multipart download completed successfully")
            logger.info(f"running download_directory_multipart_via_runpod_api{trial_info} ... Success rate: {successful_parts}/{total_parts} parts")
            logger.info(f"running download_directory_multipart_via_runpod_api{trial_info} ... Load-balanced multipart approach successful - worker isolation avoided")
            return True
        elif successful_parts > 0:
            logger.warning(f"⚠️ Partial multipart download{trial_info}: {successful_parts}/{total_parts} parts succeeded")
            logger.warning(f"running download_directory_multipart_via_runpod_api{trial_info} ... Partial success rate: {successful_parts}/{total_parts} parts")
            return True  # Consider partial success as success since some files were downloaded
        else:
            logger.error(f"❌ Multipart download failed{trial_info}: 0/{total_parts} parts succeeded")
            logger.error(f"running download_directory_multipart_via_runpod_api{trial_info} ... All parts failed for run: {run_name}")
            logger.error(f"running download_directory_multipart_via_runpod_api{trial_info} ... Load-balanced multipart approach failed")
            return False

    except Exception as e:
        logger.error(f"running download_directory_multipart_via_runpod_api{trial_info} ... Exception during multipart download: {e}")
        logger.error(f"running download_directory_multipart_via_runpod_api{trial_info} ... Load-balanced multipart approach exception")
        return False